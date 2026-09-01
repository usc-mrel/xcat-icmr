"""One-frame tissue/Gd NUFFT linearity validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import TYPE_CHECKING

import numpy as np
from scipy.io import loadmat

from xcat_icmr.coils import (
    inspect_sensitivity_map,
    load_normalized_coil_in_logical_frame,
    prepare_rss_normalization,
    sensitivity_shape_in_logical_frame,
)
from xcat_icmr.encoding import (
    SigpyNufftBackend,
    circular_shift_to_rf_center,
    encode_multicoil_frame,
    prepare_contrast_for_encoding,
    prepare_encoding_grids,
    scale_isotropic_trajectory_to_resolution,
)
from xcat_icmr.encoding.validation import _save_shifted_ground_truth
from xcat_icmr.intervention.balloon import rasterize_sparse_balloon
from xcat_icmr.intervention.debug import _atomic_savemat, _blood_properties
from xcat_icmr.intervention.gd_signal import (
    calculate_sparse_gd_bssfp_signal,
    sample_sparse_flip_angles,
)
from xcat_icmr.intervention.path import (
    interpolate_cubic_arc_length,
    load_balloon_path,
)
from xcat_icmr.phantom import plan_xcat_frames, xcat_label_shape
from xcat_icmr.sequence import build_coordinate_transforms, read_sequence
from xcat_icmr.signal import (
    generate_rf_profile_bssfp_contrast,
    generate_slice_profile,
    read_pulseq_excitation,
)
from xcat_icmr.tissue import get_tissue_library

if TYPE_CHECKING:
    from xcat_icmr.config.models import SimulationConfig


class BalloonEncodingDebugError(ValueError):
    """Raised when the one-frame balloon encoding cannot be validated."""


@dataclass(frozen=True)
class BalloonEncodingDebugResult:
    """Saved all-coil tissue/delta k-space and one-coil linearity metrics."""

    output_path: Path
    shifted_combined_path: Path
    tissue_path: Path
    delta_path: Path
    profile_path: Path
    kspace_shape: tuple[int, int, int]
    adjoint_shape: tuple[int, int, int]
    forward_relative_error: float
    adjoint_relative_error: float
    elapsed_s: float
    nonfinite_value_count: int


def _relative_l2(first: np.ndarray, second: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(second.reshape(-1))), 1e-30)
    return float(np.linalg.norm((first - second).reshape(-1)) / denominator)


def _load_image(path: Path) -> np.ndarray:
    try:
        content = loadmat(path, variable_names=["image"], squeeze_me=False)
    except (OSError, ValueError, NotImplementedError) as exc:
        raise BalloonEncodingDebugError(f"could not read {path}: {exc}") from exc
    if "image" not in content:
        raise BalloonEncodingDebugError(f"{path} has no 'image' variable")
    image = np.asarray(content["image"], dtype=np.float32)
    if image.ndim != 3 or not np.all(np.isfinite(image)):
        raise BalloonEncodingDebugError(f"{path} is not one finite 3-D image")
    return image


def validate_balloon_kspace_linearity(
    config: "SimulationConfig",
    *,
    progress=None,
) -> BalloonEncodingDebugResult:
    """Encode tissue and a point-A Gd delta, then verify NUFFT linearity."""

    started = time.perf_counter()
    balloon = config.intervention.gd_balloon
    if not balloon.enabled or balloon.path.control_points_file is None:
        raise BalloonEncodingDebugError("an enabled Gd balloon path is required")
    if balloon.composition.mode != "replace-background":
        raise BalloonEncodingDebugError(
            "validation requires composition.mode='replace-background'"
        )
    if not config.coils.enabled or config.coils.sensitivity_map is None:
        raise BalloonEncodingDebugError("an enabled sensitivity map is required")
    if not config.coils.normalize:
        raise BalloonEncodingDebugError("coils.normalize must be true")

    sequence = read_sequence(config.sequence)
    transforms = build_coordinate_transforms(
        patient_position=config.phantom.patient_position,
        coordinate_mode=config.sequence.coordinate_mode,
        sequence_orientation=config.sequence.orientation,
    )
    coil_info = inspect_sensitivity_map(config.coils.sensitivity_map)
    logical_shape = sensitivity_shape_in_logical_frame(
        coil_info,
        stored_axis_order=config.coils.axis_order,
        dcs_to_logical=transforms.dcs_to_logical,
    )
    excitation = read_pulseq_excitation(sequence.sequence_path)
    pcs_voxel = np.asarray(config.phantom.voxel_size_mm, dtype=np.float64)
    logical_voxel = np.abs(transforms.pcs_to_logical) @ pcs_voxel
    shift_mm = float(config.sequence.rf_profile.center_shift_mm)
    shift_tag = f"{shift_mm:g}".replace("-", "m").replace(".", "p")
    diameter_tag = "x".join(
        f"{float(value):g}".replace(".", "p")
        for value in balloon.geometry.diameter_mm
    )

    debug_directory = config.run.output_root / "intervention" / "balloon_debug"
    variant_directory = debug_directory / f"rf_shift_{shift_tag}mm"
    profile_path = variant_directory / f"rf_slice_profile_{shift_tag}mm.mat"
    tissue_path = variant_directory / (
        f"tissue_frame_0001_rf_shift_{shift_tag}mm.mat"
    )
    delta_path = variant_directory / (
        f"gd_delta_frame_0001_A_rf_shift_{shift_tag}mm_"
        f"diameter_{diameter_tag}mm.mat"
    )

    profile = generate_slice_profile(
        excitation,
        matrix_size=logical_shape[excitation.logical_axis],
        voxel_size_mm=float(logical_voxel[excitation.logical_axis]),
        center_shift_mm=shift_mm,
    )
    frame = plan_xcat_frames(config, debug_one_frame=False).frames[0]
    if frame.label_path is None:
        raise BalloonEncodingDebugError("XCAT frame 1 has no label path")
    generate_rf_profile_bssfp_contrast(
        label_path=frame.label_path,
        profile=profile,
        transforms=transforms,
        pcs_voxel_size_mm=config.phantom.voxel_size_mm,
        library=get_tissue_library(config.sequence.contrast.tissue_library),
        te_ms=sequence.te_ms,
        tr_ms=sequence.tr_ms,
        profile_output_path=profile_path,
        image_output_path=tissue_path,
        overwrite=True,
    )
    tissue_pcs = _load_image(tissue_path)
    expected_shape = xcat_label_shape(config)
    if tissue_pcs.shape != expected_shape:
        raise BalloonEncodingDebugError(
            f"tissue shape {tissue_pcs.shape} != {expected_shape}"
        )

    loaded_path = load_balloon_path(
        balloon.path.control_points_file,
        coordinate_system=balloon.path.coordinate_system,
    )
    path = interpolate_cubic_arc_length(loaded_path.control_points_lps_mm)
    center = path.positions_at_distances_mm(np.asarray([0.0]))[0]
    support = rasterize_sparse_balloon(
        center,
        volume_shape=expected_shape,
        voxel_size_mm=config.phantom.voxel_size_mm,
        diameter_mm=balloon.geometry.diameter_mm,
        shape=balloon.geometry.shape,
    )
    gd = calculate_sparse_gd_bssfp_signal(
        support,
        carrier=_blood_properties(config),
        concentration_mM=balloon.contrast_agent.concentration.value_mM,
        flip_angle_deg=sample_sparse_flip_angles(profile_path, support),
        te_ms=sequence.te_ms,
        tr_ms=sequence.tr_ms,
        relaxivity_library=balloon.contrast_agent.relaxivity_library,
    )
    delta_pcs = np.zeros(expected_shape, dtype=np.float32)
    slices = support.bounding_box_slices
    delta_pcs[slices] = np.asarray(
        support.occupancy * (gd.values - tissue_pcs[slices]),
        dtype=np.float32,
    )
    _atomic_savemat(
        delta_path,
        {
            "image": delta_pcs,
            "center_lps_mm": np.asarray(center[None, :], dtype=np.float32),
            "bounding_box_start_ijk_zero_based": np.asarray(
                support.bounding_box_start_ijk[None, :], dtype=np.int32
            ),
            "occupied_volume_mm3": np.asarray(
                [[support.occupied_volume_mm3]], dtype=np.float32
            ),
        },
    )

    prepare_kwargs = {
        "target_shape": logical_shape,
        "source_to_target": transforms.pcs_to_logical,
        "source_frame": "XCAT PCS [Sag, Cor, Tra]",
        "target_frame": "Pulseq logical [x, y, z]",
        "target_axis_patient_directions": (
            transforms.logical_axis_patient_directions
        ),
    }
    tissue = prepare_contrast_for_encoding(tissue_path, **prepare_kwargs).image
    delta = prepare_contrast_for_encoding(delta_path, **prepare_kwargs).image

    resolution_values = np.asarray(sequence.resolution_mm, dtype=np.float64).reshape(-1)
    if resolution_values.size != 1:
        raise BalloonEncodingDebugError("isotropic sequence resolution is required")
    scaled_k, scale_factor, target_kmax = scale_isotropic_trajectory_to_resolution(
        sequence.kx,
        sequence.ky,
        sequence.kz,
        resolution_mm=float(resolution_values[0]),
    )
    grids = prepare_encoding_grids(
        ground_truth_shape=tissue.shape,
        ground_truth_voxel_size_mm=config.phantom.voxel_size_mm,
        sequence_resolution_mm=sequence.resolution_mm,
    )
    normalization = prepare_rss_normalization(
        coil_info,
        config.run.output_root / "kspace" / "cache" / "sensitivity_rss.npy",
    )

    def load_coil(index: int) -> np.ndarray:
        return load_normalized_coil_in_logical_frame(
            coil_info,
            index,
            normalization,
            stored_axis_order=config.coils.axis_order,
            dcs_to_logical=transforms.dcs_to_logical,
        )

    common = dict(
        coil_count=coil_info.coil_count,
        coil_loader=load_coil,
        kx_per_m=scaled_k[0],
        ky_per_m=scaled_k[1],
        kz_per_m=scaled_k[2],
        density_compensation=sequence.density_compensation,
        encoding_grids=grids,
        rf_center_shift_mm=shift_mm,
        rf_axis_voxel_size_mm=float(logical_voxel[excitation.logical_axis]),
        rf_logical_axis=excitation.logical_axis,
        device_id=config.compute.device_id,
        compute_adjoint=True,
    )
    tissue_encoding = encode_multicoil_frame(
        tissue,
        progress=(lambda done, total: progress("tissue", done, total))
        if progress is not None
        else None,
        **common,
    )
    delta_encoding = encode_multicoil_frame(
        delta,
        progress=(lambda done, total: progress("Gd delta", done, total))
        if progress is not None
        else None,
        **common,
    )
    if tissue_encoding.adjoint_coils is None or delta_encoding.adjoint_coils is None:
        raise BalloonEncodingDebugError("requested adjoints were not generated")

    combined_kspace = np.asarray(
        tissue_encoding.kspace + delta_encoding.kspace, dtype=np.complex64
    )
    combined_adjoint_coils = np.asarray(
        tissue_encoding.adjoint_coils + delta_encoding.adjoint_coils,
        dtype=np.complex64,
    )
    combined_adjoint_rss = np.sqrt(
        np.sum(np.abs(combined_adjoint_coils) ** 2, axis=3, dtype=np.float64)
    ).astype(np.float32)

    coil0 = load_coil(0)
    combined_coil0, applied_shift = circular_shift_to_rf_center(
        np.asarray((tissue + delta) * coil0, dtype=np.complex64),
        center_shift_mm=shift_mm,
        voxel_size_mm=float(logical_voxel[excitation.logical_axis]),
        logical_axis=excitation.logical_axis,
    )
    backend = SigpyNufftBackend(device_id=config.compute.device_id)
    direct_flat = backend.forward(combined_coil0, tissue_encoding.trajectory)
    summed_flat = combined_kspace[:, :, 0].T.reshape(-1)
    forward_error = _relative_l2(direct_flat, summed_flat)
    normalized_dcf = tissue_encoding.normalized_dcf.T.reshape(-1)
    direct_adjoint = backend.adjoint(
        direct_flat * normalized_dcf, tissue_encoding.trajectory
    )
    summed_adjoint = combined_adjoint_coils[:, :, :, 0]
    adjoint_error = _relative_l2(direct_adjoint, summed_adjoint)

    shifted_combined = np.asarray(
        tissue_encoding.shifted_ground_truth
        + delta_encoding.shifted_ground_truth,
        dtype=np.float32,
    )
    shifted_combined_path = (
        config.run.output_root
        / "kspace"
        / "debug"
        / (
            f"shifted_high_resolution_gt_frame_0001_gd_A_"
            f"rf_shift_{shift_tag}mm_diameter_{diameter_tag}mm.mat"
        )
    )
    _save_shifted_ground_truth(
        shifted_combined,
        shifted_combined_path,
        center_shift_mm=shift_mm,
        logical_axis=excitation.logical_axis,
        applied_shift_voxels=applied_shift,
    )

    output_path = (
        config.run.output_root
        / "kspace"
        / "debug"
        / (
            f"gd_balloon_linearity_frame_0001_rf_shift_{shift_tag}mm_"
            f"diameter_{diameter_tag}mm.mat"
        )
    )
    nonfinite = sum(
        int(np.count_nonzero(~np.isfinite(values)))
        for values in (
            tissue_encoding.kspace,
            delta_encoding.kspace,
            combined_kspace,
            tissue_encoding.adjoint_rss,
            delta_encoding.adjoint_rss,
            combined_adjoint_rss,
            direct_adjoint,
        )
    )
    _atomic_savemat(
        output_path,
        {
            "tissue_kspace": tissue_encoding.kspace,
            "gd_delta_kspace": delta_encoding.kspace,
            "combined_kspace": combined_kspace,
            "tissue_adjoint_rss": tissue_encoding.adjoint_rss,
            "gd_delta_adjoint_rss": delta_encoding.adjoint_rss,
            "combined_adjoint_rss": combined_adjoint_rss,
            "direct_combined_coil0_kspace": tissue_encoding.trajectory.reshape_kspace(
                direct_flat
            ),
            "summed_combined_coil0_kspace": combined_kspace[:, :, 0],
            "direct_combined_coil0_adjoint": direct_adjoint,
            "summed_combined_coil0_adjoint": summed_adjoint,
            "forward_relative_l2_error": np.asarray([[forward_error]]),
            "adjoint_relative_l2_error": np.asarray([[adjoint_error]]),
            "coordinates": tissue_encoding.trajectory.coordinates,
            "dcf_normalized": tissue_encoding.normalized_dcf,
            "fov_mm": np.asarray([grids.acquisition_fov_mm]),
            "matrix_shape": np.asarray(
                [grids.acquisition_matrix_shape], dtype=np.int32
            ),
            "logical_axis_patient_directions": np.asarray(
                transforms.logical_axis_patient_directions, dtype=object
            ),
            "rf_center_shift_mm": np.asarray([[shift_mm]], dtype=np.float32),
            "applied_circular_shift_voxels": np.asarray(
                [[applied_shift]], dtype=np.int32
            ),
            "balloon_center_lps_mm": np.asarray(center[None, :], dtype=np.float32),
            "balloon_occupied_volume_mm3": np.asarray(
                [[support.occupied_volume_mm3]], dtype=np.float32
            ),
            "trajectory_scale_factor": np.asarray([[scale_factor]]),
            "target_kmax_per_m": np.asarray([[target_kmax]]),
            "nufft_oversampling": np.asarray([[backend.oversampling]]),
            "nufft_kernel_width": np.asarray([[backend.kernel_width]]),
            "nonfinite_value_count": np.asarray([[nonfinite]], dtype=np.int64),
            "elapsed_s": np.asarray([[time.perf_counter() - started]]),
        },
    )
    return BalloonEncodingDebugResult(
        output_path=output_path,
        shifted_combined_path=shifted_combined_path,
        tissue_path=tissue_path,
        delta_path=delta_path,
        profile_path=profile_path,
        kspace_shape=combined_kspace.shape,
        adjoint_shape=combined_adjoint_rss.shape,
        forward_relative_error=forward_error,
        adjoint_relative_error=adjoint_error,
        elapsed_s=time.perf_counter() - started,
        nonfinite_value_count=nonfinite,
    )


def format_balloon_encoding_debug(result: BalloonEncodingDebugResult) -> str:
    """Format the one-frame sparse Gd linearity result."""

    return "\n".join(
        (
            "One-frame tissue + sparse Gd NUFFT validation",
            f"K-space shape:       {result.kspace_shape}",
            f"Adjoint shape:       {result.adjoint_shape}",
            f"Forward relative L2: {result.forward_relative_error:.6g}",
            f"Adjoint relative L2: {result.adjoint_relative_error:.6g}",
            f"Non-finite values:   {result.nonfinite_value_count}",
            f"Elapsed:             {result.elapsed_s:.3f} s",
            f"Output:              {result.output_path}",
            f"Shifted combined GT: {result.shifted_combined_path}",
            f"Tissue input:        {result.tissue_path}",
            f"Sparse delta input:  {result.delta_path}",
            f"RF profile:          {result.profile_path}",
            "Overall:             "
            + (
                "PASS"
                if result.nonfinite_value_count == 0
                and result.forward_relative_error < 1e-5
                and result.adjoint_relative_error < 1e-5
                else "FAIL"
            ),
        )
    )
