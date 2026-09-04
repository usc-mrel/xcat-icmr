"""Three-position catheter alignment diagnostic on the image-reference grid."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import TYPE_CHECKING, Callable

import h5py
import numpy as np
from scipy.io import loadmat

from xcat_icmr.cache import (
    contrast_frame_path,
    contrast_profile_path,
    fullysampled_reference_cache_entry,
)
from xcat_icmr.coils import (
    inspect_sensitivity_map,
    load_normalized_coil_in_logical_frame,
    prepare_rss_normalization,
)
from xcat_icmr.encoding.fullysampled_reference import (
    _atomic_savemat,
    _resample_complex,
    centered_resize,
    reconstruction_shape_for_trajectory,
)
from xcat_icmr.encoding.inputs import load_contrast_image
from xcat_icmr.encoding.sigpy_backend import SigpyNufftBackend
from xcat_icmr.encoding.trajectory import (
    prepare_physical_sigpy_trajectory,
    scale_isotropic_trajectory_to_resolution,
)
from xcat_icmr.intervention.balloon import (
    centered_origin_lps_mm,
    rasterize_sparse_balloon,
)
from xcat_icmr.intervention.debug import _blood_properties
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
from xcat_icmr.sequence.orientation import (
    map_spatial_indices,
    reorient_spatial_array,
    reoriented_spatial_shape,
)

if TYPE_CHECKING:
    from xcat_icmr.config.models import SimulationConfig


class ThreePositionReferenceDebugError(ValueError):
    """Raised when the three-position alignment diagnostic cannot be made."""


@dataclass(frozen=True)
class ThreePositionReferenceDebugResult:
    output_path: Path
    path_length_mm: float
    centers_lps_mm: np.ndarray
    center_labels: np.ndarray
    high_resolution_shape: tuple[int, int, int]
    reconstruction_shape: tuple[int, int, int]
    elapsed_s: float


def map_pcs_centers_to_reference_grids(
    center_ijk: np.ndarray,
    *,
    pcs_shape: tuple[int, int, int],
    pcs_to_logical: np.ndarray,
    high_resolution_shape: tuple[int, int, int],
    high_resolution_voxel_mm: tuple[float, float, float],
    reconstruction_shape: tuple[int, int, int],
    reconstruction_voxel_mm: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Map PCS center samples through reorientation, crop, and low-res grid."""

    oriented, valid = map_spatial_indices(
        center_ijk,
        source_shape=pcs_shape,
        source_to_target=pcs_to_logical,
    )
    oriented_shape = np.asarray(
        reoriented_spatial_shape(pcs_shape, pcs_to_logical), dtype=np.int64
    )
    high_shape = np.asarray(high_resolution_shape, dtype=np.int64)
    high = np.asarray(oriented, dtype=np.int64)
    high += (high_shape // 2 - oriented_shape // 2)[None, :]
    valid &= np.all(high >= 0, axis=1) & np.all(high < high_shape[None, :], axis=1)
    if not np.all(valid):
        raise ThreePositionReferenceDebugError(
            "one or more catheter positions fall outside the target FOV"
        )

    high_voxel = np.asarray(high_resolution_voxel_mm, dtype=np.float64)
    low_shape = np.asarray(reconstruction_shape, dtype=np.int64)
    low_voxel = np.asarray(reconstruction_voxel_mm, dtype=np.float64)
    centered_mm = (high - high_shape[None, :] // 2) * high_voxel[None, :]
    low = np.rint(centered_mm / low_voxel[None, :]).astype(np.int64)
    low += low_shape[None, :] // 2
    if np.any(low < 0) or np.any(low >= low_shape[None, :]):
        raise ThreePositionReferenceDebugError(
            "one or more catheter positions fall outside the reconstruction grid"
        )
    return np.asarray(high, dtype=np.int32), np.asarray(low, dtype=np.int32)


def _load_frame_one_labels(config: "SimulationConfig") -> np.ndarray:
    frame = plan_xcat_frames(config, debug_one_frame=False).frames[0]
    if frame.label_path is None or not frame.label_path.is_file():
        raise ThreePositionReferenceDebugError("XCAT label frame 1 is unavailable")
    try:
        content = loadmat(frame.label_path, variable_names=["P"], squeeze_me=False)
    except (OSError, ValueError, NotImplementedError) as exc:
        raise ThreePositionReferenceDebugError(
            f"could not load label frame 1: {exc}"
        ) from exc
    if "P" not in content:
        raise ThreePositionReferenceDebugError("label frame 1 has no 'P' variable")
    labels = np.asarray(content["P"], dtype=np.uint16)
    if labels.shape != xcat_label_shape(config):
        raise ThreePositionReferenceDebugError("label frame 1 has the wrong shape")
    return labels


def generate_three_position_reference_debug(
    config: "SimulationConfig",
    *,
    overwrite: bool = False,
    progress: Callable[[str], None] | None = None,
) -> ThreePositionReferenceDebugResult:
    """Add A/mid/B balloons to frame 1 and encode their signal difference."""

    started = time.perf_counter()
    balloon = config.intervention.gd_balloon
    if not balloon.enabled or balloon.path.control_points_file is None:
        raise ThreePositionReferenceDebugError("an enabled Gd balloon path is required")
    if balloon.composition.mode != "replace-background":
        raise ThreePositionReferenceDebugError(
            "composition.mode must be 'replace-background'"
        )
    if not config.coils.enabled or config.coils.sensitivity_map is None:
        raise ThreePositionReferenceDebugError("an enabled sensitivity map is required")
    if not config.coils.normalize:
        raise ThreePositionReferenceDebugError("coils.normalize must be true")

    cache_entry = fullysampled_reference_cache_entry(config)
    reference_path = cache_entry.directory / "fullysampled_reference_4d.h5"
    output_path = cache_entry.directory / (
        "fullysampled_reference_frame_0001_three_cath_positions_debug.mat"
    )
    if output_path.exists() and not overwrite:
        raise ThreePositionReferenceDebugError(
            f"output already exists: {output_path}; pass --overwrite"
        )
    if not reference_path.is_file():
        raise ThreePositionReferenceDebugError(
            f"fully sampled tissue reference is unavailable: {reference_path}"
        )
    with h5py.File(reference_path, "r") as handle:
        if "image" not in handle or "frame_complete" not in handle:
            raise ThreePositionReferenceDebugError("reference HDF5 is incomplete")
        if not bool(handle["frame_complete"][0]):
            raise ThreePositionReferenceDebugError("reference frame 1 is not complete")
        tissue_reference = np.asarray(handle["image"][:, :, :, 0], dtype=np.complex64)
        image_attributes = dict(handle["image"].attrs)
    try:
        intensity_scale = float(image_attributes["adjoint_intensity_scale"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ThreePositionReferenceDebugError(
            "reference cache has no valid NUFFT intensity scale"
        ) from exc
    if not np.isfinite(intensity_scale) or intensity_scale <= 0:
        raise ThreePositionReferenceDebugError("invalid NUFFT intensity scale")

    sequence = read_sequence(config.sequence)
    resolution_values = np.asarray(sequence.resolution_mm, dtype=np.float64).reshape(-1)
    if resolution_values.size != 1:
        raise ThreePositionReferenceDebugError("isotropic sequence resolution is required")
    resolution_mm = float(resolution_values[0])
    scaled_k, scale_factor, target_kmax = scale_isotropic_trajectory_to_resolution(
        sequence.kx, sequence.ky, sequence.kz, resolution_mm=resolution_mm
    )
    target_fov = tuple(float(value) for value in config.encoding.target_fov_mm)
    maximum_k = tuple(float(np.max(np.abs(values))) for values in scaled_k)
    reconstruction_shape = reconstruction_shape_for_trajectory(
        target_fov, resolution_mm, maximum_k
    )
    if tissue_reference.shape != reconstruction_shape:
        raise ThreePositionReferenceDebugError(
            "cached frame shape does not match the current sequence/FOV"
        )
    reconstruction_voxel = tuple(
        fov / size for fov, size in zip(target_fov, reconstruction_shape, strict=True)
    )

    transforms = build_coordinate_transforms(
        patient_position=config.phantom.patient_position,
        coordinate_mode=config.sequence.coordinate_mode,
        sequence_orientation=config.sequence.orientation,
    )
    pcs_voxel = np.asarray(config.phantom.voxel_size_mm, dtype=np.float64)
    logical_voxel = np.abs(transforms.pcs_to_logical) @ pcs_voxel
    high_shape_values = np.asarray(target_fov) / logical_voxel
    high_shape_array = np.rint(high_shape_values).astype(np.int64)
    if not np.allclose(high_shape_values, high_shape_array, atol=1e-6, rtol=0.0):
        raise ThreePositionReferenceDebugError(
            "target FOV is not an integer number of high-resolution voxels"
        )
    high_shape = tuple(int(value) for value in high_shape_array)

    _, tissue_pcs = load_contrast_image(contrast_frame_path(config, 1))
    pcs_shape = xcat_label_shape(config)
    if tissue_pcs.shape != pcs_shape:
        raise ThreePositionReferenceDebugError("contrast frame 1 has the wrong shape")
    labels = _load_frame_one_labels(config)

    loaded_path = load_balloon_path(
        balloon.path.control_points_file,
        coordinate_system=balloon.path.coordinate_system,
    )
    path = interpolate_cubic_arc_length(loaded_path.control_points_lps_mm)
    distances = np.asarray([0.0, 0.5, 1.0], dtype=np.float64) * path.total_length_mm
    centers = path.positions_at_distances_mm(distances)
    origin = centered_origin_lps_mm(pcs_shape, config.phantom.voxel_size_mm)
    center_ijk_float = (centers - origin[None, :]) / pcs_voxel[None, :]
    center_ijk = np.rint(center_ijk_float).astype(np.int32)
    if np.any(center_ijk < 0) or np.any(center_ijk >= np.asarray(pcs_shape)[None, :]):
        raise ThreePositionReferenceDebugError("a catheter center lies outside frame 1")
    center_labels = labels[tuple(center_ijk.T)]

    delta_pcs = np.zeros(pcs_shape, dtype=np.float32)
    occupancy_sum = np.zeros(pcs_shape, dtype=np.float32)
    occupied_volumes = np.empty(3, dtype=np.float32)
    gd_t1_ms = np.empty(3, dtype=np.float32)
    gd_t2_ms = np.empty(3, dtype=np.float32)
    for index, center in enumerate(centers):
        support = rasterize_sparse_balloon(
            center,
            volume_shape=pcs_shape,
            voxel_size_mm=config.phantom.voxel_size_mm,
            diameter_mm=balloon.geometry.diameter_mm,
            shape=balloon.geometry.shape,
        )
        gd = calculate_sparse_gd_bssfp_signal(
            support,
            carrier=_blood_properties(config),
            concentration_mM=balloon.contrast_agent.concentration.value_mM,
            flip_angle_deg=sample_sparse_flip_angles(
                contrast_profile_path(config), support
            ),
            te_ms=sequence.te_ms,
            tr_ms=sequence.tr_ms,
            relaxivity_library=balloon.contrast_agent.relaxivity_library,
        )
        slices = support.bounding_box_slices
        if np.any(occupancy_sum[slices] + support.occupancy > 1.0 + 1e-6):
            raise ThreePositionReferenceDebugError(
                "the three debug balloons overlap; cannot compose independently"
            )
        delta_pcs[slices] += np.asarray(
            support.occupancy * (gd.values - tissue_pcs[slices]), dtype=np.float32
        )
        occupancy_sum[slices] += support.occupancy
        occupied_volumes[index] = support.occupied_volume_mm3
        gd_t1_ms[index] = gd.t1_ms
        gd_t2_ms[index] = gd.t2_ms

    combined_pcs = np.asarray(tissue_pcs + delta_pcs, dtype=np.float32)
    high_delta = centered_resize(
        reorient_spatial_array(delta_pcs, transforms.pcs_to_logical), high_shape
    ).astype(np.float32, copy=False)
    high_combined = centered_resize(
        reorient_spatial_array(combined_pcs, transforms.pcs_to_logical), high_shape
    ).astype(np.float32, copy=False)
    high_indices, low_indices = map_pcs_centers_to_reference_grids(
        center_ijk,
        pcs_shape=pcs_shape,
        pcs_to_logical=transforms.pcs_to_logical,
        high_resolution_shape=high_shape,
        high_resolution_voxel_mm=tuple(float(value) for value in logical_voxel),
        reconstruction_shape=reconstruction_shape,
        reconstruction_voxel_mm=reconstruction_voxel,
    )
    high_markers = np.zeros(high_shape, dtype=np.uint8)
    low_markers = np.zeros(reconstruction_shape, dtype=np.uint8)
    for marker, (high_index, low_index) in enumerate(
        zip(high_indices, low_indices, strict=True), start=1
    ):
        high_markers[tuple(high_index)] = marker
        low_markers[tuple(low_index)] = marker

    trajectory = prepare_physical_sigpy_trajectory(
        scaled_k[0], scaled_k[1], scaled_k[2],
        fov_mm=target_fov,
        matrix_shape=reconstruction_shape,
    )
    dcf = np.asarray(sequence.density_compensation, dtype=np.float32)
    flattened_dcf = dcf.T.reshape(-1)
    dcf_maximum = float(np.max(flattened_dcf))
    if not np.isfinite(dcf_maximum) or dcf_maximum <= 0:
        raise ThreePositionReferenceDebugError("invalid density compensation")
    flattened_dcf = np.asarray(flattened_dcf / dcf_maximum, dtype=np.float32)

    coil_info = inspect_sensitivity_map(config.coils.sensitivity_map)
    normalization = prepare_rss_normalization(
        coil_info,
        config.run.output_root / "kspace" / "cache" / "sensitivity_rss.npy",
    )
    low_sensitivities = np.empty(
        (coil_info.coil_count,) + reconstruction_shape, dtype=np.complex64
    )
    delta_adjoints = np.empty_like(low_sensitivities)
    backend = SigpyNufftBackend(device_id=config.compute.device_id)
    for coil_index in range(coil_info.coil_count):
        logical_coil = load_normalized_coil_in_logical_frame(
            coil_info,
            coil_index,
            normalization,
            stored_axis_order=config.coils.axis_order,
            dcs_to_logical=transforms.dcs_to_logical,
        )
        high_coil = centered_resize(logical_coil, high_shape).astype(
            np.complex64, copy=False
        )
        low_sensitivities[coil_index] = _resample_complex(
            high_coil, reconstruction_shape
        )
        flattened = backend.forward(
            np.asarray(high_delta * high_coil, dtype=np.complex64), trajectory
        )
        delta_adjoints[coil_index] = backend.adjoint(
            flattened * flattened_dcf, trajectory
        )
        if progress is not None:
            progress(f"Three-position Gd delta: coil {coil_index + 1}/{coil_info.coil_count}")

    low_rss = np.sqrt(
        np.sum(np.abs(low_sensitivities) ** 2, axis=0, dtype=np.float64)
    ).astype(np.float32)
    supported = low_rss > np.finfo(np.float32).eps
    low_sensitivities = np.divide(
        low_sensitivities,
        low_rss[None, ...],
        out=np.zeros_like(low_sensitivities),
        where=supported[None, ...],
    )
    delta_reference = np.sum(
        np.conj(low_sensitivities) * delta_adjoints,
        axis=0,
        dtype=np.complex64,
    )
    delta_reference /= np.complex64(intensity_scale)
    combined_reference = np.asarray(
        tissue_reference + delta_reference, dtype=np.complex64
    )
    if not np.all(np.isfinite(combined_reference)):
        raise ThreePositionReferenceDebugError("combined reference is non-finite")

    _atomic_savemat(
        output_path,
        {
            "tissue_reference_complex": tissue_reference,
            "tissue_reference_magnitude": np.abs(tissue_reference).astype(np.float32),
            "three_positions_reference_complex": combined_reference,
            "three_positions_reference_magnitude": np.abs(combined_reference).astype(np.float32),
            "gd_difference_complex": delta_reference,
            "gd_difference_magnitude": np.abs(delta_reference).astype(np.float32),
            "high_resolution_three_positions": high_combined,
            "high_resolution_position_markers": high_markers,
            "reference_position_markers": low_markers,
            "position_names": np.asarray(["first", "middle", "last"], dtype=object),
            "path_fractions": np.asarray([[0.0, 0.5, 1.0]], dtype=np.float32),
            "path_distances_mm": np.asarray([distances], dtype=np.float64),
            "path_length_mm": np.asarray([[path.total_length_mm]], dtype=np.float64),
            "centers_lps_mm": np.asarray(centers, dtype=np.float64),
            "centers_pcs_ijk_zero_based": center_ijk,
            "centers_pcs_ijk_one_based": center_ijk + 1,
            "centers_logical_highres_ijk_zero_based": high_indices,
            "centers_logical_highres_ijk_one_based": high_indices + 1,
            "centers_logical_reference_ijk_zero_based": low_indices,
            "centers_logical_reference_ijk_one_based": low_indices + 1,
            "tissue_labels_at_centers": np.asarray([center_labels], dtype=np.uint16),
            "occupied_volumes_mm3": np.asarray([occupied_volumes], dtype=np.float32),
            "gd_t1_ms": np.asarray([gd_t1_ms], dtype=np.float32),
            "gd_t2_ms": np.asarray([gd_t2_ms], dtype=np.float32),
            "diameter_mm": np.asarray([balloon.geometry.diameter_mm], dtype=np.float32),
            "target_fov_mm": np.asarray([target_fov], dtype=np.float64),
            "high_resolution_shape": np.asarray([high_shape], dtype=np.int32),
            "reconstruction_shape": np.asarray([reconstruction_shape], dtype=np.int32),
            "reconstruction_voxel_mm": np.asarray([reconstruction_voxel], dtype=np.float64),
            "logical_axis_patient_directions": np.asarray(
                transforms.logical_axis_patient_directions, dtype=object
            ),
            "pcs_to_logical": transforms.pcs_to_logical,
            "pulseq_sequence_filename": sequence.sequence_path.name,
            "control_points_file": str(loaded_path.source_path),
            "trajectory_scale_factor": np.asarray([[scale_factor]], dtype=np.float64),
            "target_kmax_per_m": np.asarray([[target_kmax]], dtype=np.float64),
            "adjoint_intensity_scale": np.asarray([[intensity_scale]], dtype=np.float64),
            "elapsed_s": np.asarray([[time.perf_counter() - started]], dtype=np.float64),
        },
    )
    return ThreePositionReferenceDebugResult(
        output_path=output_path,
        path_length_mm=path.total_length_mm,
        centers_lps_mm=centers,
        center_labels=np.asarray(center_labels, dtype=np.uint16),
        high_resolution_shape=high_shape,
        reconstruction_shape=reconstruction_shape,
        elapsed_s=time.perf_counter() - started,
    )


def format_three_position_reference_debug(
    result: ThreePositionReferenceDebugResult,
) -> str:
    return "\n".join(
        (
            "Three-position catheter reference debug",
            f"Path length (mm):     {result.path_length_mm:.3f}",
            f"Center labels:        {tuple(int(v) for v in result.center_labels)}",
            f"High-resolution grid: {result.high_resolution_shape}",
            f"Reference grid:       {result.reconstruction_shape}",
            f"Elapsed:              {result.elapsed_s:.1f} s",
            f"Output:               {result.output_path}",
        )
    )
