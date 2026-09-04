"""Image-only fully sampled tissue-plus-Gd reference with motion blur."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import TYPE_CHECKING, Callable

import h5py
import numpy as np

from xcat_icmr.acquisition.dynamic import (
    DynamicAcquisitionError,
    _carrier,
    _map_pcs_sparse_to_high,
    plan_dynamic_acquisition,
)
from xcat_icmr.cache import dynamic_acquisition_cache_entry, tissue_kspace_cache_entry
from xcat_icmr.coils import (
    inspect_sensitivity_map,
    load_normalized_coil_roi_in_logical_frame,
    prepare_rss_normalization,
    sensitivity_shape_in_logical_frame,
)
from xcat_icmr.encoding.fullysampled_reference import _resample_complex
from xcat_icmr.encoding.tissue_library import _grid
from xcat_icmr.encoding.tissue_reference import tissue_adjoint_reference_path
from xcat_icmr.intervention import (
    calculate_sparse_gd_bssfp_signal,
    interpolate_cubic_arc_length,
    load_balloon_path,
    rasterize_sparse_balloon,
    sample_sparse_flip_angles_from_profile,
)
from xcat_icmr.intervention.roi_encoding import PersistentSparseRoiEncoder
from xcat_icmr.phantom import plan_xcat_frames
from xcat_icmr.sequence import build_coordinate_transforms, read_sequence
from xcat_icmr.signal import calculate_rf_profile_bssfp_contrast, generate_slice_profile, read_pulseq_excitation
from xcat_icmr.tissue import get_tissue_library

if TYPE_CHECKING:
    from xcat_icmr.config.models import SimulationConfig


@dataclass(frozen=True)
class DynamicReferenceResult:
    output_path: Path
    shape: tuple[int, int, int, int]
    generated_frames: int
    reused_frames: int
    trs_averaged_per_frame: int
    elapsed_s: float


def _datasets(handle: h5py.File, shape: tuple[int, int, int, int], *, overwrite: bool):
    if overwrite:
        for name in ("image", "frame_complete"):
            if name in handle:
                del handle[name]
    image = handle.get("image")
    complete = handle.get("frame_complete")
    if image is None and complete is None:
        image = handle.create_dataset("image", shape=shape, dtype=np.complex64, chunks=shape[:3] + (1,))
        complete = handle.create_dataset("frame_complete", shape=(shape[3],), dtype=np.uint8)
        image.attrs["axis_order"] = "logical_x,logical_y,logical_z,time"
        image.attrs["contains"] = "fully sampled tissue + motion-averaged additive Gd"
    if not isinstance(image, h5py.Dataset) or image.shape != shape or not isinstance(complete, h5py.Dataset):
        raise DynamicAcquisitionError("existing dynamic reference has an incompatible schema")
    return image, complete


def generate_dynamic_fullysampled_reference(
    config: "SimulationConfig",
    *,
    overwrite: bool = False,
    progress: Callable[[str], None] | None = None,
) -> DynamicReferenceResult:
    """Save only the full-trajectory combined image series.

    The approved tissue adjoint is reused directly. Gd is averaged over the
    TR-level positions in each output frame, encoded over the complete
    trajectory, adjoint reconstructed, and added to the tissue reference. No
    full reference k-space is retained.
    """

    started = time.perf_counter()
    plan = plan_dynamic_acquisition(config, check_free_space=False)
    sequence = read_sequence(config.sequence)
    transforms = build_coordinate_transforms(
        patient_position=config.phantom.patient_position,
        coordinate_mode=config.sequence.coordinate_mode,
        sequence_orientation=config.sequence.orientation,
    )
    high_shape, reconstruction_shape, logical_voxel, scaled_k = _grid(config, sequence, transforms)
    dcf = np.asarray(sequence.density_compensation, dtype=np.float32).T.reshape(-1)
    dcf_maximum = float(np.max(dcf))
    if not np.isfinite(dcf_maximum) or dcf_maximum <= 0:
        raise DynamicAcquisitionError("density compensation must have a positive maximum")
    dcf /= np.float32(dcf_maximum)

    output_dir = dynamic_acquisition_cache_entry(config).directory
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "fullysampled_tissue_gd_reference_4d.h5"
    output_shape = reconstruction_shape + (plan.schedule.frame_count,)
    if output_path.is_file() and not overwrite:
        try:
            with h5py.File(output_path, "r") as existing:
                image = existing.get("image")
                complete = existing.get("frame_complete")
                if (
                    isinstance(image, h5py.Dataset)
                    and image.shape == output_shape
                    and image.dtype == np.dtype(np.complex64)
                    and isinstance(complete, h5py.Dataset)
                    and complete.shape == (plan.schedule.frame_count,)
                    and np.all(complete[:])
                ):
                    return DynamicReferenceResult(
                        output_path,
                        output_shape,
                        0,
                        plan.schedule.frame_count,
                        plan.schedule.trs_per_frame,
                        time.perf_counter() - started,
                    )
        except OSError:
            pass

    phases = plan_xcat_frames(config, debug_one_frame=False).frames
    tissue_path = tissue_adjoint_reference_path(config)
    if not tissue_path.is_file():
        raise DynamicAcquisitionError(
            "the complete tissue adjoint reference is required"
        )
    with h5py.File(tissue_path, "r") as tissue_handle:
        tissue_images = tissue_handle.get("image")
        tissue_complete = tissue_handle.get("frame_complete")
        if (
            not isinstance(tissue_images, h5py.Dataset)
            or tissue_images.shape != reconstruction_shape + (len(phases),)
            or tissue_images.dtype != np.dtype(np.complex64)
            or not isinstance(tissue_complete, h5py.Dataset)
            or tissue_complete.shape != (len(phases),)
            or not np.all(tissue_complete[:])
        ):
            raise DynamicAcquisitionError(
                "the tissue adjoint reference is incomplete or incompatible"
            )
        intensity_scale = float(
            tissue_images.attrs.get("adjoint_intensity_scale", 0.0)
        )
    if not np.isfinite(intensity_scale) or intensity_scale <= 0:
        raise DynamicAcquisitionError(
            "the tissue adjoint reference has no valid intensity scale"
        )

    coil_info = inspect_sensitivity_map(config.coils.sensitivity_map)
    logical_coil_shape = sensitivity_shape_in_logical_frame(
        coil_info, stored_axis_order=config.coils.axis_order, dcs_to_logical=transforms.dcs_to_logical
    )
    normalization = prepare_rss_normalization(coil_info, tissue_kspace_cache_entry(config).directory / "sensitivity_rss.npy")
    high_offset = (np.asarray(logical_coil_shape) - np.asarray(high_shape)) // 2

    def coil_roi_loader(coil_index: int, slices: tuple[slice, slice, slice]) -> np.ndarray:
        full_slices = tuple(slice(int(item.start) + int(high_offset[a]), int(item.stop) + int(high_offset[a])) for a, item in enumerate(slices))
        return load_normalized_coil_roi_in_logical_frame(
            coil_info, coil_index, normalization, full_slices,
            stored_axis_order=config.coils.axis_order, dcs_to_logical=transforms.dcs_to_logical,
        )

    first_label = phases[0].label_path
    if first_label is None:
        raise DynamicAcquisitionError("the first XCAT label path is unavailable")
    excitation = read_pulseq_excitation(sequence.sequence_path)
    profile = generate_slice_profile(
        excitation, matrix_size=logical_coil_shape[excitation.logical_axis],
        voxel_size_mm=logical_voxel[excitation.logical_axis],
        center_shift_mm=config.sequence.rf_profile.center_shift_mm,
    )
    library = get_tissue_library(config.sequence.contrast.tissue_library)
    scratch, pcs_axis, _, applied_flip = calculate_rf_profile_bssfp_contrast(
        label_path=first_label, profile=profile, transforms=transforms,
        pcs_voxel_size_mm=config.phantom.voxel_size_mm, library=library,
        te_ms=sequence.te_ms, tr_ms=sequence.tr_ms,
    )
    pcs_shape = tuple(int(v) for v in scratch.shape)
    del scratch
    path_config = config.intervention.gd_balloon.path
    loaded = load_balloon_path(
        path_config.control_points_file,
        coordinate_system=path_config.coordinate_system,
    )
    curve = interpolate_cubic_arc_length(loaded.control_points_lps_mm)
    positions = curve.positions_at_times_s(
        plan.schedule.time_s,
        velocity_cm_per_s=config.intervention.gd_balloon.movement.velocity_cm_per_s,
        start_time_s=config.intervention.gd_balloon.movement.start_time_s,
        traversal=config.intervention.gd_balloon.movement.traversal,
    )
    carrier = _carrier(
        library, config.intervention.gd_balloon.contrast_agent.carrier_tissue
    )
    encoder = PersistentSparseRoiEncoder(
        global_shape=high_shape,
        voxel_size_mm=logical_voxel,
        coil_count=coil_info.coil_count,
        coil_roi_loader=coil_roi_loader,
        kx_per_m=scaled_k[0],
        ky_per_m=scaled_k[1],
        kz_per_m=scaled_k[2],
        acquisition_matrix_shape=reconstruction_shape,
        rf_center_shift_mm=config.sequence.rf_profile.center_shift_mm,
        rf_axis_voxel_size_mm=logical_voxel[excitation.logical_axis],
        rf_logical_axis=excitation.logical_axis,
        device_id=config.compute.device_id,
        progress=progress,
    )
    resident = encoder.session
    device_dcf = resident.upload(dcf, dtype=np.float32)
    device_low_coils = resident.empty(
        (coil_info.coil_count,) + reconstruction_shape, dtype=np.complex64
    )
    full_slices = tuple(slice(0, size) for size in high_shape)
    for coil_index in range(coil_info.coil_count):
        low_coil = _resample_complex(
            coil_roi_loader(coil_index, full_slices), reconstruction_shape
        )
        with resident.device:
            device_low_coils[coil_index] = resident.upload(
                low_coil, dtype=np.complex64
            )
        if progress:
            progress(
                f"Fully sampled reference sensitivity: coil {coil_index + 1}/"
                f"{coil_info.coil_count}"
            )
    generated = reused = 0
    with h5py.File(tissue_path, "r") as tissue_handle:
        tissue_images = tissue_handle["image"]
        with h5py.File(output_path, "a") as output_handle:
            output, complete = _datasets(
                output_handle, output_shape, overwrite=overwrite
            )
            output.attrs["source_tissue_adjoint"] = str(tissue_path)
            output.attrs["trs_averaged_per_frame"] = plan.schedule.trs_per_frame
            output.attrs["frame_duration_s"] = plan.schedule.frame_duration_s
            output.attrs["adjoint_intensity_scale"] = intensity_scale
            output.attrs["device"] = resident.device_name
            output.attrs["gd_fov_centering"] = "rf-profile"
            output.attrs["gd_rf_center_shift_mm"] = (
                config.sequence.rf_profile.center_shift_mm
            )
            output.attrs["gd_rf_logical_axis_zero_based"] = (
                excitation.logical_axis
            )
            for frame_index in range(plan.schedule.frame_count):
                if complete[frame_index] and not overwrite:
                    reused += 1
                    continue
                start = frame_index * plan.schedule.trs_per_frame
                stop = start + plan.schedule.trs_per_frame
                phase_indices = plan.schedule.cardiac_phase_index_zero_based[start:stop]
                tissue_reference = np.mean(
                    np.stack(
                        [
                            np.asarray(tissue_images[..., int(phase)], dtype=np.complex64)
                            for phase in phase_indices
                        ],
                        axis=-1,
                    ),
                    axis=3,
                    dtype=np.complex64,
                )
                all_indices: list[np.ndarray] = []
                all_values: list[np.ndarray] = []
                for center in positions[start:stop]:
                    support = rasterize_sparse_balloon(
                        center, volume_shape=pcs_shape, voxel_size_mm=config.phantom.voxel_size_mm,
                        diameter_mm=config.intervention.gd_balloon.geometry.diameter_mm,
                        shape=config.intervention.gd_balloon.geometry.shape,
                    )
                    flip = sample_sparse_flip_angles_from_profile(applied_flip, pcs_axis=pcs_axis, pcs_image_shape=pcs_shape, support=support)
                    gd = calculate_sparse_gd_bssfp_signal(
                        support, carrier=carrier,
                        concentration_mM=config.intervention.gd_balloon.contrast_agent.concentration.value_mM,
                        flip_angle_deg=flip, te_ms=sequence.te_ms, tr_ms=sequence.tr_ms,
                        relaxivity_library=config.intervention.gd_balloon.contrast_agent.relaxivity_library,
                    )
                    occupied = support.occupancy > 0
                    indices, values = _map_pcs_sparse_to_high(
                        support.occupied_indices_ijk(), support.occupancy[occupied] * gd.values[occupied],
                        pcs_shape=pcs_shape, pcs_to_logical=transforms.pcs_to_logical, high_shape=high_shape,
                    )
                    all_indices.append(indices)
                    all_values.append(values / np.float32(plan.schedule.trs_per_frame))
                gd_kspace = encoder.encode_full(
                    np.concatenate(all_indices), np.concatenate(all_values)
                ).kspace
                with resident.device:
                    gd_reference_device = resident.xp.zeros(
                        reconstruction_shape, dtype=resident.xp.complex64
                    )
                for batch_start in range(0, coil_info.coil_count, 4):
                    batch = list(
                        range(
                            batch_start,
                            min(batch_start + 4, coil_info.coil_count),
                        )
                    )
                    flat = gd_kspace[:, :, batch].transpose(2, 1, 0).reshape(
                        len(batch), encoder.trajectory.point_count
                    )
                    device_values = resident.upload(flat, dtype=np.complex64)
                    with resident.device:
                        adjoint = resident.adjoint_device(
                            device_values * device_dcf[None, :]
                        )
                        gd_reference_device += resident.xp.sum(
                            resident.xp.conj(device_low_coils[batch]) * adjoint,
                            axis=0,
                        )
                    del flat, device_values, adjoint
                gd_reference = resident.download(gd_reference_device).astype(
                    np.complex64, copy=False
                )
                del gd_reference_device, gd_kspace
                output[..., frame_index] = tissue_reference + gd_reference / np.complex64(intensity_scale)
                complete[frame_index] = 1
                output_handle.flush()
                generated += 1
                if progress:
                    progress(f"Fully sampled combined reference: {frame_index + 1}/{plan.schedule.frame_count}")
    del encoder
    return DynamicReferenceResult(
        output_path, output_shape, generated, reused,
        plan.schedule.trs_per_frame, time.perf_counter() - started,
    )
