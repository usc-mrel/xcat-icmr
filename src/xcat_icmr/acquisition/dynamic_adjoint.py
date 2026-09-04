"""GPU adjoint images for a bounded TR-level dynamic acquisition."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

import h5py
import numpy as np

from xcat_icmr.coils import (
    inspect_sensitivity_map,
    load_normalized_coil_roi_in_logical_frame,
    prepare_rss_normalization,
    sensitivity_shape_in_logical_frame,
)
from xcat_icmr.encoding.fullysampled_reference import _resample_complex
from xcat_icmr.encoding.sigpy_backend import SigpyNufftSession
from xcat_icmr.encoding.tissue_library import _grid
from xcat_icmr.encoding.tissue_reference import tissue_adjoint_reference_path
from xcat_icmr.encoding.trajectory import prepare_physical_sigpy_trajectory
from xcat_icmr.sequence import build_coordinate_transforms, read_sequence
from xcat_icmr.cache import tissue_kspace_cache_entry

if TYPE_CHECKING:
    from xcat_icmr.acquisition.dynamic import DynamicAcquisitionPlan
    from xcat_icmr.config.models import SimulationConfig


class DynamicAdjointDebugError(ValueError):
    """Raised when bounded dynamic k-space cannot be adjoint reconstructed."""


def _ensure_output(
    path: Path,
    shape: tuple[int, int, int, int],
    *,
    overwrite: bool,
) -> tuple[h5py.File, h5py.Dataset, h5py.Dataset]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = h5py.File(path, "w" if overwrite else "a")
    image = handle.get("image")
    complete = handle.get("frame_complete")
    if image is None and complete is None:
        image = handle.create_dataset(
            "image",
            shape=shape,
            dtype=np.complex64,
            chunks=shape[:3] + (1,),
        )
        complete = handle.create_dataset(
            "frame_complete", shape=(shape[3],), dtype=np.uint8
        )
        image.attrs["axis_order"] = (
            "logical_x,logical_y,logical_z,acquisition_frame"
        )
        image.attrs["contains"] = (
            "DCF-weighted adjoint of combined tissue + additive Gd k-space"
        )
        image.attrs["coil_combination"] = (
            "sum(conj(normalized_sensitivity) * subset_adjoint)"
        )
        image.attrs["sensitivity_denominator_after_adjoint"] = "none"
    valid = (
        isinstance(image, h5py.Dataset)
        and image.shape == shape
        and image.dtype == np.dtype(np.complex64)
        and isinstance(complete, h5py.Dataset)
        and complete.shape == (shape[3],)
    )
    if not valid:
        handle.close()
        raise DynamicAdjointDebugError(
            "existing dynamic adjoint debug has an incompatible schema; "
            "use --overwrite"
        )
    return handle, image, complete


def generate_dynamic_adjoint_debug(
    config: "SimulationConfig",
    *,
    plan: "DynamicAcquisitionPlan",
    overwrite: bool = False,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Save one GPU adjoint image per acquisition frame."""

    if plan.view_order_cycles is None:
        raise DynamicAdjointDebugError(
            "dynamic adjoint debug requires a bounded view-order-cycle run"
        )
    output_path = plan.output_path.with_name("combined_adjoint_4d.h5")
    if output_path.is_file() and not overwrite:
        try:
            with h5py.File(output_path, "r") as existing:
                image = existing.get("image")
                complete = existing.get("frame_complete")
                if (
                    isinstance(image, h5py.Dataset)
                    and image.shape[-1] == plan.schedule.frame_count
                    and isinstance(complete, h5py.Dataset)
                    and complete.shape == (plan.schedule.frame_count,)
                    and np.all(complete[:])
                ):
                    if progress:
                        progress(
                            "Dynamic adjoint: all frames already complete"
                        )
                    return output_path
        except OSError:
            pass
    sequence = read_sequence(config.sequence)
    transforms = build_coordinate_transforms(
        patient_position=config.phantom.patient_position,
        coordinate_mode=config.sequence.coordinate_mode,
        sequence_orientation=config.sequence.orientation,
    )
    high_shape, reconstruction_shape, _, scaled_k = _grid(
        config, sequence, transforms
    )
    dcf = np.asarray(sequence.density_compensation, dtype=np.float32)
    if dcf.shape != sequence.trajectory_shape:
        raise DynamicAdjointDebugError("trajectory and DCF shapes differ")
    dcf_maximum = float(np.max(dcf))
    if not np.isfinite(dcf_maximum) or dcf_maximum <= 0:
        raise DynamicAdjointDebugError("DCF maximum must be positive")
    dcf /= np.float32(dcf_maximum)

    approved_path = tissue_adjoint_reference_path(config)
    if not approved_path.is_file():
        raise DynamicAdjointDebugError(
            "the approved tissue adjoint reference is required"
        )
    with h5py.File(approved_path, "r") as approved:
        approved_image = approved.get("image")
        if not isinstance(approved_image, h5py.Dataset):
            raise DynamicAdjointDebugError(
                "approved tissue adjoint reference has no image dataset"
            )
        intensity_scale = float(
            approved_image.attrs.get("adjoint_intensity_scale", 0.0)
        )
    if not np.isfinite(intensity_scale) or intensity_scale <= 0:
        raise DynamicAdjointDebugError(
            "approved tissue adjoint intensity scale is invalid"
        )

    full_trajectory = prepare_physical_sigpy_trajectory(
        *scaled_k,
        fov_mm=config.encoding.target_fov_mm,
        matrix_shape=reconstruction_shape,
    )
    resident = SigpyNufftSession(
        full_trajectory, device_id=config.compute.device_id
    )
    coil_info = inspect_sensitivity_map(config.coils.sensitivity_map)
    logical_coil_shape = sensitivity_shape_in_logical_frame(
        coil_info,
        stored_axis_order=config.coils.axis_order,
        dcs_to_logical=transforms.dcs_to_logical,
    )
    coil_offset = (
        np.asarray(logical_coil_shape, dtype=np.int64)
        - np.asarray(high_shape, dtype=np.int64)
    ) // 2
    coil_slices = tuple(
        slice(int(start), int(start + size))
        for start, size in zip(coil_offset, high_shape, strict=True)
    )
    normalization = prepare_rss_normalization(
        coil_info,
        tissue_kspace_cache_entry(config).directory / "sensitivity_rss.npy",
    )
    device_low_coils = resident.empty(
        (coil_info.coil_count,) + reconstruction_shape,
        dtype=np.complex64,
    )
    for coil_index in range(coil_info.coil_count):
        logical = load_normalized_coil_roi_in_logical_frame(
            coil_info,
            coil_index,
            normalization,
            coil_slices,
            stored_axis_order=config.coils.axis_order,
            dcs_to_logical=transforms.dcs_to_logical,
        )
        low_coil = _resample_complex(logical, reconstruction_shape)
        with resident.device:
            device_low_coils[coil_index] = resident.upload(
                low_coil, dtype=np.complex64
            )
        if progress:
            progress(
                f"Dynamic adjoint sensitivity: coil {coil_index + 1}/"
                f"{coil_info.coil_count}"
            )

    output_shape = reconstruction_shape + (plan.schedule.frame_count,)
    with h5py.File(plan.output_path, "r") as source:
        kspace = source.get("kspace")
        tr_complete = source.get("tr_complete")
        if (
            not isinstance(kspace, h5py.Dataset)
            or kspace.shape != plan.shape
            or not isinstance(tr_complete, h5py.Dataset)
            or not np.all(tr_complete[:])
        ):
            raise DynamicAdjointDebugError(
                "combined dynamic k-space is incomplete"
            )
        handle, image, frame_complete = _ensure_output(
            output_path, output_shape, overwrite=overwrite
        )
        image.attrs["effective_tr_s"] = plan.schedule.effective_tr_s
        image.attrs["frame_duration_s"] = plan.schedule.frame_duration_s
        image.attrs["trs_per_frame"] = plan.schedule.trs_per_frame
        image.attrs["adjoint_intensity_scale"] = intensity_scale
        image.attrs["target_fov_mm"] = config.encoding.target_fov_mm
        image.attrs["reconstruction_shape"] = reconstruction_shape
        image.attrs["device"] = resident.device_name
        try:
            for frame_index in range(plan.schedule.frame_count):
                if bool(frame_complete[frame_index]) and not overwrite:
                    continue
                start = frame_index * plan.schedule.trs_per_frame
                stop = start + plan.schedule.trs_per_frame
                trajectory_trs = np.asarray(
                    plan.schedule.trajectory_tr_index_zero_based[start:stop],
                    dtype=np.int64,
                )
                frame_trajectory = prepare_physical_sigpy_trajectory(
                    *scaled_k,
                    fov_mm=config.encoding.target_fov_mm,
                    matrix_shape=reconstruction_shape,
                    arm_indices=trajectory_trs,
                )
                session = SigpyNufftSession(
                    frame_trajectory, device_id=config.compute.device_id
                )
                values = np.asarray(
                    kspace[:, start:stop, :], dtype=np.complex64
                ).transpose(2, 1, 0).reshape(
                    coil_info.coil_count, frame_trajectory.point_count
                )
                frame_dcf = np.asarray(
                    dcf[:, trajectory_trs].T.reshape(-1),
                    dtype=np.float32,
                )
                device_values = session.upload(values, dtype=np.complex64)
                device_dcf = session.upload(frame_dcf, dtype=np.float32)
                with session.device:
                    adjoint_coils = session.adjoint_device(
                        device_values * device_dcf[None, :]
                    )
                    combined = session.xp.sum(
                        session.xp.conj(device_low_coils) * adjoint_coils,
                        axis=0,
                    )
                combined_cpu = session.download(combined).astype(
                    np.complex64, copy=False
                )
                combined_cpu /= np.complex64(intensity_scale)
                if not np.all(np.isfinite(combined_cpu)):
                    raise DynamicAdjointDebugError(
                        f"adjoint frame {frame_index + 1} is non-finite"
                    )
                image[..., frame_index] = combined_cpu
                frame_complete[frame_index] = 1
                handle.flush()
                if progress:
                    progress(
                        f"Dynamic adjoint: frame {frame_index + 1}/"
                        f"{plan.schedule.frame_count}"
                    )
        finally:
            handle.close()
    return output_path
