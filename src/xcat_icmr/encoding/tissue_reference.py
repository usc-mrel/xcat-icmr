"""Coil-combined adjoint reference reconstructed from cached tissue k-space."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import TYPE_CHECKING, Callable

import h5py
import numpy as np

from xcat_icmr.cache import tissue_kspace_cache_entry
from xcat_icmr.coils import (
    inspect_sensitivity_map,
    load_normalized_coil_roi_in_logical_frame,
    prepare_rss_normalization,
    sensitivity_shape_in_logical_frame,
)
from xcat_icmr.encoding.fullysampled_reference import _resample_complex
from xcat_icmr.encoding.sigpy_backend import SigpyNufftSession
from xcat_icmr.encoding.tissue_library import _grid, tissue_library_frame_path
from xcat_icmr.encoding.trajectory import prepare_physical_sigpy_trajectory
from xcat_icmr.phantom import plan_xcat_frames
from xcat_icmr.sequence import build_coordinate_transforms, read_sequence

if TYPE_CHECKING:
    from xcat_icmr.config.models import SimulationConfig


class TissueAdjointReferenceError(ValueError):
    """Raised when tissue k-space cannot be reconstructed consistently."""


@dataclass(frozen=True)
class TissueAdjointReferenceResult:
    output_path: Path
    image_shape: tuple[int, int, int, int]
    selected_frame_count: int
    generated_frame_count: int
    reused_frame_count: int
    missing_kspace_frames: tuple[int, ...]
    intensity_scale: float
    elapsed_s: float


def tissue_adjoint_reference_path(config: "SimulationConfig") -> Path:
    return tissue_kspace_cache_entry(config).directory / (
        "tissue_fullysampled_adjoint_reference_4d.h5"
    )


def _ensure_datasets(
    handle: h5py.File,
    shape: tuple[int, int, int, int],
    *,
    overwrite: bool,
) -> tuple[h5py.Dataset, h5py.Dataset]:
    if overwrite:
        for name in ("image", "frame_complete"):
            if name in handle:
                del handle[name]
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
        image.attrs["axis_order"] = "logical_x,logical_y,logical_z,cardiac_phase"
        image.attrs["dtype_name"] = "complex64"
        image.attrs["coil_combination"] = (
            "sum(conj(normalized_sensitivity) * dcf_weighted_adjoint)"
        )
        image.attrs["sensitivity_denominator_after_adjoint"] = "none"
    valid = (
        isinstance(image, h5py.Dataset)
        and image.shape == shape
        and image.dtype == np.dtype(np.complex64)
        and isinstance(complete, h5py.Dataset)
        and complete.shape == (shape[3],)
        and complete.dtype == np.dtype(np.uint8)
    )
    if not valid:
        raise TissueAdjointReferenceError(
            "existing tissue adjoint reference has an incompatible schema; "
            "pass --overwrite to replace it"
        )
    return image, complete


def generate_tissue_adjoint_reference(
    config: "SimulationConfig",
    *,
    start_frame: int = 1,
    end_frame: int | None = None,
    overwrite: bool = False,
    allow_missing: bool = False,
    progress: Callable[[str], None] | None = None,
) -> TissueAdjointReferenceResult:
    """Reconstruct selected cached phases into one resumable 4-D image."""

    started = time.perf_counter()
    frames = plan_xcat_frames(config, debug_one_frame=False).frames
    total = len(frames)
    resolved_end = total if end_frame is None else end_frame
    if not 1 <= start_frame <= resolved_end <= total:
        raise TissueAdjointReferenceError(
            f"frame interval must lie within 1..{total}"
        )
    selected = tuple(range(start_frame, resolved_end + 1))
    missing = tuple(
        index
        for index in selected
        if not tissue_library_frame_path(config, index).is_file()
    )
    if missing and not allow_missing:
        preview = ", ".join(str(value) for value in missing[:8])
        suffix = "..." if len(missing) > 8 else ""
        raise TissueAdjointReferenceError(
            f"{len(missing)} selected tissue k-space phase(s) are missing "
            f"({preview}{suffix}); generate them first or pass --allow-missing"
        )
    sequence = read_sequence(config.sequence)
    transforms = build_coordinate_transforms(
        patient_position=config.phantom.patient_position,
        coordinate_mode=config.sequence.coordinate_mode,
        sequence_orientation=config.sequence.orientation,
    )
    high_shape, reconstruction_shape, _, scaled_k = _grid(
        config, sequence, transforms
    )
    trajectory = prepare_physical_sigpy_trajectory(
        *scaled_k,
        fov_mm=config.encoding.target_fov_mm,
        matrix_shape=reconstruction_shape,
    )
    dcf = np.asarray(sequence.density_compensation, dtype=np.float32)
    if dcf.shape != sequence.trajectory_shape:
        raise TissueAdjointReferenceError("trajectory and DCF shapes differ")
    flattened_dcf = dcf.T.reshape(-1)
    maximum = float(np.max(flattened_dcf))
    if not np.isfinite(maximum) or maximum <= 0:
        raise TissueAdjointReferenceError("DCF maximum must be positive")
    flattened_dcf = np.asarray(flattened_dcf / maximum, dtype=np.float32)
    session = SigpyNufftSession(
        trajectory, device_id=config.compute.device_id
    )
    coil_batch_size = 4
    device_dcf = session.upload(flattened_dcf, dtype=np.float32)
    if progress:
        progress(
            "Adjoint reference: trajectory and DCF resident on "
            f"{session.device_name}; calculating one-time intensity scale"
        )
    calibration_input = session.upload(
        np.ones(high_shape, dtype=np.complex64)
    )
    with session.device:
        calibration_kspace = session.forward_device(calibration_input)
        calibration = session.adjoint_device(
            calibration_kspace * device_dcf
        )
    center = tuple(size // 2 for size in reconstruction_shape)
    intensity_scale = float(
        np.abs(session.download(calibration[center])).reshape(()))
    del calibration_input, calibration_kspace, calibration
    if not np.isfinite(intensity_scale) or intensity_scale <= 0:
        raise TissueAdjointReferenceError("invalid NUFFT intensity scale")

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
    if progress:
        progress(
            f"Adjoint reference: preparing {coil_info.coil_count} low-resolution "
            "normalized sensitivity maps"
        )
    device_low_coils = session.empty(
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
        low_coil = _resample_complex(
            logical, reconstruction_shape
        )
        with session.device:
            device_low_coils[coil_index] = session.upload(
                low_coil, dtype=np.complex64
            )
        del logical, low_coil
        if progress:
            progress(
                f"Adjoint reference sensitivity: coil {coil_index + 1}/"
                f"{coil_info.coil_count}"
            )

    output_path = tissue_adjoint_reference_path(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generated = reused = 0
    # Opening with "w" truncates the old 4-D reference immediately, so a
    # one-frame validation does not retain a second full reference in storage.
    with h5py.File(output_path, "w" if overwrite else "a") as handle:
        image, complete = _ensure_datasets(
            handle, reconstruction_shape + (total,), overwrite=False
        )
        image.attrs["target_fov_mm"] = config.encoding.target_fov_mm
        image.attrs["reconstruction_shape"] = reconstruction_shape
        image.attrs["voxel_size_mm"] = tuple(
            fov / size
            for fov, size in zip(
                config.encoding.target_fov_mm, reconstruction_shape, strict=True
            )
        )
        image.attrs["pulseq_sequence_filename"] = sequence.sequence_path.name
        image.attrs["pulseq_signature"] = sequence.signature
        image.attrs["adjoint_intensity_scale"] = intensity_scale
        image.attrs["fov_centering"] = "rf-profile"
        image.attrs["fov_centering_algorithm"] = 1
        image.attrs["rf_center_shift_mm"] = (
            config.sequence.rf_profile.center_shift_mm
        )
        image.attrs["coil_sensitivity_spatial_shift_mm"] = 0.0
        for frame_index in selected:
            zero_based = frame_index - 1
            if frame_index in missing:
                continue
            if bool(complete[zero_based]) and not overwrite:
                reused += 1
                if progress:
                    progress(f"Adjoint phase {frame_index}/{total}: reused")
                continue
            with session.device:
                combined_device = session.xp.zeros(
                    reconstruction_shape, dtype=session.xp.complex64
                )
            source_path = tissue_library_frame_path(config, frame_index)
            with h5py.File(source_path, "r") as source:
                kspace = source.get("kspace")
                expected = (
                    sequence.sample_count,
                    sequence.arm_count,
                    coil_info.coil_count,
                )
                if not isinstance(kspace, h5py.Dataset) or kspace.shape != expected:
                    raise TissueAdjointReferenceError(
                        f"invalid tissue k-space phase: {source_path}"
                    )
                if int(kspace.attrs.get("fov_centering_algorithm", 0)) != 1:
                    raise TissueAdjointReferenceError(
                        f"tissue k-space phase is not RF-centered: {source_path}"
                    )
                for batch_start in range(
                    0, coil_info.coil_count, coil_batch_size
                ):
                    batch_indices = list(
                        range(
                            batch_start,
                            min(
                                batch_start + coil_batch_size,
                                coil_info.coil_count,
                            ),
                        )
                    )
                    values = np.asarray(
                        kspace[:, :, batch_indices], dtype=np.complex64
                    ).transpose(2, 1, 0).reshape(
                        len(batch_indices), trajectory.point_count
                    )
                    device_values = session.upload(values, dtype=np.complex64)
                    with session.device:
                        adjoint_device = session.adjoint_device(
                            device_values * device_dcf[None, :]
                        )
                        combined_device += session.xp.sum(
                            session.xp.conj(
                                device_low_coils[batch_indices]
                            )
                            * adjoint_device,
                            axis=0,
                        )
                    del values, device_values, adjoint_device
                    if progress:
                        shown = ",".join(
                            str(index + 1) for index in batch_indices
                        )
                        progress(
                            f"Adjoint phase {frame_index}/{total}: GPU coils "
                            f"[{shown}]"
                        )
            combined = session.download(combined_device).astype(
                np.complex64, copy=False
            )
            del combined_device
            combined /= np.complex64(intensity_scale)
            if not np.all(np.isfinite(combined)):
                raise TissueAdjointReferenceError(
                    f"adjoint phase {frame_index} contains non-finite values"
                )
            image[..., zero_based] = combined
            complete[zero_based] = 1
            handle.flush()
            generated += 1
            if progress:
                progress(f"Adjoint phase {frame_index}/{total}: saved")
    return TissueAdjointReferenceResult(
        output_path=output_path,
        image_shape=reconstruction_shape + (total,),
        selected_frame_count=len(selected),
        generated_frame_count=generated,
        reused_frame_count=reused,
        missing_kspace_frames=missing,
        intensity_scale=intensity_scale,
        elapsed_s=time.perf_counter() - started,
    )


def format_tissue_adjoint_reference(result: TissueAdjointReferenceResult) -> str:
    return "\n".join(
        (
            "Fully sampled coil-combined tissue adjoint reference",
            f"4-D image shape:    {result.image_shape} complex64",
            f"Selected phases:    {result.selected_frame_count}",
            f"Generated/reused:   {result.generated_frame_count}/{result.reused_frame_count}",
            f"Missing skipped:    {len(result.missing_kspace_frames)}",
            f"Intensity scale:    {result.intensity_scale:.9g}",
            f"Elapsed:            {result.elapsed_s:.3f} s",
            f"Output:             {result.output_path}",
        )
    )
