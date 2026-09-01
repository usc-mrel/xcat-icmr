"""Streaming fully sampled forward/adjoint tissue reference generation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import tempfile
import time
from typing import TYPE_CHECKING, Callable

import h5py
import numpy as np
from scipy.ndimage import zoom
from scipy.io import savemat

from xcat_icmr.cache import (
    artifact_cache_status,
    contrast_cache_entry,
    contrast_frame_path,
    fullysampled_reference_cache_entry,
    write_artifact_manifest,
    write_stage_manifest,
)
from xcat_icmr.coils import (
    inspect_sensitivity_map,
    load_normalized_coil_in_logical_frame,
    prepare_rss_normalization,
    sensitivity_shape_in_logical_frame,
)
from xcat_icmr.encoding.inputs import load_contrast_image
from xcat_icmr.encoding.sigpy_backend import SigpyNufftBackend
from xcat_icmr.encoding.trajectory import (
    prepare_physical_sigpy_trajectory,
    scale_isotropic_trajectory_to_resolution,
)
from xcat_icmr.phantom import XcatFrame, plan_xcat_frames
from xcat_icmr.sequence import build_coordinate_transforms, read_sequence
from xcat_icmr.sequence.orientation import reorient_spatial_array

if TYPE_CHECKING:
    from xcat_icmr.config.models import SimulationConfig


class FullysampledReferenceError(ValueError):
    """Raised when a fully sampled reference cannot be generated safely."""


@dataclass(frozen=True)
class TemporalGroup:
    index: int
    xcat_frames: tuple[XcatFrame, ...]
    window_start_s: float
    window_end_s: float
    time_s: float


@dataclass(frozen=True)
class FullysampledReferenceResult:
    output_directory: Path
    reference_path: Path
    metadata_path: Path
    manifest_path: Path | None
    total_frame_count: int
    selected_frame_count: int
    generated_frame_count: int
    reused_frame_count: int
    target_fov_mm: tuple[float, float, float]
    high_resolution_shape: tuple[int, int, int]
    reconstruction_shape: tuple[int, int, int]
    reconstruction_voxel_size_mm: tuple[float, float, float]
    saved_kspace: bool
    elapsed_s: float


def _temporal_groups(
    frames: tuple[XcatFrame, ...],
    *,
    frames_per_group: int,
    xcat_time_step_s: float,
) -> tuple[TemporalGroup, ...]:
    if not frames or frames_per_group <= 0:
        raise FullysampledReferenceError("invalid XCAT temporal grouping")
    if len(frames) % frames_per_group:
        raise FullysampledReferenceError(
            f"{len(frames)} XCAT frames cannot be divided into groups of "
            f"{frames_per_group}"
        )
    groups = []
    for start in range(0, len(frames), frames_per_group):
        selected = frames[start : start + frames_per_group]
        groups.append(
            TemporalGroup(
                index=len(groups) + 1,
                xcat_frames=selected,
                window_start_s=selected[0].time_s,
                window_end_s=(
                    selected[0].time_s
                    + len(selected) * xcat_time_step_s
                ),
                time_s=float(
                    np.mean([frame.time_s for frame in selected])
                ),
            )
        )
    return tuple(groups)


def _average_images(images: list[np.ndarray]) -> np.ndarray:
    if not images:
        raise FullysampledReferenceError("cannot average an empty frame group")
    result = np.asarray(images[0], dtype=np.float32).copy()
    for image in images[1:]:
        values = np.asarray(image, dtype=np.float32)
        if values.shape != result.shape:
            raise FullysampledReferenceError("contrast frame shapes differ")
        np.add(result, values, out=result)
    result *= np.float32(1.0 / len(images))
    return result


def centered_resize(
    array: np.ndarray,
    target_shape: tuple[int, int, int],
) -> np.ndarray:
    """Center-crop and/or zero-pad while preserving the sampled zero index."""

    source = np.asarray(array)
    if source.ndim != 3 or len(target_shape) != 3:
        raise FullysampledReferenceError("centered resize requires 3-D arrays")
    if any(size <= 0 for size in target_shape):
        raise FullysampledReferenceError("target shape must be positive")
    destination = np.zeros(target_shape, dtype=source.dtype)
    source_slices = []
    destination_slices = []
    for source_size, target_size in zip(
        source.shape, target_shape, strict=True
    ):
        lower_coordinate = max(-(source_size // 2), -(target_size // 2))
        upper_coordinate = min(
            source_size - source_size // 2,
            target_size - target_size // 2,
        )
        source_slices.append(
            slice(
                lower_coordinate + source_size // 2,
                upper_coordinate + source_size // 2,
            )
        )
        destination_slices.append(
            slice(
                lower_coordinate + target_size // 2,
                upper_coordinate + target_size // 2,
            )
        )
    destination[tuple(destination_slices)] = source[tuple(source_slices)]
    return destination


def reconstruction_shape_for_trajectory(
    target_fov_mm: tuple[float, float, float],
    resolution_mm: float,
    maximum_absolute_k_per_m: tuple[float, float, float],
) -> tuple[int, int, int]:
    """Choose the closest nominal matrix that also contains every k sample."""

    fov = np.asarray(target_fov_mm, dtype=np.float64)
    maximum_k = np.asarray(maximum_absolute_k_per_m, dtype=np.float64)
    if (
        fov.shape != (3,)
        or np.any(~np.isfinite(fov))
        or np.any(fov <= 0)
        or not np.isfinite(resolution_mm)
        or resolution_mm <= 0
        or maximum_k.shape != (3,)
        or np.any(~np.isfinite(maximum_k))
        or np.any(maximum_k < 0)
    ):
        raise FullysampledReferenceError("invalid target-grid inputs")
    nominal = np.rint(fov / resolution_mm).astype(np.int64)
    nyquist_safe = np.ceil(2.0 * maximum_k * fov * 1e-3).astype(
        np.int64
    )
    matrix = np.maximum(nominal, nyquist_safe)
    return tuple(int(value) for value in matrix)


def _resample_complex(
    array: np.ndarray,
    target_shape: tuple[int, int, int],
) -> np.ndarray:
    factors = tuple(
        target / source
        for source, target in zip(array.shape, target_shape, strict=True)
    )
    result = zoom(
        np.asarray(array, dtype=np.complex64),
        factors,
        order=1,
        mode="constant",
        prefilter=False,
    ).astype(np.complex64)
    if result.shape != target_shape:
        raise FullysampledReferenceError(
            f"resampled sensitivity shape {result.shape} != {target_shape}"
        )
    return result


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _atomic_savemat(path: Path, variables: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        savemat(
            temporary_path,
            variables,
            appendmat=False,
            do_compression=False,
        )
        temporary_path.chmod(0o644)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _ensure_reference_datasets(
    handle: h5py.File,
    shape: tuple[int, int, int, int],
) -> tuple[h5py.Dataset, h5py.Dataset]:
    if "image" in handle:
        image = handle["image"]
        complete = handle.get("frame_complete")
        if image.shape != shape or image.dtype != np.dtype(np.complex64):
            raise FullysampledReferenceError(
                "existing fully sampled reference has incompatible shape/dtype"
            )
        if complete is None or complete.shape != (shape[3],):
            raise FullysampledReferenceError(
                "existing fully sampled completion map is invalid"
            )
        return image, complete
    image = handle.create_dataset(
        "image",
        shape=shape,
        dtype=np.complex64,
        chunks=shape[:3] + (1,),
    )
    complete = handle.create_dataset(
        "frame_complete", shape=(shape[3],), dtype=np.uint8
    )
    image.attrs["axis_order"] = "logical_x,logical_y,logical_z,time"
    image.attrs["dtype_name"] = "complex64"
    return image, complete


def generate_fullysampled_reference(
    config: "SimulationConfig",
    *,
    start_frame: int = 1,
    end_frame: int | None = None,
    overwrite: bool = False,
    progress: Callable[[str], None] | None = None,
) -> FullysampledReferenceResult:
    """Forward/adjoint encode selected tissue frames and retain the image."""

    started = time.perf_counter()
    contrast_status = artifact_cache_status(contrast_cache_entry(config))
    if contrast_status.state != "HIT":
        raise FullysampledReferenceError(
            "the high-resolution contrast cache is not reusable: "
            f"{contrast_status.state} ({contrast_status.reason})"
        )
    if not config.outputs.save_fullysampled_contrast:
        raise FullysampledReferenceError(
            "outputs.save_fullysampled_contrast must be true"
        )
    if not config.coils.enabled or config.coils.sensitivity_map is None:
        raise FullysampledReferenceError("an enabled sensitivity map is required")
    if not config.coils.normalize:
        raise FullysampledReferenceError("coils.normalize must be true")
    if config.timeline.xcat_to_kspace == "trajectory-aware":
        raise NotImplementedError(
            "trajectory-aware temporal aggregation is not implemented"
        )

    frame_plan = plan_xcat_frames(config, debug_one_frame=False)
    groups = _temporal_groups(
        frame_plan.frames,
        frames_per_group=config.timeline.xcat_frames_per_kspace_frame,
        xcat_time_step_s=config.timeline.xcat_time_step_s,
    )
    total = len(groups)
    resolved_end = total if end_frame is None else end_frame
    if not 1 <= start_frame <= total:
        raise FullysampledReferenceError(
            f"start frame must be between 1 and {total}"
        )
    if not start_frame <= resolved_end <= total:
        raise FullysampledReferenceError(
            f"end frame must be between {start_frame} and {total}"
        )

    sequence = read_sequence(config.sequence)
    resolution_values = np.asarray(
        sequence.resolution_mm, dtype=np.float64
    ).reshape(-1)
    if resolution_values.size != 1:
        raise FullysampledReferenceError(
            "the fully sampled reference currently requires isotropic resolution"
        )
    resolution_mm = float(resolution_values[0])
    scaled_k, scale_factor, target_kmax = scale_isotropic_trajectory_to_resolution(
        sequence.kx,
        sequence.ky,
        sequence.kz,
        resolution_mm=resolution_mm,
    )
    maximum_k = tuple(float(np.max(np.abs(values))) for values in scaled_k)
    target_fov = tuple(float(value) for value in config.encoding.target_fov_mm)
    reconstruction_shape = reconstruction_shape_for_trajectory(
        target_fov, resolution_mm, maximum_k
    )
    reconstruction_voxel = tuple(
        fov / size
        for fov, size in zip(target_fov, reconstruction_shape, strict=True)
    )

    transforms = build_coordinate_transforms(
        patient_position=config.phantom.patient_position,
        coordinate_mode=config.sequence.coordinate_mode,
        sequence_orientation=config.sequence.orientation,
    )
    pcs_voxel = np.asarray(config.phantom.voxel_size_mm, dtype=np.float64)
    logical_voxel = np.abs(transforms.pcs_to_logical) @ pcs_voxel
    high_shape_values = np.asarray(target_fov) / logical_voxel
    high_shape = np.rint(high_shape_values).astype(np.int64)
    if not np.allclose(high_shape_values, high_shape, atol=1e-6, rtol=0.0):
        raise FullysampledReferenceError(
            "target FOV must be an integer number of high-resolution voxels"
        )
    high_resolution_shape = tuple(int(value) for value in high_shape)

    coil_info = inspect_sensitivity_map(config.coils.sensitivity_map)
    logical_coil_shape = sensitivity_shape_in_logical_frame(
        coil_info,
        stored_axis_order=config.coils.axis_order,
        dcs_to_logical=transforms.dcs_to_logical,
    )
    if any(
        target > available
        for target, available in zip(
            high_resolution_shape, logical_coil_shape, strict=True
        )
    ):
        raise FullysampledReferenceError(
            f"target high-resolution shape {high_resolution_shape} exceeds "
            f"logical coil support {logical_coil_shape}"
        )

    cache_entry = fullysampled_reference_cache_entry(config)
    output_directory = cache_entry.directory
    output_directory.mkdir(parents=True, exist_ok=True)
    normalization = prepare_rss_normalization(
        coil_info,
        config.run.output_root / "kspace" / "cache" / "sensitivity_rss.npy",
    )
    reference_path = output_directory / "fullysampled_reference_4d.h5"
    metadata_path = output_directory / "fullysampled_reference_metadata.mat"
    kspace_directory = output_directory / "kspace"
    save_kspace = config.outputs.save_fully_sampled_kspace
    if save_kspace:
        kspace_directory.mkdir(parents=True, exist_ok=True)

    trajectory = prepare_physical_sigpy_trajectory(
        scaled_k[0],
        scaled_k[1],
        scaled_k[2],
        fov_mm=target_fov,
        matrix_shape=reconstruction_shape,
    )
    dcf = np.asarray(sequence.density_compensation, dtype=np.float32)
    if dcf.shape != sequence.kx.shape:
        raise FullysampledReferenceError("trajectory and DCF shapes differ")
    flattened_dcf = dcf.T.reshape(-1)
    dcf_maximum = float(np.max(flattened_dcf))
    if not np.isfinite(dcf_maximum) or dcf_maximum <= 0:
        raise FullysampledReferenceError("DCF maximum must be positive")
    flattened_dcf = np.asarray(
        flattened_dcf / dcf_maximum, dtype=np.float32
    )

    backend = SigpyNufftBackend(device_id=config.compute.device_id)
    # SigPy's paired NUFFT normalization and the trajectory DCF have a global
    # gain. Measure it once with a constant object so reconstructed signal
    # remains on the high-resolution contrast scale.
    calibration_image = np.ones(high_resolution_shape, dtype=np.complex64)
    calibration_kspace = backend.forward(calibration_image, trajectory)
    calibration_response = backend.adjoint(
        calibration_kspace * flattened_dcf, trajectory
    )
    calibration_center = tuple(size // 2 for size in reconstruction_shape)
    intensity_scale_value = calibration_response[calibration_center]
    intensity_scale = float(np.abs(intensity_scale_value))
    if not np.isfinite(intensity_scale) or intensity_scale <= 0:
        raise FullysampledReferenceError(
            f"invalid NUFFT intensity calibration: {intensity_scale_value}"
        )
    del calibration_image, calibration_kspace, calibration_response

    metadata = {
        "pulseq_sequence_filename": sequence.sequence_path.name,
        "pulseq_path_at_generation": str(sequence.sequence_path),
        "pulseq_signature_type": sequence.signature_type,
        "pulseq_signature": sequence.signature,
        "pulseq_sequence_sha256": _sha256(sequence.sequence_path),
        "pulseq_metadata_filename": sequence.metadata_path.name,
        "pulseq_metadata_sha256": _sha256(sequence.metadata_path),
        "sequence_fov_mm": np.asarray([sequence.fov_mm], dtype=np.float64),
        "sequence_resolution_mm": np.asarray(
            [sequence.resolution_mm], dtype=np.float64
        ),
        "target_fov_mm": np.asarray([target_fov], dtype=np.float64),
        "high_resolution_shape": np.asarray(
            [high_resolution_shape], dtype=np.int32
        ),
        "reconstruction_shape": np.asarray(
            [reconstruction_shape], dtype=np.int32
        ),
        "reconstruction_voxel_mm": np.asarray(
            [reconstruction_voxel], dtype=np.float64
        ),
        "logical_axis_patient_directions": np.asarray(
            transforms.logical_axis_patient_directions, dtype=object
        ),
        "pcs_to_logical": transforms.pcs_to_logical,
        "frame_indices_one_based": np.arange(1, total + 1, dtype=np.int32),
        "frame_times_s": np.asarray([group.time_s for group in groups]),
        "frame_window_start_s": np.asarray(
            [group.window_start_s for group in groups]
        ),
        "frame_window_end_s": np.asarray(
            [group.window_end_s for group in groups]
        ),
        "xcat_time_step_s": np.asarray(
            [[config.timeline.xcat_time_step_s]], dtype=np.float64
        ),
        "reference_time_step_s": np.asarray(
            [[config.timeline.kspace_time_step_s]], dtype=np.float64
        ),
        "xcat_to_reference": config.timeline.xcat_to_kspace,
        "kx_per_m": np.asarray(scaled_k[0], dtype=np.float32),
        "ky_per_m": np.asarray(scaled_k[1], dtype=np.float32),
        "kz_per_m": np.asarray(scaled_k[2], dtype=np.float32),
        "dcf": dcf,
        "trajectory_scale_factor": np.asarray([[scale_factor]]),
        "target_kmax_per_m": np.asarray([[target_kmax]]),
        "nufft_oversampling": np.asarray([[1.5]], dtype=np.float64),
        "nufft_kernel_width": np.asarray([[4.0]], dtype=np.float64),
        "adjoint_intensity_scale": np.asarray(
            [[intensity_scale]], dtype=np.float64
        ),
        "coil_combination": "sum(conj(S_normalized) * adjoint_coil)",
        "rf_shift_application": "already-applied-in-high-resolution-contrast",
        "save_fully_sampled_kspace": np.asarray(
            [[int(save_kspace)]], dtype=np.uint8
        ),
    }
    _atomic_savemat(metadata_path, metadata)

    reference_handle = h5py.File(reference_path, "a")
    generated = 0
    reused = 0
    try:
        reference_dataset, completion_dataset = _ensure_reference_datasets(
            reference_handle, reconstruction_shape + (total,)
        )
        reference_dataset.attrs["target_fov_mm"] = target_fov
        reference_dataset.attrs["voxel_size_mm"] = reconstruction_voxel
        reference_dataset.attrs["pulseq_sequence_filename"] = (
            sequence.sequence_path.name
        )
        reference_dataset.attrs["pulseq_signature"] = sequence.signature
        reference_dataset.attrs["coil_combination"] = (
            "sum(conj(S_normalized) * adjoint_coil)"
        )
        reference_dataset.attrs["adjoint_intensity_scale"] = intensity_scale
        for zero_based in range(start_frame - 1, resolved_end):
            group = groups[zero_based]
            destination = kspace_directory / (
                f"tissue_kspace_frame_{group.index:04d}.mat"
            )
            reference_valid = bool(completion_dataset[zero_based])
            kspace_valid = destination.is_file()
            need_reference = overwrite or not reference_valid
            need_kspace = save_kspace and (overwrite or not kspace_valid)
            if not need_reference and not need_kspace:
                reused += 1
                if progress is not None:
                    progress(
                        f"Reference frame {group.index}/{total}: reused"
                    )
                continue

            images = [
                load_contrast_image(contrast_frame_path(config, frame.index))[1]
                for frame in group.xcat_frames
            ]
            pcs_image = (
                _average_images(images)
                if config.timeline.xcat_to_kspace == "average"
                else images[len(images) // 2]
            )
            logical_image = reorient_spatial_array(
                pcs_image, transforms.pcs_to_logical
            )
            high_resolution_image = centered_resize(
                logical_image, high_resolution_shape
            ).astype(np.float32, copy=False)
            del images, pcs_image, logical_image

            low_sensitivities = np.empty(
                (coil_info.coil_count,) + reconstruction_shape,
                dtype=np.complex64,
            )
            adjoint_coils = np.empty_like(low_sensitivities)
            saved_values = (
                np.empty(
                    (
                        trajectory.sample_count,
                        trajectory.arm_count,
                        coil_info.coil_count,
                    ),
                    dtype=np.complex64,
                )
                if save_kspace
                else None
            )
            for coil_index in range(coil_info.coil_count):
                logical_coil = load_normalized_coil_in_logical_frame(
                    coil_info,
                    coil_index,
                    normalization,
                    stored_axis_order=config.coils.axis_order,
                    dcs_to_logical=transforms.dcs_to_logical,
                )
                high_resolution_coil = centered_resize(
                    logical_coil, high_resolution_shape
                ).astype(np.complex64, copy=False)
                low_sensitivities[coil_index] = _resample_complex(
                    high_resolution_coil, reconstruction_shape
                )
                flattened = backend.forward(
                    np.asarray(
                        high_resolution_image * high_resolution_coil,
                        dtype=np.complex64,
                    ),
                    trajectory,
                )
                adjoint_coils[coil_index] = backend.adjoint(
                    flattened * flattened_dcf, trajectory
                )
                if saved_values is not None:
                    saved_values[:, :, coil_index] = (
                        trajectory.reshape_kspace(flattened)
                    )
                del logical_coil, high_resolution_coil, flattened
                if progress is not None:
                    progress(
                        f"Reference frame {group.index}/{total}: coil "
                        f"{coil_index + 1}/{coil_info.coil_count}"
                    )

            low_rss = np.sqrt(
                np.sum(
                    np.abs(low_sensitivities) ** 2,
                    axis=0,
                    dtype=np.float64,
                )
            ).astype(np.float32)
            supported = low_rss > np.finfo(np.float32).eps
            low_sensitivities = np.divide(
                low_sensitivities,
                low_rss[None, ...],
                out=np.zeros_like(low_sensitivities),
                where=supported[None, ...],
            )
            reference = np.sum(
                np.conj(low_sensitivities) * adjoint_coils,
                axis=0,
                dtype=np.complex64,
            )
            reference /= np.complex64(intensity_scale)
            if not np.all(np.isfinite(reference)):
                raise FullysampledReferenceError(
                    f"reference frame {group.index} contains non-finite values"
                )
            reference_dataset[:, :, :, zero_based] = reference
            completion_dataset[zero_based] = 1
            reference_handle.flush()
            if saved_values is not None:
                _atomic_savemat(destination, {"kspace": saved_values})
            generated += 1
            del (
                high_resolution_image,
                low_sensitivities,
                adjoint_coils,
                low_rss,
                supported,
                reference,
                saved_values,
            )
    finally:
        reference_handle.close()

    with h5py.File(reference_path, "r") as handle:
        completed = np.asarray(handle["frame_complete"], dtype=np.uint8)
    complete_indices = [
        index + 1 for index, value in enumerate(completed) if value
    ]
    outputs: list[Path] = [metadata_path, reference_path]
    if save_kspace:
        outputs.extend(
            path
            for path in sorted(kspace_directory.glob("tissue_kspace_frame_*.mat"))
        )
    status = "complete" if len(complete_indices) == total else "partial"
    write_artifact_manifest(
        cache_entry,
        status=status,
        frame_count=total,
        completed_frame_indices=complete_indices,
        outputs=outputs,
    )
    stage_manifest = (
        write_stage_manifest(
            config, "fullysampled_reference", outputs
        )
        if status == "complete"
        else None
    )
    return FullysampledReferenceResult(
        output_directory=output_directory,
        reference_path=reference_path,
        metadata_path=metadata_path,
        manifest_path=stage_manifest,
        total_frame_count=total,
        selected_frame_count=resolved_end - start_frame + 1,
        generated_frame_count=generated,
        reused_frame_count=reused,
        target_fov_mm=target_fov,
        high_resolution_shape=high_resolution_shape,
        reconstruction_shape=reconstruction_shape,
        reconstruction_voxel_size_mm=reconstruction_voxel,
        saved_kspace=save_kspace,
        elapsed_s=time.perf_counter() - started,
    )


def format_fullysampled_reference(
    result: FullysampledReferenceResult,
) -> str:
    return "\n".join(
        (
            "Fully sampled tissue reference",
            f"Frames:              {result.total_frame_count}",
            f"Selected:            {result.selected_frame_count}",
            f"Generated/reused:    {result.generated_frame_count}/"
            f"{result.reused_frame_count}",
            f"Target FOV (mm):     {result.target_fov_mm}",
            f"High-res grid:       {result.high_resolution_shape}",
            f"Reference grid:      {result.reconstruction_shape}",
            f"Voxel size (mm):     {result.reconstruction_voxel_size_mm}",
            f"K-space retained:    {result.saved_kspace}",
            f"Reference:           {result.reference_path}",
            f"Metadata:            {result.metadata_path}",
            f"Elapsed:             {result.elapsed_s:.1f} s",
        )
    )
