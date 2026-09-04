"""Resumable per-frame cache of fully sampled tissue-only k-space."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
import time
from typing import TYPE_CHECKING, Callable, Iterable

import h5py
import numpy as np
from scipy.ndimage import zoom
from scipy.io import savemat, whosmat

from xcat_icmr.cache import (
    artifact_cache_status,
    contrast_cache_entry,
    contrast_frame_path,
    tissue_kspace_cache_entry,
    write_artifact_manifest,
    write_stage_manifest,
)
from xcat_icmr.coils import (
    inspect_sensitivity_map,
    load_normalized_coil_in_logical_frame,
    prepare_rss_normalization,
    sensitivity_shape_in_logical_frame,
)
from xcat_icmr.encoding.inputs import (
    load_contrast_image,
    prepare_contrast_array_for_encoding,
)
from xcat_icmr.encoding.sigpy_backend import SigpyNufftBackend
from xcat_icmr.encoding.trajectory import (
    prepare_encoding_grids,
    scale_isotropic_trajectory_to_resolution,
)
from xcat_icmr.encoding.validation import encode_multicoil_frame
from xcat_icmr.phantom import XcatFrame, plan_xcat_frames
from xcat_icmr.sequence import build_coordinate_transforms, read_sequence
from xcat_icmr.signal import read_pulseq_excitation

if TYPE_CHECKING:
    from xcat_icmr.config.models import SimulationConfig


class TissueKspaceCacheError(ValueError):
    """Raised when tissue-only k-space cannot be cached safely."""


@dataclass(frozen=True)
class TissueKspaceCacheResult:
    """Completed or resumed tissue-only frame-cache generation."""

    output_directory: Path
    metadata_path: Path
    reference_path: Path
    manifest_path: Path | None
    xcat_frame_count: int
    total_frame_count: int
    xcat_frames_per_reference_frame: int
    aggregation_method: str
    selected_frame_count: int
    generated_frame_count: int
    reused_frame_count: int
    kspace_shape: tuple[int, int, int]
    elapsed_s: float


@dataclass(frozen=True)
class TissueTemporalGroup:
    """High-resolution XCAT frames represented by one k-space time point."""

    index: int
    xcat_frames: tuple[XcatFrame, ...]
    window_start_s: float
    window_end_s: float
    representative_time_s: float


def _build_temporal_groups(
    frames: tuple[XcatFrame, ...],
    *,
    frames_per_group: int,
    xcat_time_step_s: float,
) -> tuple[TissueTemporalGroup, ...]:
    """Partition one motion cycle into uniform k-space time windows."""

    if not frames:
        raise TissueKspaceCacheError("the XCAT motion cycle has no frames")
    if frames_per_group <= 0:
        raise TissueKspaceCacheError("frames_per_group must be positive")
    if len(frames) == 1:
        groups = (frames,)
    else:
        if len(frames) % frames_per_group:
            raise TissueKspaceCacheError(
                f"{len(frames)} XCAT frames cannot be divided into uniform "
                f"groups of {frames_per_group}; choose a reference_time_step_s "
                "that divides the complete motion cycle"
            )
        groups = tuple(
            frames[start : start + frames_per_group]
            for start in range(0, len(frames), frames_per_group)
        )
    return tuple(
        TissueTemporalGroup(
            index=index,
            xcat_frames=group,
            window_start_s=group[0].time_s,
            window_end_s=(
                group[0].time_s + len(group) * xcat_time_step_s
            ),
            representative_time_s=(
                group[0].time_s + len(group) * xcat_time_step_s / 2.0
            ),
        )
        for index, group in enumerate(groups, start=1)
    )


def _average_temporal_images(images: Iterable[np.ndarray]) -> np.ndarray:
    """Average images in-place while retaining at most two volumes at once."""

    accumulator: np.ndarray | None = None
    count = 0
    for image in images:
        values = np.asarray(image, dtype=np.float32)
        if accumulator is None:
            accumulator = values
        else:
            if values.shape != accumulator.shape:
                raise TissueKspaceCacheError(
                    "XCAT contrast shapes differ within a temporal group"
                )
            np.add(accumulator, values, out=accumulator)
            del values
        count += 1
    if accumulator is None or count == 0:
        raise TissueKspaceCacheError("cannot average an empty temporal group")
    accumulator *= np.float32(1.0 / count)
    if not np.all(np.isfinite(accumulator)):
        raise TissueKspaceCacheError(
            "temporally averaged contrast contains non-finite values"
        )
    return accumulator


def _valid_kspace(path: Path, shape: tuple[int, int, int]) -> bool:
    if not path.is_file():
        return False
    try:
        entries = {
            name: (saved_shape, dtype)
            for name, saved_shape, dtype in whosmat(path)
        }
    except (OSError, ValueError):
        return False
    return entries == {"kspace": (shape, "single")}


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
    *,
    overwrite: bool,
) -> tuple[h5py.Dataset, h5py.Dataset]:
    """Create or validate the resumable 4-D complex reference schema."""

    image = handle.get("image")
    completion = handle.get("frame_complete")
    image_valid = (
        isinstance(image, h5py.Dataset)
        and image.shape == shape
        and image.dtype == np.dtype(np.complex64)
    )
    completion_valid = (
        isinstance(completion, h5py.Dataset)
        and completion.shape == (shape[3],)
        and completion.dtype == np.dtype(np.uint8)
    )
    if image_valid and completion_valid:
        return image, completion
    if (image is not None or completion is not None) and not overwrite:
        raise TissueKspaceCacheError(
            "existing 4-D reference has an incompatible schema; "
            "pass --overwrite to replace it"
        )
    for name in ("image", "frame_complete"):
        if name in handle:
            del handle[name]
    image = handle.create_dataset(
        "image",
        shape=shape,
        dtype=np.complex64,
        chunks=(shape[0], shape[1], min(8, shape[2]), 1),
    )
    completion = handle.create_dataset(
        "frame_complete", shape=(shape[3],), dtype=np.uint8
    )
    image.attrs["axis_order"] = "logical_x,y,z,time"
    image.attrs["dtype_name"] = "complex64"
    return image, completion


def generate_tissue_kspace_cache(
    config: "SimulationConfig",
    *,
    start_frame: int = 1,
    end_frame: int | None = None,
    overwrite: bool = False,
    progress: Callable[[str], None] | None = None,
) -> TissueKspaceCacheResult:
    """Encode selected temporally aggregated tissue k-space frames."""

    started = time.perf_counter()
    contrast_status = artifact_cache_status(contrast_cache_entry(config))
    if contrast_status.state != "HIT":
        raise TissueKspaceCacheError(
            "the high-resolution contrast cache does not match the current "
            f"configuration: {contrast_status.state} ({contrast_status.reason}); "
            "generate or adopt the complete "
            "contrast cycle before encoding tissue k-space"
        )
    if not config.coils.enabled or config.coils.sensitivity_map is None:
        raise TissueKspaceCacheError("an enabled sensitivity map is required")
    if not config.coils.normalize:
        raise TissueKspaceCacheError("coils.normalize must be true")
    aggregation_method = config.timeline.xcat_to_reference
    if aggregation_method == "trajectory-aware":
        raise NotImplementedError(
            "trajectory-aware XCAT-to-k-space aggregation is not implemented"
        )

    frames = plan_xcat_frames(config, debug_one_frame=False)
    xcat_frame_count = len(frames.frames)
    frames_per_group = config.timeline.xcat_frames_per_reference_frame
    temporal_groups = _build_temporal_groups(
        frames.frames,
        frames_per_group=frames_per_group,
        xcat_time_step_s=config.timeline.xcat_time_step_s,
    )
    total = len(temporal_groups)
    resolved_end = total if end_frame is None else end_frame
    if not 1 <= start_frame <= total:
        raise TissueKspaceCacheError(
            f"start k-space frame must be between 1 and {total}"
        )
    if not start_frame <= resolved_end <= total:
        raise TissueKspaceCacheError(
            f"end k-space frame must be between {start_frame} and {total}"
        )

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
    cache_entry = tissue_kspace_cache_entry(config)
    normalization = prepare_rss_normalization(
        coil_info,
        cache_entry.directory / "sensitivity_rss.npy",
    )
    excitation = read_pulseq_excitation(sequence.sequence_path)
    pcs_voxel = np.asarray(config.phantom.voxel_size_mm, dtype=np.float64)
    logical_voxel = np.abs(transforms.pcs_to_logical) @ pcs_voxel
    resolution = np.asarray(sequence.resolution_mm, dtype=np.float64).reshape(-1)
    if resolution.size != 1:
        raise TissueKspaceCacheError("isotropic sequence resolution is required")
    scaled_k, scale_factor, target_kmax = scale_isotropic_trajectory_to_resolution(
        sequence.kx,
        sequence.ky,
        sequence.kz,
        resolution_mm=float(resolution[0]),
    )
    grids = prepare_encoding_grids(
        ground_truth_shape=logical_shape,
        ground_truth_voxel_size_mm=config.phantom.voxel_size_mm,
        sequence_resolution_mm=sequence.resolution_mm,
    )
    expected_kspace_shape = (
        sequence.sample_count,
        sequence.arm_count,
        coil_info.coil_count,
    )
    output_directory = cache_entry.directory
    kspace_directory = output_directory / "kspace"
    output_directory.mkdir(parents=True, exist_ok=True)
    frame_paths = tuple(
        kspace_directory / f"tissue_kspace_frame_{index:04d}.mat"
        for index in range(1, total + 1)
    )
    reference_path = output_directory / "fullysampled_reference_4d.h5"
    reference_shape = grids.acquisition_matrix_shape + (total,)

    def load_coil(index: int) -> np.ndarray:
        return load_normalized_coil_in_logical_frame(
            coil_info,
            index,
            normalization,
            stored_axis_order=config.coils.axis_order,
            dcs_to_logical=transforms.dcs_to_logical,
        )

    reference_handle = h5py.File(reference_path, "a")
    try:
        reference_dataset, completion_dataset = _ensure_reference_datasets(
            reference_handle,
            reference_shape,
            overwrite=overwrite,
        )

        generated = 0
        reused = 0
        selected_indices = range(start_frame - 1, resolved_end)
        for zero_based in selected_indices:
            group = temporal_groups[zero_based]
            destination = frame_paths[zero_based]
            kspace_valid = _valid_kspace(destination, expected_kspace_shape)
            reference_valid = bool(completion_dataset[zero_based])
            need_kspace = overwrite or not kspace_valid
            need_reference = overwrite or not reference_valid
            if not need_kspace and not need_reference:
                reused += 1
                if progress is not None:
                    progress(
                        f"Tissue frame {group.index}/{total}: reused {destination}"
                    )
                continue
            if destination.exists() and not kspace_valid and not overwrite:
                raise TissueKspaceCacheError(
                    f"existing frame is invalid: {destination}; pass --overwrite"
                )

            def load_xcat_frame(frame: XcatFrame) -> np.ndarray:
                return load_contrast_image(
                    contrast_frame_path(config, frame.index)
                )[1]

            if aggregation_method == "average":
                pcs_image = _average_temporal_images(
                    load_xcat_frame(frame) for frame in group.xcat_frames
                )
            elif aggregation_method == "center":
                center_frame = group.xcat_frames[len(group.xcat_frames) // 2]
                pcs_image = load_xcat_frame(center_frame)
            else:
                raise TissueKspaceCacheError(
                    f"unsupported XCAT aggregation method: {aggregation_method}"
                )
            encoding_image = prepare_contrast_array_for_encoding(
                pcs_image,
                logical_shape,
                source_path=contrast_frame_path(
                    config, group.xcat_frames[0].index
                ),
                source_to_target=transforms.pcs_to_logical,
                source_frame="XCAT PCS [Sag, Cor, Tra]",
                target_frame="Pulseq logical [x, y, z]",
                target_axis_patient_directions=(
                    transforms.logical_axis_patient_directions
                ),
            ).image
            del pcs_image

            def coil_progress(completed: int, coil_total: int) -> None:
                if progress is not None:
                    progress(
                        f"Tissue frame {group.index}/{total}: "
                        f"coil {completed}/{coil_total}"
                    )

            shifted_image = None
            if need_kspace:
                encoding = encode_multicoil_frame(
                    encoding_image,
                    coil_count=coil_info.coil_count,
                    coil_loader=load_coil,
                    kx_per_m=scaled_k[0],
                    ky_per_m=scaled_k[1],
                    kz_per_m=scaled_k[2],
                    density_compensation=sequence.density_compensation,
                    encoding_grids=grids,
                    rf_center_shift_mm=config.sequence.rf_profile.center_shift_mm,
                    rf_axis_voxel_size_mm=float(
                        logical_voxel[excitation.logical_axis]
                    ),
                    rf_logical_axis=excitation.logical_axis,
                    device_id=config.compute.device_id,
                    compute_adjoint=False,
                    progress=coil_progress,
                )
                _atomic_savemat(destination, {"kspace": encoding.kspace})
                if not _valid_kspace(destination, expected_kspace_shape):
                    raise TissueKspaceCacheError(
                        f"saved tissue k-space failed verification: {destination}"
                    )
                shifted_image = encoding.shifted_ground_truth
            if need_reference:
                if shifted_image is None:
                    from xcat_icmr.encoding.validation import (
                        circular_shift_to_rf_center,
                    )

                    shifted_image, _ = circular_shift_to_rf_center(
                        encoding_image,
                        center_shift_mm=config.sequence.rf_profile.center_shift_mm,
                        voxel_size_mm=float(
                            logical_voxel[excitation.logical_axis]
                        ),
                        logical_axis=excitation.logical_axis,
                    )
                factors = tuple(
                    target / source
                    for source, target in zip(
                        shifted_image.shape,
                        grids.acquisition_matrix_shape,
                        strict=True,
                    )
                )
                reference = zoom(
                    shifted_image,
                    factors,
                    order=1,
                    mode="constant",
                    prefilter=False,
                ).astype(np.complex64)
                if reference.shape != grids.acquisition_matrix_shape:
                    raise TissueKspaceCacheError(
                        f"reference shape {reference.shape} does not match "
                        f"{grids.acquisition_matrix_shape}"
                    )
                reference_dataset[:, :, :, zero_based] = reference
                completion_dataset[zero_based] = 1
                reference_handle.flush()
            generated += 1
            if progress is not None:
                progress(
                    f"Tissue frame {group.index}/{total}: cached "
                    f"{len(group.xcat_frames)} XCAT frame(s)"
                )
    finally:
        reference_handle.close()

    backend = SigpyNufftBackend(device_id=config.compute.device_id)
    metadata_path = output_directory / "tissue_kspace_metadata.mat"
    frame_times = np.asarray(
        [group.representative_time_s for group in temporal_groups],
        dtype=np.float64,
    )
    window_start_times = np.asarray(
        [group.window_start_s for group in temporal_groups], dtype=np.float64
    )
    window_end_times = np.asarray(
        [group.window_end_s for group in temporal_groups], dtype=np.float64
    )
    source_frame_indices = np.asarray(
        [
            [frame.index for frame in group.xcat_frames]
            for group in temporal_groups
        ],
        dtype=np.int32,
    )
    source_frame_times = np.asarray(
        [
            [frame.time_s for frame in group.xcat_frames]
            for group in temporal_groups
        ],
        dtype=np.float64,
    )
    applied_shift = -int(
        np.rint(
            config.sequence.rf_profile.center_shift_mm
            / float(logical_voxel[excitation.logical_axis])
        )
    )
    _atomic_savemat(
        metadata_path,
        {
            "frame_indices_one_based": np.arange(1, total + 1, dtype=np.int32),
            "frame_times_s": frame_times,
            "frame_window_start_s": window_start_times,
            "frame_window_end_s": window_end_times,
            "source_xcat_frame_indices_one_based": source_frame_indices,
            "source_xcat_frame_times_s": source_frame_times,
            "xcat_time_step_s": np.asarray(
                [[config.timeline.xcat_time_step_s]], dtype=np.float64
            ),
            "reference_time_step_s": np.asarray(
                [[config.timeline.reference_time_step_s]], dtype=np.float64
            ),
            "xcat_frames_per_reference_frame": np.asarray(
                [[frames_per_group]], dtype=np.int32
            ),
            "xcat_to_reference": aggregation_method,
            "kspace_shape": np.asarray([expected_kspace_shape], dtype=np.int32),
            "kspace_dtype": "complex64",
            "kx_per_m": np.asarray(scaled_k[0], dtype=np.float32),
            "ky_per_m": np.asarray(scaled_k[1], dtype=np.float32),
            "kz_per_m": np.asarray(scaled_k[2], dtype=np.float32),
            "dcf": np.asarray(sequence.density_compensation, dtype=np.float32),
            "fov_mm": np.asarray([grids.acquisition_fov_mm]),
            "matrix_shape": np.asarray(
                [grids.acquisition_matrix_shape], dtype=np.int32
            ),
            "logical_axis_patient_directions": np.asarray(
                transforms.logical_axis_patient_directions, dtype=object
            ),
            "pcs_to_logical": transforms.pcs_to_logical,
            "rf_center_shift_mm": np.asarray(
                [[config.sequence.rf_profile.center_shift_mm]], dtype=np.float32
            ),
            "rf_logical_axis_zero_based": np.asarray(
                [[excitation.logical_axis]], dtype=np.int32
            ),
            "applied_circular_shift_voxels": np.asarray(
                [[applied_shift]], dtype=np.int32
            ),
            "trajectory_scale_factor": np.asarray([[scale_factor]]),
            "target_kmax_per_m": np.asarray([[target_kmax]]),
            "nufft_oversampling": np.asarray([[backend.oversampling]]),
            "nufft_kernel_width": np.asarray([[backend.kernel_width]]),
        },
    )

    manifest_path = None
    with h5py.File(reference_path, "r") as reference_handle:
        references_complete = bool(
            np.all(np.asarray(reference_handle["frame_complete"], dtype=np.uint8))
        )
    complete_indices = [
        index
        for index, path in enumerate(frame_paths, start=1)
        if _valid_kspace(path, expected_kspace_shape)
    ]
    if len(complete_indices) == total and references_complete:
        write_artifact_manifest(
            cache_entry,
            status="complete",
            frame_count=total,
            completed_frame_indices=complete_indices,
            outputs=[metadata_path, reference_path, *frame_paths],
        )
        manifest_path = write_stage_manifest(
            config,
            "fullysampled_kspace",
            [metadata_path, reference_path, *frame_paths],
        )
    return TissueKspaceCacheResult(
        output_directory=output_directory,
        metadata_path=metadata_path,
        reference_path=reference_path,
        manifest_path=manifest_path,
        xcat_frame_count=xcat_frame_count,
        total_frame_count=total,
        xcat_frames_per_reference_frame=frames_per_group,
        aggregation_method=aggregation_method,
        selected_frame_count=resolved_end - start_frame + 1,
        generated_frame_count=generated,
        reused_frame_count=reused,
        kspace_shape=expected_kspace_shape,
        elapsed_s=time.perf_counter() - started,
    )


def format_tissue_kspace_cache(result: TissueKspaceCacheResult) -> str:
    """Format a completed or partially completed tissue-cache run."""

    return "\n".join(
        (
            "Tissue-only fully sampled k-space cache",
            f"XCAT frames:       {result.xcat_frame_count}",
            f"Aggregation:       {result.xcat_frames_per_reference_frame} XCAT "
            f"frame(s), {result.aggregation_method}",
            f"K-space frames:    {result.total_frame_count}",
            f"Selected frames:   {result.selected_frame_count}",
            f"Generated:         {result.generated_frame_count}",
            f"Reused:            {result.reused_frame_count}",
            f"K-space shape:     {result.kspace_shape}",
            f"Elapsed:           {result.elapsed_s:.3f} s",
            f"Output directory:  {result.output_directory}",
            f"Shared metadata:   {result.metadata_path}",
            f"4-D reference:     {result.reference_path}",
            "Stage manifest:    "
            + (
                str(result.manifest_path)
                if result.manifest_path is not None
                else "not written until all frames exist"
            ),
        )
    )
