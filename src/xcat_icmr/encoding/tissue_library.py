"""Persistent full-trajectory tissue k-space library, one file per cardiac phase."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
import time
from typing import TYPE_CHECKING, Callable

import h5py
import numpy as np
from scipy.io import savemat

from xcat_icmr.acquisition.storage import (
    StorageEstimate,
    estimate_tissue_library_storage,
    require_free_space,
)
from xcat_icmr.cache import (
    artifact_cache_status,
    label_cache_entry,
    tissue_kspace_cache_entry,
    stage_manifest_path,
    write_artifact_manifest,
    write_stage_manifest,
)
from xcat_icmr.coils import (
    inspect_sensitivity_map,
    load_normalized_coil_roi_in_logical_frame,
    prepare_rss_normalization,
    sensitivity_shape_in_logical_frame,
)
from xcat_icmr.encoding.fullysampled_reference import centered_resize
from xcat_icmr.encoding.fov_centering import rf_centering_phase_ramp
from xcat_icmr.encoding.sigpy_backend import SigpyNufftSession
from xcat_icmr.encoding.trajectory import (
    prepare_physical_sigpy_trajectory,
    scale_isotropic_trajectory_to_resolution,
)
from xcat_icmr.phantom import plan_xcat_frames
from xcat_icmr.sequence import build_coordinate_transforms, read_sequence
from xcat_icmr.sequence.orientation import reorient_spatial_array
from xcat_icmr.signal import (
    calculate_rf_profile_bssfp_contrast,
    generate_slice_profile,
    read_pulseq_excitation,
)
from xcat_icmr.tissue import get_tissue_library

if TYPE_CHECKING:
    from xcat_icmr.config.models import SimulationConfig


class TissueKspaceLibraryError(ValueError):
    """Raised when the reusable tissue library cannot be generated safely."""


@dataclass(frozen=True)
class TissueKspaceLibraryPlan:
    output_directory: Path
    frame_count: int
    high_resolution_shape: tuple[int, int, int]
    kspace_shape_per_frame: tuple[int, int, int]
    storage: StorageEstimate
    existing_valid_frames: int
    missing_bytes: int
    free_bytes: int | None


@dataclass(frozen=True)
class TissueKspaceLibraryResult:
    plan: TissueKspaceLibraryPlan
    generated_frame_count: int
    reused_frame_count: int
    debug_contrast_path: Path | None
    manifest_path: Path | None
    elapsed_s: float


def tissue_library_frame_path(config: "SimulationConfig", frame_index: int) -> Path:
    return tissue_kspace_cache_entry(config).directory / "frames" / (
        f"tissue_kspace_phase_{frame_index:04d}.h5"
    )


def _valid_frame(path: Path, expected_shape: tuple[int, int, int]) -> bool:
    if not path.is_file():
        return False
    try:
        with h5py.File(path, "r") as handle:
            data = handle.get("kspace")
            return bool(
                isinstance(data, h5py.Dataset)
                and data.shape == expected_shape
                and data.dtype == np.dtype(np.complex64)
                and bool(data.attrs.get("complete", False))
                and int(data.attrs.get("fov_centering_algorithm", 0)) == 1
            )
    except OSError:
        return False


def _grid(config: "SimulationConfig", sequence, transforms) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[float, float, float], tuple[np.ndarray, ...]]:
    resolution = np.asarray(sequence.resolution_mm, dtype=np.float64).reshape(-1)
    if resolution.size != 1:
        raise TissueKspaceLibraryError("isotropic sequence resolution is required")
    scaled_k, _, _ = scale_isotropic_trajectory_to_resolution(
        sequence.kx, sequence.ky, sequence.kz, resolution_mm=float(resolution[0])
    )
    target_fov = tuple(float(value) for value in config.encoding.target_fov_mm)
    pcs_voxel = np.asarray(config.phantom.voxel_size_mm, dtype=np.float64)
    logical_voxel = np.abs(transforms.pcs_to_logical) @ pcs_voxel
    high_values = np.asarray(target_fov) / logical_voxel
    high = np.rint(high_values).astype(np.int64)
    if not np.allclose(high_values, high, atol=1e-6, rtol=0.0):
        raise TissueKspaceLibraryError(
            "encoding.target_fov_mm must be an integer number of XCAT voxels"
        )
    maximum_k = np.asarray([np.max(np.abs(item)) for item in scaled_k])
    reconstruction = np.maximum(1, np.ceil(2e-3 * np.asarray(target_fov) * maximum_k)).astype(int)
    reconstruction += reconstruction % 2
    return tuple(int(v) for v in high), tuple(int(v) for v in reconstruction), tuple(float(v) for v in logical_voxel), scaled_k


def plan_tissue_kspace_library(
    config: "SimulationConfig", *, check_free_space: bool = False
) -> TissueKspaceLibraryPlan:
    if not config.outputs.cache_full_tissue_kspace_library:
        raise TissueKspaceLibraryError(
            "outputs.cache_full_tissue_kspace_library must be true"
        )
    if not config.coils.enabled or config.coils.sensitivity_map is None:
        raise TissueKspaceLibraryError("an enabled sensitivity map is required")
    frames = plan_xcat_frames(config, debug_one_frame=False).frames
    sequence = read_sequence(config.sequence)
    transforms = build_coordinate_transforms(
        patient_position=config.phantom.patient_position,
        coordinate_mode=config.sequence.coordinate_mode,
        sequence_orientation=config.sequence.orientation,
    )
    high_shape, _, _, _ = _grid(config, sequence, transforms)
    coil_info = inspect_sensitivity_map(config.coils.sensitivity_map)
    frame_shape = (sequence.sample_count, sequence.arm_count, coil_info.coil_count)
    storage = estimate_tissue_library_storage(*frame_shape, len(frames))
    valid = sum(_valid_frame(tissue_library_frame_path(config, i), frame_shape) for i in range(1, len(frames) + 1))
    missing = storage.bytes * (len(frames) - valid) // len(frames)
    free = None
    if check_free_space and missing:
        free = require_free_space(tissue_kspace_cache_entry(config).directory, missing)
    return TissueKspaceLibraryPlan(
        output_directory=tissue_kspace_cache_entry(config).directory,
        frame_count=len(frames),
        high_resolution_shape=high_shape,
        kspace_shape_per_frame=frame_shape,
        storage=storage,
        existing_valid_frames=valid,
        missing_bytes=missing,
        free_bytes=free,
    )


def _atomic_h5(path: Path, kspace: np.ndarray, metadata: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
        with h5py.File(temporary_path, "w") as handle:
            data = handle.create_dataset(
                "kspace",
                data=np.asarray(kspace, dtype=np.complex64),
                chunks=(kspace.shape[0], 1, kspace.shape[2]),
            )
            data.attrs["axis_order"] = "sample,trajectory_TR,coil"
            for key, value in metadata.items():
                data.attrs[key] = value
            data.attrs["complete"] = True
        temporary_path.chmod(0o644)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def generate_tissue_kspace_library(
    config: "SimulationConfig",
    *,
    start_frame: int = 1,
    end_frame: int | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
    progress: Callable[[str], None] | None = None,
) -> TissueKspaceLibraryResult:
    """Compute contrast one phase at a time, encode it, then discard it."""

    started = time.perf_counter()
    if progress:
        progress("Tissue library: validating label cache and storage requirements")
    label_status = artifact_cache_status(label_cache_entry(config))
    if label_status.state != "HIT":
        raise TissueKspaceLibraryError(
            f"tissue-label cache is not complete: {label_status.state} ({label_status.reason})"
        )
    plan = plan_tissue_kspace_library(config, check_free_space=not dry_run)
    if dry_run:
        return TissueKspaceLibraryResult(plan, 0, plan.existing_valid_frames, None, None, time.perf_counter() - started)
    frames = plan_xcat_frames(config, debug_one_frame=False).frames
    end = len(frames) if end_frame is None else end_frame
    if not 1 <= start_frame <= end <= len(frames):
        raise TissueKspaceLibraryError(f"frame interval must lie within 1..{len(frames)}")
    # Invalidate old completion claims before changing any phase. This prevents
    # downstream acquisition stages from consuming a mixture of legacy
    # uncentered and newly RF-centered k-space while regeneration is in flight.
    cache_entry = tissue_kspace_cache_entry(config)
    centered_before = [
        index
        for index in range(1, len(frames) + 1)
        if _valid_frame(
            tissue_library_frame_path(config, index),
            plan.kspace_shape_per_frame,
        )
    ]
    write_artifact_manifest(
        cache_entry,
        status="partial",
        frame_count=len(frames),
        completed_frame_indices=centered_before,
        outputs=[tissue_library_frame_path(config, index) for index in centered_before],
    )
    stage_manifest_path(config, "fullysampled_kspace").unlink(missing_ok=True)
    sequence = read_sequence(config.sequence)
    transforms = build_coordinate_transforms(
        patient_position=config.phantom.patient_position,
        coordinate_mode=config.sequence.coordinate_mode,
        sequence_orientation=config.sequence.orientation,
    )
    high_shape, reconstruction_shape, logical_voxel, scaled_k = _grid(config, sequence, transforms)
    trajectory = prepare_physical_sigpy_trajectory(
        *scaled_k, fov_mm=config.encoding.target_fov_mm, matrix_shape=reconstruction_shape
    )
    coil_info = inspect_sensitivity_map(config.coils.sensitivity_map)
    logical_coil_shape = sensitivity_shape_in_logical_frame(
        coil_info,
        stored_axis_order=config.coils.axis_order,
        dcs_to_logical=transforms.dcs_to_logical,
    )
    if any(a > b for a, b in zip(high_shape, logical_coil_shape, strict=True)):
        raise TissueKspaceLibraryError("target FOV exceeds sensitivity-map support")
    coil_offset = (
        np.asarray(logical_coil_shape, dtype=np.int64)
        - np.asarray(high_shape, dtype=np.int64)
    ) // 2
    coil_slices = tuple(
        slice(int(start), int(start + size))
        for start, size in zip(coil_offset, high_shape, strict=True)
    )
    if progress:
        progress(
            "Tissue library: preparing/reusing sensitivity RSS normalization"
        )
    normalization = prepare_rss_normalization(
        coil_info, plan.output_directory / "sensitivity_rss.npy"
    )
    if progress:
        progress("Tissue library: sensitivity RSS normalization ready")
    excitation = read_pulseq_excitation(sequence.sequence_path)
    profile = generate_slice_profile(
        excitation,
        matrix_size=logical_coil_shape[excitation.logical_axis],
        voxel_size_mm=logical_voxel[excitation.logical_axis],
        center_shift_mm=config.sequence.rf_profile.center_shift_mm,
    )
    tissue = get_tissue_library(config.sequence.contrast.tissue_library)
    session = SigpyNufftSession(
        trajectory, device_id=config.compute.device_id
    )
    coil_batch_size = 4
    if progress:
        progress(
            f"Tissue library: uploading {coil_info.coil_count} static coil "
            f"ROIs to {session.device_name}"
        )
    device_coils = session.empty(
        (coil_info.coil_count,) + high_shape, dtype=np.complex64
    )
    for coil_index in range(coil_info.coil_count):
        coil = load_normalized_coil_roi_in_logical_frame(
            coil_info,
            coil_index,
            normalization,
            coil_slices,
            stored_axis_order=config.coils.axis_order,
            dcs_to_logical=transforms.dcs_to_logical,
        )
        if coil.shape != high_shape:
            raise TissueKspaceLibraryError(
                f"coil ROI shape {coil.shape} does not match {high_shape}"
            )
        with session.device:
            device_coils[coil_index] = session.upload(
                coil, dtype=np.complex64
            )
        del coil
        if progress:
            progress(
                f"Tissue library GPU coil preparation: {coil_index + 1}/"
                f"{coil_info.coil_count}"
            )
    phase_ramp = rf_centering_phase_ramp(
        scaled_k[excitation.logical_axis],
        rf_center_shift_mm=config.sequence.rf_profile.center_shift_mm,
    ).T.reshape(-1)
    device_phase_ramp = session.upload(phase_ramp, dtype=np.complex64)
    del phase_ramp
    if progress:
        progress(
            f"Tissue library: trajectory, RF phase ramp, and coil maps are "
            f"resident on {session.device_name}; coil batch size {coil_batch_size}"
        )
    generated = reused = 0
    debug_path = plan.output_directory / "debug_contrast_frame.mat" if config.outputs.save_debug_contrast_frame else None
    for frame in frames[start_frame - 1 : end]:
        destination = tissue_library_frame_path(config, frame.index)
        partial = destination.with_name(f"{destination.stem}.partial.h5")
        if _valid_frame(destination, plan.kspace_shape_per_frame) and not overwrite:
            reused += 1
            if progress:
                progress(f"Tissue phase {frame.index}/{len(frames)}: reused")
            continue
        if overwrite:
            destination.unlink(missing_ok=True)
            partial.unlink(missing_ok=True)
        if frame.label_path is None or not frame.label_path.is_file():
            raise TissueKspaceLibraryError(f"missing label frame: {frame.label_path}")
        if progress:
            progress(
                f"Tissue phase {frame.index}/{len(frames)}: calculating "
                "transient high-resolution bSSFP contrast"
            )
        pcs_contrast, _, _, _ = calculate_rf_profile_bssfp_contrast(
            label_path=frame.label_path,
            profile=profile,
            transforms=transforms,
            pcs_voxel_size_mm=config.phantom.voxel_size_mm,
            library=tissue,
            te_ms=sequence.te_ms,
            tr_ms=sequence.tr_ms,
        )
        logical = reorient_spatial_array(pcs_contrast, transforms.pcs_to_logical)
        high = centered_resize(logical, high_shape).astype(np.float32, copy=False)
        if progress:
            progress(
                f"Tissue phase {frame.index}/{len(frames)}: contrast ready; "
                f"encoding {coil_info.coil_count} coils"
            )
        if debug_path is not None and frame.index == config.outputs.debug_contrast_frame:
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            savemat(debug_path, {"contrast_pcs": pcs_contrast, "contrast_logical": high}, appendmat=False, do_compression=False)
        del pcs_contrast, logical
        device_high = session.upload(high, dtype=np.float32)
        partial.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(partial, "a") as handle:
            kspace = handle.get("kspace")
            coil_complete = handle.get("coil_complete")
            if kspace is None and coil_complete is None:
                kspace = handle.create_dataset(
                    "kspace",
                    shape=plan.kspace_shape_per_frame,
                    dtype=np.complex64,
                    # One arm is the random-access unit used by the later
                    # dynamic acquisition; retain every coil in that chunk.
                    chunks=(
                        plan.kspace_shape_per_frame[0],
                        1,
                        plan.kspace_shape_per_frame[2],
                    ),
                )
                coil_complete = handle.create_dataset(
                    "coil_complete",
                    shape=(coil_info.coil_count,),
                    dtype=np.uint8,
                )
                kspace.attrs["axis_order"] = "sample,trajectory_TR,coil"
                kspace.attrs["phase_index_one_based"] = frame.index
                kspace.attrs["phase_time_s"] = frame.time_s
                kspace.attrs["pulseq_sequence_filename"] = sequence.sequence_path.name
                kspace.attrs["pulseq_signature"] = sequence.signature
                kspace.attrs["fov_centering"] = "rf-profile"
                kspace.attrs["fov_centering_algorithm"] = 1
                kspace.attrs["rf_center_shift_mm"] = (
                    config.sequence.rf_profile.center_shift_mm
                )
                kspace.attrs["rf_logical_axis_zero_based"] = (
                    excitation.logical_axis
                )
                kspace.attrs["coil_sensitivity_spatial_shift_mm"] = 0.0
            valid_partial = (
                isinstance(kspace, h5py.Dataset)
                and kspace.shape == plan.kspace_shape_per_frame
                and kspace.dtype == np.dtype(np.complex64)
                and isinstance(coil_complete, h5py.Dataset)
                and coil_complete.shape == (coil_info.coil_count,)
            )
            if not valid_partial:
                raise TissueKspaceLibraryError(
                    f"partial phase checkpoint is incompatible: {partial}; "
                    "pass --overwrite to replace it"
                )
            incomplete = [
                index
                for index in range(coil_info.coil_count)
                if not bool(coil_complete[index])
            ]
            for coil_index in range(coil_info.coil_count):
                if coil_index not in incomplete and progress:
                    progress(
                        f"Tissue phase {frame.index}/{len(frames)}: coil "
                        f"{coil_index + 1}/{coil_info.coil_count} reused"
                    )
            for batch_start in range(0, len(incomplete), coil_batch_size):
                batch_indices = incomplete[
                    batch_start : batch_start + coil_batch_size
                ]
                if progress:
                    shown = ",".join(str(index + 1) for index in batch_indices)
                    progress(
                        f"Tissue phase {frame.index}/{len(frames)}: GPU forward "
                        f"NUFFT coils [{shown}]"
                    )
                with session.device:
                    weighted = (
                        device_high[None, ...] * device_coils[batch_indices]
                    )
                    encoded_device = session.forward_device(weighted)
                    encoded_device *= device_phase_ramp[None, :]
                encoded_batch = session.download(encoded_device).astype(
                    np.complex64, copy=False
                )
                del weighted, encoded_device
                for batch_position, coil_index in enumerate(batch_indices):
                    kspace[:, :, coil_index] = trajectory.reshape_kspace(
                        encoded_batch[batch_position]
                    )
                    coil_complete[coil_index] = 1
                handle.flush()
                del encoded_batch
                if progress:
                    shown = ",".join(str(index + 1) for index in batch_indices)
                    progress(
                        f"Tissue phase {frame.index}/{len(frames)}: coils "
                        f"[{shown}] checkpointed"
                    )
            kspace.attrs["complete"] = True
            del kspace, coil_complete
        os.replace(partial, destination)
        del high, device_high
        generated += 1
        if progress:
            progress(
                f"Tissue phase {frame.index}/{len(frames)}: saved {destination}"
            )
    valid_indices = [
        i
        for i in range(1, len(frames) + 1)
        if _valid_frame(
            tissue_library_frame_path(config, i), plan.kspace_shape_per_frame
        )
    ]
    valid_paths = [tissue_library_frame_path(config, i) for i in valid_indices]
    manifest = None
    if len(valid_paths) == len(frames):
        write_artifact_manifest(
            tissue_kspace_cache_entry(config),
            status="complete",
            frame_count=len(frames),
            completed_frame_indices=list(range(1, len(frames) + 1)),
            outputs=valid_paths,
        )
        manifest = write_stage_manifest(config, "fullysampled_kspace", valid_paths)
    else:
        write_artifact_manifest(
            tissue_kspace_cache_entry(config),
            status="partial",
            frame_count=len(frames),
            completed_frame_indices=valid_indices,
            outputs=valid_paths,
        )
    return TissueKspaceLibraryResult(plan, generated, reused, debug_path if debug_path and debug_path.is_file() else None, manifest, time.perf_counter() - started)


def format_tissue_kspace_library(result: TissueKspaceLibraryResult) -> str:
    plan = result.plan
    return "\n".join(
        (
            "Full-trajectory tissue k-space library",
            f"Cardiac phases:     {plan.frame_count}",
            f"High-res grid:      {plan.high_resolution_shape}",
            f"Per-phase k-space:  {plan.kspace_shape_per_frame} complex64",
            f"Estimated total:    {plan.storage.gib:.2f} GiB",
            f"Already valid:      {plan.existing_valid_frames}",
            f"Missing allocation: {plan.missing_bytes / 1024**3:.2f} GiB",
            f"Generated/reused:   {result.generated_frame_count}/{result.reused_frame_count}",
            f"Output:             {plan.output_directory}",
        )
    )
