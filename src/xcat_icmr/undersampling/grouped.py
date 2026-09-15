"""Integer temporal grouping of the canonical dynamic acquisition stream."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import tempfile
import time
from typing import TYPE_CHECKING, Callable

import h5py
import numpy as np

from xcat_icmr.acquisition.dynamic import (
    DynamicAcquisitionPlan,
    plan_dynamic_acquisition,
)
from xcat_icmr.acquisition.dynamic_adjoint import generate_dynamic_adjoint_debug
from xcat_icmr.acquisition.schema import embed_reconstruction_contract
from xcat_icmr.cache import (
    dynamic_acquisition_cache_entry,
    undersampled_acquisition_cache_entry,
    write_artifact_manifest,
)

if TYPE_CHECKING:
    from xcat_icmr.config.models import SimulationConfig


class GroupedAcquisitionError(ValueError):
    """Raised when canonical frames cannot form the requested output frames."""


@dataclass(frozen=True)
class TemporalGrouping:
    source_frame_duration_s: float
    source_frames_per_output_frame: int
    output_frame_duration_s: float
    source_trs_per_frame: int
    trs_per_output_frame: int
    available_source_frames: int
    retained_source_frames: int
    dropped_source_frames: int
    output_frame_count: int
    retained_tr_count: int


@dataclass(frozen=True)
class GroupedAcquisitionResult:
    cache_id: str
    descriptor_path: Path
    fullysampled_reference_path: Path
    adjoint_path: Path | None
    curved_profile_path: Path | None
    experiment_manifest_path: Path
    grouping: TemporalGrouping
    elapsed_s: float


def resolve_bracketing_multiples(
    *, target_frame_duration_s: float, source_frame_duration_s: float
) -> tuple[int, ...]:
    """Return unique floor/ceil source-frame multiples around a target."""

    if target_frame_duration_s < source_frame_duration_s:
        raise GroupedAcquisitionError(
            "target frame duration must be at least one canonical frame"
        )
    ratio = target_frame_duration_s / source_frame_duration_s
    lower = int(np.floor(ratio + 1e-12))
    upper = int(np.ceil(ratio - 1e-12))
    return tuple(sorted({lower, upper}))


def resolve_temporal_grouping(
    *,
    source_frame_duration_s: float,
    source_trs_per_frame: int,
    available_tr_count: int,
    source_frames_per_output_frame: int,
) -> TemporalGrouping:
    """Resolve complete output frames and explicitly account for the tail."""

    if source_frame_duration_s <= 0:
        raise GroupedAcquisitionError("source frame duration must be positive")
    if source_trs_per_frame <= 0 or source_frames_per_output_frame <= 0:
        raise GroupedAcquisitionError("frame grouping factors must be positive")
    available_source_frames, partial_source_trs = divmod(
        available_tr_count, source_trs_per_frame
    )
    if partial_source_trs:
        raise GroupedAcquisitionError(
            "the selected TR range does not contain complete canonical frames"
        )
    output_frame_count, dropped_source_frames = divmod(
        available_source_frames, source_frames_per_output_frame
    )
    if output_frame_count <= 0:
        raise GroupedAcquisitionError(
            "the selected acquisition does not contain one complete grouped frame"
        )
    retained_source_frames = output_frame_count * source_frames_per_output_frame
    trs_per_output_frame = source_trs_per_frame * source_frames_per_output_frame
    return TemporalGrouping(
        source_frame_duration_s=float(source_frame_duration_s),
        source_frames_per_output_frame=int(source_frames_per_output_frame),
        output_frame_duration_s=(
            float(source_frame_duration_s) * source_frames_per_output_frame
        ),
        source_trs_per_frame=int(source_trs_per_frame),
        trs_per_output_frame=int(trs_per_output_frame),
        available_source_frames=int(available_source_frames),
        retained_source_frames=int(retained_source_frames),
        dropped_source_frames=int(dropped_source_frames),
        output_frame_count=int(output_frame_count),
        retained_tr_count=int(retained_source_frames * source_trs_per_frame),
    )


def _number_slug(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _experiment_directory(
    config: "SimulationConfig", grouping: TemporalGrouping
) -> Path:
    control_points = config.intervention.gd_balloon.path.control_points_file
    if control_points is None:
        raise GroupedAcquisitionError("a control-point file is required")
    velocity = config.intervention.gd_balloon.movement.velocity_cm_per_s
    actual_ms = grouping.output_frame_duration_s * 1e3
    return (
        config.run.output_root
        / "experiments"
        / control_points.stem
        / f"velocity_{_number_slug(velocity)}_cmps"
        / (
            f"grouped_{grouping.source_frames_per_output_frame}x_"
            f"{_number_slug(actual_ms)}ms"
        )
    )


def _write_json_atomic(path: Path, content: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
            mode="w", encoding="utf-8", delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(content, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _write_virtual_kspace(
    destination: Path,
    *,
    source_path: Path,
    source_shape: tuple[int, int, int],
    source_schedule,
    grouping: TemporalGrouping,
    overwrite: bool,
) -> None:
    if destination.is_file() and not overwrite:
        with h5py.File(destination, "r") as existing:
            data = existing.get("kspace")
            complete = existing.get("tr_complete")
            if (
                isinstance(data, h5py.Dataset)
                and data.shape
                == (source_shape[0], grouping.retained_tr_count, source_shape[2])
                and isinstance(complete, h5py.Dataset)
                and np.all(complete[:])
            ):
                return
        raise GroupedAcquisitionError(
            "existing grouped k-space descriptor is incompatible; use --overwrite"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    layout = h5py.VirtualLayout(
        shape=(source_shape[0], grouping.retained_tr_count, source_shape[2]),
        dtype=np.complex64,
    )
    virtual_source = h5py.VirtualSource(
        str(source_path), "kspace", shape=source_shape
    )
    layout[:] = virtual_source[:, : grouping.retained_tr_count, :]
    with h5py.File(destination, "w", libver="latest") as output:
        data = output.create_virtual_dataset("kspace", layout)
        data.attrs["axis_order"] = "sample,simulation_TR,coil"
        data.attrs["storage"] = "HDF5 virtual dataset; samples remain in source"
        data.attrs["source_dynamic_kspace"] = str(source_path)
        data.attrs["source_frame_duration_s"] = grouping.source_frame_duration_s
        data.attrs["source_frames_per_output_frame"] = (
            grouping.source_frames_per_output_frame
        )
        data.attrs["frame_duration_s"] = grouping.output_frame_duration_s
        data.attrs["trs_per_frame"] = grouping.trs_per_output_frame
        data.attrs["dropped_source_frames"] = grouping.dropped_source_frames
        output.create_dataset(
            "tr_complete",
            data=np.ones(grouping.retained_tr_count, dtype=np.uint8),
        )
        output.create_dataset(
            "time_s",
            data=source_schedule.time_s[: grouping.retained_tr_count],
        )
        output.create_dataset(
            "cardiac_phase_index_zero_based",
            data=source_schedule.cardiac_phase_index_zero_based[
                : grouping.retained_tr_count
            ],
        )
        output.create_dataset(
            "trajectory_tr_index_zero_based",
            data=source_schedule.trajectory_tr_index_zero_based[
                : grouping.retained_tr_count
            ],
        )
        output.create_dataset(
            "frame_index_zero_based",
            data=np.arange(grouping.retained_tr_count, dtype=np.int64)
            // grouping.trs_per_output_frame,
        )
        output.create_dataset(
            "frame_start_tr_zero_based",
            data=np.arange(grouping.output_frame_count, dtype=np.int64)
            * grouping.trs_per_output_frame,
        )
        output.create_dataset(
            "frame_stop_tr_exclusive",
            data=(np.arange(grouping.output_frame_count, dtype=np.int64) + 1)
            * grouping.trs_per_output_frame,
        )


def _write_grouped_reference(
    destination: Path,
    *,
    source_path: Path,
    grouping: TemporalGrouping,
    overwrite: bool,
    progress: Callable[[str], None] | None,
) -> None:
    mode = "w" if overwrite else "a"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(source_path, "r") as source:
        source_image = source.get("image")
        source_complete = source.get("frame_complete")
        if not isinstance(source_image, h5py.Dataset) or source_image.ndim != 4:
            raise GroupedAcquisitionError(
                "the canonical fully sampled reference has no 4-D image dataset"
            )
        if source_image.shape[3] < grouping.retained_source_frames:
            raise GroupedAcquisitionError(
                "the canonical fully sampled reference has too few frames"
            )
        if (
            not isinstance(source_complete, h5py.Dataset)
            or not np.all(source_complete[: grouping.retained_source_frames])
        ):
            raise GroupedAcquisitionError(
                "the required canonical fully sampled reference frames are incomplete"
            )
        shape = source_image.shape[:3] + (grouping.output_frame_count,)
        if destination.is_file() and not overwrite:
            with h5py.File(destination, "r") as existing:
                existing_image = existing.get("image")
                existing_complete = existing.get("frame_complete")
                existing_times = existing.get("frame_center_time_s")
                if (
                    isinstance(existing_image, h5py.Dataset)
                    and existing_image.shape == shape
                    and existing_image.dtype == np.dtype(np.complex64)
                    and isinstance(existing_complete, h5py.Dataset)
                    and existing_complete.shape == (grouping.output_frame_count,)
                    and np.all(existing_complete[:])
                    and isinstance(existing_times, h5py.Dataset)
                    and existing_times.shape == (grouping.output_frame_count,)
                ):
                    return
            raise GroupedAcquisitionError(
                "existing grouped reference is incompatible; use --overwrite"
            )
        with h5py.File(destination, mode) as output:
            image = output.get("image")
            complete = output.get("frame_complete")
            if image is None and complete is None:
                image = output.create_dataset(
                    "image", shape=shape, dtype=np.complex64,
                    chunks=shape[:3] + (1,),
                )
                complete = output.create_dataset(
                    "frame_complete",
                    shape=(grouping.output_frame_count,), dtype=np.uint8,
                )
                image.attrs["axis_order"] = "logical_x,logical_y,logical_z,time"
                image.attrs["contains"] = (
                    "complex average of canonical fully sampled tissue + Gd frames"
                )
                image.attrs["source_reference"] = str(source_path)
                image.attrs["source_frame_duration_s"] = (
                    grouping.source_frame_duration_s
                )
                image.attrs["source_frames_per_output_frame"] = (
                    grouping.source_frames_per_output_frame
                )
                image.attrs["frame_duration_s"] = grouping.output_frame_duration_s
                output.create_dataset(
                    "frame_center_time_s",
                    data=(
                        np.arange(grouping.output_frame_count, dtype=np.float64)
                        + 0.5
                    )
                    * grouping.output_frame_duration_s,
                )
            if (
                not isinstance(image, h5py.Dataset)
                or image.shape != shape
                or not isinstance(complete, h5py.Dataset)
            ):
                raise GroupedAcquisitionError(
                    "existing grouped reference is incompatible; use --overwrite"
                )
            for frame in range(grouping.output_frame_count):
                if complete[frame] and not overwrite:
                    continue
                start = frame * grouping.source_frames_per_output_frame
                stop = start + grouping.source_frames_per_output_frame
                image[..., frame] = np.mean(
                    np.asarray(source_image[..., start:stop], dtype=np.complex64),
                    axis=3,
                    dtype=np.complex64,
                )
                complete[frame] = 1
                output.flush()
                if progress:
                    progress(
                        f"Grouped fully sampled reference: {frame + 1}/"
                        f"{grouping.output_frame_count}"
                    )


def generate_grouped_acquisition(
    config: "SimulationConfig",
    *,
    source_frames_per_output_frame: int,
    view_order_cycles: int | None = None,
    save_adjoint: bool = False,
    analyze_curve: bool = True,
    overwrite: bool = False,
    progress: Callable[[str], None] | None = None,
) -> GroupedAcquisitionResult:
    """Group a canonical acquisition without duplicating its multicoil samples."""

    started = time.perf_counter()
    if not config.undersampling.enabled:
        raise GroupedAcquisitionError("undersampling is disabled in the configuration")
    if view_order_cycles is not None and view_order_cycles <= 0:
        raise GroupedAcquisitionError("view-order cycles must be positive")
    source_plan = plan_dynamic_acquisition(config, check_free_space=False)
    requested_trs = (
        source_plan.schedule.acquisition_count
        if view_order_cycles is None
        else view_order_cycles * source_plan.schedule.view_order_cycle_length
    )
    if requested_trs > source_plan.schedule.acquisition_count:
        raise GroupedAcquisitionError(
            "the canonical acquisition is shorter than the requested view-order cycles"
        )
    grouping = resolve_temporal_grouping(
        source_frame_duration_s=source_plan.schedule.frame_duration_s,
        source_trs_per_frame=source_plan.schedule.trs_per_frame,
        available_tr_count=requested_trs,
        source_frames_per_output_frame=source_frames_per_output_frame,
    )
    with h5py.File(source_plan.output_path, "r") as source:
        data = source.get("kspace")
        complete = source.get("tr_complete")
        if (
            not isinstance(data, h5py.Dataset)
            or data.shape != source_plan.shape
            or not isinstance(complete, h5py.Dataset)
            or not np.all(complete[: grouping.retained_tr_count])
        ):
            raise GroupedAcquisitionError(
                "the required canonical dynamic k-space samples are incomplete"
            )

    cache_entry = undersampled_acquisition_cache_entry(
        config,
        source_frames_per_output_frame=source_frames_per_output_frame,
        view_order_cycles=view_order_cycles,
    )
    descriptor = cache_entry.directory / "grouped_multicoil_kspace.h5"
    reference = cache_entry.directory / "fullysampled_reference_4d.h5"
    _write_virtual_kspace(
        descriptor,
        source_path=source_plan.output_path,
        source_shape=source_plan.shape,
        source_schedule=source_plan.schedule,
        grouping=grouping,
        overwrite=overwrite,
    )
    source_reference = (
        dynamic_acquisition_cache_entry(config).directory
        / "fullysampled_tissue_gd_reference_4d.h5"
    )
    if not source_reference.is_file():
        raise GroupedAcquisitionError(
            "the canonical fully sampled tissue-plus-Gd reference is required"
        )
    _write_grouped_reference(
        reference,
        source_path=source_reference,
        grouping=grouping,
        overwrite=overwrite,
        progress=progress,
    )
    embed_reconstruction_contract(
        descriptor,
        config,
        schedule=source_plan.schedule,
        matching_reference=reference,
        acquisition_id=cache_entry.cache_id,
        experiment_directory=_experiment_directory(config, grouping),
        overwrite_metadata=overwrite,
        progress=progress,
    )

    adjoint: Path | None = None
    if save_adjoint:
        if view_order_cycles is None:
            raise GroupedAcquisitionError(
                "a full-experiment adjoint belongs to reconstruction; use a bounded debug"
            )
        grouped_schedule = replace(
            source_plan.schedule,
            trs_per_frame=grouping.trs_per_output_frame,
            frame_duration_s=grouping.output_frame_duration_s,
            frame_count=grouping.output_frame_count,
            acquisition_count=grouping.retained_tr_count,
            retained_duration_s=(
                grouping.output_frame_count * grouping.output_frame_duration_s
            ),
            dropped_duration_s=(
                grouping.dropped_source_frames * grouping.source_frame_duration_s
            ),
            time_s=source_plan.schedule.time_s[: grouping.retained_tr_count],
            frame_index_zero_based=(
                np.arange(grouping.retained_tr_count, dtype=np.int64)
                // grouping.trs_per_output_frame
            ),
            cardiac_phase_index_zero_based=(
                source_plan.schedule.cardiac_phase_index_zero_based[
                    : grouping.retained_tr_count
                ]
            ),
            trajectory_tr_index_zero_based=(
                source_plan.schedule.trajectory_tr_index_zero_based[
                    : grouping.retained_tr_count
                ]
            ),
        )
        grouped_plan = DynamicAcquisitionPlan(
            output_path=descriptor,
            schedule=grouped_schedule,
            shape=(
                source_plan.shape[0],
                grouping.retained_tr_count,
                source_plan.shape[2],
            ),
            storage=source_plan.storage,
            completed_trs=grouping.retained_tr_count,
            view_order_cycles=view_order_cycles,
        )
        adjoint = generate_dynamic_adjoint_debug(
            config,
            plan=grouped_plan,
            overwrite=overwrite,
            progress=progress,
        )

    curved_profile_path: Path | None = None
    if analyze_curve and config.analysis.curved_line_profile.enabled:
        from xcat_icmr.analysis.curved_profile import generate_curved_line_profile

        profile = generate_curved_line_profile(
            config, input_path=reference, overwrite=overwrite
        )
        curved_profile_path = profile.heatmap_path
    outputs = [descriptor, reference]
    if adjoint is not None:
        outputs.append(adjoint)
    if curved_profile_path is not None:
        outputs.append(curved_profile_path)
    write_artifact_manifest(
        cache_entry,
        status="complete",
        frame_count=grouping.output_frame_count,
        completed_frame_indices=list(range(1, grouping.output_frame_count + 1)),
        outputs=outputs,
    )

    control_points = config.intervention.gd_balloon.path.control_points_file
    assert control_points is not None
    experiment_manifest = _experiment_directory(config, grouping) / "result.json"
    dynamic_entry = dynamic_acquisition_cache_entry(config)
    balloon_payload = dynamic_entry.payload["balloon"]
    assert isinstance(balloon_payload, dict)
    path_payload = balloon_payload["path"]
    assert isinstance(path_payload, dict)
    file_signature = path_payload["control_points_file"]
    assert isinstance(file_signature, dict)
    _write_json_atomic(
        experiment_manifest,
        {
            "trajectory_name": control_points.stem,
            "control_points_filename": control_points.name,
            "control_points_path": str(control_points),
            "control_points_sha256": file_signature.get("sha256"),
            "velocity_cm_per_s": (
                config.intervention.gd_balloon.movement.velocity_cm_per_s
            ),
            "source_dynamic_acquisition_cache_id": (
                dynamic_entry.cache_id
            ),
            "undersampled_acquisition_cache_id": cache_entry.cache_id,
            "source_frame_duration_s": grouping.source_frame_duration_s,
            "source_frames_per_output_frame": (
                grouping.source_frames_per_output_frame
            ),
            "actual_temporal_resolution_s": grouping.output_frame_duration_s,
            "requested_temporal_resolution_s": (
                config.undersampling.target_frame_duration_s
            ),
            "available_source_frames": grouping.available_source_frames,
            "retained_source_frames": grouping.retained_source_frames,
            "dropped_source_frames": grouping.dropped_source_frames,
            "view_order_cycles": view_order_cycles,
            "outputs": {
                "undersampled_multicoil_kspace": str(descriptor),
                "matching_fullysampled_reference": str(reference),
                "adjoint_debug": str(adjoint) if adjoint is not None else None,
                "ground_truth_curved_line_profile": (
                    str(curved_profile_path)
                    if curved_profile_path is not None
                    else None
                ),
            },
        },
    )
    return GroupedAcquisitionResult(
        cache_id=cache_entry.cache_id,
        descriptor_path=descriptor,
        fullysampled_reference_path=reference,
        adjoint_path=adjoint,
        curved_profile_path=curved_profile_path,
        experiment_manifest_path=experiment_manifest,
        grouping=grouping,
        elapsed_s=time.perf_counter() - started,
    )


def generate_bracketed_acquisitions(
    config: "SimulationConfig",
    *,
    view_order_cycles: int | None = None,
    save_adjoint: bool = False,
    analyze_curve: bool = True,
    overwrite: bool = False,
    progress: Callable[[str], None] | None = None,
) -> tuple[GroupedAcquisitionResult, ...]:
    """Generate the unique floor/ceil multiples around the requested duration."""

    factors = resolve_bracketing_multiples(
        target_frame_duration_s=config.undersampling.target_frame_duration_s,
        source_frame_duration_s=config.acquisition.frame_duration_s,
    )
    results = []
    for factor in factors:
        if progress:
            progress(
                f"Temporal bracket: {factor} x "
                f"{config.acquisition.frame_duration_s * 1e3:g} ms = "
                f"{factor * config.acquisition.frame_duration_s * 1e3:g} ms"
            )
        results.append(
            generate_grouped_acquisition(
                config,
                source_frames_per_output_frame=factor,
                view_order_cycles=view_order_cycles,
                save_adjoint=save_adjoint,
                analyze_curve=analyze_curve,
                overwrite=overwrite,
                progress=progress,
            )
        )
    return tuple(results)


def format_grouped_acquisition(result: GroupedAcquisitionResult) -> str:
    grouping = result.grouping
    return "\n".join(
        (
            "Grouped undersampled acquisition",
            f"Source frame:       {grouping.source_frame_duration_s * 1e3:g} ms",
            f"Grouping:           {grouping.source_frames_per_output_frame} source frames",
            f"Actual output:      {grouping.output_frame_duration_s * 1e3:g} ms",
            f"Output frames:      {grouping.output_frame_count}",
            f"Retained/dropped:   {grouping.retained_source_frames}/{grouping.dropped_source_frames} source frames",
            f"Cache ID:           {result.cache_id}",
            f"K-space descriptor: {result.descriptor_path}",
            f"Full reference:     {result.fullysampled_reference_path}",
            f"Adjoint debug:      {result.adjoint_path or 'not generated'}",
            f"Curve profile:      {result.curved_profile_path or 'not generated'}",
            f"Experiment record:  {result.experiment_manifest_path}",
            f"Elapsed:             {result.elapsed_s:.3f} s",
        )
    )
