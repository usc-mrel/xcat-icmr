"""Small, inspectable image-space checkpoints for the moving Gd balloon."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING

import numpy as np
from scipy.io import loadmat, savemat

from xcat_icmr.intervention.balloon import (
    SparseBalloonSupport,
    rasterize_sparse_balloon,
)
from xcat_icmr.intervention.gd_signal import (
    calculate_sparse_gd_bssfp_signal,
    sample_sparse_flip_angles,
)
from xcat_icmr.intervention.path import (
    interpolate_cubic_arc_length,
    load_balloon_path,
    resolve_simulation_duration,
)
from xcat_icmr.phantom import plan_xcat_frames, xcat_label_shape
from xcat_icmr.sequence import read_sequence
from xcat_icmr.tissue import get_tissue_library

if TYPE_CHECKING:
    from xcat_icmr.config.models import SimulationConfig


class BalloonDebugError(ValueError):
    """Raised when balloon image-space checkpoints cannot be generated."""


@dataclass(frozen=True)
class BalloonDebugFrame:
    """One saved combined tissue/Gd checkpoint."""

    sample_name: str
    time_s: float
    tissue_frame_index: int
    center_lps_mm: np.ndarray
    center_ijk: np.ndarray
    tissue_label_at_center: int
    occupied_volume_mm3: float
    signal_difference_range: tuple[float, float]
    output_path: Path


@dataclass(frozen=True)
class BalloonDebugResult:
    """Representative image-space checkpoints over one configured traversal."""

    duration_s: float
    traversal: str
    velocity_cm_per_s: float
    frames: tuple[BalloonDebugFrame, ...]
    summary_path: Path


@dataclass(frozen=True)
class BalloonPathDebugResult:
    """One anatomy frame containing the complete swept A-to-B balloon path."""

    path_length_mm: float
    center_count: int
    center_spacing_mm: float
    tissue_frame_index: int
    swept_volume_mm3: float
    encountered_labels: tuple[int, ...]
    output_path: Path
    summary_path: Path


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


def _load_mat_array(path: Path, variable: str) -> np.ndarray:
    if not path.is_file():
        raise BalloonDebugError(f"required input does not exist: {path}")
    try:
        content = loadmat(path, variable_names=[variable], squeeze_me=False)
    except (OSError, ValueError, NotImplementedError) as exc:
        raise BalloonDebugError(f"could not read {path}: {exc}") from exc
    if variable not in content:
        raise BalloonDebugError(f"{path} does not contain {variable!r}")
    return np.asarray(content[variable])


def _blood_properties(config: SimulationConfig):
    library = get_tissue_library(config.sequence.contrast.tissue_library)
    group = next(
        (item for item in library.groups if item.name.lower() == "blood"), None
    )
    if group is None:
        raise BalloonDebugError("the selected tissue library has no Blood group")
    return group.properties


def generate_balloon_debug_frames(
    config: SimulationConfig,
    *,
    overwrite: bool = False,
) -> BalloonDebugResult:
    """Generate full-volume A/middle/B images using local partial volumes."""

    balloon = config.intervention.gd_balloon
    if not balloon.enabled:
        raise BalloonDebugError("the Gd balloon is disabled")
    if balloon.path.control_points_file is None:
        raise BalloonDebugError("the Gd balloon has no control-point file")
    if balloon.composition.mode != "replace-background":
        raise BalloonDebugError(
            "balloon debug requires composition.mode='replace-background'"
        )

    loaded_path = load_balloon_path(
        balloon.path.control_points_file,
        coordinate_system=balloon.path.coordinate_system,
    )
    path = interpolate_cubic_arc_length(loaded_path.control_points_lps_mm)
    duration = resolve_simulation_duration(config).duration_s
    fractions = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
    times_s = fractions * duration
    centres = path.positions_at_times_s(
        times_s,
        velocity_cm_per_s=balloon.movement.velocity_cm_per_s,
        start_time_s=balloon.movement.start_time_s,
        traversal=balloon.movement.traversal,
    )
    names = (
        ("A", "B_turnaround", "A_return")
        if balloon.movement.traversal == "round-trip"
        else ("A", "middle", "B")
    )

    frame_plan = plan_xcat_frames(config, debug_one_frame=False)
    frame_count = len(frame_plan.frames)
    if frame_count < 1:
        raise BalloonDebugError("the XCAT motion cycle contains no frames")
    expected_shape = xcat_label_shape(config)
    voxel_size = config.phantom.voxel_size_mm
    output_directory = config.run.output_root / "intervention" / "balloon_debug"
    output_directory.mkdir(parents=True, exist_ok=True)
    profile_path = (
        config.run.output_root
        / "contrast"
        / f"phantom_{config.run.id}_rf_slice_profile.mat"
    )
    sequence = read_sequence(config.sequence)
    carrier = _blood_properties(config)
    concentration = balloon.contrast_agent.concentration.value_mM
    reports: list[BalloonDebugFrame] = []

    for sample_index, (name, time_s, center) in enumerate(
        zip(names, times_s, centres, strict=True), start=1
    ):
        zero_based_tissue = int(
            np.rint(time_s / config.timeline.xcat_time_step_s)
        ) % frame_count
        planned = frame_plan.frames[zero_based_tissue]
        if planned.label_path is None:
            raise BalloonDebugError("the XCAT frame has no saved label path")
        contrast_path = (
            config.run.output_root
            / "contrast"
            / (
                f"phantom_{config.run.id}_act_{planned.index}_"
                f"{config.sequence.contrast.model}.mat"
            )
        )
        labels = _load_mat_array(planned.label_path, "P")
        tissue = np.asarray(
            _load_mat_array(contrast_path, "image"), dtype=np.float32
        )
        if labels.shape != expected_shape or tissue.shape != expected_shape:
            raise BalloonDebugError(
                f"frame {planned.index} shape does not match {expected_shape}"
            )

        support = rasterize_sparse_balloon(
            center,
            volume_shape=expected_shape,
            voxel_size_mm=voxel_size,
            diameter_mm=balloon.geometry.diameter_mm,
            shape=balloon.geometry.shape,
        )
        flip = sample_sparse_flip_angles(profile_path, support)
        gd = calculate_sparse_gd_bssfp_signal(
            support,
            carrier=carrier,
            concentration_mM=concentration,
            flip_angle_deg=flip,
            te_ms=sequence.te_ms,
            tr_ms=sequence.tr_ms,
            relaxivity_library=balloon.contrast_agent.relaxivity_library,
        )
        slices = support.bounding_box_slices
        tissue_patch = tissue[slices]
        delta = np.asarray(
            support.occupancy * (gd.values - tissue_patch), dtype=np.float32
        )
        combined = tissue.copy()
        combined[slices] = tissue_patch + delta

        center_ijk_float = (
            (center - support.origin_lps_mm) / np.asarray(voxel_size)
        )
        center_ijk = np.rint(center_ijk_float).astype(np.int32)
        center_ijk = np.clip(
            center_ijk, 0, np.asarray(expected_shape, dtype=np.int32) - 1
        )
        label = int(labels[tuple(center_ijk)])
        destination = output_directory / (
            f"balloon_debug_{sample_index:02d}_{name}.mat"
        )
        if destination.exists() and not overwrite:
            raise BalloonDebugError(
                f"output already exists: {destination}; pass --overwrite"
            )
        _atomic_savemat(destination, {"image": combined})
        reopened = _load_mat_array(destination, "image")
        if reopened.shape != expected_shape or reopened.dtype != np.float32:
            raise BalloonDebugError(
                f"saved debug frame failed verification: {destination}"
            )
        reports.append(
            BalloonDebugFrame(
                sample_name=name,
                time_s=float(time_s),
                tissue_frame_index=planned.index,
                center_lps_mm=np.asarray(center, dtype=np.float64),
                center_ijk=center_ijk,
                tissue_label_at_center=label,
                occupied_volume_mm3=support.occupied_volume_mm3,
                signal_difference_range=(float(delta.min()), float(delta.max())),
                output_path=destination,
            )
        )

    summary_path = output_directory / "balloon_debug_summary.mat"
    if summary_path.exists() and not overwrite:
        raise BalloonDebugError(
            f"output already exists: {summary_path}; pass --overwrite"
        )
    _atomic_savemat(
        summary_path,
        {
            "sample_names": np.asarray(names, dtype=object)[None, :],
            "times_s": np.asarray([item.time_s for item in reports]),
            "centers_lps_mm": np.asarray(
                [item.center_lps_mm for item in reports], dtype=np.float32
            ),
            "centers_ijk_zero_based": np.asarray(
                [item.center_ijk for item in reports], dtype=np.int32
            ),
            "tissue_frame_indices_one_based": np.asarray(
                [item.tissue_frame_index for item in reports], dtype=np.int32
            ),
            "tissue_labels_at_centers": np.asarray(
                [item.tissue_label_at_center for item in reports], dtype=np.int32
            ),
            "occupied_volume_mm3": np.asarray(
                [item.occupied_volume_mm3 for item in reports], dtype=np.float32
            ),
            "signal_difference_range": np.asarray(
                [item.signal_difference_range for item in reports],
                dtype=np.float32,
            ),
            "duration_s": np.asarray([[duration]], dtype=np.float32),
            "velocity_cm_per_s": np.asarray(
                [[balloon.movement.velocity_cm_per_s]], dtype=np.float32
            ),
            "traversal": balloon.movement.traversal,
        },
    )
    return BalloonDebugResult(
        duration_s=float(duration),
        traversal=balloon.movement.traversal,
        velocity_cm_per_s=float(balloon.movement.velocity_cm_per_s),
        frames=tuple(reports),
        summary_path=summary_path,
    )


def generate_balloon_path_debug(
    config: SimulationConfig,
    *,
    center_spacing_mm: float = 0.5,
    overwrite: bool = False,
    contrast_path_override: str | Path | None = None,
    profile_path_override: str | Path | None = None,
    output_tag: str | None = None,
) -> BalloonPathDebugResult:
    """Show the complete A-to-B swept balloon volume on tissue frame one."""

    if not np.isfinite(center_spacing_mm) or center_spacing_mm <= 0.0:
        raise BalloonDebugError("center_spacing_mm must be positive")
    balloon = config.intervention.gd_balloon
    if not balloon.enabled or balloon.path.control_points_file is None:
        raise BalloonDebugError("an enabled Gd balloon path is required")
    if balloon.composition.mode != "replace-background":
        raise BalloonDebugError(
            "balloon path debug requires composition.mode='replace-background'"
        )

    loaded_path = load_balloon_path(
        balloon.path.control_points_file,
        coordinate_system=balloon.path.coordinate_system,
    )
    path = interpolate_cubic_arc_length(loaded_path.control_points_lps_mm)
    distances = np.arange(
        0.0, path.total_length_mm, center_spacing_mm, dtype=np.float64
    )
    distances = np.concatenate((distances, [path.total_length_mm]))
    centres = path.positions_at_distances_mm(distances)

    expected_shape = xcat_label_shape(config)
    voxel_size = config.phantom.voxel_size_mm
    supports = [
        rasterize_sparse_balloon(
            center,
            volume_shape=expected_shape,
            voxel_size_mm=voxel_size,
            diameter_mm=balloon.geometry.diameter_mm,
            shape=balloon.geometry.shape,
        )
        for center in centres
    ]
    starts = np.asarray(
        [support.bounding_box_start_ijk for support in supports], dtype=np.int64
    )
    stops = np.asarray(
        [
            support.bounding_box_start_ijk
            + np.asarray(support.occupancy.shape, dtype=np.int64)
            for support in supports
        ]
    )
    union_start = np.min(starts, axis=0)
    union_stop = np.max(stops, axis=0)
    union_occupancy = np.zeros(
        tuple(int(value) for value in union_stop - union_start),
        dtype=np.float32,
    )
    for support in supports:
        local_start = support.bounding_box_start_ijk - union_start
        local_slices = tuple(
            slice(int(start), int(start) + int(size))
            for start, size in zip(
                local_start, support.occupancy.shape, strict=True
            )
        )
        np.maximum(
            union_occupancy[local_slices],
            support.occupancy,
            out=union_occupancy[local_slices],
        )
    union_support = SparseBalloonSupport(
        center_lps_mm=np.asarray(centres[len(centres) // 2], dtype=np.float64),
        bounding_box_start_ijk=np.asarray(union_start, dtype=np.int32),
        occupancy=union_occupancy,
        volume_shape=expected_shape,
        voxel_size_mm=tuple(float(value) for value in voxel_size),
        origin_lps_mm=supports[0].origin_lps_mm,
    )

    frame_plan = plan_xcat_frames(config, debug_one_frame=False)
    planned = frame_plan.frames[0]
    if planned.label_path is None:
        raise BalloonDebugError("tissue frame one has no label path")
    contrast_path = (
        Path(contrast_path_override)
        if contrast_path_override is not None
        else (
            config.run.output_root
            / "contrast"
            / (
                f"phantom_{config.run.id}_act_{planned.index}_"
                f"{config.sequence.contrast.model}.mat"
            )
        )
    )
    labels = _load_mat_array(planned.label_path, "P")
    tissue = np.asarray(
        _load_mat_array(contrast_path, "image"), dtype=np.float32
    )
    if labels.shape != expected_shape or tissue.shape != expected_shape:
        raise BalloonDebugError("tissue frame one has an unexpected shape")

    profile_path = (
        Path(profile_path_override)
        if profile_path_override is not None
        else (
            config.run.output_root
            / "contrast"
            / f"phantom_{config.run.id}_rf_slice_profile.mat"
        )
    )
    flip = sample_sparse_flip_angles(profile_path, union_support)
    sequence = read_sequence(config.sequence)
    gd = calculate_sparse_gd_bssfp_signal(
        union_support,
        carrier=_blood_properties(config),
        concentration_mM=balloon.contrast_agent.concentration.value_mM,
        flip_angle_deg=flip,
        te_ms=sequence.te_ms,
        tr_ms=sequence.tr_ms,
        relaxivity_library=balloon.contrast_agent.relaxivity_library,
    )
    slices = union_support.bounding_box_slices
    tissue_patch = tissue[slices]
    delta = np.asarray(
        union_support.occupancy * (gd.values - tissue_patch), dtype=np.float32
    )
    combined = tissue.copy()
    combined[slices] = tissue_patch + delta

    origin = union_support.origin_lps_mm
    center_indices = np.rint(
        (centres - origin[None, :]) / np.asarray(voxel_size)[None, :]
    ).astype(np.int32)
    center_indices = np.clip(
        center_indices,
        0,
        np.asarray(expected_shape, dtype=np.int32)[None, :] - 1,
    )
    center_labels = labels[
        center_indices[:, 0], center_indices[:, 1], center_indices[:, 2]
    ].astype(np.int32)
    encountered_labels = tuple(
        int(value) for value in np.unique(center_labels)
    )

    output_directory = config.run.output_root / "intervention" / "balloon_debug"
    suffix = "" if output_tag is None else f"_{output_tag}"
    output_path = output_directory / f"balloon_debug_path_A_to_B{suffix}.mat"
    summary_path = (
        output_directory / f"balloon_debug_path_A_to_B{suffix}_summary.mat"
    )
    for destination in (output_path, summary_path):
        if destination.exists() and not overwrite:
            raise BalloonDebugError(
                f"output already exists: {destination}; pass --overwrite"
            )
    _atomic_savemat(output_path, {"image": combined})
    _atomic_savemat(
        summary_path,
        {
            "centers_lps_mm": np.asarray(centres, dtype=np.float32),
            "centers_ijk_zero_based": center_indices,
            "labels_at_centers": center_labels,
            "unique_labels_at_centers": np.asarray(
                encountered_labels, dtype=np.int32
            ),
            "path_length_mm": np.asarray(
                [[path.total_length_mm]], dtype=np.float32
            ),
            "center_spacing_mm": np.asarray(
                [[center_spacing_mm]], dtype=np.float32
            ),
            "swept_volume_mm3": np.asarray(
                [[union_support.occupied_volume_mm3]], dtype=np.float32
            ),
            "tissue_frame_index_one_based": np.asarray(
                [[planned.index]], dtype=np.int32
            ),
        },
    )
    reopened = _load_mat_array(output_path, "image")
    if reopened.shape != expected_shape or reopened.dtype != np.float32:
        raise BalloonDebugError("saved A-to-B path frame failed verification")
    return BalloonPathDebugResult(
        path_length_mm=float(path.total_length_mm),
        center_count=len(centres),
        center_spacing_mm=float(center_spacing_mm),
        tissue_frame_index=planned.index,
        swept_volume_mm3=union_support.occupied_volume_mm3,
        encountered_labels=encountered_labels,
        output_path=output_path,
        summary_path=summary_path,
    )


def format_balloon_debug(result: BalloonDebugResult) -> str:
    """Format the inspectable frame locations and key measurements."""

    lines = [
        "Gd balloon image-space debug",
        f"Traversal: {result.traversal}",
        f"Velocity:  {result.velocity_cm_per_s:g} cm/s",
        f"Duration:  {result.duration_s:g} s",
    ]
    for frame in result.frames:
        center = ", ".join(f"{value:.3f}" for value in frame.center_lps_mm)
        lines.extend(
            (
                "",
                f"{frame.sample_name}: t={frame.time_s:.6g} s, "
                f"tissue frame={frame.tissue_frame_index}",
                f"  center LPS mm: [{center}]",
                f"  center label:  {frame.tissue_label_at_center}",
                f"  volume:        {frame.occupied_volume_mm3:.3f} mm^3",
                f"  output:        {frame.output_path}",
            )
        )
    lines.extend(("", f"Summary: {result.summary_path}"))
    return "\n".join(lines)


def format_balloon_path_debug(result: BalloonPathDebugResult) -> str:
    """Format one complete swept-path geometry checkpoint."""

    return "\n".join(
        (
            "Gd balloon complete A-to-B path debug",
            f"Path length:       {result.path_length_mm:.3f} mm",
            f"Centre spacing:    {result.center_spacing_mm:.3f} mm",
            f"Sampled centres:   {result.center_count}",
            f"Tissue frame:      {result.tissue_frame_index}",
            f"Swept volume:      {result.swept_volume_mm3:.3f} mm^3",
            f"Centreline labels: {result.encountered_labels}",
            f"Output:            {result.output_path}",
            f"Summary:           {result.summary_path}",
        )
    )
