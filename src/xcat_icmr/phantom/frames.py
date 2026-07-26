"""Deterministic XCAT frame timing and output-path planning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from xcat_icmr.config.models import SimulationConfig
from xcat_icmr.phantom.parameters import (
    XcatMotionPlan,
    plan_xcat_motion_cycle,
)


class XcatFramePlanError(Exception):
    """Raised when deterministic XCAT frame paths cannot be planned."""


@dataclass(frozen=True)
class XcatFrame:
    """One XCAT output frame and its expected files."""

    index: int
    time_s: float
    binary_path: Path
    label_path: Path | None
    binary_exists: bool
    label_exists: bool


@dataclass(frozen=True)
class XcatFramePlan:
    """The XCAT frames required for one generated anatomy cycle."""

    run_id: str
    output_root: Path
    raw_directory: Path
    label_directory: Path
    output_prefix: Path
    retain_binary_files: bool
    save_tissue_labels: bool
    motion: XcatMotionPlan
    frames: tuple[XcatFrame, ...]

    @property
    def time_axis_s(self) -> tuple[float, ...]:
        return tuple(frame.time_s for frame in self.frames)


_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def plan_xcat_frames(
    config: SimulationConfig,
    *,
    debug_one_frame: bool = True,
) -> XcatFramePlan:
    """Plan one-based XCAT frames on a half-open periodic time axis."""

    run_id = config.run.id
    if not _SAFE_RUN_ID.fullmatch(run_id):
        raise XcatFramePlanError(
            "run.id may contain only letters, numbers, '.', '_', and '-', "
            "and must start with a letter or number"
        )

    motion = plan_xcat_motion_cycle(
        config, debug_one_frame=debug_one_frame
    )
    xcat_directory = config.run.output_root / "xcat"
    raw_directory = xcat_directory / "raw"
    label_directory = xcat_directory / "labels"
    stem = f"phantom_{run_id}"
    output_prefix = raw_directory / stem

    frames = []
    for zero_based_index in range(motion.generated_frame_count):
        index = zero_based_index + 1
        time_s = zero_based_index * motion.time_step_s
        binary_path = raw_directory / f"{stem}_act_{index}.bin"
        label_path = (
            label_directory / f"{stem}_act_{index}.mat"
            if config.outputs.save_tissue_labels
            else None
        )
        frames.append(
            XcatFrame(
                index=index,
                time_s=time_s,
                binary_path=binary_path,
                label_path=label_path,
                binary_exists=binary_path.is_file(),
                label_exists=(
                    label_path.is_file() if label_path is not None else False
                ),
            )
        )

    return XcatFramePlan(
        run_id=run_id,
        output_root=config.run.output_root,
        raw_directory=raw_directory,
        label_directory=label_directory,
        output_prefix=output_prefix,
        retain_binary_files=config.outputs.retain_xcat_binary_files,
        save_tissue_labels=config.outputs.save_tissue_labels,
        motion=motion,
        frames=tuple(frames),
    )


def _display_path(path: Path | None, root: Path) -> str:
    if path is None:
        return "not retained"
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _displayed_frames(
    frames: tuple[XcatFrame, ...],
    *,
    maximum: int = 8,
) -> tuple[XcatFrame | None, ...]:
    if len(frames) <= maximum:
        return frames
    side = maximum // 2
    return frames[:side] + (None,) + frames[-side:]


def format_xcat_frame_plan(plan: XcatFramePlan) -> str:
    """Format a concise frame timeline and the expected output paths."""

    duration = (
        "static"
        if plan.motion.cycle_duration_s is None
        else f"{plan.motion.cycle_duration_s:g} s"
    )
    lines = [
        "XCAT frame plan",
        f"Run:                   {plan.run_id}",
        f"Motion mode:           {plan.motion.mode}",
        f"XCAT time step:        {plan.motion.time_step_s:g} s",
        f"Reusable motion cycle: {duration}",
        f"Motion-cycle frames:   {plan.motion.cycle_frame_count}",
        f"Frames planned now:    {len(plan.frames)}",
        (
            "Debug one-frame mode:  enabled"
            if plan.motion.debug_one_frame
            else "Debug one-frame mode:  disabled"
        ),
        f"XCAT output prefix:    {plan.output_prefix}",
        (
            "Raw binaries retained: yes"
            if plan.retain_binary_files
            else "Raw binaries retained: no (transient)"
        ),
        (
            "Tissue labels saved:   yes"
            if plan.save_tissue_labels
            else "Tissue labels saved:   no"
        ),
        "",
        "Frames use XCAT's one-based names and a half-open time axis:",
    ]
    for frame in _displayed_frames(plan.frames):
        if frame is None:
            lines.append("  ...")
            continue
        binary_status = " [exists]" if frame.binary_exists else ""
        label_status = " [exists]" if frame.label_exists else ""
        lines.extend(
            (
                f"  Frame {frame.index}: t={frame.time_s:.9g} s",
                (
                    f"    binary: "
                    f"{_display_path(frame.binary_path, plan.output_root)}"
                    f"{binary_status}"
                ),
                (
                    f"    label:  "
                    f"{_display_path(frame.label_path, plan.output_root)}"
                    f"{label_status}"
                ),
            )
        )
    return "\n".join(lines)
