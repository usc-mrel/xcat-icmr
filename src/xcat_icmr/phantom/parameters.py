"""Preparation of auditable, run-specific XCAT parameter files."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import TypeAlias

from xcat_icmr.config.models import SimulationConfig


class XcatParameterError(Exception):
    """Raised when an XCAT parameter file cannot be prepared safely."""


ParameterValue: TypeAlias = int | float | str


@dataclass(frozen=True)
class XcatMotionPlan:
    """One non-repeating XCAT anatomy cycle on the configured time grid."""

    mode: str
    time_step_s: float
    cardiac_period_s: float
    respiratory_period_s: float | None
    cycle_duration_s: float | None
    cycle_frame_count: int
    generated_frame_count: int
    debug_one_frame: bool


@dataclass(frozen=True)
class XcatParameterFile:
    """Result of preparing a run-specific XCAT parameter file."""

    template_path: Path
    output_path: Path
    motion_plan: XcatMotionPlan
    parameters: dict[str, ParameterValue]
    command_line_parameters: dict[str, ParameterValue]
    appended_parameters: tuple[str, ...]


_PARAMETER_LINE = re.compile(
    r"^(?P<indent>\s*)(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
    r"(?P<assignment>\s*=\s*)(?P<value>.*?)\s*$"
)

_COMMAND_LINE_ONLY_PARAMETERS = (
    "phan_rotx",
    "phan_roty",
    "phan_rotz",
)


def _period_frames(period_s: float, time_step_s: float, name: str) -> int:
    ratio = period_s / time_step_s
    frame_count = round(ratio)
    if not math.isclose(ratio, frame_count, rel_tol=0.0, abs_tol=1e-9):
        raise XcatParameterError(
            f"{name} period ({period_s:g} s) must be an integer multiple "
            f"of xcat_time_step_s ({time_step_s:g} s); got {ratio:g} frames"
        )
    return frame_count


def plan_xcat_motion_cycle(
    config: SimulationConfig,
    *,
    debug_one_frame: bool = True,
) -> XcatMotionPlan:
    """Find the shortest reusable XCAT motion cycle on the XCAT time grid."""

    motion = config.phantom.motion
    time_step = config.timeline.xcat_time_step_s
    cardiac_period = 60.0 / motion.cardiac.heart_rate_bpm
    breaths_per_minute = motion.respiratory.breaths_per_minute
    respiratory_period = (
        60.0 / breaths_per_minute
        if breaths_per_minute is not None
        else None
    )

    if motion.mode == "no-motion":
        cycle_duration = None
        cycle_frames = 1
    elif motion.mode == "breath-hold":
        cycle_frames = _period_frames(
            cardiac_period, time_step, "cardiac"
        )
        cycle_duration = cycle_frames * time_step
    else:
        if respiratory_period is None:
            raise XcatParameterError(
                "free-breathing motion requires breaths_per_minute"
            )
        cardiac_frames = _period_frames(
            cardiac_period, time_step, "cardiac"
        )
        respiratory_frames = _period_frames(
            respiratory_period, time_step, "respiratory"
        )
        cycle_frames = math.lcm(cardiac_frames, respiratory_frames)
        cycle_duration = cycle_frames * time_step

    generated_frames = 1 if debug_one_frame else cycle_frames
    return XcatMotionPlan(
        mode=motion.mode,
        time_step_s=time_step,
        cardiac_period_s=cardiac_period,
        respiratory_period_s=respiratory_period,
        cycle_duration_s=cycle_duration,
        cycle_frame_count=cycle_frames,
        generated_frame_count=generated_frames,
        debug_one_frame=debug_one_frame,
    )


def _motion_parameters(config: SimulationConfig) -> dict[str, ParameterValue]:
    motion = config.phantom.motion
    respiratory = motion.respiratory

    if motion.mode == "no-motion":
        motion_option = 1  # respiratory-only, with respiratory amplitudes zero
        diaphragm_motion = 0.0
        ap_expansion = 0.0
    elif motion.mode == "breath-hold":
        motion_option = 0  # beating heart only
        diaphragm_motion = respiratory.diaphragm_motion_cm
        ap_expansion = respiratory.anterior_posterior_expansion_cm
    else:
        motion_option = 2  # beating heart and respiratory motion
        diaphragm_motion = respiratory.diaphragm_motion_cm
        ap_expansion = respiratory.anterior_posterior_expansion_cm

    breaths_per_minute = respiratory.breaths_per_minute
    resp_period = (
        60.0 / breaths_per_minute
        if breaths_per_minute is not None
        else 5.0
    )
    return {
        "motion_option": motion_option,
        "hrt_period": 60.0 / motion.cardiac.heart_rate_bpm,
        "hrt_start_ph_index": motion.cardiac.start_phase,
        "resp_period": resp_period,
        "resp_start_ph_index": respiratory.start_phase,
        "max_diaphragm_motion": diaphragm_motion,
        "max_AP_exp": ap_expansion,
    }


def build_xcat_parameter_values(
    config: SimulationConfig,
    *,
    debug_one_frame: bool = True,
) -> dict[str, ParameterValue]:
    """Translate the current YAML sections into XCAT parameter values."""

    motion_plan = plan_xcat_motion_cycle(
        config, debug_one_frame=debug_one_frame
    )

    voxel_x, voxel_y, voxel_z = config.phantom.voxel_size_mm
    if abs(voxel_x - voxel_y) > 1e-12:
        raise XcatParameterError(
            "XCAT uses one in-plane pixel_width; phantom voxel x and y "
            "sizes must match"
        )

    rotation_x, rotation_y, rotation_z = (
        config.phantom.transform.additional_rotation_deg_xyz
    )
    translation_x, translation_y, translation_z = (
        config.phantom.transform.translation_mm_xyz
    )
    if config.phantom.patient_position != "HFS":
        raise XcatParameterError(
            "only the HFS patient position is currently implemented"
        )

    is_female = config.phantom.anatomy.sex == "female"
    parameters: dict[str, ParameterValue] = {
        # Native XCAT coordinates are used for HFS:
        # +x left, +y posterior, +z superior (LPS).
        "phan_rotx": 0,
        "phan_roty": 0,
        "phan_rotz": 0,
        "d_ZY_rotation": rotation_x,
        "d_XZ_rotation": rotation_y,
        "d_YX_rotation": rotation_z,
        "X_tr": translation_x,
        "Y_tr": translation_y,
        "Z_tr": translation_z,
        "pixel_width": voxel_x / 10.0,
        "slice_width": voxel_z / 10.0,
        "array_size": config.phantom.matrix_size_xy,
        "startslice": config.phantom.head_foot_slice_range.start,
        "endslice": config.phantom.head_foot_slice_range.end,
        "gender": int(is_female),
        "heart_base": (
            "vfemale50_heart.nrb" if is_female else "vmale50_heart.nrb"
        ),
        "organ_file": "vfemale50.nrb" if is_female else "vmale50.nrb",
        "papillary_flag": int(config.phantom.anatomy.papillary_muscles),
        "out_period": 0.0,
        "time_per_frame": config.timeline.xcat_time_step_s,
        "out_frames": motion_plan.generated_frame_count,
    }
    parameters.update(_motion_parameters(config))
    return parameters


def _format_parameter_value(value: ParameterValue) -> str:
    if isinstance(value, float):
        return format(value, ".15g")
    return str(value)


def render_xcat_parameter_file(
    template_text: str,
    parameters: dict[str, ParameterValue],
) -> tuple[str, tuple[str, ...]]:
    """Apply parameters while preserving untouched template lines/comments."""

    replaced: set[str] = set()
    rendered: list[str] = []
    for line in template_text.splitlines():
        content, marker, comment = line.partition("#")
        match = _PARAMETER_LINE.match(content)
        if match is None or match.group("name") not in parameters:
            rendered.append(line)
            continue

        name = match.group("name")
        if name in replaced:
            raise XcatParameterError(
                f"XCAT template contains duplicate parameter {name!r}"
            )
        value = _format_parameter_value(parameters[name])
        updated = (
            f"{match.group('indent')}{name}{match.group('assignment')}{value}"
        )
        if marker:
            updated += f"\t#{comment}"
        rendered.append(updated)
        replaced.add(name)

    appended = tuple(name for name in parameters if name not in replaced)
    if appended:
        rendered.extend(
            (
                "",
                "# XCAT-iCMR run-specific parameters absent from source template",
            )
        )
        rendered.extend(
            f"{name} = {_format_parameter_value(parameters[name])}"
            for name in appended
        )

    return "\n".join(rendered) + "\n", appended


def prepare_xcat_parameter_file(
    config: SimulationConfig,
    *,
    output_path: str | Path | None = None,
    debug_one_frame: bool = True,
) -> XcatParameterFile:
    """Create a run-specific XCAT parameter file without launching XCAT."""

    template_path = config.resources.xcat.parameter_template
    try:
        template_text = template_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise XcatParameterError(
            f"could not read XCAT parameter template {template_path}: {exc}"
        ) from exc

    motion_plan = plan_xcat_motion_cycle(
        config, debug_one_frame=debug_one_frame
    )
    parameters = build_xcat_parameter_values(
        config, debug_one_frame=debug_one_frame
    )
    command_line_parameters = {
        name: parameters[name] for name in _COMMAND_LINE_ONLY_PARAMETERS
    }
    file_parameters = {
        name: value
        for name, value in parameters.items()
        if name not in command_line_parameters
    }
    rendered, appended = render_xcat_parameter_file(
        template_text, file_parameters
    )
    destination = (
        Path(output_path).expanduser().resolve(strict=False)
        if output_path is not None
        else config.run.output_root / "xcat" / "parameters.par"
    )
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
    except OSError as exc:
        raise XcatParameterError(
            f"could not write XCAT parameter file {destination}: {exc}"
        ) from exc

    return XcatParameterFile(
        template_path=template_path,
        output_path=destination,
        motion_plan=motion_plan,
        parameters=parameters,
        command_line_parameters=command_line_parameters,
        appended_parameters=appended,
    )


def format_xcat_parameter_summary(result: XcatParameterFile) -> str:
    """Format the values applied to a run-specific XCAT parameter file."""

    lines = [
        "XCAT parameter file prepared.",
        f"Template: {result.template_path}",
        f"Output:   {result.output_path}",
        f"Motion mode:          {result.motion_plan.mode}",
        f"XCAT time step:       {result.motion_plan.time_step_s:g} s",
        (
            "Motion cycle:         static"
            if result.motion_plan.cycle_duration_s is None
            else (
                f"Motion cycle:         "
                f"{result.motion_plan.cycle_duration_s:g} s"
            )
        ),
        (
            f"Motion-derived frames: "
            f"{result.motion_plan.cycle_frame_count}"
        ),
        (
            f"Frames written:        "
            f"{result.motion_plan.generated_frame_count}"
        ),
        (
            "Debug one-frame mode: enabled"
            if result.motion_plan.debug_one_frame
            else "Debug one-frame mode: disabled"
        ),
        "",
        "Applied parameters:",
    ]
    for name, value in result.parameters.items():
        if name in result.command_line_parameters:
            suffix = " (command-line override)"
        elif name in result.appended_parameters:
            suffix = " (appended)"
        else:
            suffix = ""
        lines.append(f"  {name:<26} {_format_parameter_value(value)}{suffix}")
    return "\n".join(lines)
