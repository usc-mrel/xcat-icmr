"""Environment validation and human-readable configuration summaries."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from xcat_icmr.config.models import SimulationConfig


@dataclass(frozen=True)
class ValidationIssue:
    field: str
    message: str

    def format(self) -> str:
        return f"{self.field}: {self.message}"


def _require_file(
    issues: list[ValidationIssue], field: str, path: Path | None
) -> None:
    if path is None:
        return
    if not path.is_file():
        issues.append(ValidationIssue(field, f"file does not exist: {path}"))


def _require_directory(
    issues: list[ValidationIssue], field: str, path: Path
) -> None:
    if not path.is_dir():
        issues.append(ValidationIssue(field, f"directory does not exist: {path}"))


def validate_paths(config: SimulationConfig) -> list[ValidationIssue]:
    """Validate required external files and directories."""

    issues: list[ValidationIssue] = []
    xcat = config.resources.xcat

    _require_file(issues, "resources.xcat.executable", xcat.executable)
    if xcat.executable.is_file() and not os.access(xcat.executable, os.X_OK):
        issues.append(
            ValidationIssue(
                "resources.xcat.executable",
                f"file is not executable: {xcat.executable}",
            )
        )

    _require_file(
        issues,
        "resources.xcat.parameter_template",
        xcat.parameter_template,
    )
    _require_directory(issues, "sequence.folder", config.sequence.folder)
    _require_file(issues, "sequence.file", config.sequence.resolved_file)
    _require_directory(
        issues,
        "sequence.metadata_directory",
        config.sequence.metadata_directory,
    )

    off_resonance = config.scanner.effects.off_resonance
    if off_resonance.enabled:
        _require_file(
            issues,
            "scanner.effects.off_resonance.field_map",
            off_resonance.field_map,
        )

    balloon = config.intervention.gd_balloon
    if balloon.enabled:
        _require_file(
            issues,
            "intervention.gd_balloon.path.control_points_file",
            balloon.path.control_points_file,
        )

    if config.coils.enabled:
        _require_file(
            issues,
            "coils.sensitivity_map",
            config.coils.sensitivity_map,
        )

    if isinstance(config.noise.coil_covariance, Path):
        _require_file(
            issues,
            "noise.coil_covariance",
            config.noise.coil_covariance,
        )

    return issues


def format_summary(config: SimulationConfig) -> str:
    """Return a concise summary of a validated configuration."""

    balloon = config.intervention.gd_balloon
    device = "CPU" if config.compute.device_id == -1 else f"GPU {config.compute.device_id}"
    duration = (
        "derived from balloon path"
        if config.timeline.duration_s == "auto"
        else f"{config.timeline.duration_s:g} s"
    )

    lines = (
        ("Run", config.run.id),
        ("Device", device),
        ("Scanner field", f"{config.scanner.field_strength_t:g} T"),
        ("Motion", config.phantom.motion.mode),
        ("XCAT time step", f"{config.timeline.xcat_time_step_s * 1e3:g} ms"),
        (
            "K-space time step",
            f"{config.timeline.kspace_time_step_s * 1e3:g} ms",
        ),
        (
            "XCAT frames/k-space frame",
            str(config.timeline.xcat_frames_per_kspace_frame),
        ),
        ("XCAT aggregation", config.timeline.xcat_to_kspace),
        ("Gd balloon", "enabled" if balloon.enabled else "disabled"),
        ("Duration", duration),
        (
            "Undersampling",
            "enabled" if config.undersampling.enabled else "disabled",
        ),
        ("Noise", "enabled" if config.noise.enabled else "disabled"),
    )
    width = max(len(label) for label, _ in lines)
    return "\n".join(f"{label + ':':<{width + 2}} {value}" for label, value in lines)
