"""Gd-balloon path geometry and automatic simulation duration."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import CubicSpline

if TYPE_CHECKING:
    from xcat_icmr.config.models import SimulationConfig


AUTO_DURATION_REFERENCE_SPEED_CM_PER_S = 0.5
AUTO_DURATION_REFERENCE_SPEED_MM_PER_S = (
    AUTO_DURATION_REFERENCE_SPEED_CM_PER_S * 10.0
)


class BalloonPathError(Exception):
    """Raised when a configured balloon path cannot be interpreted."""


@dataclass(frozen=True)
class SimulationDuration:
    """Resolved total simulation time and its derivation."""

    duration_s: float
    automatic: bool
    path_length_mm: float | None
    reference_speed_cm_per_s: float | None


@dataclass(frozen=True)
class BalloonPath:
    """One Slicer curve represented in patient LPS millimetres."""

    source_path: Path
    source_coordinate_system: str
    control_points_lps_mm: np.ndarray


@dataclass(frozen=True)
class CubicArcLengthPath:
    """Cubic path with a dense, monotonic physical arc-length map."""

    control_points_lps_mm: np.ndarray
    parameter_samples: np.ndarray
    arc_length_samples_mm: np.ndarray
    curve_samples_lps_mm: np.ndarray
    total_length_mm: float

    def positions_at_distances_mm(self, distances_mm: np.ndarray) -> np.ndarray:
        distances = np.asarray(distances_mm, dtype=np.float64)
        if not np.all(np.isfinite(distances)):
            raise BalloonPathError("distances must be finite")
        clipped = np.clip(distances, 0.0, self.total_length_mm)
        positions = np.column_stack(
            [
                np.interp(
                    clipped,
                    self.arc_length_samples_mm,
                    self.curve_samples_lps_mm[:, axis],
                )
                for axis in range(3)
            ]
        )
        return positions.reshape(distances.shape + (3,))

    def positions_at_times_s(
        self,
        times_s: np.ndarray,
        *,
        velocity_cm_per_s: float,
        start_time_s: float,
        traversal: str = "one-way",
    ) -> np.ndarray:
        times = np.asarray(times_s, dtype=np.float64)
        if not np.all(np.isfinite(times)) or np.any(times < 0.0):
            raise BalloonPathError("times must be finite and non-negative")
        if not np.isfinite(velocity_cm_per_s) or velocity_cm_per_s <= 0.0:
            raise BalloonPathError("velocity_cm_per_s must be positive")
        if not np.isfinite(start_time_s) or start_time_s < 0.0:
            raise BalloonPathError("start_time_s must be non-negative")
        if traversal not in {"one-way", "round-trip"}:
            raise BalloonPathError(
                "traversal must be 'one-way' or 'round-trip'"
            )
        travelled_mm = (
            np.maximum(times - start_time_s, 0.0)
            * velocity_cm_per_s
            * 10.0
        )
        if traversal == "one-way":
            distances_mm = travelled_mm
        else:
            distances_mm = np.where(
                travelled_mm <= self.total_length_mm,
                travelled_mm,
                np.maximum(2.0 * self.total_length_mm - travelled_mm, 0.0),
            )
        return self.positions_at_distances_mm(distances_mm)


def load_balloon_path(
    path: str | Path,
    *,
    coordinate_system: str = "auto",
) -> BalloonPath:
    """Load one Slicer curve and return control points in LPS millimetres."""

    path = Path(path).expanduser().resolve(strict=False)
    if not path.is_file():
        raise BalloonPathError(f"control-point file does not exist: {path}")
    try:
        document: dict[str, Any] = json.loads(
            path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BalloonPathError(
            f"could not read control-point JSON {path}: {exc}"
        ) from exc

    markups = document.get("markups")
    if not isinstance(markups, list) or len(markups) != 1:
        raise BalloonPathError(
            "control-point JSON must contain exactly one markup"
        )
    markup = markups[0]
    if not isinstance(markup, dict) or markup.get("type") != "Curve":
        raise BalloonPathError("the markup must have type 'Curve'")
    if markup.get("coordinateUnits", "mm") != "mm":
        raise BalloonPathError("control-point coordinates must use mm")
    declared_coordinate_system = markup.get("coordinateSystem")
    if declared_coordinate_system not in {"LPS", "RAS"}:
        raise BalloonPathError(
            "markup coordinateSystem must be 'LPS' or 'RAS'"
        )
    if coordinate_system not in {"auto", "LPS", "RAS"}:
        raise BalloonPathError(
            "coordinate_system must be 'auto', 'LPS', or 'RAS'"
        )
    source_coordinate_system = (
        declared_coordinate_system
        if coordinate_system == "auto"
        else coordinate_system
    )

    control_points = markup.get("controlPoints")
    if not isinstance(control_points, list) or len(control_points) < 2:
        raise BalloonPathError("the curve must contain at least two points")
    try:
        positions = np.asarray(
            [point["position"] for point in control_points], dtype=np.float64
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise BalloonPathError("curve positions must be numeric 3-D points") from exc
    if positions.shape != (len(control_points), 3) or not np.all(
        np.isfinite(positions)
    ):
        raise BalloonPathError("curve positions must be finite 3-D points")
    positions_lps = positions.copy()
    if source_coordinate_system == "RAS":
        positions_lps *= np.asarray([-1.0, -1.0, 1.0])
    return BalloonPath(
        source_path=path,
        source_coordinate_system=source_coordinate_system,
        control_points_lps_mm=positions_lps,
    )


def interpolate_cubic_arc_length(
    positions_lps_mm: np.ndarray,
    *,
    samples_per_segment: int = 4096,
) -> CubicArcLengthPath:
    """Build the constant-distance lookup used for balloon motion."""

    positions = np.asarray(positions_lps_mm, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) < 2:
        raise BalloonPathError("positions must have shape (N, 3), N >= 2")
    if not np.all(np.isfinite(positions)):
        raise BalloonPathError("positions must be finite")
    if samples_per_segment < 2:
        raise BalloonPathError("samples_per_segment must be at least 2")

    parameter = np.arange(len(positions), dtype=np.float64)
    spline = CubicSpline(parameter, positions, axis=0)
    sample_count = (len(positions) - 1) * samples_per_segment + 1
    parameter_samples = np.linspace(
        0.0, float(len(positions) - 1), sample_count
    )
    curve_samples = np.asarray(spline(parameter_samples), dtype=np.float64)
    segment_lengths = np.linalg.norm(np.diff(curve_samples, axis=0), axis=1)
    arc_lengths = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    exact_length = cubic_path_length_mm(positions)
    if arc_lengths[-1] <= 0.0:
        raise BalloonPathError("the path must have positive finite length")
    arc_lengths *= exact_length / arc_lengths[-1]
    return CubicArcLengthPath(
        control_points_lps_mm=positions.copy(),
        parameter_samples=parameter_samples,
        arc_length_samples_mm=arc_lengths,
        curve_samples_lps_mm=curve_samples,
        total_length_mm=exact_length,
    )


def cubic_path_length_mm(positions_mm: np.ndarray) -> float:
    """Return the arc length of the configured cubic interpolation."""

    positions = np.asarray(positions_mm, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) < 2:
        raise BalloonPathError("positions must have shape (N, 3), N >= 2")
    if not np.all(np.isfinite(positions)):
        raise BalloonPathError("positions must be finite")

    parameter = np.arange(len(positions), dtype=np.float64)
    spline = CubicSpline(parameter, positions, axis=0)
    derivative = spline.derivative()
    length_mm = 0.0
    for segment in range(len(positions) - 1):
        segment_length, _ = quad(
            lambda value: float(np.linalg.norm(derivative(value))),
            float(segment),
            float(segment + 1),
            epsabs=1e-8,
            epsrel=1e-10,
            limit=100,
        )
        length_mm += segment_length
    if not np.isfinite(length_mm) or length_mm <= 0.0:
        raise BalloonPathError("the path must have positive finite length")
    return float(length_mm)


def resolve_simulation_duration(config: SimulationConfig) -> SimulationDuration:
    """Resolve a numeric duration, using 0.5 cm/s for ``auto``."""

    configured = config.timeline.duration_s
    if configured != "auto":
        return SimulationDuration(
            duration_s=float(configured),
            automatic=False,
            path_length_mm=None,
            reference_speed_cm_per_s=None,
        )

    path = config.intervention.gd_balloon.path.control_points_file
    if path is None:
        raise BalloonPathError(
            "automatic duration requires a balloon control-point file"
        )
    loaded_path = load_balloon_path(
        path,
        coordinate_system=config.intervention.gd_balloon.path.coordinate_system,
    )
    length_mm = cubic_path_length_mm(loaded_path.control_points_lps_mm)
    return SimulationDuration(
        duration_s=length_mm / AUTO_DURATION_REFERENCE_SPEED_MM_PER_S,
        automatic=True,
        path_length_mm=length_mm,
        reference_speed_cm_per_s=AUTO_DURATION_REFERENCE_SPEED_CM_PER_S,
    )
