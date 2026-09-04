"""Generic TR-level acquisition schedule driven by a user view-order file."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from xcat_icmr.config.models import SimulationConfig


class AcquisitionScheduleError(ValueError):
    """Raised when timing or view ordering cannot form an acquisition."""


@dataclass(frozen=True)
class AcquisitionSchedule:
    """Resolved TR-level sampling instructions for one virtual experiment."""

    actual_tr_s: float
    effective_tr_s: float
    tr_mismatch_percent: float
    trs_per_frame: int
    frame_duration_s: float
    frame_count: int
    acquisition_count: int
    retained_duration_s: float
    dropped_duration_s: float
    trajectory_tr_count: int
    view_order_cycle_length: int
    complete_view_order_cycles: int
    partial_view_order_cycle_tr_count: int
    time_s: np.ndarray
    frame_index_zero_based: np.ndarray
    cardiac_phase_index_zero_based: np.ndarray
    trajectory_tr_index_zero_based: np.ndarray


def load_view_order(
    path: str | Path,
    *,
    variable: str,
    trajectory_tr_count: int,
) -> np.ndarray:
    """Load one repeatable cycle of zero-based trajectory-TR indices."""

    resolved = Path(path).expanduser().resolve(strict=False)
    suffix = resolved.suffix.lower()
    if suffix == ".mat":
        content = loadmat(resolved)
        if variable not in content:
            raise AcquisitionScheduleError(
                f"view-order variable {variable!r} is absent from {resolved}"
            )
        values = content[variable]
    elif suffix == ".npy":
        values = np.load(resolved, allow_pickle=False)
    elif suffix in {".csv", ".txt"}:
        delimiter = "," if suffix == ".csv" else None
        try:
            values = np.loadtxt(resolved, delimiter=delimiter, dtype=np.int64)
        except ValueError:
            values = np.genfromtxt(
                resolved,
                delimiter=delimiter,
                names=True,
                dtype=None,
                encoding="utf-8",
            )
            if values.dtype.names is None or variable not in values.dtype.names:
                raise AcquisitionScheduleError(
                    f"view-order column {variable!r} is absent from {resolved}"
                )
            values = values[variable]
    else:
        raise AcquisitionScheduleError(
            "view-order files must use .csv, .txt, .npy, or .mat"
        )
    order = np.asarray(values)
    if order.ndim == 0:
        order = order.reshape(1)
    elif order.ndim == 2 and 1 in order.shape:
        order = order.reshape(-1)
    elif order.ndim != 1:
        raise AcquisitionScheduleError(
            "view order must be a one-dimensional integer list"
        )
    if order.size == 0:
        raise AcquisitionScheduleError("view order must not be empty")
    if not np.issubdtype(order.dtype, np.number):
        raise AcquisitionScheduleError("view-order indices must be numeric")
    if not np.issubdtype(order.dtype, np.integer):
        if not np.all(np.isfinite(order)) or not np.all(order == np.rint(order)):
            raise AcquisitionScheduleError("view-order indices must be integers")
    order = order.astype(np.int64)
    if np.any(order < 0) or np.any(order >= trajectory_tr_count):
        raise AcquisitionScheduleError(
            "view-order indices must be between 0 and "
            f"{trajectory_tr_count - 1}"
        )
    return order


def write_view_order_csv(path: str | Path, order: np.ndarray, *, variable: str) -> Path:
    """Write a simple, portable one-column view-order file."""

    destination = Path(path).expanduser().resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        destination,
        np.asarray(order, dtype=np.int64).reshape(-1),
        fmt="%d",
        delimiter=",",
        header=variable,
        comments="",
    )
    return destination


def _within_percent(value: float, reference: float, tolerance: float) -> tuple[bool, float]:
    mismatch = abs(value - reference) / reference * 100.0
    return mismatch <= tolerance, mismatch


def build_acquisition_schedule(
    config: SimulationConfig,
    *,
    actual_tr_s: float,
    trajectory_tr_count: int,
    cardiac_phase_count: int,
    view_order_cycles: int | None = None,
) -> AcquisitionSchedule:
    """Resolve timing and repeat one complete view-order cycle as needed."""

    if actual_tr_s <= 0 or trajectory_tr_count <= 0 or cardiac_phase_count <= 0:
        raise AcquisitionScheduleError(
            "TR, trajectory-TR count, and phase count must be positive"
        )
    if view_order_cycles is not None and view_order_cycles <= 0:
        raise AcquisitionScheduleError("view_order_cycles must be positive")
    tolerance = config.acquisition.tr_snap_tolerance_percent
    snapped, mismatch = _within_percent(
        actual_tr_s, config.timeline.xcat_time_step_s, tolerance
    )
    effective_tr = config.timeline.xcat_time_step_s if snapped else actual_tr_s
    ratio = config.acquisition.frame_duration_s / effective_tr
    trs_per_frame = int(round(ratio))
    if trs_per_frame <= 0:
        raise AcquisitionScheduleError("acquisition frame must contain at least one TR")
    resolved_frame_duration = trs_per_frame * effective_tr
    compatible, frame_mismatch = _within_percent(
        config.acquisition.frame_duration_s,
        resolved_frame_duration,
        tolerance,
    )
    if not compatible:
        raise AcquisitionScheduleError(
            f"frame_duration_s={config.acquisition.frame_duration_s:g} is not "
            f"an integer number of effective TRs ({effective_tr:g} s); nearest "
            f"is {resolved_frame_duration:g} s ({frame_mismatch:.3f}% mismatch)"
        )
    order = load_view_order(
        config.acquisition.view_order.file,
        variable=config.acquisition.view_order.variable,
        trajectory_tr_count=trajectory_tr_count,
    )
    if view_order_cycles is None:
        duration = config.timeline.duration_s
        if duration == "auto":
            from xcat_icmr.intervention.path import resolve_simulation_duration

            duration_s = resolve_simulation_duration(config).duration_s
        else:
            duration_s = float(duration)
        if config.acquisition.frame_count == "auto":
            frame_count = int(
                np.floor((duration_s + 1e-12) / resolved_frame_duration)
            )
        else:
            frame_count = int(config.acquisition.frame_count)
        if frame_count <= 0:
            raise AcquisitionScheduleError(
                "duration does not contain one complete frame"
            )
        acquisition_count = frame_count * trs_per_frame
        retained_duration = acquisition_count * effective_tr
        dropped_duration = max(0.0, duration_s - retained_duration)
    else:
        acquisition_count = int(view_order_cycles) * int(order.size)
        frame_count, incomplete_trs = divmod(acquisition_count, trs_per_frame)
        if incomplete_trs:
            raise AcquisitionScheduleError(
                f"{view_order_cycles} view-order cycle(s) contain "
                f"{acquisition_count} TRs, which is not divisible by "
                f"{trs_per_frame} TRs per frame"
            )
        retained_duration = acquisition_count * effective_tr
        dropped_duration = 0.0
    global_tr = np.arange(acquisition_count, dtype=np.int64)
    complete_cycles, partial_cycle = divmod(acquisition_count, order.size)
    return AcquisitionSchedule(
        actual_tr_s=float(actual_tr_s),
        effective_tr_s=float(effective_tr),
        tr_mismatch_percent=float(mismatch),
        trs_per_frame=trs_per_frame,
        frame_duration_s=float(resolved_frame_duration),
        frame_count=frame_count,
        acquisition_count=acquisition_count,
        retained_duration_s=float(retained_duration),
        dropped_duration_s=float(dropped_duration),
        trajectory_tr_count=int(trajectory_tr_count),
        view_order_cycle_length=int(order.size),
        complete_view_order_cycles=int(complete_cycles),
        partial_view_order_cycle_tr_count=int(partial_cycle),
        time_s=global_tr.astype(np.float64) * effective_tr,
        frame_index_zero_based=global_tr // trs_per_frame,
        cardiac_phase_index_zero_based=global_tr % cardiac_phase_count,
        trajectory_tr_index_zero_based=order[global_tr % order.size],
    )
