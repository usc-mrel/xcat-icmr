"""Paired diagnostics for a reconstruction and its fully sampled reference."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Callable

import h5py
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import map_coordinates, uniform_filter

from xcat_icmr.analysis.curved_profile import (
    _disk_offsets,
    _normal_frame,
    _sample_frame,
)


class ReconstructionAssessmentError(ValueError):
    """Raised when a reconstruction/reference pair cannot be assessed."""


@dataclass(frozen=True)
class ReconstructionAssessmentResult:
    output_directory: Path
    metrics_path: Path
    curved_profile_path: Path
    summary_figure_path: Path
    profile_arrays_path: Path
    frame_count: int


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _resolve_result(path: str | Path) -> tuple[Path, dict]:
    candidate = Path(path).expanduser().resolve(strict=False)
    if candidate.is_dir():
        result_path = candidate / "result.json"
    elif candidate.name == "result.json":
        result_path = candidate
    elif candidate.suffix.lower() in {".h5", ".hdf5"}:
        result_path = candidate.parent / "result.json"
    else:
        raise ReconstructionAssessmentError(
            "input must be a reconstruction directory, result.json, or reconstruction HDF5"
        )
    if not result_path.is_file():
        raise ReconstructionAssessmentError(f"result metadata does not exist: {result_path}")
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionAssessmentError(
            f"could not read reconstruction result: {result_path}"
        ) from exc
    if result.get("status") != "complete":
        raise ReconstructionAssessmentError("reconstruction is not complete")
    return result_path, result


def _pair_paths(result: dict) -> tuple[Path, Path, Path]:
    reconstruction = Path(result["reconstruction_file"]).resolve(strict=False)
    acquisition = Path(result["acquisition_file"]).resolve(strict=False)
    if not reconstruction.is_file() or not acquisition.is_file():
        raise ReconstructionAssessmentError(
            "reconstruction or source acquisition file is missing"
        )
    with h5py.File(acquisition, "r") as handle:
        reference_value = str(handle.attrs.get("matching_fullysampled_reference", ""))
    reference = Path(reference_value).resolve(strict=False)
    if not reference.is_file():
        raise ReconstructionAssessmentError(
            f"matching fully sampled reference is missing: {reference}"
        )
    return reconstruction, acquisition, reference


def _validate_pair(reconstruction: Path, reference: Path):
    with h5py.File(reconstruction, "r") as recon, h5py.File(reference, "r") as ref:
        image = recon.get("image_complex")
        truth = ref.get("image")
        if not isinstance(image, h5py.Dataset) or image.ndim != 4:
            raise ReconstructionAssessmentError(
                "reconstruction must contain image_complex [time,x,y,z]"
            )
        if not isinstance(truth, h5py.Dataset) or truth.ndim != 4:
            raise ReconstructionAssessmentError(
                "reference must contain image [x,y,z,time]"
            )
        expected = (truth.shape[3], *truth.shape[:3])
        if image.shape != expected:
            raise ReconstructionAssessmentError(
                f"shape mismatch: reconstruction {image.shape}, reference {expected}"
            )
        complete = recon.get("frame_complete")
        reference_complete = ref.get("frame_complete")
        if (
            not isinstance(complete, h5py.Dataset)
            or not np.all(complete[:])
            or not isinstance(reference_complete, h5py.Dataset)
            or not np.all(reference_complete[:])
        ):
            raise ReconstructionAssessmentError("reconstruction/reference pair is incomplete")
        times = ref.get("frame_center_time_s")
        frame_duration_s = float(truth.attrs.get("frame_duration_s", 0.0))
        frame_times = (
            np.asarray(times[:], dtype=np.float64)
            if isinstance(times, h5py.Dataset)
            else (np.arange(truth.shape[3]) + 0.5) * frame_duration_s
        )
        if frame_times.shape != (truth.shape[3],):
            raise ReconstructionAssessmentError("reference frame times are invalid")
        return tuple(int(value) for value in truth.shape[:3]), frame_times, frame_duration_s


def _global_magnitude_scale(reconstruction: Path, reference: Path) -> float:
    numerator = 0.0
    denominator = 0.0
    with h5py.File(reconstruction, "r") as recon, h5py.File(reference, "r") as ref:
        image = recon["image_complex"]
        truth = ref["image"]
        for frame in range(image.shape[0]):
            estimate = np.abs(np.asarray(image[frame])).astype(np.float64, copy=False)
            target = np.abs(np.asarray(truth[..., frame])).astype(np.float64, copy=False)
            numerator += float(np.sum(estimate * target, dtype=np.float64))
            denominator += float(np.sum(estimate * estimate, dtype=np.float64))
    if not np.isfinite(denominator) or denominator <= 0:
        raise ReconstructionAssessmentError("reconstruction has zero or invalid energy")
    scale = numerator / denominator
    if not np.isfinite(scale) or scale <= 0:
        raise ReconstructionAssessmentError("could not determine a positive global scale")
    return float(scale)


def _curve_ssim(estimate: np.ndarray, truth: np.ndarray, window: int = 7) -> float:
    valid = np.isfinite(estimate) & np.isfinite(truth)
    if np.count_nonzero(valid) < 2:
        return float("nan")
    estimate_values = estimate[valid]
    truth_values = truth[valid]
    data_range = float(
        max(np.max(estimate_values), np.max(truth_values))
        - min(np.min(estimate_values), np.min(truth_values))
    )
    if data_range <= 0:
        return 1.0 if np.allclose(estimate_values, truth_values) else float("nan")
    estimate_fill = np.where(valid, estimate, np.mean(estimate_values))
    truth_fill = np.where(valid, truth, np.mean(truth_values))
    size = tuple(max(1, min(window, value)) for value in estimate.shape)
    mu_e = uniform_filter(estimate_fill, size=size, mode="nearest")
    mu_t = uniform_filter(truth_fill, size=size, mode="nearest")
    var_e = uniform_filter(estimate_fill**2, size=size, mode="nearest") - mu_e**2
    var_t = uniform_filter(truth_fill**2, size=size, mode="nearest") - mu_t**2
    cov = uniform_filter(estimate_fill * truth_fill, size=size, mode="nearest") - mu_e * mu_t
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2 * mu_e * mu_t + c1) * (2 * cov + c2)
    denominator = (mu_e**2 + mu_t**2 + c1) * (var_e + var_t + c2)
    values = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator != 0,
    )
    return float(np.nanmean(values[valid]))


def _crossing(x0: float, y0: float, x1: float, y1: float, level: float) -> float:
    if np.isclose(y1, y0):
        return 0.5 * (x0 + x1)
    return x0 + (level - y0) * (x1 - x0) / (y1 - y0)


def _local_peak_and_fwhm(
    profile: np.ndarray,
    axis_mm: np.ndarray,
    center_mm: float,
    search_radius_mm: float,
) -> tuple[float, float]:
    mask = (
        np.isfinite(profile)
        & (axis_mm >= center_mm - search_radius_mm)
        & (axis_mm <= center_mm + search_radius_mm)
    )
    indices = np.flatnonzero(mask)
    if len(indices) < 3:
        return float("nan"), float("nan")
    peak_index = int(indices[np.argmax(profile[indices])])
    peak_position = float(axis_mm[peak_index])
    edge = max(1, int(np.ceil(0.2 * len(indices))))
    baseline = float(np.median(np.concatenate((profile[indices[:edge]], profile[indices[-edge:]]))))
    amplitude = float(profile[peak_index] - baseline)
    if amplitude <= 0:
        return peak_position, float("nan")
    level = baseline + 0.5 * amplitude
    left = None
    for index in range(peak_index - 1, indices[0] - 1, -1):
        if profile[index] <= level <= profile[index + 1] or profile[index] >= level >= profile[index + 1]:
            left = _crossing(axis_mm[index], profile[index], axis_mm[index + 1], profile[index + 1], level)
            break
    right = None
    for index in range(peak_index, indices[-1]):
        if profile[index] >= level >= profile[index + 1] or profile[index] <= level <= profile[index + 1]:
            right = _crossing(axis_mm[index], profile[index], axis_mm[index + 1], profile[index + 1], level)
            break
    width = float(right - left) if left is not None and right is not None else float("nan")
    return peak_position, width


def _sample_line(
    magnitude: np.ndarray,
    center_mm: np.ndarray,
    direction: np.ndarray,
    offsets_mm: np.ndarray,
    voxel_size_mm: np.ndarray,
) -> np.ndarray:
    points = center_mm[None, :] + offsets_mm[:, None] * direction[None, :]
    voxel = points / voxel_size_mm[None, :] + (np.asarray(magnitude.shape) // 2)[None, :]
    return map_coordinates(
        magnitude,
        voxel.T,
        order=1,
        mode="constant",
        cval=np.nan,
        prefilter=False,
    )


def _regular_disk_offsets(radius_mm: float, step_mm: float) -> np.ndarray:
    axis = np.arange(-radius_mm, radius_mm + 0.5 * step_mm, step_mm)
    first, second = np.meshgrid(axis, axis, indexing="ij")
    keep = first**2 + second**2 <= radius_mm**2 + 1e-9
    return np.column_stack((first[keep], second[keep])).astype(np.float64)


def _sample_plane(
    magnitude: np.ndarray,
    center_mm: np.ndarray,
    voxel_size_mm: np.ndarray,
    normal_one: np.ndarray,
    normal_two: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    points = (
        center_mm[None, :]
        + offsets[:, 0, None] * normal_one[None, :]
        + offsets[:, 1, None] * normal_two[None, :]
    )
    voxel = points / voxel_size_mm[None, :] + (
        np.asarray(magnitude.shape) // 2
    )[None, :]
    return map_coordinates(
        magnitude,
        voxel.T,
        order=1,
        mode="constant",
        cval=np.nan,
        prefilter=False,
    )


def _causal_arc_update(
    profile: np.ndarray,
    arc_mm: np.ndarray,
    previous_s_mm: float | None,
    recent_steps_mm: list[float],
    *,
    max_candidates: int = 8,
    jump_scale_mm: float = 15.0,
    jump_weight: float = 0.75,
) -> tuple[float, float, str]:
    """Select an along-path position using only current and past profiles."""

    finite = np.isfinite(profile) & np.isfinite(arc_mm)
    if np.count_nonzero(finite) < 3:
        if previous_s_mm is None:
            return float("nan"), float("nan"), "no_candidates"
        prediction = previous_s_mm + (
            float(np.median(recent_steps_mm)) if recent_steps_mm else 0.0
        )
        return float(prediction), 0.0, "predicted_no_candidates"
    fill = float(np.median(profile[finite]))
    values = np.where(finite, profile, fill)
    spacing = float(np.median(np.diff(arc_mm)))
    smooth_size = max(1, int(round(1.0 / max(spacing, 1e-6))))
    background_size = max(3, int(round(14.0 / max(spacing, 1e-6))))
    smooth = uniform_filter(values, size=smooth_size, mode="nearest")
    contrast = smooth - uniform_filter(smooth, size=background_size, mode="nearest")
    contrast[~finite] = np.nan
    maxima = np.zeros(len(profile), dtype=bool)
    maxima[1:-1] = (
        (contrast[1:-1] >= contrast[:-2])
        & (contrast[1:-1] > contrast[2:])
        & np.isfinite(contrast[1:-1])
        & (contrast[1:-1] > 0)
    )
    candidates = np.flatnonzero(maxima)
    if not len(candidates):
        candidates = np.asarray([int(np.nanargmax(contrast))])
    candidates = candidates[np.argsort(contrast[candidates])[-max_candidates:]]
    positive = np.clip(contrast[np.isfinite(contrast)], 0.0, None)
    scale = float(np.percentile(positive, 95.0)) if positive.size else 0.0
    if not np.isfinite(scale) or scale <= 0:
        scale = max(float(np.nanmax(contrast)), 1e-12)
    raw_scores = contrast[candidates] / scale
    if previous_s_mm is None:
        selected = int(np.argmax(raw_scores))
        score = float(raw_scores[selected])
    else:
        prediction = previous_s_mm + (
            float(np.median(recent_steps_mm)) if recent_steps_mm else 0.0
        )
        scores = raw_scores - jump_weight * np.abs(
            arc_mm[candidates] - prediction
        ) / max(jump_scale_mm, 1e-6)
        selected = int(np.argmax(scores))
        score = float(scores[selected])
    selected_index = int(candidates[selected])
    selected_contrast = float(contrast[selected_index])
    local = (
        np.isfinite(contrast)
        & (np.abs(arc_mm - arc_mm[selected_index]) <= 4.0)
    )
    weights = np.clip(contrast[local] - 0.5 * max(selected_contrast, 0.0), 0.0, None)
    refined_s = (
        float(np.sum(arc_mm[local] * weights) / np.sum(weights))
        if np.sum(weights) > 0
        else float(arc_mm[selected_index])
    )
    return refined_s, score, "ok" if score >= -0.25 else "low_score"


def _summary(values: np.ndarray) -> dict[str, float | int]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return {
        "mean": float(np.mean(finite)) if finite.size else float("nan"),
        "median": float(np.median(finite)) if finite.size else float("nan"),
        "rmse": float(np.sqrt(np.mean(finite**2))) if finite.size else float("nan"),
        "valid_frames": int(finite.size),
    }


def _directional_extrema(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return finite row-wise minimum, maximum, and max/min anisotropy."""

    rows = np.asarray(values, dtype=np.float64)
    minimum = np.full(rows.shape[0], np.nan, dtype=np.float64)
    maximum = np.full(rows.shape[0], np.nan, dtype=np.float64)
    ratio = np.full(rows.shape[0], np.nan, dtype=np.float64)
    for index, row in enumerate(rows):
        finite = row[np.isfinite(row)]
        if not finite.size:
            continue
        minimum[index] = float(np.min(finite))
        maximum[index] = float(np.max(finite))
        if minimum[index] > 0:
            ratio[index] = maximum[index] / minimum[index]
    return minimum, maximum, ratio


def _error_summary(values: np.ndarray) -> dict[str, float | int]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    absolute = np.abs(finite)
    return {
        "mean_signed_mm": float(np.mean(finite)) if finite.size else float("nan"),
        "mean_absolute_mm": float(np.mean(absolute)) if finite.size else float("nan"),
        "median_absolute_mm": float(np.median(absolute)) if finite.size else float("nan"),
        "rmse_mm": float(np.sqrt(np.mean(finite**2))) if finite.size else float("nan"),
        "maximum_absolute_mm": float(np.max(absolute)) if finite.size else float("nan"),
        "valid_frames": int(finite.size),
    }


def _finite_fwhm_summary(values: np.ndarray) -> dict[str, float | int]:
    """Summarize all finite frame-wise FWHM measurements."""

    data = np.asarray(values, dtype=np.float64)
    selected = data[np.isfinite(data)]
    return {
        "median_mm": (
            float(np.median(selected)) if selected.size else float("nan")
        ),
        "mean_mm": (
            float(np.mean(selected)) if selected.size else float("nan")
        ),
        "finite_frames": int(selected.size),
    }


def _profile_metrics(
    estimate: np.ndarray,
    truth: np.ndarray,
    arc_mm: np.ndarray,
    expected_arc_mm: np.ndarray,
) -> dict:
    correlation = []
    estimate_peak = np.full(len(estimate), np.nan)
    truth_peak = np.full(len(estimate), np.nan)
    estimate_fwhm = np.full(len(estimate), np.nan)
    truth_fwhm = np.full(len(estimate), np.nan)
    for frame in range(len(estimate)):
        valid = np.isfinite(estimate[frame]) & np.isfinite(truth[frame])
        if np.count_nonzero(valid) > 1 and np.std(estimate[frame, valid]) > 0 and np.std(truth[frame, valid]) > 0:
            correlation.append(float(np.corrcoef(estimate[frame, valid], truth[frame, valid])[0, 1]))
        estimate_peak[frame], estimate_fwhm[frame] = _local_peak_and_fwhm(
            estimate[frame], arc_mm, expected_arc_mm[frame], 15.0
        )
        truth_peak[frame], truth_fwhm[frame] = _local_peak_and_fwhm(
            truth[frame], arc_mm, expected_arc_mm[frame], 15.0
        )
    difference = estimate - truth
    finite = np.isfinite(difference)
    valid_fwhm = np.isfinite(estimate_fwhm) & np.isfinite(truth_fwhm)
    valid_peak = np.isfinite(estimate_peak) & np.isfinite(truth_peak)
    return {
        "correlation_mean_per_frame": float(np.mean(correlation)) if correlation else float("nan"),
        "ssim_time_arc_map": _curve_ssim(estimate, truth),
        "rmse": float(np.sqrt(np.mean(difference[finite] ** 2))),
        "tracking_mae_mm": float(np.mean(np.abs(estimate_peak[valid_peak] - truth_peak[valid_peak]))) if np.any(valid_peak) else float("nan"),
        "tracking_valid_frames": int(np.count_nonzero(valid_peak)),
        "longitudinal_fwhm_truth_mean_mm": float(np.mean(truth_fwhm[np.isfinite(truth_fwhm)])),
        "longitudinal_fwhm_reconstruction_mean_mm": float(np.mean(estimate_fwhm[np.isfinite(estimate_fwhm)])),
        "longitudinal_fwhm_mae_mm": float(np.mean(np.abs(estimate_fwhm[valid_fwhm] - truth_fwhm[valid_fwhm]))) if np.any(valid_fwhm) else float("nan"),
        "longitudinal_fwhm_valid_frames": int(np.count_nonzero(valid_fwhm)),
        "truth_peak_arc_mm": truth_peak,
        "reconstruction_peak_arc_mm": estimate_peak,
        "truth_longitudinal_fwhm_mm": truth_fwhm,
        "reconstruction_longitudinal_fwhm_mm": estimate_fwhm,
    }


def _write_curved_comparison(
    output: Path,
    times_s: np.ndarray,
    truth_profile: np.ndarray,
    estimate_profile: np.ndarray,
    arc_mm: np.ndarray,
) -> Path:
    """Compare both line profiles using a perceptual false-color fusion."""

    mpl_root = Path(tempfile.gettempdir()) / "xcat-icmr-matplotlib"
    mpl_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_root))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    finite = np.concatenate((truth_profile[np.isfinite(truth_profile)], estimate_profile[np.isfinite(estimate_profile)]))
    vmax = float(np.percentile(finite, 99.5))
    extent = (times_s[0], times_s[-1], arc_mm[0], arc_mm[-1])
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for axis, values, title in zip(
        axes[:2],
        (truth_profile, estimate_profile),
        ("Fully sampled ground truth", "Causal IRLS reconstruction"),
        strict=True,
    ):
        axis.imshow(values.T, origin="lower", aspect="auto", extent=extent, cmap="gray", vmin=0, vmax=vmax)
        axis.set(title=title, xlabel="Time (s)", ylabel="Arc length (mm)")
    display_max = max(vmax, 1e-12)
    truth_normalized = np.clip(truth_profile / display_max, 0.0, 1.0)
    reconstruction_normalized = np.clip(
        estimate_profile / display_max, 0.0, 1.0
    )
    fusion = np.stack(
        (
            truth_normalized.T,
            reconstruction_normalized.T,
            truth_normalized.T,
        ),
        axis=-1,
    )
    axes[2].imshow(fusion, origin="lower", aspect="auto", extent=extent)
    axes[2].set(
        title=(
            "False-color fusion\nGT magenta | reconstruction green | "
            "agreement white"
        ),
        xlabel="Time (s)",
        ylabel="Arc length (mm)",
    )
    curved_path = output / "curved_profile_comparison.png"
    figure.savefig(curved_path, dpi=180)
    plt.close(figure)
    return curved_path


def _write_summary_figure(
    output: Path,
    metrics: dict,
    arrays_path: Path,
) -> Path:
    """Write a compact review figure for image quality and localization metrics."""

    mpl_root = Path(tempfile.gettempdir()) / "xcat-icmr-matplotlib"
    mpl_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_root))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with np.load(arrays_path) as arrays:
        times_s = np.asarray(arrays["frame_center_time_s"], dtype=np.float64)
        arc_mm = np.asarray(arrays["arc_length_mm"], dtype=np.float64)
        truth = np.asarray(arrays["truth_tube_max"], dtype=np.float32)
        estimate = np.asarray(arrays["reconstruction_tube_max"], dtype=np.float32)
        truth_peak = np.asarray(arrays["truth_peak_arc_mm"], dtype=np.float64)
        estimate_peak = np.asarray(arrays["reconstruction_peak_arc_mm"], dtype=np.float64)
        apparent_cnr = np.asarray(
            arrays["reconstruction_apparent_cnr"], dtype=np.float64
        )
        expected_arc = np.asarray(arrays["expected_arc_length_mm"], dtype=np.float64)
        causal_s = np.asarray(
            arrays["causal_tracked_arc_length_mm"]
            if "causal_tracked_arc_length_mm" in arrays.files
            else estimate_peak,
            dtype=np.float64,
        )
        tracked_cnr = np.asarray(
            arrays["tracked_position_apparent_cnr"]
            if "tracked_position_apparent_cnr" in arrays.files
            else apparent_cnr,
            dtype=np.float64,
        )
        tracker_status = np.asarray(
            arrays["causal_tracker_status"]
            if "causal_tracker_status" in arrays.files
            else np.full(times_s.shape, "ok"),
            dtype=str,
        )
        causal_parallel = np.asarray(
            arrays["causal_tracking_parallel_error_mm"]
            if "causal_tracking_parallel_error_mm" in arrays.files
            else np.full_like(times_s, np.nan),
            dtype=np.float64,
        )
        causal_perpendicular = np.asarray(
            arrays["causal_tracking_perpendicular_error_mm"]
            if "causal_tracking_perpendicular_error_mm" in arrays.files
            else np.full_like(times_s, np.nan),
            dtype=np.float64,
        )
        causal_total = np.asarray(
            arrays["causal_tracking_total_3d_error_mm"]
            if "causal_tracking_total_3d_error_mm" in arrays.files
            else np.full_like(times_s, np.nan),
            dtype=np.float64,
        )
        truth_parallel_fwhm = np.asarray(
            arrays["truth_longitudinal_fwhm_mm"], dtype=np.float64
        )
        estimate_parallel_fwhm = np.asarray(
            arrays["reconstruction_longitudinal_fwhm_mm"], dtype=np.float64
        )
        truth_perpendicular_fwhm = np.asarray(
            arrays["truth_transverse_fwhm_mm"], dtype=np.float64
        )
        estimate_perpendicular_fwhm = np.asarray(
            arrays["reconstruction_transverse_fwhm_mm"], dtype=np.float64
        )

    finite_intensity = np.concatenate(
        (truth[np.isfinite(truth)], estimate[np.isfinite(estimate)])
    )
    vmax = float(np.percentile(finite_intensity, 99.5))
    extent = (times_s[0], times_s[-1], arc_mm[0], arc_mm[-1])
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(18, 10.0),
        constrained_layout=True,
    )
    for axis, values, title in zip(
        axes[0],
        (truth, estimate, estimate),
        (
            "Fully sampled ground truth",
            "Causal IRLS reconstruction",
            "Reconstruction with tracking and apparent CNR",
        ),
        strict=True,
    ):
        axis.imshow(
            values.T,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="gray",
            vmin=0,
            vmax=max(vmax, 1e-12),
        )
        axis.set_title(title)
        axis.set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Arc length (mm)")
    axes[0, 1].sharex(axes[0, 0])
    axes[0, 2].sharex(axes[0, 0])
    axes[0, 1].sharey(axes[0, 0])
    axes[0, 2].sharey(axes[0, 0])

    valid_track = (
        np.isfinite(causal_s)
        & np.isfinite(tracked_cnr)
    )
    valid_cnr = tracked_cnr[valid_track]
    cnr_max = max(float(np.percentile(valid_cnr, 95.0)), 3.0) if valid_cnr.size else 3.0
    axes[0, 2].plot(
        times_s,
        expected_arc,
        color="#00D5E8",
        linestyle="--",
        linewidth=1.4,
        label="Known 3D centre",
        zorder=3,
    )
    axes[0, 2].plot(
        times_s,
        estimate_peak,
        color="#FF6B35",
        linestyle=":",
        linewidth=1.0,
        label="GT-assisted peak",
        zorder=3,
    )
    scatter = axes[0, 2].scatter(
        times_s[valid_track],
        causal_s[valid_track],
        c=tracked_cnr[valid_track],
        cmap="viridis",
        vmin=0.0,
        vmax=cnr_max,
        s=13,
        linewidths=0,
        label="Causal 3D tracker",
        zorder=4,
    )
    colorbar = figure.colorbar(scatter, ax=axes[0, 2], pad=0.02)
    colorbar.set_label("Tracker-guided apparent CNR")
    axes[0, 2].legend(loc="upper left", fontsize=8, framealpha=0.8)

    axes[1, 0].plot(times_s, apparent_cnr, color="#0072B2", label="GT-guided")
    axes[1, 0].plot(times_s, tracked_cnr, color="#009E73", label="Tracker-guided")
    axes[1, 0].axhline(3.0, color="0.35", linestyle="--", linewidth=1.0, label="CNR = 3")
    axes[1, 0].set(title="Apparent CNR", xlabel="Time (s)", ylabel="Apparent CNR")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.2)

    gt_assisted_error = np.abs(estimate_peak - truth_peak)
    axes[1, 1].plot(times_s, gt_assisted_error, color="#FF6B35", label="GT-assisted |arc error|")
    axes[1, 1].plot(times_s, np.abs(causal_parallel), color="#0072B2", label="Causal |parallel|")
    axes[1, 1].plot(times_s, causal_perpendicular, color="#CC79A7", label="Causal perpendicular")
    axes[1, 1].plot(times_s, causal_total, color="black", linewidth=1.2, label="Causal total 3D")
    axes[1, 1].set(title="Localization error", xlabel="Time (s)", ylabel="Error (mm)")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.2)

    axes[1, 2].plot(times_s, truth_parallel_fwhm, color="#0072B2", linestyle="--", label="GT parallel")
    axes[1, 2].plot(times_s, estimate_parallel_fwhm, color="#0072B2", label="Recon parallel")
    axes[1, 2].plot(times_s, truth_perpendicular_fwhm, color="#009E73", linestyle="--", label="GT perpendicular")
    axes[1, 2].plot(times_s, estimate_perpendicular_fwhm, color="#009E73", label="Recon perpendicular")
    axes[1, 2].set(title="Apparent balloon width", xlabel="Time (s)", ylabel="FWHM (mm)")
    axes[1, 2].legend(fontsize=8, ncol=2)
    axes[1, 2].grid(alpha=0.2)

    volume = metrics["volume_magnitude"]
    profile = metrics["curved_profile"]
    cnr = metrics["apparent_cnr"]
    transverse = metrics["transverse_fwhm"]
    causal = metrics.get("causal_3d_tracking", {})
    causal_parallel_mae = causal.get("parallel_error", {}).get("mean_absolute_mm", float("nan"))
    causal_perpendicular_mae = causal.get("perpendicular_error", {}).get("mean_absolute_mm", float("nan"))
    causal_total_mae = causal.get("total_3d_error", {}).get("mean_absolute_mm", float("nan"))
    tracked_cnr_metrics = metrics.get("tracked_position_apparent_cnr", {})
    parallel_width = metrics.get("parallel_fwhm", {})
    perpendicular_width = metrics.get("perpendicular_fwhm", {}).get(
        "median_across_directions", {}
    )
    figure.suptitle(
        (
            f"{1000.0 * metrics['frame_duration_s']:.0f} ms/frame | "
            f"volume corr={volume['correlation']:.3f}, "
            f"NRMSE={100.0 * volume['normalized_rmse_by_truth_rms']:.1f}% | "
            f"path corr={profile['correlation_mean_per_frame']:.3f}, "
            f"SSIM={profile['ssim_time_arc_map']:.3f}\n"
            f"GT-assisted tracking MAE={profile['tracking_mae_mm']:.1f} mm | "
            f"causal 3D MAE parallel/perpendicular/total="
            f"{causal_parallel_mae:.1f}/{causal_perpendicular_mae:.1f}/{causal_total_mae:.1f} mm\n"
            f"GT/tracker-guided apparent CNR="
            f"{cnr['reconstruction_median_at_expected_position']:.1f}/"
            f"{tracked_cnr_metrics.get('reconstruction_median', float('nan')):.1f} | "
            f"parallel/perpendicular FWHM="
            f"{profile['longitudinal_fwhm_reconstruction_mean_mm']:.1f}/"
            f"{transverse['reconstruction_mean_mm']:.1f} mm"
        ),
        fontsize=12,
    )
    plt.close(figure)

    # Presentation-oriented summary: preserve the quantitative plots above as
    # a detailed diagnostic, but make the primary figure communicate the
    # reference, reconstruction, tracking, and directional balloon width
    # directly. Reliability is determined without consulting ground truth.
    parallel_color = "#0072B2"
    perpendicular_color = "#D55E00"
    known_color = "#009E73"
    tracked_color = "#CC79A7"
    reliable_detection = (
        (tracker_status == "ok")
        & np.isfinite(causal_s)
        & np.isfinite(tracked_cnr)
        & (tracked_cnr >= 3.0)
    )
    displayed_parallel = estimate_parallel_fwhm
    displayed_perpendicular = estimate_perpendicular_fwhm

    figure = plt.figure(figsize=(18.0, 10.5), constrained_layout=True)
    grid = figure.add_gridspec(2, 6, height_ratios=(1.0, 1.05))
    reference_axis = figure.add_subplot(grid[0, 0:2])
    reconstruction_axis = figure.add_subplot(grid[0, 2:4])
    metrics_axis = figure.add_subplot(grid[0, 4:6])
    tracking_axis = figure.add_subplot(grid[1, 0:3])
    fwhm_axis = figure.add_subplot(grid[1, 3:6])

    for axis, values, title in (
        (reference_axis, truth, "Fully sampled ground truth LIP"),
        (reconstruction_axis, estimate, "Causal IRLS reconstruction LIP"),
    ):
        axis.imshow(
            values.T,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="gray",
            vmin=0.0,
            vmax=max(vmax, 1e-12),
        )
        axis.set(title=title, xlabel="Time (s)", ylabel="Arc length (mm)")

    tracking_axis.imshow(
        estimate.T,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="gray",
        vmin=0.0,
        vmax=max(vmax, 1e-12),
    )
    tracking_axis.plot(
        times_s,
        expected_arc,
        color=known_color,
        linestyle="--",
        linewidth=1.6,
        label="Known 3D centre",
    )
    tracking_axis.plot(
        times_s,
        np.where(reliable_detection, causal_s, np.nan),
        color=tracked_color,
        linewidth=1.5,
        label="Causal 3D tracker",
    )
    tracking_axis.set(
        title=(
            "Balloon-tip tracking | "
            f"parallel/perpendicular/total MAE "
            f"{causal_parallel_mae:.1f}/{causal_perpendicular_mae:.1f}/"
            f"{causal_total_mae:.1f} mm"
        ),
        xlabel="Time (s)",
        ylabel="Arc length (mm)",
    )
    tracking_axis.legend(loc="upper left", fontsize=9, framealpha=0.85)

    fwhm_axis.plot(
        times_s,
        truth_parallel_fwhm,
        color=parallel_color,
        linestyle="--",
        linewidth=1.0,
        alpha=0.75,
        label="Ground truth parallel",
    )
    fwhm_axis.plot(
        times_s,
        displayed_parallel,
        color=parallel_color,
        marker="o",
        markersize=2.8,
        linewidth=1.3,
        label="Reconstruction parallel",
    )
    fwhm_axis.plot(
        times_s,
        truth_perpendicular_fwhm,
        color=perpendicular_color,
        linestyle="--",
        linewidth=1.0,
        alpha=0.75,
        label="Ground truth perpendicular",
    )
    fwhm_axis.plot(
        times_s,
        displayed_perpendicular,
        color=perpendicular_color,
        marker="o",
        markersize=2.8,
        linewidth=1.3,
        label="Reconstruction perpendicular",
    )
    reliable_fraction = float(np.mean(reliable_detection))
    fwhm_axis.set(
        title="Directional apparent balloon width",
        xlabel="Time (s)",
        ylabel="FWHM (mm)",
    )
    fwhm_axis.grid(alpha=0.2)
    fwhm_axis.legend(loc="best", fontsize=9, ncol=2)

    metrics_axis.axis("off")
    metric_lines = (
        "IMAGE QUALITY ASSESSMENT",
        "",
        f"Volume correlation       {volume['correlation']:.3f}",
        f"Volume NRMSE             {100.0 * volume['normalized_rmse_by_truth_rms']:.1f}%",
        f"Time-arc SSIM            {profile['ssim_time_arc_map']:.3f}",
        f"Time-arc correlation     {profile['correlation_mean_per_frame']:.3f}",
        "",
        f"GT-guided apparent CNR   {cnr['reconstruction_median_at_expected_position']:.1f}",
        f"Tracker-guided CNR       {tracked_cnr_metrics.get('reconstruction_median', float('nan')):.1f}",
        f"Reliable detections      {100.0 * reliable_fraction:.1f}%",
        "",
        f"Parallel tracking MAE    {causal_parallel_mae:.1f} mm",
        f"Perpendicular MAE        {causal_perpendicular_mae:.1f} mm",
        f"Total 3D tracking MAE    {causal_total_mae:.1f} mm",
        "",
        "FWHM                    median [mean]",
        (
            "GT parallel              "
            f"{parallel_width.get('truth_median_mm', float('nan')):.1f} "
            f"[{parallel_width.get('truth_mean_mm', float('nan')):.1f}] mm"
        ),
        (
            "Recon parallel           "
            f"{parallel_width.get('reconstruction_median_mm', float('nan')):.1f} "
            f"[{parallel_width.get('reconstruction_mean_mm', float('nan')):.1f}] mm"
        ),
        (
            "GT perpendicular         "
            f"{perpendicular_width.get('truth_median_mm', float('nan')):.1f} "
            f"[{perpendicular_width.get('truth_mean_mm', float('nan')):.1f}] mm"
        ),
        (
            "Recon perpendicular      "
            f"{perpendicular_width.get('reconstruction_median_mm', float('nan')):.1f} "
            f"[{perpendicular_width.get('reconstruction_mean_mm', float('nan')):.1f}] mm"
        ),
    )
    metrics_axis.text(
        0.05,
        0.95,
        "\n".join(metric_lines),
        transform=metrics_axis.transAxes,
        va="top",
        ha="left",
        fontsize=12,
        linespacing=1.35,
        family="monospace",
        bbox={
            "boxstyle": "round,pad=0.8",
            "facecolor": "#F3F0F7",
            "edgecolor": "#6B5B73",
            "linewidth": 1.5,
        },
    )
    figure.suptitle(
        f"Reconstruction IQA summary | {1000.0 * metrics['frame_duration_s']:.0f} ms/frame",
        fontsize=16,
    )
    summary_path = output / "summary_line_profiles.png"
    figure.savefig(summary_path, dpi=180)
    plt.close(figure)
    return summary_path


def _write_tracking_overlay(
    output: Path,
    metrics: dict,
    arrays_path: Path,
) -> Path:
    """Overlay the known and causally estimated positions on the line profile."""

    mpl_root = Path(tempfile.gettempdir()) / "xcat-icmr-matplotlib"
    mpl_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_root))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with np.load(arrays_path) as arrays:
        times_s = np.asarray(arrays["frame_center_time_s"], dtype=np.float64)
        arc_mm = np.asarray(arrays["arc_length_mm"], dtype=np.float64)
        profile = np.asarray(arrays["reconstruction_tube_max"], dtype=np.float32)
        truth_profile = np.asarray(arrays["truth_tube_max"], dtype=np.float32)
        ground_truth_s = np.asarray(
            arrays["expected_arc_length_mm"], dtype=np.float64
        )
        estimated_s = np.asarray(
            arrays["causal_tracked_arc_length_mm"], dtype=np.float64
        )

    finite_intensity = np.concatenate(
        (profile[np.isfinite(profile)], truth_profile[np.isfinite(truth_profile)])
    )
    if not finite_intensity.size:
        raise ReconstructionAssessmentError("line profile has no finite intensities")
    vmax = float(np.percentile(finite_intensity, 99.5))
    extent = (times_s[0], times_s[-1], arc_mm[0], arc_mm[-1])
    figure, axes = plt.subplots(1, 2, figsize=(18.0, 6.8), constrained_layout=True)
    image = axes[0].imshow(
        profile.T,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="gray",
        vmin=0.0,
        vmax=max(vmax, 1e-12),
    )
    axes[0].plot(
        times_s,
        ground_truth_s,
        color="#00D5E8",
        linestyle="--",
        linewidth=2.2,
        label="Known ground-truth position",
        zorder=3,
    )
    axes[0].plot(
        times_s,
        estimated_s,
        color="#FF6B35",
        linewidth=1.6,
        label="Causal 3D estimated position",
        zorder=4,
    )
    causal = metrics["causal_3d_tracking"]
    parallel_mae = causal["parallel_error"]["mean_absolute_mm"]
    perpendicular_mae = causal["perpendicular_error"]["mean_absolute_mm"]
    total_mae = causal["total_3d_error"]["mean_absolute_mm"]
    axes[0].set(
        xlabel="Time (s)",
        ylabel="Arc length (mm)",
        title="Position overlay on reconstruction",
    )
    axes[0].legend(loc="upper left", framealpha=0.9)
    figure.colorbar(
        image, ax=axes[0], pad=0.02, label="Reconstructed magnitude (a.u.)"
    )

    display_max = max(vmax, 1e-12)
    truth_normalized = np.clip(truth_profile / display_max, 0.0, 1.0)
    reconstruction_normalized = np.clip(profile / display_max, 0.0, 1.0)
    fusion = np.stack(
        (
            truth_normalized.T,
            reconstruction_normalized.T,
            truth_normalized.T,
        ),
        axis=-1,
    )
    axes[1].imshow(
        fusion,
        origin="lower",
        aspect="auto",
        extent=extent,
    )
    axes[1].set(
        xlabel="Time (s)",
        ylabel="Arc length (mm)",
        title="False-color fusion | GT magenta | reconstruction green | agreement gray/white",
    )
    figure.suptitle(
        "Causal 3D balloon tracking\n"
        f"parallel MAE={parallel_mae:.2f} mm | "
        f"perpendicular MAE={perpendicular_mae:.2f} mm | "
        f"total 3D MAE={total_mae:.2f} mm",
        fontsize=14,
    )
    path = output / "causal_tracking_overlay.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def assess_reconstruction(
    path: str | Path,
    *,
    overwrite: bool = False,
    progress: Callable[[str], None] | None = None,
) -> ReconstructionAssessmentResult:
    """Assess a completed reconstruction against its recorded reference."""

    result_path, result = _resolve_result(path)
    reconstruction, acquisition, reference = _pair_paths(result)
    shape, frame_times, frame_duration = _validate_pair(reconstruction, reference)
    output = result_path.parent / "diagnostics"
    metrics_path = output / "metrics.json"
    if metrics_path.is_file() and not overwrite:
        metadata = json.loads(metrics_path.read_text(encoding="utf-8"))
        arrays_path = output / "curve_profiles.npz"
        summary_path = output / "summary_line_profiles.png"
        curved_path = output / "curved_profile_comparison.png"
        if not arrays_path.is_file():
            raise ReconstructionAssessmentError(
                f"assessment cache is missing: {arrays_path}; pass --overwrite"
            )
        visualization_schema = int(
            metadata.get("visualization_schema_version", 1)
        )
        if visualization_schema < 6:
            with np.load(arrays_path) as arrays:
                parallel_summary = _finite_fwhm_summary(
                    arrays["reconstruction_longitudinal_fwhm_mm"]
                )
                perpendicular_summary = _finite_fwhm_summary(
                    arrays["reconstruction_transverse_fwhm_mm"]
                )
                truth_parallel_median = float(
                    np.nanmedian(arrays["truth_longitudinal_fwhm_mm"])
                )
                truth_perpendicular_median = float(
                    np.nanmedian(arrays["truth_transverse_fwhm_mm"])
                )
            metadata.pop("reliable_frame_fwhm", None)
            metadata["parallel_fwhm"]["reconstruction_mean_mm"] = (
                parallel_summary["mean_mm"]
            )
            metadata["parallel_fwhm"]["reconstruction_median_mm"] = (
                parallel_summary["median_mm"]
            )
            metadata["parallel_fwhm"]["truth_median_mm"] = (
                truth_parallel_median
            )
            metadata["perpendicular_fwhm"]["median_across_directions"][
                "reconstruction_mean_mm"
            ] = perpendicular_summary["mean_mm"]
            metadata["perpendicular_fwhm"]["median_across_directions"][
                "reconstruction_median_mm"
            ] = perpendicular_summary["median_mm"]
            metadata["perpendicular_fwhm"]["median_across_directions"][
                "truth_median_mm"
            ] = truth_perpendicular_median
            metadata["visualization_schema_version"] = 6
            _write_json_atomic(metrics_path, metadata)
            summary_path = _write_summary_figure(
                output, metadata, arrays_path
            )
        elif not summary_path.is_file():
            summary_path = _write_summary_figure(output, metadata, arrays_path)
        if (
            not curved_path.is_file()
            or visualization_schema < 3
        ):
            with np.load(arrays_path) as arrays:
                curved_path = _write_curved_comparison(
                    output,
                    np.asarray(arrays["frame_center_time_s"], dtype=np.float64),
                    np.asarray(arrays["truth_tube_max"], dtype=np.float32),
                    np.asarray(
                        arrays["reconstruction_tube_max"], dtype=np.float32
                    ),
                    np.asarray(arrays["arc_length_mm"], dtype=np.float64),
                )
        result["assessment"] = {
            "metrics_file": str(metrics_path),
            "curved_profile_comparison": str(curved_path),
            "summary_line_profiles": str(summary_path),
        }
        _write_json_atomic(result_path, result)
        return ReconstructionAssessmentResult(
            output,
            metrics_path,
            curved_path,
            summary_path,
            arrays_path,
            int(metadata["frame_count"]),
        )
    output.mkdir(parents=True, exist_ok=True)
    scale = _global_magnitude_scale(reconstruction, reference)
    if progress:
        progress(f"Global magnitude scale: {scale:.8g}")

    profile_path = reference.parent / "analysis" / "curved_line_profile" / "curved_line_profile.mat"
    if not profile_path.is_file():
        raise ReconstructionAssessmentError(
            f"verified ground-truth curve geometry is missing: {profile_path}"
        )
    profile = loadmat(profile_path, squeeze_me=True)
    arc_mm = np.asarray(profile["arc_length_mm"], dtype=np.float64).reshape(-1)
    curve_mm = np.asarray(profile["curve_logical_mm"], dtype=np.float64)
    curve_voxel = np.asarray(profile["curve_logical_voxel"], dtype=np.float64)
    if curve_mm.shape != (len(arc_mm), 3) or curve_voxel.shape != curve_mm.shape:
        raise ReconstructionAssessmentError("saved curve geometry is incompatible")
    with h5py.File(acquisition, "r") as acquisition_handle:
        voxel_size = np.asarray(
            acquisition_handle.attrs["reconstruction_voxel_size_mm"],
            dtype=np.float64,
        )
        velocity_mm_s = float(
            acquisition_handle.attrs["velocity_cm_per_s"]
        ) * 10.0
    normal_one, normal_two = _normal_frame(curve_mm)
    tangent = np.gradient(curve_mm, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    tube_offsets = _disk_offsets(7.0, float(np.min(voxel_size)), 16)
    tracker_search_radius_mm = 10.5
    tracker_search_offsets = _regular_disk_offsets(
        tracker_search_radius_mm, float(np.min(voxel_size)) / 2.0
    )
    background_offsets = []
    for radius in np.arange(10.5, 17.5 + 0.5 * np.min(voxel_size), np.min(voxel_size)):
        for angle in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            background_offsets.append((radius * np.cos(angle), radius * np.sin(angle)))
    background_offsets = np.asarray(background_offsets, dtype=np.float64)
    frame_count = len(frame_times)
    truth_tube = np.empty((frame_count, len(arc_mm)), dtype=np.float32)
    estimate_tube = np.empty_like(truth_tube)
    truth_cnr = np.full(frame_count, np.nan, dtype=np.float32)
    estimate_cnr = np.full(frame_count, np.nan, dtype=np.float32)
    truth_transverse = np.full(frame_count, np.nan, dtype=np.float32)
    estimate_transverse = np.full(frame_count, np.nan, dtype=np.float32)
    perpendicular_direction_count = 8
    perpendicular_angles_rad = np.linspace(
        0.0, np.pi, perpendicular_direction_count, endpoint=False
    )
    perpendicular_angles_deg = np.rad2deg(perpendicular_angles_rad)
    truth_perpendicular = np.full(
        (frame_count, perpendicular_direction_count), np.nan, dtype=np.float32
    )
    estimate_perpendicular = np.full_like(truth_perpendicular, np.nan)
    tracked_s = np.full(frame_count, np.nan, dtype=np.float32)
    tracked_xyz = np.full((frame_count, 3), np.nan, dtype=np.float32)
    tracker_confidence = np.full(frame_count, np.nan, dtype=np.float32)
    tracker_status: list[str] = []
    tracker_cnr = np.full(frame_count, np.nan, dtype=np.float32)
    tracker_error_parallel = np.full(frame_count, np.nan, dtype=np.float32)
    tracker_error_normal_one = np.full(frame_count, np.nan, dtype=np.float32)
    tracker_error_normal_two = np.full(frame_count, np.nan, dtype=np.float32)
    tracker_error_perpendicular = np.full(frame_count, np.nan, dtype=np.float32)
    tracker_error_total = np.full(frame_count, np.nan, dtype=np.float32)
    expected_arc = np.clip(frame_times * velocity_mm_s, 0.0, arc_mm[-1])
    sums = {name: 0.0 for name in ("e", "t", "ee", "tt", "et", "sq")}
    count = 0
    transverse_axis = np.arange(-20.0, 20.0 + 0.25, 0.5)
    previous_s: float | None = None
    previous_uv: np.ndarray | None = None
    recent_s_steps: list[float] = []
    recent_uv_steps: list[np.ndarray] = []
    with h5py.File(reconstruction, "r") as recon, h5py.File(reference, "r") as ref:
        for frame in range(frame_count):
            truth = np.abs(np.asarray(ref["image"][..., frame])).astype(np.float32)
            estimate = scale * np.abs(np.asarray(recon["image_complex"][frame])).astype(np.float32)
            for key, value in (("e", estimate), ("t", truth), ("ee", estimate * estimate), ("tt", truth * truth), ("et", estimate * truth), ("sq", (estimate - truth) ** 2)):
                sums[key] += float(np.sum(value, dtype=np.float64))
            count += truth.size
            truth_samples = _sample_frame(truth, curve_mm, voxel_size, normal_one, normal_two, tube_offsets)
            estimate_samples = _sample_frame(estimate, curve_mm, voxel_size, normal_one, normal_two, tube_offsets)
            truth_tube[frame] = np.nanmax(truth_samples, axis=0)
            estimate_tube[frame] = np.nanmax(estimate_samples, axis=0)

            current_s, arc_score, status = _causal_arc_update(
                estimate_tube[frame], arc_mm, previous_s, recent_s_steps
            )
            current_uv = None
            if np.isfinite(current_s):
                track_index = int(np.argmin(np.abs(arc_mm - current_s)))
                track_curve_center = np.asarray(
                    [np.interp(current_s, arc_mm, curve_mm[:, axis]) for axis in range(3)]
                )
                transverse_values = _sample_plane(
                    estimate,
                    track_curve_center,
                    voxel_size,
                    normal_one[track_index],
                    normal_two[track_index],
                    tracker_search_offsets,
                )
                finite_transverse = np.isfinite(transverse_values)
                if np.any(finite_transverse):
                    baseline = float(np.nanmedian(transverse_values))
                    contrast = transverse_values - baseline
                    positive = np.clip(contrast[finite_transverse], 0.0, None)
                    contrast_scale = (
                        float(np.percentile(positive, 95.0))
                        if positive.size
                        else 0.0
                    )
                    if not np.isfinite(contrast_scale) or contrast_scale <= 0:
                        contrast_scale = max(float(np.nanmax(contrast)), 1e-12)
                    transverse_score = contrast / contrast_scale
                    if previous_uv is not None:
                        predicted_uv = previous_uv + (
                            np.median(np.asarray(recent_uv_steps), axis=0)
                            if recent_uv_steps
                            else 0.0
                        )
                        transverse_score -= 0.75 * np.linalg.norm(
                            tracker_search_offsets - predicted_uv[None, :], axis=1
                        ) / tracker_search_radius_mm
                    transverse_score[~finite_transverse] = -np.inf
                    selected_offset = int(np.argmax(transverse_score))
                    peak_uv = tracker_search_offsets[selected_offset]
                    local = (
                        finite_transverse
                        & (
                            np.linalg.norm(
                                tracker_search_offsets - peak_uv[None, :], axis=1
                            )
                            <= float(np.min(voxel_size))
                        )
                    )
                    half_level = 0.5 * max(float(contrast[selected_offset]), 0.0)
                    weights = np.clip(contrast[local] - half_level, 0.0, None)
                    current_uv = (
                        np.sum(tracker_search_offsets[local] * weights[:, None], axis=0)
                        / np.sum(weights)
                        if np.sum(weights) > 0
                        else peak_uv
                    )
                    tracked_s[frame] = float(current_s)
                    tracked_xyz[frame] = (
                        track_curve_center
                        + current_uv[0] * normal_one[track_index]
                        + current_uv[1] * normal_two[track_index]
                    )
                    tracker_confidence[frame] = float(
                        min(arc_score, transverse_score[selected_offset])
                    )
                    signal_at_tracker = _sample_plane(
                        estimate,
                        tracked_xyz[frame],
                        voxel_size,
                        normal_one[track_index],
                        normal_two[track_index],
                        tube_offsets,
                    )
                    background_at_tracker = _sample_plane(
                        estimate,
                        tracked_xyz[frame],
                        voxel_size,
                        normal_one[track_index],
                        normal_two[track_index],
                        background_offsets,
                    )
                    tracker_background = float(np.nanmedian(background_at_tracker))
                    tracker_sigma = 1.4826 * float(
                        np.nanmedian(np.abs(background_at_tracker - tracker_background))
                    )
                    if tracker_sigma > 0:
                        tracker_cnr[frame] = (
                            float(np.nanmax(signal_at_tracker)) - tracker_background
                        ) / tracker_sigma
                else:
                    status = "no_transverse_candidates"
            tracker_status.append(status)
            if current_uv is not None and np.all(np.isfinite(current_uv)):
                if previous_s is not None:
                    recent_s_steps.append(float(tracked_s[frame] - previous_s))
                    recent_s_steps = recent_s_steps[-4:]
                if previous_uv is not None:
                    recent_uv_steps.append(current_uv - previous_uv)
                    recent_uv_steps = recent_uv_steps[-4:]
                previous_s = float(tracked_s[frame])
                previous_uv = np.asarray(current_uv, dtype=np.float64)

                expected_point = np.asarray(
                    [np.interp(expected_arc[frame], arc_mm, curve_mm[:, axis]) for axis in range(3)]
                )
                expected_tangent = np.asarray(
                    [np.interp(expected_arc[frame], arc_mm, tangent[:, axis]) for axis in range(3)]
                )
                expected_tangent /= max(np.linalg.norm(expected_tangent), 1e-12)
                expected_index = int(np.argmin(np.abs(arc_mm - expected_arc[frame])))
                difference_xyz = tracked_xyz[frame] - expected_point
                tracker_error_parallel[frame] = float(
                    np.dot(difference_xyz, expected_tangent)
                )
                tracker_error_normal_one[frame] = float(
                    np.dot(difference_xyz, normal_one[expected_index])
                )
                tracker_error_normal_two[frame] = float(
                    np.dot(difference_xyz, normal_two[expected_index])
                )
                tracker_error_perpendicular[frame] = float(
                    np.linalg.norm(
                        difference_xyz
                        - tracker_error_parallel[frame] * expected_tangent
                    )
                )
                tracker_error_total[frame] = float(np.linalg.norm(difference_xyz))
            truth_background = _sample_frame(truth, curve_mm, voxel_size, normal_one, normal_two, background_offsets)
            estimate_background = _sample_frame(estimate, curve_mm, voxel_size, normal_one, normal_two, background_offsets)
            index = int(np.argmin(np.abs(arc_mm - expected_arc[frame])))
            for values, background, destination in (
                (truth_tube, truth_background, truth_cnr),
                (estimate_tube, estimate_background, estimate_cnr),
            ):
                samples = background[:, index]
                median = float(np.nanmedian(samples))
                sigma = 1.4826 * float(np.nanmedian(np.abs(samples - median)))
                destination[frame] = (values[frame, index] - median) / sigma if sigma > 0 else np.nan
            for magnitude, destination, directional in (
                (truth, truth_transverse, truth_perpendicular),
                (estimate, estimate_transverse, estimate_perpendicular),
            ):
                widths: list[float] = []
                center = curve_mm[index]
                for direction_index, angle in enumerate(perpendicular_angles_rad):
                    direction = (
                        np.cos(angle) * normal_one[index]
                        + np.sin(angle) * normal_two[index]
                    )
                    line = _sample_line(magnitude, center, direction, transverse_axis, voxel_size)
                    _, width = _local_peak_and_fwhm(line, transverse_axis, 0.0, 10.0)
                    if np.isfinite(width):
                        directional[frame, direction_index] = width
                        widths.append(width)
                if widths:
                    destination[frame] = float(np.median(widths))
            if progress and (frame == 0 or (frame + 1) % 20 == 0 or frame + 1 == frame_count):
                progress(f"Assessment frame {frame + 1}/{frame_count}")

    mean_e, mean_t = sums["e"] / count, sums["t"] / count
    variance_e = sums["ee"] / count - mean_e**2
    variance_t = sums["tt"] / count - mean_t**2
    covariance = sums["et"] / count - mean_e * mean_t
    volume_correlation = covariance / np.sqrt(max(variance_e * variance_t, 0.0))
    volume_rmse = np.sqrt(sums["sq"] / count)
    path_metrics = _profile_metrics(estimate_tube, truth_tube, arc_mm, expected_arc)
    path_metrics["gt_assisted_tracking_mae_mm"] = path_metrics["tracking_mae_mm"]
    path_metrics["gt_assisted_tracking_valid_frames"] = path_metrics[
        "tracking_valid_frames"
    ]
    valid_transverse = np.isfinite(truth_transverse) & np.isfinite(estimate_transverse)
    valid_tracker_cnr = np.isfinite(tracker_cnr)
    parallel_fwhm_valid = np.isfinite(path_metrics["truth_longitudinal_fwhm_mm"]) & np.isfinite(
        path_metrics["reconstruction_longitudinal_fwhm_mm"]
    )
    perpendicular_direction_metrics = []
    for direction in range(perpendicular_direction_count):
        valid = np.isfinite(truth_perpendicular[:, direction]) & np.isfinite(
            estimate_perpendicular[:, direction]
        )
        perpendicular_direction_metrics.append(
            {
                "angle_deg": float(perpendicular_angles_deg[direction]),
                "truth_mean_mm": float(np.nanmean(truth_perpendicular[:, direction])),
                "reconstruction_mean_mm": float(
                    np.nanmean(estimate_perpendicular[:, direction])
                ),
                "mae_mm": float(
                    np.mean(
                        np.abs(
                            estimate_perpendicular[valid, direction]
                            - truth_perpendicular[valid, direction]
                        )
                    )
                )
                if np.any(valid)
                else float("nan"),
                "valid_frames": int(np.count_nonzero(valid)),
            }
        )
    truth_perpendicular_min, truth_perpendicular_max, truth_perpendicular_ratio = (
        _directional_extrema(truth_perpendicular)
    )
    (
        estimate_perpendicular_min,
        estimate_perpendicular_max,
        estimate_perpendicular_ratio,
    ) = _directional_extrema(estimate_perpendicular)
    parallel_fwhm_summary = _finite_fwhm_summary(
        path_metrics["reconstruction_longitudinal_fwhm_mm"],
    )
    perpendicular_fwhm_summary = _finite_fwhm_summary(
        estimate_transverse,
    )
    metrics = {
        "schema_version": 2,
        "visualization_schema_version": 6,
        "reconstruction_id": result["reconstruction_id"],
        "reconstruction_file": reconstruction,
        "acquisition_file": acquisition,
        "reference_file": reference,
        "frame_count": frame_count,
        "frame_duration_s": frame_duration,
        "image_shape": shape,
        "global_magnitude_scale": scale,
        "volume_magnitude": {
            "correlation": float(volume_correlation),
            "rmse": float(volume_rmse),
            "normalized_rmse_by_truth_rms": float(volume_rmse / np.sqrt(sums["tt"] / count)),
        },
        "curved_profile": {key: value for key, value in path_metrics.items() if not isinstance(value, np.ndarray)},
        "apparent_cnr": {
            "definition": "tube maximum minus annulus median, divided by 1.4826 times annulus MAD",
            "guidance": "ground_truth_position",
            "tube_radius_mm": 7.0,
            "background_annulus_mm": [10.5, 17.5],
            "truth_median_at_expected_position": float(np.nanmedian(truth_cnr)),
            "reconstruction_median_at_expected_position": float(np.nanmedian(estimate_cnr)),
            "reconstruction_visible_fraction_cnr_ge_3": float(np.mean(estimate_cnr >= 3.0)),
        },
        "tracked_position_apparent_cnr": {
            "definition": "tube maximum minus annulus median, divided by 1.4826 times annulus MAD",
            "guidance": "causal_3d_tracker_position",
            "tube_radius_mm": 7.0,
            "background_annulus_mm": [10.5, 17.5],
            "reconstruction_median": float(np.nanmedian(tracker_cnr)),
            "visible_fraction_all_frames_cnr_ge_3": float(np.mean(tracker_cnr >= 3.0)),
            "visible_fraction_valid_frames_cnr_ge_3": float(
                np.mean(tracker_cnr[valid_tracker_cnr] >= 3.0)
            )
            if np.any(valid_tracker_cnr)
            else float("nan"),
            "valid_frames": int(np.count_nonzero(valid_tracker_cnr)),
        },
        "causal_3d_tracking": {
            "definition": "causal along-path contrast tracker followed by a causal local normal-plane search",
            "coordinate_system": "reconstruction logical physical coordinates in millimetres",
            "parallel_search_scale_mm": 15.0,
            "perpendicular_search_radius_mm": tracker_search_radius_mm,
            "parallel_error": _error_summary(tracker_error_parallel),
            "normal_one_error": _error_summary(tracker_error_normal_one),
            "normal_two_error": _error_summary(tracker_error_normal_two),
            "perpendicular_error": _error_summary(tracker_error_perpendicular),
            "total_3d_error": _error_summary(tracker_error_total),
            "status_counts": {
                status: tracker_status.count(status) for status in sorted(set(tracker_status))
            },
        },
        "parallel_fwhm": {
            "truth_mean_mm": float(
                np.nanmean(path_metrics["truth_longitudinal_fwhm_mm"])
            ),
            "truth_median_mm": float(
                np.nanmedian(path_metrics["truth_longitudinal_fwhm_mm"])
            ),
            "reconstruction_mean_mm": float(
                parallel_fwhm_summary["mean_mm"]
            ),
            "reconstruction_median_mm": float(
                parallel_fwhm_summary["median_mm"]
            ),
            "mae_mm": float(
                np.mean(
                    np.abs(
                        path_metrics["reconstruction_longitudinal_fwhm_mm"][parallel_fwhm_valid]
                        - path_metrics["truth_longitudinal_fwhm_mm"][parallel_fwhm_valid]
                    )
                )
            )
            if np.any(parallel_fwhm_valid)
            else float("nan"),
            "valid_frames": int(np.count_nonzero(parallel_fwhm_valid)),
        },
        "perpendicular_fwhm": {
            "direction_count": perpendicular_direction_count,
            "angles_deg": perpendicular_angles_deg.tolist(),
            "directions": perpendicular_direction_metrics,
            "normal_one": perpendicular_direction_metrics[0],
            "normal_two": perpendicular_direction_metrics[
                perpendicular_direction_count // 2
            ],
            "median_across_directions": {
                "truth_mean_mm": float(np.nanmean(truth_transverse)),
                "truth_median_mm": float(np.nanmedian(truth_transverse)),
                "reconstruction_mean_mm": float(
                    perpendicular_fwhm_summary["mean_mm"]
                ),
                "reconstruction_median_mm": float(
                    perpendicular_fwhm_summary["median_mm"]
                ),
                "mae_mm": float(np.mean(np.abs(estimate_transverse[valid_transverse] - truth_transverse[valid_transverse]))) if np.any(valid_transverse) else float("nan"),
                "valid_frames": int(np.count_nonzero(valid_transverse)),
            },
            "directional_spread": {
                "truth_minimum_mean_mm": float(
                    np.nanmean(truth_perpendicular_min)
                ),
                "truth_maximum_mean_mm": float(
                    np.nanmean(truth_perpendicular_max)
                ),
                "truth_anisotropy_ratio_median": float(
                    np.nanmedian(truth_perpendicular_ratio)
                ),
                "reconstruction_minimum_mean_mm": float(
                    np.nanmean(estimate_perpendicular_min)
                ),
                "reconstruction_maximum_mean_mm": float(
                    np.nanmean(estimate_perpendicular_max)
                ),
                "reconstruction_anisotropy_ratio_median": float(
                    np.nanmedian(estimate_perpendicular_ratio)
                ),
            },
        },
        "transverse_fwhm": {
            "truth_mean_mm": float(np.nanmean(truth_transverse)),
            "reconstruction_mean_mm": float(np.nanmean(estimate_transverse)),
            "mae_mm": float(np.mean(np.abs(estimate_transverse[valid_transverse] - truth_transverse[valid_transverse]))) if np.any(valid_transverse) else float("nan"),
            "valid_frames": int(np.count_nonzero(valid_transverse)),
        },
        "definitions": {
            "ssim": "2-D local SSIM on the time-by-arc-length tube-maximum maps using a 7x7 uniform window",
            "tracking": "ground-truth-assisted absolute difference between local reconstructed and reference peaks within +/-15 mm of the known position",
            "causal_3d_tracking": "tracker uses only current and previous reconstructed frames; errors are evaluated against the known 3D balloon centre",
            "longitudinal_fwhm": "baseline-corrected FWHM along arc length within +/-15 mm of the known position",
            "transverse_fwhm": "median baseline-corrected FWHM across eight uniformly spaced lines in the plane normal to the known path position",
        },
    }
    _write_json_atomic(metrics_path, metrics)
    arrays_path = output / "curve_profiles.npz"
    np.savez_compressed(
        arrays_path,
        arc_length_mm=arc_mm,
        frame_center_time_s=frame_times,
        expected_arc_length_mm=expected_arc,
        truth_tube_max=truth_tube,
        reconstruction_tube_max=estimate_tube,
        truth_apparent_cnr=truth_cnr,
        reconstruction_apparent_cnr=estimate_cnr,
        tracked_position_apparent_cnr=tracker_cnr,
        causal_tracked_arc_length_mm=tracked_s,
        causal_tracked_logical_xyz_mm=tracked_xyz,
        causal_tracker_confidence=tracker_confidence,
        causal_tracker_status=np.asarray(tracker_status),
        causal_tracking_parallel_error_mm=tracker_error_parallel,
        causal_tracking_normal_one_error_mm=tracker_error_normal_one,
        causal_tracking_normal_two_error_mm=tracker_error_normal_two,
        causal_tracking_perpendicular_error_mm=tracker_error_perpendicular,
        causal_tracking_total_3d_error_mm=tracker_error_total,
        truth_transverse_fwhm_mm=truth_transverse,
        reconstruction_transverse_fwhm_mm=estimate_transverse,
        truth_perpendicular_fwhm_mm=truth_perpendicular,
        reconstruction_perpendicular_fwhm_mm=estimate_perpendicular,
        perpendicular_fwhm_angles_deg=perpendicular_angles_deg,
        truth_perpendicular_fwhm_min_mm=truth_perpendicular_min,
        truth_perpendicular_fwhm_max_mm=truth_perpendicular_max,
        truth_perpendicular_fwhm_anisotropy_ratio=truth_perpendicular_ratio,
        reconstruction_perpendicular_fwhm_min_mm=estimate_perpendicular_min,
        reconstruction_perpendicular_fwhm_max_mm=estimate_perpendicular_max,
        reconstruction_perpendicular_fwhm_anisotropy_ratio=(
            estimate_perpendicular_ratio
        ),
        truth_peak_arc_mm=path_metrics["truth_peak_arc_mm"],
        reconstruction_peak_arc_mm=path_metrics["reconstruction_peak_arc_mm"],
        truth_longitudinal_fwhm_mm=path_metrics["truth_longitudinal_fwhm_mm"],
        reconstruction_longitudinal_fwhm_mm=path_metrics["reconstruction_longitudinal_fwhm_mm"],
    )
    curved_path = _write_curved_comparison(
        output,
        frame_times,
        truth_tube,
        estimate_tube,
        arc_mm,
    )
    summary_path = _write_summary_figure(output, metrics, arrays_path)
    result["assessment"] = {
        "metrics_file": str(metrics_path),
        "curved_profile_comparison": str(curved_path),
        "summary_line_profiles": str(summary_path),
    }
    _write_json_atomic(result_path, result)
    return ReconstructionAssessmentResult(
        output,
        metrics_path,
        curved_path,
        summary_path,
        arrays_path,
        frame_count,
    )
