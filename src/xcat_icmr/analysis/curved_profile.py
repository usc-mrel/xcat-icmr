"""Streaming curved-tube intensity profiles from fully sampled 4-D images."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING

import h5py
import numpy as np
from scipy.io import savemat
from scipy.ndimage import map_coordinates

from xcat_icmr.acquisition.dynamic import plan_dynamic_acquisition
from xcat_icmr.cache import dynamic_acquisition_cache_entry
from xcat_icmr.encoding.tissue_library import _grid
from xcat_icmr.intervention import interpolate_cubic_arc_length, load_balloon_path
from xcat_icmr.sequence import build_coordinate_transforms, read_sequence
from xcat_icmr.signal import read_pulseq_excitation

if TYPE_CHECKING:
    from xcat_icmr.config.models import SimulationConfig


class CurvedLineProfileError(ValueError):
    """Raised when a curved-line profile cannot be generated consistently."""


@dataclass(frozen=True)
class CurvedLineProfileResult:
    source_path: Path
    output_directory: Path
    mat_path: Path
    heatmap_path: Path
    geometry_path: Path
    frame_count: int
    arc_sample_count: int
    curve_length_mm: float
    reused: bool


def map_lps_to_reconstruction_voxels(
    points_lps_mm: np.ndarray,
    *,
    pcs_to_logical: np.ndarray,
    rf_logical_axis: int,
    rf_center_shift_mm: float,
    target_fov_mm: tuple[float, float, float],
    reconstruction_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Map patient-LPS positions into the RF-centered logical image grid."""

    points = np.asarray(points_lps_mm, dtype=np.float64)
    transform = np.asarray(pcs_to_logical, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise CurvedLineProfileError("LPS points must have shape [point, 3]")
    if transform.shape != (3, 3):
        raise CurvedLineProfileError("pcs_to_logical must have shape [3, 3]")
    if rf_logical_axis not in {0, 1, 2}:
        raise CurvedLineProfileError("RF logical axis must be 0, 1, or 2")
    logical_mm = points @ transform.T
    logical_mm[:, rf_logical_axis] -= float(rf_center_shift_mm)
    fov = np.asarray(target_fov_mm, dtype=np.float64)
    shape = np.asarray(reconstruction_shape, dtype=np.int64)
    voxel_mm = fov / shape
    voxel = logical_mm / voxel_mm[None, :] + (shape // 2)[None, :]
    return voxel, logical_mm


def _curve_samples(curve, step_mm: float) -> tuple[np.ndarray, np.ndarray]:
    arc = np.arange(0.0, curve.total_length_mm, step_mm, dtype=np.float64)
    if not len(arc) or arc[-1] < curve.total_length_mm:
        arc = np.concatenate((arc, [curve.total_length_mm]))
    return curve.positions_at_distances_mm(arc), arc


def _normal_frame(curve_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tangent = np.gradient(curve_mm, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    normal_one = np.empty_like(tangent)
    for index, direction in enumerate(tangent):
        reference = np.asarray((0.0, 0.0, 1.0))
        if abs(float(np.dot(direction, reference))) > 0.9:
            reference = np.asarray((0.0, 1.0, 0.0))
        normal = np.cross(direction, reference)
        normal_one[index] = normal / np.linalg.norm(normal)
    normal_two = np.cross(tangent, normal_one)
    normal_two /= np.maximum(
        np.linalg.norm(normal_two, axis=1, keepdims=True), 1e-12
    )
    return normal_one, normal_two


def _disk_offsets(
    radius_mm: float, radial_step_mm: float, angular_samples: int
) -> np.ndarray:
    radii = np.arange(radial_step_mm, radius_mm, radial_step_mm)
    radii = np.concatenate((radii, [radius_mm]))
    offsets = [(0.0, 0.0)]
    for radius in radii:
        for angle in np.linspace(
            0.0, 2.0 * np.pi, angular_samples, endpoint=False
        ):
            offsets.append((radius * np.cos(angle), radius * np.sin(angle)))
    return np.asarray(offsets, dtype=np.float64)


def _sample_frame(
    magnitude: np.ndarray,
    curve_logical_mm: np.ndarray,
    voxel_size_mm: np.ndarray,
    normal_one: np.ndarray,
    normal_two: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    physical = (
        curve_logical_mm[None, :, :]
        + offsets[:, 0, None, None] * normal_one[None, :, :]
        + offsets[:, 1, None, None] * normal_two[None, :, :]
    )
    shape = np.asarray(magnitude.shape, dtype=np.int64)
    voxel = physical / voxel_size_mm[None, None, :] + (shape // 2)[None, None, :]
    values = map_coordinates(
        magnitude,
        voxel.reshape(-1, 3).T,
        order=1,
        mode="constant",
        cval=np.nan,
        prefilter=False,
    )
    return values.reshape(len(offsets), len(curve_logical_mm))


def _path_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_figures(
    tube_max: np.ndarray,
    arc_mm: np.ndarray,
    frame_time_s: np.ndarray,
    temporal_max: np.ndarray,
    curve_voxel: np.ndarray,
    heatmap_path: Path,
    geometry_path: Path,
    velocity_cm_per_s: float,
    frame_duration_s: float,
) -> None:
    mpl_root = Path(tempfile.gettempdir()) / "xcat-icmr-matplotlib"
    mpl_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_root))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    finite = tube_max[np.isfinite(tube_max)]
    if not finite.size:
        raise CurvedLineProfileError("tube profile contains no finite samples")
    # Magnitude images have a physical zero. Keeping zero as the black point
    # preserves the gray tissue baseline used by the legacy comparison plots.
    vmin = 0.0
    vmax = float(np.percentile(finite, 99.5))
    if vmax <= vmin:
        vmax = vmin + 1.0
    time_extent = (
        max(0.0, float(frame_time_s[0]) - frame_duration_s / 2.0),
        float(frame_time_s[-1]) + frame_duration_s / 2.0,
    )
    figure, axis = plt.subplots(figsize=(5.2, 4.0), constrained_layout=True)
    axis.imshow(
        tube_max.T,
        origin="lower",
        aspect="auto",
        extent=(time_extent[0], time_extent[1], arc_mm[0], arc_mm[-1]),
        cmap="gray",
        vmin=float(vmin),
        vmax=float(vmax),
    )
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Arc length (mm)")
    axis.set_title(
        "Ground Truth\n"
        f"{velocity_cm_per_s:g} cm/s | {frame_duration_s * 1e3:g} ms/frame",
        fontsize=10,
    )
    figure.savefig(heatmap_path, dpi=180)
    plt.close(figure)

    midpoint = np.rint(curve_voxel[len(curve_voxel) // 2]).astype(int)
    midpoint = np.clip(midpoint, 0, np.asarray(temporal_max.shape) - 1)
    panels = (
        (temporal_max[midpoint[0], :, :].T, 1, 2, "logical x"),
        (temporal_max[:, midpoint[1], :].T, 0, 2, "logical y"),
        (temporal_max[:, :, midpoint[2]].T, 0, 1, "logical z"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), constrained_layout=True)
    for axis, (image, horizontal, vertical, title) in zip(axes, panels, strict=True):
        display_max = float(np.percentile(image[np.isfinite(image)], 99.5))
        axis.imshow(image, origin="lower", cmap="gray", vmin=0.0, vmax=display_max)
        axis.plot(
            curve_voxel[:, horizontal], curve_voxel[:, vertical],
            color="lime", linewidth=1.0,
        )
        axis.set_title(title)
    figure.savefig(geometry_path, dpi=180)
    plt.close(figure)


def generate_curved_line_profile(
    config: "SimulationConfig",
    *,
    input_path: str | Path | None = None,
    overwrite: bool = False,
) -> CurvedLineProfileResult:
    """Generate a curved-tube maximum profile from a compatible 4-D reference."""

    settings = config.analysis.curved_line_profile
    source = (
        Path(input_path).expanduser().resolve(strict=False)
        if input_path is not None
        else dynamic_acquisition_cache_entry(config).directory
        / "fullysampled_tissue_gd_reference_4d.h5"
    )
    if not source.is_file():
        raise CurvedLineProfileError(f"fully sampled input does not exist: {source}")
    balloon_path = config.intervention.gd_balloon.path.control_points_file
    if balloon_path is None:
        raise CurvedLineProfileError("the configuration has no balloon path")
    loaded = load_balloon_path(
        balloon_path,
        coordinate_system=config.intervention.gd_balloon.path.coordinate_system,
    )
    curve = interpolate_cubic_arc_length(loaded.control_points_lps_mm)
    curve_lps, arc_mm = _curve_samples(curve, settings.sample_step_mm)
    sequence = read_sequence(config.sequence)
    transforms = build_coordinate_transforms(
        patient_position=config.phantom.patient_position,
        coordinate_mode=config.sequence.coordinate_mode,
        sequence_orientation=config.sequence.orientation,
    )
    _, reconstruction_shape, _, _ = _grid(config, sequence, transforms)
    excitation = read_pulseq_excitation(sequence.sequence_path)
    curve_voxel, curve_logical_mm = map_lps_to_reconstruction_voxels(
        curve_lps,
        pcs_to_logical=transforms.pcs_to_logical,
        rf_logical_axis=excitation.logical_axis,
        rf_center_shift_mm=config.sequence.rf_profile.center_shift_mm,
        target_fov_mm=config.encoding.target_fov_mm,
        reconstruction_shape=reconstruction_shape,
    )
    in_bounds = np.all(
        (curve_voxel >= 0.0)
        & (curve_voxel <= np.asarray(reconstruction_shape)[None, :] - 1.0),
        axis=1,
    )
    if not np.all(in_bounds):
        raise CurvedLineProfileError(
            f"{np.count_nonzero(~in_bounds)} interpolated path samples leave "
            "the reconstructed FOV"
        )
    plan = plan_dynamic_acquisition(config, check_free_space=False)
    frame_times = np.asarray(
        [
            np.mean(
                plan.schedule.time_s[
                    frame * plan.schedule.trs_per_frame :
                    (frame + 1) * plan.schedule.trs_per_frame
                ]
            )
            for frame in range(plan.schedule.frame_count)
        ],
        dtype=np.float64,
    )
    output_dir = source.parent / "analysis" / "curved_line_profile"
    mat_path = output_dir / "curved_line_profile.mat"
    heatmap_path = output_dir / "tube_max_time_distance.png"
    geometry_path = output_dir / "geometry_overlay.png"
    metadata_path = output_dir / "metadata.json"
    source_stat = source.stat()
    fingerprint_content = {
        "source": str(source),
        "source_size_bytes": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "control_points_sha256": _path_sha256(loaded.source_path),
        "sample_step_mm": settings.sample_step_mm,
        "tube_radius_mm": settings.tube_radius_mm,
        "angular_samples": settings.angular_samples,
        "pcs_to_logical": transforms.pcs_to_logical.tolist(),
        "rf_logical_axis": excitation.logical_axis,
        "rf_center_shift_mm": config.sequence.rf_profile.center_shift_mm,
        "target_fov_mm": list(config.encoding.target_fov_mm),
        "reconstruction_shape": list(reconstruction_shape),
        "velocity_cm_per_s": config.intervention.gd_balloon.movement.velocity_cm_per_s,
        "frame_duration_s": config.acquisition.frame_duration_s,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_content, sort_keys=True).encode("utf-8")
    ).hexdigest()
    outputs = (mat_path, heatmap_path, geometry_path, metadata_path)
    if all(path.is_file() for path in outputs) and not overwrite:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("fingerprint") == fingerprint:
                return CurvedLineProfileResult(
                    source, output_dir, mat_path, heatmap_path, geometry_path,
                    plan.schedule.frame_count, len(arc_mm), curve.total_length_mm,
                    True,
                )
        except (OSError, json.JSONDecodeError):
            pass
        raise CurvedLineProfileError(
            "curved-profile inputs changed; pass --overwrite"
        )
    if any(path.exists() for path in outputs) and not overwrite:
        raise CurvedLineProfileError(
            "curved-profile output is incomplete; pass --overwrite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    voxel_size = np.asarray(config.encoding.target_fov_mm) / np.asarray(
        reconstruction_shape
    )
    normal_one, normal_two = _normal_frame(curve_logical_mm)
    offsets = _disk_offsets(
        settings.tube_radius_mm,
        float(np.min(voxel_size)),
        settings.angular_samples,
    )
    tube_max = np.empty((plan.schedule.frame_count, len(arc_mm)), dtype=np.float32)
    tube_mean = np.empty_like(tube_max)
    centerline = np.empty_like(tube_max)
    temporal_max = np.zeros(reconstruction_shape, dtype=np.float32)
    with h5py.File(source, "r") as handle:
        image = handle.get("image")
        complete = handle.get("frame_complete")
        if (
            not isinstance(image, h5py.Dataset)
            or image.shape != reconstruction_shape + (plan.schedule.frame_count,)
            or not isinstance(complete, h5py.Dataset)
            or not np.all(complete[:])
        ):
            raise CurvedLineProfileError(
                "fully sampled input is incomplete or has an incompatible shape"
            )
        for frame in range(plan.schedule.frame_count):
            magnitude = np.abs(np.asarray(image[..., frame])).astype(
                np.float32, copy=False
            )
            np.maximum(temporal_max, magnitude, out=temporal_max)
            sampled = _sample_frame(
                magnitude,
                curve_logical_mm,
                voxel_size,
                normal_one,
                normal_two,
                offsets,
            )
            centerline[frame] = sampled[0]
            tube_max[frame] = np.nanmax(sampled, axis=0)
            tube_mean[frame] = np.nanmean(sampled, axis=0)
    arrays = {
        "centerline": centerline,
        "tube_max": tube_max,
        "tube_mean": tube_mean,
        "arc_length_mm": np.asarray(arc_mm, dtype=np.float32),
        "frame_center_time_s": np.asarray(frame_times, dtype=np.float32),
        "curve_lps_mm": np.asarray(curve_lps, dtype=np.float32),
        "curve_logical_mm": np.asarray(curve_logical_mm, dtype=np.float32),
        "curve_logical_voxel": np.asarray(curve_voxel, dtype=np.float32),
        "control_points_lps_mm": np.asarray(
            loaded.control_points_lps_mm, dtype=np.float32
        ),
    }
    savemat(mat_path, arrays, appendmat=False, do_compression=True)
    _write_figures(
        tube_max, arc_mm, frame_times, temporal_max, curve_voxel,
        heatmap_path, geometry_path,
        config.intervention.gd_balloon.movement.velocity_cm_per_s,
        config.acquisition.frame_duration_s,
    )
    metadata = dict(fingerprint_content)
    metadata.update(
        {
            "fingerprint": fingerprint,
            "axis_order": "time,arc_length",
            "analysis_values": "magnitude",
            "curve_length_mm": curve.total_length_mm,
            "arc_sample_count": len(arc_mm),
            "frame_count": plan.schedule.frame_count,
            "frame_center_time_s": frame_times.tolist(),
        }
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return CurvedLineProfileResult(
        source, output_dir, mat_path, heatmap_path, geometry_path,
        plan.schedule.frame_count, len(arc_mm), curve.total_length_mm, False,
    )


def format_curved_line_profile(result: CurvedLineProfileResult) -> str:
    return "\n".join(
        (
            "Curved-line fully sampled intensity profile",
            f"Frames:             {result.frame_count}",
            f"Arc samples:        {result.arc_sample_count}",
            f"Curve length:       {result.curve_length_mm:.3f} mm",
            f"Generated/reused:   {0 if result.reused else 1}/{1 if result.reused else 0}",
            f"Numerical output:   {result.mat_path}",
            f"Time-distance map:  {result.heatmap_path}",
            f"Geometry overlay:   {result.geometry_path}",
        )
    )
