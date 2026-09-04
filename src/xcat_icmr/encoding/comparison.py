"""Numerical and image-space checks for saved NUFFT reference products."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile

import numpy as np
from scipy.io import loadmat, savemat
from scipy.ndimage import zoom


@dataclass(frozen=True)
class DeviceParity:
    output_path: Path
    kspace_relative_l2_error: float
    kspace_maximum_absolute_error: float
    adjoint_relative_l2_error: float
    adjoint_maximum_absolute_error: float
    cpu_elapsed_s: float
    gpu_elapsed_s: float
    speedup: float
    passed: bool


@dataclass(frozen=True)
class ImageReferenceValidation:
    output_path: Path
    correlation: float
    center_offset_voxels: tuple[float, float, float]
    center_offset_mm: tuple[float, float, float]
    gt_boundary_energy_ratio: float
    adjoint_boundary_energy_ratio: float
    intended_orientation_is_best: bool


def _relative_l2(reference: np.ndarray, candidate: np.ndarray) -> float:
    difference = np.asarray(candidate, dtype=np.complex128) - np.asarray(
        reference, dtype=np.complex128
    )
    denominator = float(np.linalg.norm(reference.reshape(-1)))
    return float(np.linalg.norm(difference.reshape(-1)) / max(denominator, 1e-12))


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
        savemat(temporary_path, variables, appendmat=False, do_compression=False)
        temporary_path.chmod(0o644)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def compare_device_references(
    cpu_path: str | Path,
    gpu_path: str | Path,
    output_path: str | Path,
    *,
    relative_tolerance: float = 5e-4,
) -> DeviceParity:
    """Compare saved CPU/GPU k-space and adjoints from identical inputs."""

    cpu_source = Path(cpu_path).expanduser().resolve(strict=True)
    gpu_source = Path(gpu_path).expanduser().resolve(strict=True)
    variables = ["kspace", "adjoint", "elapsed_s"]
    cpu = loadmat(cpu_source, variable_names=variables)
    gpu = loadmat(gpu_source, variable_names=variables)
    for name in ("kspace", "adjoint"):
        if name not in cpu or name not in gpu:
            raise ValueError(f"CPU/GPU reference is missing variable {name!r}")
        if cpu[name].shape != gpu[name].shape:
            raise ValueError(
                f"CPU/GPU {name} shapes differ: {cpu[name].shape} != {gpu[name].shape}"
            )
    kspace_relative = _relative_l2(cpu["kspace"], gpu["kspace"])
    adjoint_relative = _relative_l2(cpu["adjoint"], gpu["adjoint"])
    kspace_maximum = float(np.max(np.abs(cpu["kspace"] - gpu["kspace"])))
    adjoint_maximum = float(np.max(np.abs(cpu["adjoint"] - gpu["adjoint"])))
    cpu_elapsed = float(np.asarray(cpu.get("elapsed_s", [[np.nan]])).item())
    gpu_elapsed = float(np.asarray(gpu.get("elapsed_s", [[np.nan]])).item())
    speedup = cpu_elapsed / gpu_elapsed if gpu_elapsed > 0 else float("nan")
    passed = (
        kspace_relative <= relative_tolerance
        and adjoint_relative <= relative_tolerance
    )
    destination = Path(output_path).expanduser().resolve(strict=False)
    _atomic_savemat(
        destination,
        {
            "cpu_path": str(cpu_source),
            "gpu_path": str(gpu_source),
            "kspace_relative_l2_error": np.asarray([[kspace_relative]]),
            "kspace_maximum_absolute_error": np.asarray([[kspace_maximum]]),
            "adjoint_relative_l2_error": np.asarray([[adjoint_relative]]),
            "adjoint_maximum_absolute_error": np.asarray([[adjoint_maximum]]),
            "cpu_elapsed_s": np.asarray([[cpu_elapsed]]),
            "gpu_elapsed_s": np.asarray([[gpu_elapsed]]),
            "speedup": np.asarray([[speedup]]),
            "relative_tolerance": np.asarray([[relative_tolerance]]),
            "passed": np.asarray([[passed]], dtype=np.uint8),
        },
    )
    return DeviceParity(
        destination,
        kspace_relative,
        kspace_maximum,
        adjoint_relative,
        adjoint_maximum,
        cpu_elapsed,
        gpu_elapsed,
        speedup,
        passed,
    )


def _weighted_center(values: np.ndarray) -> np.ndarray:
    weights = np.abs(np.asarray(values, dtype=np.float64))
    total = float(np.sum(weights))
    if total <= 0:
        raise ValueError("cannot measure the center of an empty image")
    return np.asarray(
        [
            np.sum(
                weights
                * np.arange(size, dtype=np.float64).reshape(
                    (1,) * axis + (size,) + (1,) * (2 - axis)
                )
            )
            / total
            for axis, size in enumerate(weights.shape)
        ]
    )


def _boundary_energy(values: np.ndarray, width: int = 5) -> float:
    energy = np.abs(values).astype(np.float64) ** 2
    boundary = np.zeros(energy.shape, dtype=bool)
    for axis in range(3):
        lower = [slice(None)] * 3
        upper = [slice(None)] * 3
        lower[axis] = slice(0, width)
        upper[axis] = slice(-width, None)
        boundary[tuple(lower)] = True
        boundary[tuple(upper)] = True
    total = float(np.sum(energy))
    return float(np.sum(energy[boundary]) / total) if total > 0 else 0.0


def validate_image_reference(
    shifted_gt_path: str | Path,
    multicoil_path: str | Path,
    output_path: str | Path,
) -> ImageReferenceValidation:
    """Compare the shifted high-resolution GT with the all-coil RSS adjoint."""

    gt_source = Path(shifted_gt_path).expanduser().resolve(strict=True)
    reference_source = Path(multicoil_path).expanduser().resolve(strict=True)
    gt_content = loadmat(gt_source, variable_names=["image"])
    reference_content = loadmat(
        reference_source, variable_names=["adjoint_rss", "fov_mm"]
    )
    if "image" not in gt_content or "adjoint_rss" not in reference_content:
        raise ValueError("reference files are missing image or adjoint_rss")
    gt = np.asarray(gt_content["image"], dtype=np.float32)
    adjoint = np.asarray(reference_content["adjoint_rss"], dtype=np.float32)
    factors = tuple(target / source for source, target in zip(gt.shape, adjoint.shape))
    gt_resampled = zoom(
        gt,
        factors,
        order=1,
        mode="grid-constant",
        cval=0.0,
        prefilter=False,
        grid_mode=True,
    ).astype(np.float32)
    if gt_resampled.shape != adjoint.shape:
        raise ValueError(
            f"resampled GT shape {gt_resampled.shape} != adjoint {adjoint.shape}"
        )
    gt_norm = gt_resampled / max(float(np.max(gt_resampled)), 1e-12)
    adjoint_norm = adjoint / max(float(np.max(adjoint)), 1e-12)
    mask = gt_norm >= 0.01
    correlation = float(np.corrcoef(gt_norm[mask], adjoint_norm[mask])[0, 1])
    center_offset = _weighted_center(adjoint_norm) - _weighted_center(gt_norm)
    fov = np.asarray(reference_content.get("fov_mm", [[500, 500, 500]])).reshape(-1)
    voxel = fov / np.asarray(adjoint.shape)
    center_offset_mm = center_offset * voxel
    intended_score = float(np.vdot(gt_norm, adjoint_norm).real)
    flip_scores = [
        float(np.vdot(np.flip(gt_norm, axis=axis), adjoint_norm).real)
        for axis in range(3)
    ]
    intended_is_best = intended_score >= max(flip_scores)
    gt_boundary = _boundary_energy(gt_norm)
    adjoint_boundary = _boundary_energy(adjoint_norm)
    destination = Path(output_path).expanduser().resolve(strict=False)
    _atomic_savemat(
        destination,
        {
            "shifted_gt_path": str(gt_source),
            "multicoil_path": str(reference_source),
            "gt_resampled": gt_resampled,
            "adjoint_rss_normalized": adjoint_norm,
            "correlation_in_gt_support": np.asarray([[correlation]]),
            "center_offset_voxels": center_offset.reshape(1, 3),
            "center_offset_mm": center_offset_mm.reshape(1, 3),
            "gt_boundary_energy_ratio": np.asarray([[gt_boundary]]),
            "adjoint_boundary_energy_ratio": np.asarray([[adjoint_boundary]]),
            "orientation_intended_score": np.asarray([[intended_score]]),
            "orientation_flip_scores": np.asarray([flip_scores]),
            "intended_orientation_is_best": np.asarray(
                [[intended_is_best]], dtype=np.uint8
            ),
        },
    )
    return ImageReferenceValidation(
        destination,
        correlation,
        tuple(float(value) for value in center_offset),
        tuple(float(value) for value in center_offset_mm),
        gt_boundary,
        adjoint_boundary,
        intended_is_best,
    )
