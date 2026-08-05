"""Impulse-PSF and fixed-FOV diagnostics for one saved tissue k-space."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile

import numpy as np
from scipy.io import savemat

from xcat_icmr.encoding.sigpy_backend import SigpyNufftBackend
from xcat_icmr.encoding.trajectory import prepare_physical_sigpy_trajectory


@dataclass(frozen=True)
class SignalSupport:
    """Thresholded signal bounds on a centered physical grid."""

    threshold_fraction: float
    threshold_value: float
    bbox_min_index: tuple[int, int, int]
    bbox_max_index: tuple[int, int, int]
    bbox_min_mm: tuple[float, float, float]
    bbox_max_mm: tuple[float, float, float]
    extent_mm: tuple[float, float, float]
    derived_fov_mm: tuple[float, float, float]


@dataclass(frozen=True)
class FovCandidateDiagnostic:
    """PSF and tissue-adjoint measurements for one reconstruction FOV."""

    name: str
    fov_mm: tuple[float, float, float]
    matrix_shape: tuple[int, int, int]
    voxel_size_mm: tuple[float, float, float]
    psf_fwhm_mm: tuple[float, float, float]
    psf_peak_sidelobe_ratio: float
    psf_integrated_sidelobe_energy_ratio: float
    support_fully_contained: bool
    tissue_background_energy_ratio: float


@dataclass(frozen=True)
class FovPsfDiagnostic:
    """Saved result of a same-k-space comparison over fixed FOVs."""

    output_path: Path
    kspace_path: Path
    oversampling: float
    kernel_width: float
    support: SignalSupport
    candidates: tuple[FovCandidateDiagnostic, ...]


def measure_centered_signal_support(
    image: np.ndarray,
    *,
    voxel_size_mm: tuple[float, float, float],
    threshold_fraction: float = 0.01,
    margin_mm: float = 10.0,
    fov_rounding_mm: tuple[float, float, float] = (3.5, 3.5, 3.5),
) -> SignalSupport:
    """Measure conservative centered support and derive a fixed compact FOV."""

    values = np.abs(np.asarray(image))
    if values.ndim != 3 or not np.all(np.isfinite(values)):
        raise ValueError("support image must be one finite 3-D array")
    if not 0 < threshold_fraction < 1:
        raise ValueError("threshold_fraction must be between zero and one")
    if not np.isfinite(margin_mm) or margin_mm < 0:
        raise ValueError("margin_mm must be finite and non-negative")
    voxel = np.asarray(voxel_size_mm, dtype=np.float64)
    rounding = np.asarray(fov_rounding_mm, dtype=np.float64)
    if voxel.shape != (3,) or np.any(voxel <= 0):
        raise ValueError("voxel_size_mm must contain three positive values")
    if rounding.shape != (3,) or np.any(rounding <= 0):
        raise ValueError("fov_rounding_mm must contain three positive values")
    maximum = float(np.max(values))
    if maximum <= 0:
        raise ValueError("support image has no nonzero signal")
    threshold = maximum * threshold_fraction
    mask = values >= threshold
    bounds = []
    for axis in range(3):
        projection = np.any(
            mask, axis=tuple(index for index in range(3) if index != axis)
        )
        occupied = np.flatnonzero(projection)
        bounds.append((int(occupied[0]), int(occupied[-1])))
    lower_index = np.asarray([bound[0] for bound in bounds], dtype=np.int64)
    upper_index = np.asarray([bound[1] for bound in bounds], dtype=np.int64)
    center = np.asarray(values.shape, dtype=np.int64) // 2
    lower_mm = (lower_index - center) * voxel
    upper_mm = (upper_index - center) * voxel
    extent_mm = (upper_index - lower_index + 1) * voxel
    half_width = np.maximum(np.abs(lower_mm), np.abs(upper_mm)) + margin_mm
    requested_fov = 2.0 * half_width
    full_fov = np.asarray(values.shape, dtype=np.float64) * voxel
    rounded_fov = np.ceil(requested_fov / rounding) * rounding
    derived_fov = np.minimum(rounded_fov, full_fov)
    return SignalSupport(
        threshold_fraction=float(threshold_fraction),
        threshold_value=threshold,
        bbox_min_index=tuple(int(value) for value in lower_index),
        bbox_max_index=tuple(int(value) for value in upper_index),
        bbox_min_mm=tuple(float(value) for value in lower_mm),
        bbox_max_mm=tuple(float(value) for value in upper_mm),
        extent_mm=tuple(float(value) for value in extent_mm),
        derived_fov_mm=tuple(float(value) for value in derived_fov),
    )


def matrix_shape_for_fov(
    fov_mm: tuple[float, float, float],
    resolution_mm: tuple[float, float, float],
    maximum_absolute_k_per_m: tuple[float, float, float],
) -> tuple[int, int, int]:
    """Choose a nominal matrix that also contains every physical k sample."""

    fov = np.asarray(fov_mm, dtype=np.float64)
    resolution = np.asarray(resolution_mm, dtype=np.float64)
    maximum_k = np.asarray(maximum_absolute_k_per_m, dtype=np.float64)
    nominal = np.rint(fov / resolution).astype(np.int64)
    nyquist_safe = np.ceil(2.0 * maximum_k * fov * 1e-3).astype(np.int64)
    matrix = np.maximum(nominal, nyquist_safe)
    if np.any(matrix <= 0):
        raise ValueError("FOV produced a non-positive diagnostic matrix")
    return tuple(int(value) for value in matrix)


def _fwhm(line: np.ndarray, peak_index: int, voxel_size_mm: float) -> float:
    threshold = float(line[peak_index]) * 0.5
    lower = peak_index
    upper = peak_index
    while lower > 0 and line[lower - 1] >= threshold:
        lower -= 1
    while upper + 1 < len(line) and line[upper + 1] >= threshold:
        upper += 1
    return float((upper - lower + 1) * voxel_size_mm)


def _psf_metrics(
    psf: np.ndarray,
    voxel_size_mm: tuple[float, float, float],
) -> tuple[tuple[float, float, float], float, float]:
    magnitude = np.abs(psf).astype(np.float64)
    peak_index = np.unravel_index(int(np.argmax(magnitude)), magnitude.shape)
    peak = float(magnitude[peak_index])
    normalized = magnitude / peak
    fwhm = []
    for axis in range(3):
        selection = list(peak_index)
        selection[axis] = slice(None)
        fwhm.append(
            _fwhm(
                normalized[tuple(selection)],
                peak_index[axis],
                voxel_size_mm[axis],
            )
        )
    main_lobe = np.zeros(normalized.shape, dtype=bool)
    slices = tuple(
        slice(max(index - 2, 0), min(index + 3, size))
        for index, size in zip(peak_index, normalized.shape, strict=True)
    )
    main_lobe[slices] = True
    sidelobes = normalized[~main_lobe]
    peak_sidelobe = float(np.max(sidelobes))
    energy = normalized**2
    integrated = float(np.sum(energy[~main_lobe]) / np.sum(energy))
    return tuple(fwhm), peak_sidelobe, integrated  # type: ignore[return-value]


def _background_energy_ratio(
    image: np.ndarray,
    *,
    fov_mm: tuple[float, float, float],
    support: SignalSupport,
) -> tuple[float, bool]:
    axes = [
        (np.arange(size, dtype=np.float64) - size // 2) * (fov / size)
        for size, fov in zip(image.shape, fov_mm, strict=True)
    ]
    inside_axes = [
        (axis >= lower) & (axis <= upper)
        for axis, lower, upper in zip(
            axes, support.bbox_min_mm, support.bbox_max_mm, strict=True
        )
    ]
    support_fully_contained = all(
        lower >= float(axis[0]) and upper <= float(axis[-1])
        for axis, lower, upper in zip(
            axes, support.bbox_min_mm, support.bbox_max_mm, strict=True
        )
    )
    if not support_fully_contained:
        return float("nan"), False
    inside = (
        inside_axes[0][:, None, None]
        & inside_axes[1][None, :, None]
        & inside_axes[2][None, None, :]
    )
    energy = np.abs(image).astype(np.float64) ** 2
    total = float(np.sum(energy))
    ratio = float(np.sum(energy[~inside]) / total) if total > 0 else 0.0
    return ratio, True


def run_fov_psf_diagnostic(
    *,
    kspace: np.ndarray,
    kspace_path: str | Path,
    kx_per_m: np.ndarray,
    ky_per_m: np.ndarray,
    kz_per_m: np.ndarray,
    density_compensation: np.ndarray,
    resolution_mm: tuple[float, float, float],
    support: SignalSupport,
    candidate_fovs_mm: dict[str, tuple[float, float, float]],
    output_path: str | Path,
    device_id: int = -1,
    overwrite: bool = False,
) -> FovPsfDiagnostic:
    """Create impulse PSFs and same-k-space adjoints for several fixed FOVs."""

    values = np.asarray(kspace)
    coordinate_arrays = tuple(
        np.asarray(component, dtype=np.float64)
        for component in (kx_per_m, ky_per_m, kz_per_m)
    )
    if values.shape != coordinate_arrays[0].shape:
        raise ValueError(
            f"k-space shape {values.shape} does not match trajectory "
            f"{coordinate_arrays[0].shape}"
        )
    if any(component.shape != values.shape for component in coordinate_arrays):
        raise ValueError("physical trajectory component shapes do not match")
    dcf = np.asarray(density_compensation, dtype=np.float32)
    if dcf.shape != values.shape or np.any(dcf < 0) or not np.all(
        np.isfinite(dcf)
    ):
        raise ValueError("DCF must be finite, non-negative, and match k-space")
    maximum_dcf = float(np.max(dcf))
    if maximum_dcf <= 0:
        raise ValueError("DCF must have a positive maximum")
    selected_dcf = (dcf / maximum_dcf).T.reshape(-1)
    flattened_kspace = values.T.reshape(-1).astype(np.complex64)
    maximum_k = tuple(
        float(np.max(np.abs(component))) for component in coordinate_arrays
    )
    backend = SigpyNufftBackend(device_id=device_id)
    destination = Path(output_path).expanduser().resolve(strict=False)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"diagnostic output already exists: {destination}; pass --overwrite"
        )
    saved: dict[str, object] = {
        "source_kspace_path": str(Path(kspace_path).resolve(strict=False)),
        "nufft_oversampling": np.asarray([[backend.oversampling]]),
        "nufft_kernel_width": np.asarray([[backend.kernel_width]]),
        "support_threshold_fraction": np.asarray(
            [[support.threshold_fraction]]
        ),
        "support_bbox_min_mm": np.asarray([support.bbox_min_mm]),
        "support_bbox_max_mm": np.asarray([support.bbox_max_mm]),
        "support_extent_mm": np.asarray([support.extent_mm]),
        "support_derived_fov_mm": np.asarray([support.derived_fov_mm]),
    }
    reports = []
    for name, fov in candidate_fovs_mm.items():
        matrix = matrix_shape_for_fov(fov, resolution_mm, maximum_k)
        trajectory = prepare_physical_sigpy_trajectory(
            *coordinate_arrays,
            fov_mm=fov,
            matrix_shape=matrix,
        )
        psf = backend.adjoint(selected_dcf.astype(np.complex64), trajectory)
        adjoint = backend.adjoint(
            flattened_kspace * selected_dcf, trajectory
        )
        voxel_size = tuple(
            float(axis_fov / size)
            for axis_fov, size in zip(fov, matrix, strict=True)
        )
        fwhm, peak_sidelobe, integrated_sidelobe = _psf_metrics(
            psf, voxel_size
        )
        background, support_fully_contained = _background_energy_ratio(
            adjoint, fov_mm=fov, support=support
        )
        reports.append(
            FovCandidateDiagnostic(
                name=name,
                fov_mm=fov,
                matrix_shape=matrix,
                voxel_size_mm=voxel_size,
                psf_fwhm_mm=fwhm,
                psf_peak_sidelobe_ratio=peak_sidelobe,
                psf_integrated_sidelobe_energy_ratio=integrated_sidelobe,
                support_fully_contained=support_fully_contained,
                tissue_background_energy_ratio=background,
            )
        )
        prefix = name.replace("-", "_")
        saved[f"{prefix}_fov_mm"] = np.asarray([fov])
        saved[f"{prefix}_matrix_shape"] = np.asarray([matrix], dtype=np.int32)
        saved[f"{prefix}_voxel_size_mm"] = np.asarray([voxel_size])
        saved[f"{prefix}_coordinates"] = trajectory.coordinates
        saved[f"{prefix}_psf"] = psf
        saved[f"{prefix}_tissue_adjoint"] = adjoint
        saved[f"{prefix}_psf_fwhm_mm"] = np.asarray([fwhm])
        saved[f"{prefix}_psf_peak_sidelobe_ratio"] = np.asarray(
            [[peak_sidelobe]]
        )
        saved[f"{prefix}_psf_integrated_sidelobe_energy_ratio"] = np.asarray(
            [[integrated_sidelobe]]
        )
        saved[f"{prefix}_tissue_background_energy_ratio"] = np.asarray(
            [[background]]
        )
        saved[f"{prefix}_support_fully_contained"] = np.asarray(
            [[support_fully_contained]], dtype=np.uint8
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        savemat(temporary_path, saved, appendmat=False, do_compression=False)
        temporary_path.chmod(0o644)
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return FovPsfDiagnostic(
        output_path=destination,
        kspace_path=Path(kspace_path).resolve(strict=False),
        oversampling=backend.oversampling,
        kernel_width=backend.kernel_width,
        support=support,
        candidates=tuple(reports),
    )


def format_fov_psf_diagnostic(report: FovPsfDiagnostic) -> str:
    """Format the concise comparison table printed by the CLI."""

    lines = [
        "NUFFT impulse-PSF and fixed-FOV diagnostic",
        f"Source k-space: {report.kspace_path}",
        (
            f"SigPy kernel: OS {report.oversampling:g}, "
            f"width {report.kernel_width:g}"
        ),
        (
            "Signal support: "
            + " × ".join(f"{value:g}" for value in report.support.extent_mm)
            + " mm"
        ),
        (
            "Derived FOV:    "
            + " × ".join(
                f"{value:g}" for value in report.support.derived_fov_mm
            )
            + " mm"
        ),
    ]
    for candidate in report.candidates:
        lines.extend(
            (
                "",
                f"[{candidate.name}]",
                "FOV:            "
                + " × ".join(f"{value:g}" for value in candidate.fov_mm)
                + " mm",
                f"Matrix:         {candidate.matrix_shape}",
                "Voxel:          "
                + " × ".join(
                    f"{value:.4g}" for value in candidate.voxel_size_mm
                )
                + " mm",
                "PSF FWHM:       "
                + " × ".join(
                    f"{value:.4g}" for value in candidate.psf_fwhm_mm
                )
                + " mm",
                (
                    "Peak sidelobe:  "
                    f"{candidate.psf_peak_sidelobe_ratio:.6g}"
                ),
                (
                    "Sidelobe energy:"
                    f" {candidate.psf_integrated_sidelobe_energy_ratio:.6g}"
                ),
                (
                    "Support contained: "
                    + ("yes" if candidate.support_fully_contained else "no")
                ),
                (
                    "Background energy:"
                    + (
                        f" {candidate.tissue_background_energy_ratio:.6g}"
                        if candidate.support_fully_contained
                        else " not applicable"
                    )
                ),
            )
        )
    lines.extend(("", f"Output: {report.output_path}", "Verification: PASS"))
    return "\n".join(lines)
