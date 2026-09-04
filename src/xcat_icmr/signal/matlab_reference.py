"""Chunked bSSFP image comparison against MATLAB-generated volumes."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import h5py
import numpy as np

from xcat_icmr.signal.bssfp import bssfp_signal_from_tissue_properties
from xcat_icmr.tissue import (
    TissueLibrary,
    XcatLabel,
    map_labels_to_tissue_properties,
)


class MatlabSignalReferenceError(Exception):
    """Raised when MATLAB label and contrast files cannot be compared."""


@dataclass(frozen=True)
class TissueSignalComparison:
    """Accumulated error metrics for one MATLAB tissue group."""

    tissue: str
    voxel_count: int
    mismatch_count: int
    max_abs_error: float
    mean_abs_error: float
    rmse: float
    max_rel_error: float


@dataclass(frozen=True)
class BssfpMatlabComparison:
    """Global and per-tissue comparison against a MATLAB contrast image."""

    label_path: Path
    image_path: Path
    volume_shape: tuple[int, ...]
    flip_angle_deg: float
    te_ms: float
    tr_ms: float
    voxel_count: int
    mismatch_count: int
    max_abs_error: float
    mean_abs_error: float
    rmse: float
    max_rel_error: float
    atol: float
    rtol: float
    tissues: tuple[TissueSignalComparison, ...]

    @property
    def passed(self) -> bool:
        return self.mismatch_count == 0


@dataclass
class _Accumulator:
    voxel_count: int = 0
    mismatch_count: int = 0
    max_abs_error: float = 0.0
    sum_abs_error: float = 0.0
    sum_squared_error: float = 0.0
    max_rel_error: float = 0.0

    def update(
        self,
        difference: np.ndarray,
        reference: np.ndarray,
        mismatched: np.ndarray,
    ) -> None:
        absolute = np.abs(difference, dtype=np.float64)
        count = absolute.size
        self.voxel_count += count
        self.mismatch_count += int(np.count_nonzero(mismatched))
        if count == 0:
            return

        self.max_abs_error = max(
            self.max_abs_error, float(np.max(absolute))
        )
        self.sum_abs_error += float(np.sum(absolute, dtype=np.float64))
        self.sum_squared_error += float(
            np.sum(absolute * absolute, dtype=np.float64)
        )

        nonzero = np.abs(reference) > np.finfo(np.float64).tiny
        if np.any(nonzero):
            relative = absolute[nonzero] / np.abs(reference[nonzero])
            self.max_rel_error = max(
                self.max_rel_error, float(np.max(relative))
            )

    def metrics(self) -> tuple[float, float, float, float]:
        if self.voxel_count == 0:
            return 0.0, 0.0, 0.0, 0.0
        mean_abs = self.sum_abs_error / self.voxel_count
        rmse = math.sqrt(self.sum_squared_error / self.voxel_count)
        return self.max_abs_error, mean_abs, rmse, self.max_rel_error


def _open_dataset(
    handle: h5py.File,
    dataset_name: str,
    path: Path,
) -> h5py.Dataset:
    if dataset_name not in handle:
        raise MatlabSignalReferenceError(
            f"MATLAB file is missing dataset {dataset_name!r}: {path}"
        )
    dataset = handle[dataset_name]
    if not isinstance(dataset, h5py.Dataset):
        raise MatlabSignalReferenceError(
            f"MATLAB object {dataset_name!r} is not a dataset: {path}"
        )
    if dataset.ndim == 0:
        raise MatlabSignalReferenceError(
            f"MATLAB dataset {dataset_name!r} must be an array: {path}"
        )
    return dataset


def _tissue_name(label_value: int, library: TissueLibrary) -> str:
    label = XcatLabel(label_value)
    group = library.group_for_label(label)
    return group.name if group is not None else "Unassigned"


def compare_bssfp_to_matlab(
    label_path: str | Path,
    image_path: str | Path,
    library: TissueLibrary,
    *,
    flip_angle_deg: float,
    te_ms: float,
    tr_ms: float,
    off_resonance_enabled: bool = False,
    label_dataset: str = "P",
    image_dataset: str = "image",
    chunk_slices: int = 8,
    atol: float = 1e-5,
    rtol: float = 1e-6,
) -> BssfpMatlabComparison:
    """Compare Python bSSFP contrast with paired MATLAB v7.3 volumes."""

    labels_file = Path(label_path).expanduser().resolve(strict=False)
    image_file = Path(image_path).expanduser().resolve(strict=False)
    for description, path in (
        ("label", labels_file),
        ("contrast image", image_file),
    ):
        if not path.is_file():
            raise MatlabSignalReferenceError(
                f"MATLAB {description} file does not exist: {path}"
            )
    if chunk_slices <= 0:
        raise ValueError("chunk_slices must be positive")
    if atol < 0 or rtol < 0:
        raise ValueError("atol and rtol must be non-negative")

    try:
        labels_handle = h5py.File(labels_file, "r")
        image_handle = h5py.File(image_file, "r")
    except OSError as exc:
        raise MatlabSignalReferenceError(
            f"could not open MATLAB v7.3 reference: {exc}"
        ) from exc

    global_stats = _Accumulator()
    tissue_stats: dict[str, _Accumulator] = {}
    with labels_handle, image_handle:
        labels = _open_dataset(labels_handle, label_dataset, labels_file)
        reference = _open_dataset(image_handle, image_dataset, image_file)
        if labels.shape != reference.shape:
            raise MatlabSignalReferenceError(
                "MATLAB label and image shapes differ: "
                f"{labels.shape} != {reference.shape}"
            )

        for start in range(0, labels.shape[0], chunk_slices):
            stop = min(start + chunk_slices, labels.shape[0])
            selection = (slice(start, stop),) + (slice(None),) * (
                labels.ndim - 1
            )
            label_chunk = np.asarray(labels[selection])
            reference_chunk = np.asarray(reference[selection], dtype=np.float64)
            properties = map_labels_to_tissue_properties(label_chunk, library)
            python_chunk = bssfp_signal_from_tissue_properties(
                properties,
                flip_angle_deg=flip_angle_deg,
                te_ms=te_ms,
                tr_ms=tr_ms,
                off_resonance_enabled=off_resonance_enabled,
                dtype=np.float32,
            ).astype(np.float64, copy=False)

            if not np.all(np.isfinite(reference_chunk)):
                raise MatlabSignalReferenceError(
                    "MATLAB contrast image contains non-finite values"
                )
            difference = python_chunk - reference_chunk
            mismatched = ~np.isclose(
                python_chunk,
                reference_chunk,
                atol=atol,
                rtol=rtol,
            )
            global_stats.update(difference, reference_chunk, mismatched)

            integer_labels = label_chunk.astype(np.int64, copy=False)
            for label_value in np.unique(integer_labels):
                tissue = _tissue_name(int(label_value), library)
                mask = integer_labels == label_value
                tissue_stats.setdefault(tissue, _Accumulator()).update(
                    difference[mask],
                    reference_chunk[mask],
                    mismatched[mask],
                )

        volume_shape = tuple(labels.shape)

    max_abs, mean_abs, rmse, max_rel = global_stats.metrics()
    tissues = []
    for name in sorted(tissue_stats):
        stats = tissue_stats[name]
        t_max_abs, t_mean_abs, t_rmse, t_max_rel = stats.metrics()
        tissues.append(
            TissueSignalComparison(
                tissue=name,
                voxel_count=stats.voxel_count,
                mismatch_count=stats.mismatch_count,
                max_abs_error=t_max_abs,
                mean_abs_error=t_mean_abs,
                rmse=t_rmse,
                max_rel_error=t_max_rel,
            )
        )

    return BssfpMatlabComparison(
        label_path=labels_file,
        image_path=image_file,
        volume_shape=volume_shape,
        flip_angle_deg=flip_angle_deg,
        te_ms=te_ms,
        tr_ms=tr_ms,
        voxel_count=global_stats.voxel_count,
        mismatch_count=global_stats.mismatch_count,
        max_abs_error=max_abs,
        mean_abs_error=mean_abs,
        rmse=rmse,
        max_rel_error=max_rel,
        atol=atol,
        rtol=rtol,
        tissues=tuple(tissues),
    )


def format_bssfp_matlab_comparison(report: BssfpMatlabComparison) -> str:
    """Format global and per-tissue MATLAB comparison metrics."""

    status = "PASS" if report.passed else "FAIL"
    lines = [
        f"MATLAB labels:   {report.label_path}",
        f"MATLAB contrast: {report.image_path}",
        f"Volume shape:    {report.volume_shape}",
        (
            f"Sequence:        FA={report.flip_angle_deg:g} deg, "
            f"TE={report.te_ms:g} ms, TR={report.tr_ms:g} ms"
        ),
        f"Voxels:          {report.voxel_count:,}",
        f"Tolerance:       atol={report.atol:g}, rtol={report.rtol:g}",
        "",
        "Tissue comparison:",
        (
            f"  {'Tissue':<14} {'Voxels':>12} {'Mismatch':>10} "
            f"{'Max |Δ|':>12} {'Mean |Δ|':>12} {'RMSE':>12}"
        ),
    ]
    for item in report.tissues:
        lines.append(
            f"  {item.tissue:<14} {item.voxel_count:>12,} "
            f"{item.mismatch_count:>10,} "
            f"{item.max_abs_error:>12.6g} "
            f"{item.mean_abs_error:>12.6g} "
            f"{item.rmse:>12.6g}"
        )
    lines.extend(
        (
            "",
            f"Overall:         {status}",
            f"Mismatch voxels: {report.mismatch_count:,}",
            f"Maximum |Δ|:     {report.max_abs_error:.6g}",
            f"Mean |Δ|:        {report.mean_abs_error:.6g}",
            f"RMSE:            {report.rmse:.6g}",
            f"Maximum rel |Δ|: {report.max_rel_error:.6g}",
        )
    )
    return "\n".join(lines)
