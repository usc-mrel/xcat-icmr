"""Chunked comparison against a saved MATLAB ``par.seq_params`` structure."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from xcat_icmr.sequence.reader import SequenceData


class MatlabReferenceError(Exception):
    """Raised when a saved MATLAB reference cannot be compared."""


@dataclass(frozen=True)
class ComparisonItem:
    name: str
    matched: bool
    max_abs_error: float
    python_shape: tuple[int, ...]
    matlab_shape: tuple[int, ...]


@dataclass(frozen=True)
class MatlabComparison:
    reference_path: Path
    items: tuple[ComparisonItem, ...]

    @property
    def passed(self) -> bool:
        return all(item.matched for item in self.items)


def _compare_values(
    name: str, python_value: np.ndarray, matlab_value: np.ndarray
) -> ComparisonItem:
    python_array = np.asarray(python_value)
    matlab_array = np.asarray(matlab_value)
    same_shape = python_array.shape == matlab_array.shape
    if not same_shape:
        return ComparisonItem(
            name,
            False,
            float("inf"),
            python_array.shape,
            matlab_array.shape,
        )
    difference = np.abs(python_array - matlab_array)
    max_error = float(np.max(difference)) if difference.size else 0.0
    return ComparisonItem(
        name,
        bool(np.array_equal(python_array, matlab_array)),
        max_error,
        python_array.shape,
        matlab_array.shape,
    )


def _compare_transposed_dataset(
    name: str,
    python_array: np.ndarray,
    matlab_dataset: h5py.Dataset,
    *,
    arm_chunk: int = 64,
) -> ComparisonItem:
    matlab_logical_shape = tuple(reversed(matlab_dataset.shape))
    if python_array.shape != matlab_logical_shape:
        return ComparisonItem(
            name,
            False,
            float("inf"),
            python_array.shape,
            matlab_logical_shape,
        )

    max_error = 0.0
    exact = True
    for start in range(0, python_array.shape[1], arm_chunk):
        stop = min(start + arm_chunk, python_array.shape[1])
        matlab_chunk = np.asarray(matlab_dataset[start:stop, :]).T
        python_chunk = python_array[:, start:stop]
        difference = np.abs(python_chunk - matlab_chunk)
        if difference.size:
            max_error = max(max_error, float(np.max(difference)))
        exact = exact and bool(np.array_equal(python_chunk, matlab_chunk))

    return ComparisonItem(
        name,
        exact,
        max_error,
        python_array.shape,
        matlab_logical_shape,
    )


def compare_to_matlab(
    data: SequenceData, reference_path: str | Path
) -> MatlabComparison:
    """Compare sequence data without loading the MATLAB coil map."""

    path = Path(reference_path).expanduser().resolve(strict=False)
    if not path.is_file():
        raise MatlabReferenceError(f"MATLAB reference does not exist: {path}")

    try:
        handle = h5py.File(path, "r")
    except OSError as exc:
        raise MatlabReferenceError(
            f"could not open MATLAB reference {path}: {exc}"
        ) from exc

    with handle:
        group_path = "par/seq_params"
        if group_path not in handle:
            raise MatlabReferenceError(
                f"MATLAB reference is missing {group_path}: {path}"
            )
        seq = handle[group_path]
        required = ("FA", "TE", "TR", "FOV", "res", "kx", "ky", "kz")
        missing = [name for name in required if name not in seq]
        if missing:
            raise MatlabReferenceError(
                f"MATLAB seq_params is missing: {', '.join(missing)}"
            )

        items = [
            _compare_values(
                "FA",
                np.asarray(data.flip_angle_deg),
                np.asarray(seq["FA"][()]).squeeze(),
            ),
            _compare_values(
                "TE",
                np.asarray(data.te_ms),
                np.asarray(seq["TE"][()]).squeeze(),
            ),
            _compare_values(
                "TR",
                np.asarray(data.tr_ms),
                np.asarray(seq["TR"][()]).squeeze(),
            ),
            _compare_values(
                "FOV",
                data.fov_mm,
                np.asarray(seq["FOV"][()]).squeeze(),
            ),
            _compare_values(
                "resolution",
                data.resolution_mm,
                np.atleast_1d(np.asarray(seq["res"][()]).squeeze()),
            ),
            _compare_transposed_dataset("kx", data.kx, seq["kx"]),
            _compare_transposed_dataset("ky", data.ky, seq["ky"]),
            _compare_transposed_dataset("kz", data.kz, seq["kz"]),
        ]

        metadata_path = "metadata/w"
        if metadata_path in seq:
            items.append(
                _compare_transposed_dataset(
                    "density compensation",
                    data.density_compensation,
                    seq[metadata_path],
                )
            )

    return MatlabComparison(path, tuple(items))


def format_matlab_comparison(report: MatlabComparison) -> str:
    """Format an exact comparison report."""

    lines = [f"MATLAB reference: {report.reference_path}"]
    for item in report.items:
        status = "PASS" if item.matched else "FAIL"
        lines.append(
            f"  {item.name:<22} {status}  "
            f"max |Δ|={item.max_abs_error:.6g}  shape={item.python_shape}"
        )
    lines.append(f"Overall: {'PASS' if report.passed else 'FAIL'}")
    return "\n".join(lines)
