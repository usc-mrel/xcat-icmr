"""Cheap storage estimates and preflight checks for large acquisitions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

import numpy as np


@dataclass(frozen=True)
class StorageEstimate:
    shape: tuple[int, ...]
    dtype: str
    bytes: int

    @property
    def gib(self) -> float:
        return self.bytes / 1024**3


def _estimate(shape: tuple[int, ...], dtype: np.dtype) -> StorageEstimate:
    count = int(np.prod(shape, dtype=np.int64))
    return StorageEstimate(shape, dtype.name, count * dtype.itemsize)


def estimate_tissue_library_storage(
    sample_count: int, arm_count: int, coil_count: int, phase_count: int
) -> StorageEstimate:
    return _estimate(
        (sample_count, arm_count, coil_count, phase_count), np.dtype(np.complex64)
    )


def estimate_dynamic_acquisition_storage(
    sample_count: int, acquisition_count: int, coil_count: int
) -> StorageEstimate:
    return _estimate(
        (sample_count, acquisition_count, coil_count), np.dtype(np.complex64)
    )


def require_free_space(
    directory: str | Path,
    required_bytes: int,
    *,
    safety_fraction: float = 0.05,
    fixed_reserve_bytes: int = 1024**3,
) -> int:
    """Return free bytes or raise before a knowingly oversized generation."""

    destination = Path(directory).expanduser().resolve(strict=False)
    destination.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(destination).free
    threshold = required_bytes + max(
        fixed_reserve_bytes, int(required_bytes * safety_fraction)
    )
    if free < threshold:
        raise OSError(
            f"insufficient free space: need {threshold / 1024**3:.2f} GiB "
            f"including reserve, found {free / 1024**3:.2f} GiB"
        )
    return free
