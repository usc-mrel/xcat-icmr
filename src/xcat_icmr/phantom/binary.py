"""Memory-mapped decoding of raw headerless XCAT label volumes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from xcat_icmr.config.models import SimulationConfig


class XcatBinaryReadError(Exception):
    """Raised when a raw XCAT binary cannot be decoded safely."""


@dataclass(frozen=True)
class XcatBinaryVolume:
    """Raw and YAML-cropped views of one XCAT float32 label volume."""

    path: Path
    raw_shape: tuple[int, int, int]
    cropped_shape: tuple[int, int, int]
    row_slice: slice
    column_slice: slice
    raw: npt.NDArray[np.float32]
    cropped: npt.NDArray[np.float32]


def xcat_raw_shape(config: SimulationConfig) -> tuple[int, int, int]:
    """Return the logical MATLAB/XCAT volume shape before cropping."""

    matrix = config.phantom.matrix_size_xy
    slices = (
        config.phantom.slice_range.end
        - config.phantom.slice_range.start
        + 1
    )
    return matrix, matrix, slices


def open_xcat_binary(
    config: SimulationConfig,
    path: str | Path,
) -> XcatBinaryVolume:
    """Memory-map XCAT output using MATLAB-compatible column-major ordering."""

    binary_path = Path(path).expanduser().resolve(strict=False)
    if not binary_path.is_file():
        raise XcatBinaryReadError(
            f"XCAT binary file does not exist: {binary_path}"
        )

    raw_shape = xcat_raw_shape(config)
    expected_bytes = int(np.prod(raw_shape, dtype=np.int64)) * 4
    actual_bytes = binary_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise XcatBinaryReadError(
            f"XCAT binary has {actual_bytes:,} bytes; "
            f"expected {expected_bytes:,} for shape {raw_shape}"
        )

    row_start, row_stop = config.phantom.crop.rows
    column_start, column_stop = config.phantom.crop.columns
    matrix = config.phantom.matrix_size_xy
    if row_stop > matrix or column_stop > matrix:
        raise XcatBinaryReadError(
            "phantom crop exceeds the raw XCAT in-plane matrix: "
            f"rows={config.phantom.crop.rows}, "
            f"columns={config.phantom.crop.columns}, matrix={matrix}"
        )

    raw = np.memmap(
        binary_path,
        dtype=np.dtype("<f4"),
        mode="r",
        shape=raw_shape,
        order="F",
    )
    row_slice = slice(row_start, row_stop)
    column_slice = slice(column_start, column_stop)
    cropped = raw[row_slice, column_slice, :]
    return XcatBinaryVolume(
        path=binary_path,
        raw_shape=raw_shape,
        cropped_shape=cropped.shape,
        row_slice=row_slice,
        column_slice=column_slice,
        raw=raw,
        cropped=cropped,
    )
