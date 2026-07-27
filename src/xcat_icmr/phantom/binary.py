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
    """Raw XCAT float32 label volume with full RL/AP coverage."""

    path: Path
    raw_shape: tuple[int, int, int]
    cropped_shape: tuple[int, int, int]
    row_slice: slice
    column_slice: slice
    raw: npt.NDArray[np.float32]
    cropped: npt.NDArray[np.float32]


def xcat_raw_shape(config: SimulationConfig) -> tuple[int, int, int]:
    """Return the XCAT volume shape with full RL/AP and configured HF extent."""

    matrix = config.phantom.matrix_size_xy
    slices = (
        config.phantom.head_foot_slice_range.end
        - config.phantom.head_foot_slice_range.start
        + 1
    )
    return matrix, matrix, slices


def _matlab_range_to_slice(
    values: tuple[int, int] | None,
    *,
    matrix: int,
    axis_name: str,
) -> slice:
    """Convert an optional MATLAB 1-based inclusive range to a Python slice."""

    if values is None:
        return slice(0, matrix)
    start, end = values
    if end > matrix:
        raise XcatBinaryReadError(
            f"{axis_name} crop {values} exceeds matrix size {matrix}"
        )
    return slice(start - 1, end)


def xcat_label_shape(config: SimulationConfig) -> tuple[int, int, int]:
    """Return the saved label shape after optional in-plane cropping."""

    matrix, _, slices = xcat_raw_shape(config)
    rl = config.phantom.in_plane_crop.right_left
    ap = config.phantom.in_plane_crop.anterior_posterior
    rl_size = matrix if rl is None else rl[1] - rl[0] + 1
    ap_size = matrix if ap is None else ap[1] - ap[0] + 1
    return rl_size, ap_size, slices


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

    matrix = config.phantom.matrix_size_xy

    raw = np.memmap(
        binary_path,
        dtype=np.dtype("<f4"),
        mode="r",
        shape=raw_shape,
        order="F",
    )
    # HFS native XCAT axes are RL, AP, HF. YAML crop values deliberately use
    # MATLAB's 1-based inclusive indexing to match the reference workflow.
    row_slice = _matlab_range_to_slice(
        config.phantom.in_plane_crop.right_left,
        matrix=matrix,
        axis_name="right_left",
    )
    column_slice = _matlab_range_to_slice(
        config.phantom.in_plane_crop.anterior_posterior,
        matrix=matrix,
        axis_name="anterior_posterior",
    )
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
