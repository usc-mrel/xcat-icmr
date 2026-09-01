"""Streaming access and safe normalization for MATLAB sensitivity maps."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Callable, Iterator

import h5py
import numpy as np

from xcat_icmr.sequence.orientation import (
    reorient_spatial_array,
    reoriented_spatial_shape,
)


class SensitivityMapError(Exception):
    """Raised when a sensitivity map cannot be read or normalized safely."""


@dataclass(frozen=True)
class SensitivityMapInfo:
    """Validated layout of a MATLAB v7.3 sensitivity-map dataset."""

    path: Path
    dataset_name: str
    coil_count: int
    stored_shape: tuple[int, int, int, int]
    logical_spatial_shape: tuple[int, int, int]
    stored_dtype: str
    block_shape: tuple[int, int, int]


@dataclass(frozen=True)
class RssNormalization:
    """Reusable voxelwise RSS denominator and its validation statistics."""

    cache_path: Path
    reused_cache: bool
    relative_epsilon: float
    absolute_epsilon: float
    minimum_supported_rss: float
    maximum_rss: float
    supported_voxel_count: int
    background_voxel_count: int
    nonfinite_value_count: int


ProgressCallback = Callable[[int, int], None]


def inspect_sensitivity_map(
    path: str | Path,
    *,
    dataset_name: str = "sens",
) -> SensitivityMapInfo:
    """Inspect a coil-first MATLAB v7.3 dataset without loading its values."""

    resolved = Path(path).expanduser().resolve(strict=False)
    if not resolved.is_file():
        raise SensitivityMapError(
            f"sensitivity-map file does not exist: {resolved}"
        )
    try:
        with h5py.File(resolved, "r") as handle:
            if dataset_name not in handle:
                raise SensitivityMapError(
                    f"sensitivity-map file is missing dataset {dataset_name!r}"
                )
            dataset = handle[dataset_name]
            if not isinstance(dataset, h5py.Dataset) or dataset.ndim != 4:
                raise SensitivityMapError(
                    f"dataset {dataset_name!r} must have four dimensions"
                )
            dtype = dataset.dtype
            names = dtype.names
            is_matlab_complex = (
                names is not None and {"real", "imag"}.issubset(names)
            )
            if not np.issubdtype(dtype, np.complexfloating) and not (
                is_matlab_complex
            ):
                raise SensitivityMapError(
                    "sensitivity-map values must be native complex or MATLAB "
                    "complex values with real/imag fields"
                )
            stored_shape = tuple(int(value) for value in dataset.shape)
            if stored_shape[0] <= 0 or any(
                value <= 0 for value in stored_shape[1:]
            ):
                raise SensitivityMapError(
                    f"invalid sensitivity-map shape: {stored_shape}"
                )
            if dataset.chunks is None:
                block_shape = tuple(
                    min(value, 16) for value in stored_shape[1:]
                )
            else:
                block_shape = tuple(
                    int(value) for value in dataset.chunks[1:]
                )
    except OSError as exc:
        raise SensitivityMapError(
            f"could not open sensitivity map {resolved}: {exc}"
        ) from exc

    # Match main_gen_kspace_fullysampled.py: h5py exposes this particular map
    # as [coil, x, y, z], and those spatial axes are used directly when the
    # map is multiplied by the [x, y, z] contrast image.
    #
    # TODO: This is a contract for the prepared legacy XCAT sensitivity map,
    # not a general MATLAB v7.3 rule. Add an explicit, validated coil-map axis
    # order to the configuration before accepting maps from other sources.
    # Never infer the order from dimensions alone: cubic maps such as
    # [500, 500, 500] make spatial-axis permutations undetectable by shape.
    return SensitivityMapInfo(
        path=resolved,
        dataset_name=dataset_name,
        coil_count=stored_shape[0],
        stored_shape=stored_shape,
        logical_spatial_shape=stored_shape[1:],
        stored_dtype=str(dtype),
        block_shape=block_shape,
    )


def _spatial_blocks(
    shape: tuple[int, int, int],
    block_shape: tuple[int, int, int],
) -> Iterator[tuple[slice, slice, slice]]:
    for first in range(0, shape[0], block_shape[0]):
        for second in range(0, shape[1], block_shape[1]):
            for third in range(0, shape[2], block_shape[2]):
                yield (
                    slice(first, min(first + block_shape[0], shape[0])),
                    slice(second, min(second + block_shape[1], shape[1])),
                    slice(third, min(third + block_shape[2], shape[2])),
                )


def _block_count(
    shape: tuple[int, int, int],
    block_shape: tuple[int, int, int],
) -> int:
    return math.prod(
        math.ceil(size / block)
        for size, block in zip(shape, block_shape, strict=True)
    )


def _as_complex(values: np.ndarray) -> np.ndarray:
    if np.issubdtype(values.dtype, np.complexfloating):
        return np.asarray(values, dtype=np.complex64)
    names = values.dtype.names
    if names is not None and {"real", "imag"}.issubset(names):
        return (
            np.asarray(values["real"], dtype=np.float32)
            + 1j * np.asarray(values["imag"], dtype=np.float32)
        )
    raise SensitivityMapError("unsupported sensitivity-map complex dtype")


def _cache_signature(info: SensitivityMapInfo) -> dict[str, object]:
    stat = info.path.stat()
    return {
        "source_path": str(info.path),
        "source_size_bytes": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "dataset_name": info.dataset_name,
        "stored_shape": list(info.stored_shape),
        "stored_dtype": info.stored_dtype,
    }


def _metadata_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(".json")


def _load_valid_cache(
    info: SensitivityMapInfo,
    cache_path: Path,
) -> tuple[np.memmap, dict[str, object]] | None:
    metadata_path = _metadata_path(cache_path)
    if not cache_path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("signature") != _cache_signature(info):
            return None
        rss = np.load(cache_path, mmap_mode="r")
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if rss.shape != info.stored_shape[1:] or rss.dtype != np.float32:
        return None
    return rss, metadata


def _rss_statistics(
    rss: np.ndarray,
    *,
    relative_epsilon: float,
    block_shape: tuple[int, int, int],
) -> tuple[float, float, int, int]:
    maximum = 0.0
    for selection in _spatial_blocks(rss.shape, block_shape):
        maximum = max(maximum, float(np.max(rss[selection])))
    absolute_epsilon = maximum * relative_epsilon
    minimum_supported = math.inf
    supported = 0
    background = 0
    for selection in _spatial_blocks(rss.shape, block_shape):
        block = np.asarray(rss[selection])
        mask = block > absolute_epsilon
        count = int(np.count_nonzero(mask))
        supported += count
        background += block.size - count
        if count:
            minimum_supported = min(
                minimum_supported, float(np.min(block[mask]))
            )
    if not math.isfinite(minimum_supported):
        minimum_supported = 0.0
    return maximum, minimum_supported, supported, background


def prepare_rss_normalization(
    info: SensitivityMapInfo,
    cache_path: str | Path,
    *,
    relative_epsilon: float = 1e-6,
    rebuild: bool = False,
    progress: ProgressCallback | None = None,
) -> RssNormalization:
    """Compute or reuse a float32 RSS denominator without changing coils."""

    if not math.isfinite(relative_epsilon) or relative_epsilon < 0:
        raise ValueError("relative_epsilon must be finite and non-negative")
    resolved_cache = Path(cache_path).expanduser().resolve(strict=False)
    resolved_cache.parent.mkdir(parents=True, exist_ok=True)

    cached = None if rebuild else _load_valid_cache(info, resolved_cache)
    nonfinite_count = 0
    reused = cached is not None
    if cached is None:
        partial = resolved_cache.with_name(f".{resolved_cache.name}.partial")
        partial.unlink(missing_ok=True)
        rss = np.lib.format.open_memmap(
            partial,
            mode="w+",
            dtype=np.float32,
            shape=info.stored_shape[1:],
        )
        total = _block_count(info.stored_shape[1:], info.block_shape)
        try:
            with h5py.File(info.path, "r") as handle:
                dataset = handle[info.dataset_name]
                for completed, selection in enumerate(
                    _spatial_blocks(
                        info.stored_shape[1:], info.block_shape
                    ),
                    start=1,
                ):
                    values = _as_complex(
                        np.asarray(
                            dataset[(slice(None),) + selection]
                        )
                    )
                    finite = np.isfinite(values)
                    nonfinite_count += values.size - int(
                        np.count_nonzero(finite)
                    )
                    if not np.all(finite):
                        raise SensitivityMapError(
                            "sensitivity map contains non-finite values"
                        )
                    squared = (
                        values.real.astype(np.float64) ** 2
                        + values.imag.astype(np.float64) ** 2
                    )
                    rss[selection] = np.sqrt(
                        np.sum(squared, axis=0, dtype=np.float64)
                    ).astype(np.float32)
                    if progress is not None:
                        progress(completed, total)
            rss.flush()
            del rss
            os.replace(partial, resolved_cache)
        finally:
            partial.unlink(missing_ok=True)

        rss = np.load(resolved_cache, mmap_mode="r")
        metadata = {
            "signature": _cache_signature(info),
            "nonfinite_value_count": nonfinite_count,
        }
        metadata_partial = _metadata_path(resolved_cache).with_name(
            f".{_metadata_path(resolved_cache).name}.partial"
        )
        metadata_partial.write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(metadata_partial, _metadata_path(resolved_cache))
    else:
        rss, metadata = cached
        nonfinite_count = int(metadata.get("nonfinite_value_count", 0))

    maximum, minimum, supported, background = _rss_statistics(
        rss,
        relative_epsilon=relative_epsilon,
        block_shape=info.block_shape,
    )
    return RssNormalization(
        cache_path=resolved_cache,
        reused_cache=reused,
        relative_epsilon=relative_epsilon,
        absolute_epsilon=maximum * relative_epsilon,
        minimum_supported_rss=minimum,
        maximum_rss=maximum,
        supported_voxel_count=supported,
        background_voxel_count=background,
        nonfinite_value_count=nonfinite_count,
    )


def load_normalized_coil(
    info: SensitivityMapInfo,
    coil_index: int,
    normalization: RssNormalization,
) -> np.ndarray:
    """Load one coil, normalize safely, and return logical ``[x, y, z]``."""

    if not 0 <= coil_index < info.coil_count:
        raise IndexError(
            f"coil_index must be between 0 and {info.coil_count - 1}"
        )
    rss = np.load(normalization.cache_path, mmap_mode="r")
    stored = np.zeros(info.stored_shape[1:], dtype=np.complex64)
    with h5py.File(info.path, "r") as handle:
        dataset = handle[info.dataset_name]
        for selection in _spatial_blocks(
            info.stored_shape[1:], info.block_shape
        ):
            values = _as_complex(
                np.asarray(dataset[(coil_index,) + selection])
            )
            denominator = np.asarray(rss[selection])
            stored[selection] = np.divide(
                values,
                denominator,
                out=np.zeros_like(values),
                where=denominator > normalization.absolute_epsilon,
            )
    return stored


def sensitivity_shape_in_logical_frame(
    info: SensitivityMapInfo,
    *,
    stored_axis_order: tuple[str, str, str],
    dcs_to_logical: np.ndarray,
) -> tuple[int, int, int]:
    """Return logical shape after declared stored→DCS and DCS→logical maps."""

    if set(stored_axis_order) != {"X", "Y", "Z"}:
        raise SensitivityMapError(
            "stored_axis_order must contain X, Y, and Z exactly once"
        )
    dcs_shape = tuple(
        info.logical_spatial_shape[stored_axis_order.index(axis)]
        for axis in ("X", "Y", "Z")
    )
    return reoriented_spatial_shape(dcs_shape, dcs_to_logical)


def load_normalized_coil_in_logical_frame(
    info: SensitivityMapInfo,
    coil_index: int,
    normalization: RssNormalization,
    *,
    stored_axis_order: tuple[str, str, str],
    dcs_to_logical: np.ndarray,
) -> np.ndarray:
    """Normalize one declared-DCS coil and reorient it to Pulseq logical axes."""

    stored = load_normalized_coil(info, coil_index, normalization)
    if set(stored_axis_order) != {"X", "Y", "Z"}:
        raise SensitivityMapError(
            "stored_axis_order must contain X, Y, and Z exactly once"
        )
    stored_to_dcs = tuple(
        stored_axis_order.index(axis) for axis in ("X", "Y", "Z")
    )
    dcs = np.transpose(stored, stored_to_dcs)
    return reorient_spatial_array(dcs, dcs_to_logical)


def load_normalized_coil_roi_in_logical_frame(
    info: SensitivityMapInfo,
    coil_index: int,
    normalization: RssNormalization,
    logical_slices: tuple[slice, slice, slice],
    *,
    stored_axis_order: tuple[str, str, str],
    dcs_to_logical: np.ndarray,
) -> np.ndarray:
    """Load and normalize only a contiguous logical sensitivity-map ROI."""

    if not 0 <= coil_index < info.coil_count:
        raise IndexError(
            f"coil_index must be between 0 and {info.coil_count - 1}"
        )
    if set(stored_axis_order) != {"X", "Y", "Z"}:
        raise SensitivityMapError(
            "stored_axis_order must contain X, Y, and Z exactly once"
        )
    if len(logical_slices) != 3:
        raise SensitivityMapError("logical_slices must contain three slices")
    stored_to_dcs = np.zeros((3, 3), dtype=np.float64)
    for dcs_axis, axis_name in enumerate(("X", "Y", "Z")):
        stored_to_dcs[dcs_axis, stored_axis_order.index(axis_name)] = 1.0
    stored_to_logical = np.asarray(dcs_to_logical) @ stored_to_dcs
    source_axes = np.argmax(np.abs(stored_to_logical), axis=1)
    logical_shape = tuple(
        int(info.stored_shape[1 + int(axis)]) for axis in source_axes
    )
    stored_selections: list[slice | None] = [None, None, None]
    output_shape = []
    for logical_axis, selection in enumerate(logical_slices):
        start = 0 if selection.start is None else int(selection.start)
        stop = logical_shape[logical_axis] if selection.stop is None else int(
            selection.stop
        )
        if selection.step not in (None, 1) or not 0 <= start < stop <= logical_shape[
            logical_axis
        ]:
            raise SensitivityMapError("logical ROI slices are invalid")
        source_axis = int(source_axes[logical_axis])
        size = int(info.stored_shape[1 + source_axis])
        sign = stored_to_logical[logical_axis, source_axis]
        if sign > 0:
            stored_selection = slice(start, stop)
        elif size % 2:
            stored_selection = slice(size - stop, size - start)
        else:
            if start == 0:
                raise SensitivityMapError(
                    "an even-grid reversed ROI cannot include the zero-filled edge"
                )
            stored_selection = slice(size - stop + 1, size - start + 1)
        stored_selections[source_axis] = stored_selection
        output_shape.append(stop - start)
    if any(selection is None for selection in stored_selections):
        raise SensitivityMapError("logical ROI does not map to every stored axis")
    stored_selection_tuple = tuple(stored_selections)  # type: ignore[arg-type]
    rss = np.load(normalization.cache_path, mmap_mode="r")
    with h5py.File(info.path, "r") as handle:
        dataset = handle[info.dataset_name]
        values = _as_complex(
            np.asarray(dataset[(coil_index,) + stored_selection_tuple])
        )
    denominator = np.asarray(rss[stored_selection_tuple])
    normalized = np.divide(
        values,
        denominator,
        out=np.zeros_like(values),
        where=denominator > normalization.absolute_epsilon,
    )
    logical = np.transpose(normalized, tuple(int(axis) for axis in source_axes))
    for logical_axis, source_axis_value in enumerate(source_axes):
        source_axis = int(source_axis_value)
        if stored_to_logical[logical_axis, source_axis] < 0:
            logical = np.flip(logical, axis=logical_axis)
    if logical.shape != tuple(output_shape):
        raise SensitivityMapError(
            f"logical ROI shape {logical.shape} != {tuple(output_shape)}"
        )
    return np.asarray(logical, dtype=np.complex64)


def format_sensitivity_preparation(
    info: SensitivityMapInfo,
    normalization: RssNormalization,
) -> str:
    """Format sensitivity-map and RSS-cache validation."""

    total = (
        normalization.supported_voxel_count
        + normalization.background_voxel_count
    )
    return "\n".join(
        (
            "Sensitivity-map preparation",
            f"Source:                {info.path}",
            f"Dataset:               {info.dataset_name}",
            f"Stored shape:          {info.stored_shape}",
            (
                "Logical shape:         "
                f"{info.logical_spatial_shape + (info.coil_count,)}"
            ),
            f"Coils used:            all {info.coil_count} file coils",
            "Coil selection:        none",
            "Normalization:          voxelwise RSS",
            f"RSS cache:             {normalization.cache_path}",
            (
                "RSS cache status:      "
                + ("reused" if normalization.reused_cache else "created")
            ),
            f"RSS range:             0 to {normalization.maximum_rss:g}",
            (
                "RSS support threshold: "
                f"{normalization.absolute_epsilon:g} "
                f"(relative {normalization.relative_epsilon:g})"
            ),
            (
                f"Supported voxels:      "
                f"{normalization.supported_voxel_count:,} / {total:,}"
            ),
            (
                f"Background voxels:     "
                f"{normalization.background_voxel_count:,} / {total:,}"
            ),
            (
                f"Non-finite values:     "
                f"{normalization.nonfinite_value_count:,}"
            ),
        )
    )
