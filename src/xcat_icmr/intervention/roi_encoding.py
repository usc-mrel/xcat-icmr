"""Small-volume 3-D NUFFT encoding in a fixed global logical grid."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from xcat_icmr.encoding.sigpy_backend import SigpyNufftBackend
from xcat_icmr.encoding.trajectory import prepare_physical_sigpy_trajectory


class RoiEncodingError(ValueError):
    """Raised when a sparse logical delta cannot be ROI encoded."""


@dataclass(frozen=True)
class RoiEncodingResult:
    kspace: np.ndarray
    roi_start_ijk: np.ndarray
    roi_shape: tuple[int, int, int]
    applied_shift_voxels: int
    normalization_scale: float
    nonfinite_value_count: int


def _roi_bounds(
    indices: np.ndarray,
    global_shape: tuple[int, int, int],
    *,
    minimum_shape: int = 32,
    margin_voxels: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(indices, dtype=np.int64)
    shape = np.asarray(global_shape, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise RoiEncodingError("indices must have non-empty shape (N, 3)")
    if np.any(points < 0) or np.any(points >= shape[None, :]):
        raise RoiEncodingError("ROI indices fall outside the global grid")
    if minimum_shape <= 0 or margin_voxels < 0:
        raise RoiEncodingError("ROI minimum shape and margin are invalid")
    lower = np.min(points, axis=0) - margin_voxels
    upper = np.max(points, axis=0) + margin_voxels + 1
    desired = np.maximum(upper - lower, minimum_shape)
    desired = np.minimum(desired, shape)
    center_twice = lower + upper
    lower = np.floor_divide(center_twice - desired, 2)
    lower = np.maximum(lower, 0)
    lower = np.minimum(lower, shape - desired)
    return lower.astype(np.int32), desired.astype(np.int32)


def encode_sparse_logical_delta_roi(
    indices_ijk: np.ndarray,
    values: np.ndarray,
    *,
    global_shape: tuple[int, int, int],
    voxel_size_mm: tuple[float, float, float],
    coil_count: int,
    coil_roi_loader: Callable[
        [int, tuple[slice, slice, slice]], np.ndarray
    ],
    kx_per_m: np.ndarray,
    ky_per_m: np.ndarray,
    kz_per_m: np.ndarray,
    acquisition_matrix_shape: tuple[int, int, int],
    rf_center_shift_mm: float,
    rf_axis_voxel_size_mm: float,
    rf_logical_axis: int,
    device_id: int,
    minimum_roi_shape: int = 32,
    roi_margin_voxels: int = 4,
    progress: Callable[[int, int], None] | None = None,
) -> RoiEncodingResult:
    """Encode a sparse delta via a local grid plus global phase correction."""

    points = np.asarray(indices_ijk, dtype=np.int64)
    signal = np.asarray(values, dtype=np.float32).reshape(-1)
    global_matrix = np.asarray(global_shape, dtype=np.int64)
    voxel = np.asarray(voxel_size_mm, dtype=np.float64)
    if points.shape != (len(signal), 3):
        raise RoiEncodingError("indices and values have incompatible shapes")
    if not np.all(np.isfinite(signal)):
        raise RoiEncodingError("sparse delta contains non-finite values")
    if coil_count <= 0:
        raise RoiEncodingError("coil_count must be positive")
    if rf_logical_axis not in {0, 1, 2}:
        raise RoiEncodingError("rf_logical_axis must be 0, 1, or 2")
    roi_start, roi_shape_array = _roi_bounds(
        points,
        global_shape,
        minimum_shape=minimum_roi_shape,
        margin_voxels=roi_margin_voxels,
    )
    roi_shape = tuple(int(value) for value in roi_shape_array)
    local_indices = points - roi_start[None, :]
    roi_delta = np.zeros(roi_shape, dtype=np.float32)
    np.add.at(roi_delta, tuple(local_indices.T), signal)
    roi_slices = tuple(
        slice(int(start), int(start + size))
        for start, size in zip(roi_start, roi_shape_array, strict=True)
    )
    global_fov_mm = tuple(
        float(value) for value in global_matrix * voxel
    )
    roi_fov_mm = tuple(float(value) for value in roi_shape_array * voxel)
    global_trajectory = prepare_physical_sigpy_trajectory(
        kx_per_m,
        ky_per_m,
        kz_per_m,
        fov_mm=global_fov_mm,
        matrix_shape=acquisition_matrix_shape,
    )
    roi_trajectory = prepare_physical_sigpy_trajectory(
        kx_per_m,
        ky_per_m,
        kz_per_m,
        fov_mm=roi_fov_mm,
        matrix_shape=roi_shape,
    )
    shift_voxels = -int(
        np.rint(rf_center_shift_mm / rf_axis_voxel_size_mm)
    )
    offset = (
        roi_start.astype(np.float64)
        + roi_shape_array // 2
        - global_matrix // 2
    )
    offset[rf_logical_axis] += shift_voxels
    phase = np.exp(
        -2j
        * np.pi
        * np.sum(
            global_trajectory.coordinates
            * (offset / global_matrix)[None, :],
            axis=1,
        )
    ).astype(np.complex64)
    normalization_scale = float(
        np.sqrt(np.prod(roi_shape_array) / np.prod(global_matrix))
    )
    backend = SigpyNufftBackend(device_id=device_id)
    kspace = np.empty(
        (
            global_trajectory.sample_count,
            global_trajectory.arm_count,
            coil_count,
        ),
        dtype=np.complex64,
    )
    nonfinite = 0
    for coil_index in range(coil_count):
        coil = np.asarray(
            coil_roi_loader(coil_index, roi_slices), dtype=np.complex64
        )
        if coil.shape != roi_shape or not np.all(np.isfinite(coil)):
            raise RoiEncodingError(
                f"coil {coil_index} ROI is invalid: {coil.shape} != {roi_shape}"
            )
        local_kspace = backend.forward(
            np.asarray(roi_delta * coil, dtype=np.complex64), roi_trajectory
        )
        corrected = np.asarray(
            local_kspace * phase * normalization_scale, dtype=np.complex64
        )
        restored = global_trajectory.reshape_kspace(corrected)
        nonfinite += int(np.count_nonzero(~np.isfinite(restored)))
        kspace[:, :, coil_index] = restored
        if progress is not None:
            progress(coil_index + 1, coil_count)
    if nonfinite:
        raise RoiEncodingError(
            f"ROI NUFFT produced {nonfinite} non-finite values"
        )
    return RoiEncodingResult(
        kspace=kspace,
        roi_start_ijk=roi_start,
        roi_shape=roi_shape,
        applied_shift_voxels=shift_voxels,
        normalization_scale=normalization_scale,
        nonfinite_value_count=nonfinite,
    )
