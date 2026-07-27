"""Conversion of physical sequence trajectories to SigPy coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class TrajectoryPreparationError(ValueError):
    """Raised when a physical trajectory cannot be mapped to an image grid."""


@dataclass(frozen=True)
class EncodingGrids:
    """Ground-truth and encoded matrices sharing one padded physical FOV."""

    ground_truth_shape: tuple[int, int, int]
    acquisition_fov_mm: tuple[float, float, float]
    acquisition_matrix_shape: tuple[int, int, int]
    reconstruction_fov_mm: tuple[float, float, float]
    reconstruction_matrix_shape: tuple[int, int, int]
    resolution_mm: tuple[float, float, float]


@dataclass(frozen=True)
class EncodingTrajectory:
    """Physical and grid-coordinate views of an arm-ordered trajectory."""

    coordinates: np.ndarray
    sample_count: int
    arm_count: int
    matrix_shape: tuple[int, int, int]
    source_maximum_absolute_k_per_m: tuple[float, float, float]
    maximum_absolute_coordinate: tuple[float, float, float]

    @property
    def point_count(self) -> int:
        return self.sample_count * self.arm_count

    def reshape_kspace(self, values: np.ndarray) -> np.ndarray:
        """Restore flattened arm-major samples to ``[sample, arm]``."""

        array = np.asarray(values)
        if array.shape != (self.point_count,):
            raise ValueError(
                f"flattened k-space shape {array.shape} does not match "
                f"{self.point_count} trajectory points"
            )
        return array.reshape(self.arm_count, self.sample_count).T


def _three_finite_positive(
    values: np.ndarray | tuple[float, float, float],
    *,
    name: str,
    allow_scalar: bool = False,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if allow_scalar and array.size == 1:
        array = np.repeat(array, 3)
    if array.size != 3 or not np.all(np.isfinite(array)) or np.any(array <= 0):
        suffix = "one or three" if allow_scalar else "three"
        raise TrajectoryPreparationError(
            f"{name} must contain {suffix} positive finite values"
        )
    return array


def prepare_encoding_grids(
    *,
    ground_truth_shape: tuple[int, int, int],
    ground_truth_voxel_size_mm: tuple[float, float, float],
    sequence_resolution_mm: np.ndarray | tuple[float, ...],
) -> EncodingGrids:
    """Derive the lower-resolution matrix on the padded ground-truth FOV."""

    if len(ground_truth_shape) != 3 or any(
        value <= 0 for value in ground_truth_shape
    ):
        raise TrajectoryPreparationError(
            "ground_truth_shape must contain three positive dimensions"
        )
    voxel = _three_finite_positive(
        ground_truth_voxel_size_mm, name="ground_truth_voxel_size_mm"
    )
    resolution = _three_finite_positive(
        sequence_resolution_mm,
        name="sequence_resolution_mm",
        allow_scalar=True,
    )
    acquisition_fov = np.asarray(ground_truth_shape) * voxel
    acquisition_shape = np.floor(acquisition_fov / resolution).astype(int)
    if np.any(acquisition_shape <= 0):
        raise TrajectoryPreparationError(
            "FOV and resolution produced a non-positive encoding matrix"
        )

    return EncodingGrids(
        ground_truth_shape=ground_truth_shape,
        acquisition_fov_mm=tuple(float(value) for value in acquisition_fov),
        acquisition_matrix_shape=tuple(int(value) for value in acquisition_shape),
        reconstruction_fov_mm=tuple(float(value) for value in acquisition_fov),
        reconstruction_matrix_shape=tuple(
            int(value) for value in acquisition_shape
        ),
        resolution_mm=tuple(float(value) for value in resolution),
    )


def prepare_sigpy_trajectory(
    kx_per_m: np.ndarray,
    ky_per_m: np.ndarray,
    kz_per_m: np.ndarray,
    *,
    matrix_shape: tuple[int, int, int],
    arm_indices: np.ndarray | None = None,
) -> EncodingTrajectory:
    """Normalize a full trajectory to SigPy matrix coordinates."""

    components = tuple(
        np.asarray(values, dtype=np.float64)
        for values in (kx_per_m, ky_per_m, kz_per_m)
    )
    shapes = {values.shape for values in components}
    if len(shapes) != 1 or components[0].ndim != 2:
        raise TrajectoryPreparationError(
            "kx, ky, and kz must have matching [sample, arm] shapes"
        )
    if not all(np.all(np.isfinite(values)) for values in components):
        raise TrajectoryPreparationError(
            "trajectory coordinates contain non-finite values"
        )
    if len(matrix_shape) != 3 or any(value <= 0 for value in matrix_shape):
        raise TrajectoryPreparationError(
            "matrix_shape must contain three positive dimensions"
        )

    sample_count, full_arm_count = components[0].shape
    if arm_indices is None:
        selected = np.arange(full_arm_count, dtype=np.intp)
    else:
        selected = np.asarray(arm_indices, dtype=np.intp)
        if selected.ndim != 1 or selected.size == 0:
            raise TrajectoryPreparationError(
                "arm_indices must be a non-empty one-dimensional array"
            )
        if np.any(selected < 0) or np.any(selected >= full_arm_count):
            raise TrajectoryPreparationError(
                f"arm_indices must be between 0 and {full_arm_count - 1}"
            )

    source_maxima = tuple(
        float(np.max(np.abs(values))) for values in components
    )
    flattened = []
    for values, maximum, size in zip(
        components, source_maxima, matrix_shape, strict=True
    ):
        selected_values = values[:, selected].T.reshape(-1)
        if maximum == 0:
            flattened.append(np.zeros_like(selected_values))
        else:
            flattened.append(selected_values * ((size / 2) / maximum))
    coordinates = np.column_stack(
        flattened
    ).astype(np.float32)
    maxima = tuple(
        float(np.max(np.abs(coordinates[:, axis])))
        for axis in range(3)
    )
    for axis, (maximum, size) in enumerate(
        zip(maxima, matrix_shape, strict=True)
    ):
        if maximum > size / 2 + 1e-5:
            raise TrajectoryPreparationError(
                f"trajectory axis {axis} exceeds the grid Nyquist range: "
                f"{maximum:g} > {size / 2:g}"
            )

    return EncodingTrajectory(
        coordinates=coordinates,
        sample_count=sample_count,
        arm_count=int(selected.size),
        matrix_shape=matrix_shape,
        source_maximum_absolute_k_per_m=source_maxima,
        maximum_absolute_coordinate=maxima,
    )


def format_encoding_trajectory(trajectory: EncodingTrajectory) -> str:
    """Format the coordinate contract used by SigPy."""

    return "\n".join(
        (
            "SigPy trajectory preparation",
            (
                f"Trajectory shape:      "
                f"{trajectory.sample_count} samples × "
                f"{trajectory.arm_count} arms"
            ),
            f"Flattened points:       {trajectory.point_count:,}",
            f"Coordinate matrix:      {trajectory.matrix_shape}",
            (
                "Full-trajectory |k|max: "
                + ", ".join(
                    f"{value:g} m^-1"
                    for value in trajectory.source_maximum_absolute_k_per_m
                )
            ),
            (
                "Maximum |coordinate|:  "
                + ", ".join(
                    f"{value:g}"
                    for value in trajectory.maximum_absolute_coordinate
                )
            ),
            "Coordinate scaling:      full-axis |k|max -> matrix size / 2",
            "Flattening order:        arm-major, samples contiguous",
            "Coordinate axis order:   kx, ky, kz",
        )
    )
