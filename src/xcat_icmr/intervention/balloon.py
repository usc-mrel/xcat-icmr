"""Local partial-volume support for the high-resolution Gd balloon."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class SparseBalloonError(ValueError):
    """Raised when sparse balloon geometry cannot be rasterized."""


@dataclass(frozen=True)
class SparseBalloonSupport:
    """A small local occupancy patch for one balloon frame."""

    center_lps_mm: np.ndarray
    bounding_box_start_ijk: np.ndarray
    occupancy: np.ndarray
    volume_shape: tuple[int, int, int]
    voxel_size_mm: tuple[float, float, float]
    origin_lps_mm: np.ndarray

    @property
    def voxel_count(self) -> int:
        """Return the number of local voxels with non-zero occupancy."""

        return int(np.count_nonzero(self.occupancy))

    @property
    def bounding_box_slices(self) -> tuple[slice, slice, slice]:
        """Return the local patch location in the complete PCS volume."""

        return tuple(
            slice(int(start), int(start) + int(size))
            for start, size in zip(
                self.bounding_box_start_ijk,
                self.occupancy.shape,
                strict=True,
            )
        )

    @property
    def occupied_volume_mm3(self) -> float:
        """Return the partial-volume estimate of balloon volume."""

        return float(np.sum(self.occupancy, dtype=np.float64)) * float(
            np.prod(self.voxel_size_mm)
        )

    def occupied_indices_ijk(self) -> np.ndarray:
        """Materialize occupied global indices only when explicitly needed."""

        local = np.argwhere(self.occupancy > 0.0)
        return np.asarray(
            local + self.bounding_box_start_ijk[None, :], dtype=np.int32
        )


def centered_origin_lps_mm(
    volume_shape: tuple[int, int, int],
    voxel_size_mm: tuple[float, float, float],
) -> np.ndarray:
    """Return the XCAT PCS origin used by the label/contrast NRRDs."""

    shape = np.asarray(volume_shape, dtype=np.int64)
    voxel = np.asarray(voxel_size_mm, dtype=np.float64)
    if shape.shape != (3,) or np.any(shape <= 0):
        raise SparseBalloonError("volume_shape must contain three sizes")
    if voxel.shape != (3,) or np.any(~np.isfinite(voxel)) or np.any(voxel <= 0):
        raise SparseBalloonError("voxel_size_mm must contain three positive values")
    return -(shape // 2) * voxel


def rasterize_sparse_balloon(
    center_lps_mm: np.ndarray,
    *,
    volume_shape: tuple[int, int, int],
    voxel_size_mm: tuple[float, float, float],
    diameter_mm: tuple[float, float, float],
    shape: str,
    boundary_samples_per_axis: int = 8,
) -> SparseBalloonSupport:
    """Rasterize fractional occupancy only inside a small local box.

    Completely inside/outside voxels are classified geometrically. Only
    boundary voxels are supersampled, which makes sub-voxel translation smooth
    without allocating a full-volume mask or storing global voxel indices.
    """

    center = np.asarray(center_lps_mm, dtype=np.float64)
    voxel = np.asarray(voxel_size_mm, dtype=np.float64)
    diameter = np.asarray(diameter_mm, dtype=np.float64)
    matrix = np.asarray(volume_shape, dtype=np.int64)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise SparseBalloonError("center_lps_mm must be one finite 3-D point")
    if voxel.shape != (3,) or np.any(~np.isfinite(voxel)) or np.any(voxel <= 0):
        raise SparseBalloonError("voxel_size_mm must contain three positive values")
    if diameter.shape != (3,) or np.any(~np.isfinite(diameter)) or np.any(diameter <= 0):
        raise SparseBalloonError("diameter_mm must contain three positive values")
    if shape not in {"sphere", "ellipsoid"}:
        raise SparseBalloonError("shape must be 'sphere' or 'ellipsoid'")
    if shape == "sphere" and not np.allclose(diameter, diameter[0]):
        raise SparseBalloonError("sphere diameter_mm values must be equal")
    if (
        not isinstance(boundary_samples_per_axis, int)
        or isinstance(boundary_samples_per_axis, bool)
        or boundary_samples_per_axis < 1
    ):
        raise SparseBalloonError(
            "boundary_samples_per_axis must be a positive integer"
        )

    origin = centered_origin_lps_mm(volume_shape, voxel_size_mm)
    radius = diameter / 2.0
    # Include every voxel cell intersected by the ellipsoid, not only voxel
    # centres that lie inside it.
    lower = np.ceil(
        (center - radius - origin) / voxel - 0.5
    ).astype(np.int64)
    upper = np.floor(
        (center + radius - origin) / voxel + 0.5
    ).astype(np.int64)
    lower = np.maximum(lower, 0)
    upper = np.minimum(upper, matrix - 1)
    if np.any(lower > upper):
        raise SparseBalloonError("balloon does not intersect the XCAT volume")

    axes = [np.arange(lower[i], upper[i] + 1) for i in range(3)]
    mesh = np.meshgrid(*axes, indexing="ij")
    indices = np.column_stack([axis.reshape(-1) for axis in mesh])
    positions = origin + indices * voxel
    distance = np.abs(positions - center)
    half_voxel = voxel / 2.0
    nearest = np.maximum(distance - half_voxel, 0.0)
    farthest = distance + half_voxel
    nearest_radius_squared = np.sum(
        (nearest / radius) ** 2, axis=1
    )
    farthest_radius_squared = np.sum(
        (farthest / radius) ** 2, axis=1
    )
    intersects = nearest_radius_squared < 1.0
    completely_inside = farthest_radius_squared <= 1.0
    occupancy = np.zeros(len(indices), dtype=np.float32)
    occupancy[completely_inside] = 1.0

    boundary_indices = np.flatnonzero(intersects & ~completely_inside)
    if len(boundary_indices):
        samples = (
            (np.arange(boundary_samples_per_axis, dtype=np.float64) + 0.5)
            / boundary_samples_per_axis
            - 0.5
        )
        offset_mesh = np.meshgrid(
            *(samples * voxel[axis] for axis in range(3)), indexing="ij"
        )
        offsets = np.column_stack(
            [component.reshape(-1) for component in offset_mesh]
        )
        boundary_centres = positions[boundary_indices]
        for start in range(0, len(boundary_indices), 1024):
            stop = min(start + 1024, len(boundary_indices))
            sample_positions = (
                boundary_centres[start:stop, None, :] + offsets[None, :, :]
            )
            normalized_radius_squared = np.sum(
                ((sample_positions - center) / radius) ** 2, axis=2
            )
            occupancy[boundary_indices[start:stop]] = np.mean(
                normalized_radius_squared <= 1.0,
                axis=1,
                dtype=np.float64,
            ).astype(np.float32)

    local_shape = tuple(int(len(axis)) for axis in axes)
    occupancy = occupancy.reshape(local_shape)
    if not np.any(occupancy > 0.0):
        raise SparseBalloonError("balloon has zero sampled occupancy")
    return SparseBalloonSupport(
        center_lps_mm=center.copy(),
        bounding_box_start_ijk=np.asarray(lower, dtype=np.int32),
        occupancy=occupancy,
        volume_shape=tuple(int(value) for value in matrix),
        voxel_size_mm=tuple(float(value) for value in voxel),
        origin_lps_mm=origin,
    )
