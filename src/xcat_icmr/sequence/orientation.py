"""Explicit coordinate-frame transforms between XCAT, scanner, and Pulseq."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class OrientationTransformError(ValueError):
    """Raised when a coordinate-frame transform is invalid or unsupported."""


_PCS_TO_DCS = {
    # PCS component order is [Sag, Cor, Tra].
    # HFS: [X_DCS, Y_DCS, Z_DCS] = [Sag, -Cor, -Tra].
    "HFS": np.diag((1.0, -1.0, -1.0)),
}

_LOGICAL_TO_DCS_XYZ_IN_TRA = {
    # Scanner-derived mappings preserved from MATLAB config.m.
    # Rows are [X_DCS, Y_DCS, Z_DCS]; columns are Pulseq [x, y, z].
    "TRA": np.asarray(
        ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    ),
    "COR": np.asarray(
        ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))
    ),
    "SAG": np.asarray(
        ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0))
    ),
}


def _validate_signed_permutation(matrix: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (3, 3):
        raise OrientationTransformError(f"{name} must have shape (3, 3)")
    if not np.all(np.isin(values, (-1.0, 0.0, 1.0))):
        raise OrientationTransformError(
            f"{name} must be a signed permutation matrix"
        )
    if not np.array_equal(values @ values.T, np.eye(3)):
        raise OrientationTransformError(f"{name} must be orthogonal")
    values.setflags(write=False)
    return values


@dataclass(frozen=True)
class CoordinateTransforms:
    """Auditable transforms for one patient position and scanner prescription."""

    patient_position: str
    coordinate_mode: str
    sequence_orientation: str
    pcs_to_dcs: np.ndarray
    logical_to_dcs: np.ndarray
    dcs_to_logical: np.ndarray
    pcs_to_logical: np.ndarray
    logical_axis_patient_directions: tuple[str, str, str]


def _axis_directions(matrix: np.ndarray) -> tuple[str, str, str]:
    source_labels = ("Sag", "Cor", "Tra")
    output = []
    for row in matrix:
        source_axis = int(np.argmax(np.abs(row)))
        prefix = "+" if row[source_axis] > 0 else "-"
        output.append(f"{prefix}{source_labels[source_axis]}")
    return tuple(output)  # type: ignore[return-value]


def build_coordinate_transforms(
    *,
    patient_position: str,
    coordinate_mode: str,
    sequence_orientation: str,
) -> CoordinateTransforms:
    """Build PCS→DCS and Pulseq-logical→DCS signed permutations."""

    pcs_to_dcs = pcs_to_dcs_matrix(patient_position)
    logical_to_dcs = logical_to_dcs_matrix(
        coordinate_mode, sequence_orientation
    )
    dcs_to_logical = _validate_signed_permutation(
        logical_to_dcs.T, "dcs_to_logical"
    )
    pcs_to_logical = _validate_signed_permutation(
        dcs_to_logical @ pcs_to_dcs, "pcs_to_logical"
    )
    return CoordinateTransforms(
        patient_position=patient_position,
        coordinate_mode=coordinate_mode,
        sequence_orientation=sequence_orientation,
        pcs_to_dcs=pcs_to_dcs,
        logical_to_dcs=logical_to_dcs,
        dcs_to_logical=dcs_to_logical,
        pcs_to_logical=pcs_to_logical,
        logical_axis_patient_directions=_axis_directions(pcs_to_logical),
    )


def pcs_to_dcs_matrix(patient_position: str) -> np.ndarray:
    """Return the PCS→DCS signed permutation for a patient position."""

    try:
        pcs_to_dcs = _PCS_TO_DCS[patient_position]
    except KeyError as exc:
        raise OrientationTransformError(
            f"unsupported patient position: {patient_position}"
        ) from exc
    return _validate_signed_permutation(pcs_to_dcs, "pcs_to_dcs")


def logical_to_dcs_matrix(
    coordinate_mode: str,
    sequence_orientation: str,
) -> np.ndarray:
    """Return the Pulseq-logical→DCS scanner-derived mapping."""

    if coordinate_mode != "XYZ-in-TRA":
        raise OrientationTransformError(
            f"unsupported sequence coordinate mode: {coordinate_mode}"
        )
    try:
        logical_to_dcs = _LOGICAL_TO_DCS_XYZ_IN_TRA[
            sequence_orientation
        ]
    except KeyError as exc:
        raise OrientationTransformError(
            f"unsupported sequence orientation: {sequence_orientation}"
        ) from exc
    return _validate_signed_permutation(
        logical_to_dcs, "logical_to_dcs"
    )


def transform_vector_components(
    matrix: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    third: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply a 3×3 component transform to three equally shaped arrays."""

    components = tuple(np.asarray(value) for value in (first, second, third))
    if len({value.shape for value in components}) != 1:
        raise OrientationTransformError(
            "vector-component arrays must have matching shapes"
        )
    transform = _validate_signed_permutation(matrix, "component transform")
    output = []
    for row in transform:
        source_axis = int(np.argmax(np.abs(row)))
        output.append(row[source_axis] * components[source_axis])
    return tuple(output)  # type: ignore[return-value]


def reoriented_spatial_shape(
    shape: tuple[int, int, int],
    source_to_target: np.ndarray,
) -> tuple[int, int, int]:
    """Return target-axis sizes for a signed spatial permutation."""

    if len(shape) != 3:
        raise OrientationTransformError("spatial shape must have three axes")
    transform = _validate_signed_permutation(
        source_to_target, "spatial transform"
    )
    source_axes = np.argmax(np.abs(transform), axis=1)
    return tuple(int(shape[int(axis)]) for axis in source_axes)


def _reverse_axis_preserving_zero(
    array: np.ndarray,
    axis: int,
) -> np.ndarray:
    """Reverse a centered axis while keeping its sampled zero at size//2."""

    reversed_array = np.flip(array, axis=axis)
    if array.shape[axis] % 2:
        return reversed_array

    # For coordinates [-N, ..., N-1], a sign reversal requests +N at the
    # first target sample. +N is outside the sampled source grid. Shift the
    # reversed data by one so zero remains fixed, and zero-fill that edge.
    shifted = np.roll(reversed_array, shift=1, axis=axis)
    edge = [slice(None)] * shifted.ndim
    edge[axis] = 0
    shifted[tuple(edge)] = 0
    return shifted


def reorient_spatial_array(
    array: np.ndarray,
    source_to_target: np.ndarray,
) -> np.ndarray:
    """Reorient the first three array axes using centered-grid conventions."""

    values = np.asarray(array)
    if values.ndim < 3:
        raise OrientationTransformError(
            "a spatial array must have at least three dimensions"
        )
    transform = _validate_signed_permutation(
        source_to_target, "spatial transform"
    )
    source_axes = tuple(
        int(axis) for axis in np.argmax(np.abs(transform), axis=1)
    )
    trailing_axes = tuple(range(3, values.ndim))
    output = np.transpose(values, source_axes + trailing_axes).copy()
    for target_axis, source_axis in enumerate(source_axes):
        if transform[target_axis, source_axis] < 0:
            output = _reverse_axis_preserving_zero(output, target_axis)
    return output


def map_spatial_indices(
    indices: np.ndarray,
    *,
    source_shape: tuple[int, int, int],
    source_to_target: np.ndarray,
    target_shape: tuple[int, int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Map zero-based voxel indices exactly like ``reorient_spatial_array``.

    If ``target_shape`` is supplied, the reoriented coordinates are centered
    in that larger grid using the same padding convention as encoding inputs.
    The returned Boolean mask identifies source samples retained by the
    zero-preserving reversal used for even matrices.
    """

    points = np.asarray(indices, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise OrientationTransformError("indices must have shape (N, 3)")
    shape = np.asarray(source_shape, dtype=np.int64)
    if shape.shape != (3,) or np.any(shape <= 0):
        raise OrientationTransformError(
            "source_shape must contain three positive dimensions"
        )
    if np.any(points < 0) or np.any(points >= shape[None, :]):
        raise OrientationTransformError("indices fall outside source_shape")
    transform = _validate_signed_permutation(
        source_to_target, "spatial transform"
    )
    source_axes = np.argmax(np.abs(transform), axis=1)
    oriented_shape = shape[source_axes]
    mapped = np.empty_like(points)
    valid = np.ones(len(points), dtype=bool)
    for target_axis, source_axis_value in enumerate(source_axes):
        source_axis = int(source_axis_value)
        source_index = points[:, source_axis]
        size = int(shape[source_axis])
        if transform[target_axis, source_axis] > 0:
            target_index = source_index
        elif size % 2:
            target_index = size - 1 - source_index
        else:
            target_index = size - source_index
            valid &= source_index != 0
        mapped[:, target_axis] = target_index
    if target_shape is not None:
        target = np.asarray(target_shape, dtype=np.int64)
        if target.shape != (3,) or np.any(target < oriented_shape):
            raise OrientationTransformError(
                "target_shape must contain the reoriented source grid"
            )
        mapped += ((target - oriented_shape) // 2)[None, :]
    return np.asarray(mapped, dtype=np.int32), valid
