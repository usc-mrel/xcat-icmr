from __future__ import annotations

import numpy as np

from xcat_icmr.sequence.orientation import (
    build_coordinate_transforms,
    map_spatial_indices,
    reorient_spatial_array,
    transform_vector_components,
)


def test_hfs_cor_coordinate_chain_matches_declared_axes() -> None:
    transforms = build_coordinate_transforms(
        patient_position="HFS",
        coordinate_mode="XYZ-in-TRA",
        sequence_orientation="COR",
    )

    np.testing.assert_array_equal(
        transforms.pcs_to_dcs,
        np.diag((1, -1, -1)),
    )
    np.testing.assert_array_equal(
        transforms.logical_to_dcs,
        np.asarray(((0, 1, 0), (0, 0, 1), (1, 0, 0))),
    )
    np.testing.assert_array_equal(
        transforms.pcs_to_logical,
        np.asarray(((0, 0, -1), (1, 0, 0), (0, -1, 0))),
    )
    assert transforms.logical_axis_patient_directions == (
        "-Tra",
        "+Sag",
        "-Cor",
    )


def test_phase_is_invariant_across_pcs_dcs_and_logical_frames() -> None:
    transforms = build_coordinate_transforms(
        patient_position="HFS",
        coordinate_mode="XYZ-in-TRA",
        sequence_orientation="COR",
    )
    rng = np.random.default_rng(42)
    for _ in range(20):
        position_pcs = rng.normal(size=3)
        k_logical = rng.normal(size=3)
        position_dcs = transforms.pcs_to_dcs @ position_pcs
        position_logical = transforms.dcs_to_logical @ position_dcs
        k_dcs = transforms.logical_to_dcs @ k_logical

        np.testing.assert_allclose(
            np.dot(k_logical, position_logical),
            np.dot(k_dcs, position_dcs),
            atol=1e-12,
        )


def test_reorientation_keeps_even_grid_zero_sample_fixed() -> None:
    transforms = build_coordinate_transforms(
        patient_position="HFS",
        coordinate_mode="XYZ-in-TRA",
        sequence_orientation="COR",
    )
    source = np.zeros((4, 6, 8), dtype=np.float32)
    source[2, 3, 4] = 7

    logical = reorient_spatial_array(
        source, transforms.pcs_to_logical
    )

    assert logical.shape == (8, 4, 6)
    assert logical[4, 2, 3] == 7
    assert np.count_nonzero(logical) == 1


def test_cor_component_mapping_preserves_logical_arrays() -> None:
    transforms = build_coordinate_transforms(
        patient_position="HFS",
        coordinate_mode="XYZ-in-TRA",
        sequence_orientation="COR",
    )
    kx = np.asarray([1.0, 2.0])
    ky = np.asarray([3.0, 4.0])
    kz = np.asarray([5.0, 6.0])

    dcs = transform_vector_components(
        transforms.logical_to_dcs, kx, ky, kz
    )

    np.testing.assert_array_equal(dcs[0], ky)
    np.testing.assert_array_equal(dcs[1], kz)
    np.testing.assert_array_equal(dcs[2], kx)


def test_index_mapping_matches_array_reorientation_and_padding() -> None:
    transforms = build_coordinate_transforms(
        patient_position="HFS",
        coordinate_mode="XYZ-in-TRA",
        sequence_orientation="COR",
    )
    source_shape = (4, 6, 8)
    target_shape = (12, 12, 12)
    source_indices = np.asarray(((2, 3, 4), (3, 4, 6)), dtype=np.int32)
    mapped, valid = map_spatial_indices(
        source_indices,
        source_shape=source_shape,
        source_to_target=transforms.pcs_to_logical,
        target_shape=target_shape,
    )

    source = np.zeros(source_shape, dtype=np.int32)
    for value, index in enumerate(source_indices, start=1):
        source[tuple(index)] = value
    oriented = reorient_spatial_array(source, transforms.pcs_to_logical)
    padding = tuple(
        ((target - size) // 2, target - size - (target - size) // 2)
        for size, target in zip(oriented.shape, target_shape, strict=True)
    )
    padded = np.pad(oriented, padding)

    assert np.all(valid)
    for value, index in enumerate(mapped, start=1):
        assert padded[tuple(index)] == value
