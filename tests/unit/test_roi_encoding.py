from __future__ import annotations

import numpy as np

from xcat_icmr.encoding.sigpy_backend import SigpyNufftBackend
from xcat_icmr.encoding.trajectory import prepare_physical_sigpy_trajectory
from xcat_icmr.encoding.validation import circular_shift_to_rf_center
from xcat_icmr.intervention.roi_encoding import (
    PersistentSparseRoiEncoder,
    encode_sparse_logical_delta_roi,
)


def test_roi_nufft_matches_shifted_full_grid_forward() -> None:
    rng = np.random.default_rng(11)
    global_shape = (32, 30, 28)
    indices = np.asarray(
        ((14, 13, 12), (15, 13, 12), (14, 14, 13), (16, 15, 12)),
        dtype=np.int32,
    )
    values = np.asarray((1.0, -0.3, 0.6, 0.2), dtype=np.float32)
    coil = np.exp(
        1j
        * rng.normal(scale=0.05, size=global_shape).astype(np.float32)
    ).astype(np.complex64)
    kx = rng.uniform(-80.0, 80.0, size=(17, 13))
    ky = rng.uniform(-80.0, 80.0, size=(17, 13))
    kz = rng.uniform(-80.0, 80.0, size=(17, 13))
    result = encode_sparse_logical_delta_roi(
        indices,
        values,
        global_shape=global_shape,
        voxel_size_mm=(1.0, 1.0, 1.0),
        coil_count=1,
        coil_roi_loader=lambda _coil, slices: coil[slices],
        kx_per_m=kx,
        ky_per_m=ky,
        kz_per_m=kz,
        acquisition_matrix_shape=global_shape,
        rf_center_shift_mm=3.0,
        rf_axis_voxel_size_mm=1.0,
        rf_logical_axis=1,
        device_id=-1,
        minimum_roi_shape=12,
        roi_margin_voxels=2,
    )

    full = np.zeros(global_shape, dtype=np.float32)
    np.add.at(full, tuple(indices.T), values)
    weighted, shift = circular_shift_to_rf_center(
        np.asarray(full * coil, dtype=np.complex64),
        center_shift_mm=3.0,
        voxel_size_mm=1.0,
        logical_axis=1,
    )
    trajectory = prepare_physical_sigpy_trajectory(
        kx,
        ky,
        kz,
        fov_mm=tuple(float(value) for value in global_shape),
        matrix_shape=global_shape,
    )
    expected = trajectory.reshape_kspace(
        SigpyNufftBackend(device_id=-1).forward(weighted, trajectory)
    )
    relative_error = np.linalg.norm(result.kspace[:, :, 0] - expected) / np.linalg.norm(
        expected
    )

    assert result.applied_shift_voxels == shift
    assert relative_error < 0.01


def test_persistent_roi_encoder_matches_existing_encoder() -> None:
    rng = np.random.default_rng(19)
    shape = (20, 18, 16)
    indices = np.asarray(
        ((8, 7, 6), (9, 7, 6), (9, 8, 7), (10, 8, 7)),
        dtype=np.int32,
    )
    values = np.asarray((0.2, 1.0, 0.5, -0.1), dtype=np.float32)
    coils = (
        rng.normal(size=(2,) + shape)
        + 1j * rng.normal(size=(2,) + shape)
    ).astype(np.complex64)
    kx = rng.uniform(-60.0, 60.0, size=(11, 3))
    ky = rng.uniform(-60.0, 60.0, size=(11, 3))
    kz = rng.uniform(-60.0, 60.0, size=(11, 3))
    loader = lambda coil, slices: coils[(coil,) + slices]
    expected = encode_sparse_logical_delta_roi(
        indices,
        values,
        global_shape=shape,
        voxel_size_mm=(1.0, 1.0, 1.0),
        coil_count=2,
        coil_roi_loader=loader,
        kx_per_m=kx[:, 1:2],
        ky_per_m=ky[:, 1:2],
        kz_per_m=kz[:, 1:2],
        acquisition_matrix_shape=shape,
        rf_center_shift_mm=2.0,
        rf_axis_voxel_size_mm=1.0,
        rf_logical_axis=2,
        device_id=-1,
        minimum_roi_shape=10,
        roi_margin_voxels=2,
    )
    encoder = PersistentSparseRoiEncoder(
        global_shape=shape,
        voxel_size_mm=(1.0, 1.0, 1.0),
        coil_count=2,
        coil_roi_loader=loader,
        kx_per_m=kx,
        ky_per_m=ky,
        kz_per_m=kz,
        acquisition_matrix_shape=shape,
        rf_center_shift_mm=2.0,
        rf_axis_voxel_size_mm=1.0,
        rf_logical_axis=2,
        device_id=-1,
    )
    actual = encoder.encode(
        indices,
        values,
        trajectory_tr=1,
        minimum_roi_shape=10,
        roi_margin_voxels=2,
    )
    np.testing.assert_allclose(actual.kspace, expected.kspace, rtol=1e-6, atol=1e-6)

    full = encoder.encode_full(
        indices,
        values,
        minimum_roi_shape=10,
        roi_margin_voxels=2,
    )
    assert full.kspace.shape == (11, 3, 2)
    np.testing.assert_allclose(
        full.kspace[:, 1:2, :], actual.kspace, rtol=1e-6, atol=1e-6
    )
