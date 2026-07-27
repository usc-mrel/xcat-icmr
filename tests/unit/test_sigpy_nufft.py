from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from xcat_icmr.encoding import (
    NufftBackendError,
    SigpyNufftBackend,
    prepare_encoding_grids,
    prepare_sigpy_trajectory,
    run_reduced_nufft_validation,
    validate_sigpy_reference,
)


def test_trajectory_scaling_and_arm_major_order() -> None:
    kx = np.array([[1, 2], [3, 4]], dtype=np.float64)
    ky = 10 * kx
    kz = 20 * kx

    trajectory = prepare_sigpy_trajectory(
        kx,
        ky,
        kz,
        matrix_shape=(20, 40, 60),
    )

    # Each complete trajectory axis is scaled independently to matrix_size/2.
    # Samples remain contiguous within each arm.
    np.testing.assert_allclose(
        trajectory.coordinates,
        np.column_stack(
            (
                np.array([1, 3, 2, 4]) * 2.5,
                np.array([10, 30, 20, 40]) * 0.5,
                np.array([20, 60, 40, 80]) * 0.375,
            )
        ),
    )
    assert trajectory.maximum_absolute_coordinate == (10.0, 20.0, 30.0)
    restored = trajectory.reshape_kspace(np.arange(4))
    np.testing.assert_array_equal(restored, [[0, 2], [1, 3]])


def test_trajectory_uses_full_axes_for_scaling_before_arm_selection() -> None:
    kx = np.array([[1, 4], [2, 8]], dtype=np.float64)
    ky = np.zeros_like(kx)
    kz = -kx

    trajectory = prepare_sigpy_trajectory(
        kx,
        ky,
        kz,
        matrix_shape=(16, 12, 20),
        arm_indices=np.array([0]),
    )

    np.testing.assert_allclose(
        trajectory.coordinates,
        [[1.0, 0.0, -1.25], [2.0, 0.0, -2.5]],
    )
    assert trajectory.source_maximum_absolute_k_per_m == (8.0, 0.0, 8.0)


def test_encoding_grids_keep_padded_fov_at_lower_resolution() -> None:
    grids = prepare_encoding_grids(
        ground_truth_shape=(500, 500, 500),
        ground_truth_voxel_size_mm=(1.0, 1.0, 1.0),
        sequence_resolution_mm=np.array([3.5]),
    )

    assert grids.acquisition_fov_mm == (500.0, 500.0, 500.0)
    assert grids.acquisition_matrix_shape == (142, 142, 142)
    assert grids.reconstruction_fov_mm == (500.0, 500.0, 500.0)
    assert grids.reconstruction_matrix_shape == (142, 142, 142)


def test_sigpy_matches_direct_dft_and_is_a_paired_adjoint() -> None:
    report = validate_sigpy_reference()

    assert report.passed
    assert report.direct_dft_relative_error < 1e-2
    assert report.impulse_magnitude_spread < 1e-3
    assert report.adjoint_relative_error < 1e-5


def test_gpu_request_without_cupy_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "xcat_icmr.encoding.sigpy_backend.importlib.util.find_spec",
        lambda name: None,
    )
    with pytest.raises(NufftBackendError, match="CuPy"):
        SigpyNufftBackend(device_id=0)


def test_reduced_forward_and_adjoint_save_debug_data(
    tmp_path: Path,
) -> None:
    shape = (8, 6, 4)
    image = np.zeros(shape, dtype=np.float32)
    image[2:6, 2:5, 1:3] = 2
    coil = np.ones(shape, dtype=np.complex64)
    samples, arms = 5, 3
    kx = np.linspace(-1, 1, samples)[:, None] * np.ones((1, arms))
    ky = np.zeros_like(kx)
    kz = np.zeros_like(kx)
    dcf = np.ones_like(kx)
    destination = tmp_path / "debug.mat"
    grids = prepare_encoding_grids(
        ground_truth_shape=shape,
        ground_truth_voxel_size_mm=(1, 1, 1),
        sequence_resolution_mm=np.array([2]),
    )

    report = run_reduced_nufft_validation(
        image,
        coil,
        kx_per_m=kx,
        ky_per_m=ky,
        kz_per_m=kz,
        density_compensation=dcf,
        encoding_grids=grids,
        coil_index=0,
        arm_count=2,
        output_path=destination,
        save_full_adjoint=True,
    )

    saved = loadmat(destination)
    assert report.kspace_shape == (samples, 2)
    assert report.nonfinite_value_count == 0
    assert saved["kspace"].shape == (samples, 2)
    assert saved["acquisition_coordinates"].shape == (samples * 2, 3)
    assert saved["reconstruction_coordinates"].shape == (samples * 2, 3)
    assert saved["adjoint_magnitude_center"].shape == (4, 3)
    assert saved["adjoint"].shape == (4, 3, 2)
    np.testing.assert_array_equal(
        saved["acquisition_matrix_shape"].squeeze(), [4, 3, 2]
    )
    np.testing.assert_array_equal(
        saved["reconstruction_matrix_shape"].squeeze(), [4, 3, 2]
    )
    assert report.full_adjoint_saved
