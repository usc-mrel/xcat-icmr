from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from xcat_icmr.encoding import (
    DEFAULT_NUFFT_OVERSAMPLING,
    NufftBackendError,
    SigpyNufftBackend,
    circular_shift_to_rf_center,
    encode_multicoil_frame,
    matrix_shape_for_fov,
    measure_centered_signal_support,
    prepare_encoding_grids,
    prepare_physical_sigpy_trajectory,
    prepare_sigpy_trajectory,
    run_multicoil_nufft_debug,
    run_reduced_nufft_validation,
    scale_isotropic_trajectory_to_resolution,
    validate_sigpy_reference,
)


def test_default_nufft_oversampling_is_one_point_five() -> None:
    backend = SigpyNufftBackend(device_id=-1)

    assert DEFAULT_NUFFT_OVERSAMPLING == 1.5
    assert backend.oversampling == 1.5


def test_rf_center_shift_moves_signal_in_negative_local_direction() -> None:
    image = np.zeros((7, 5, 3), dtype=np.float32)
    image[5, 2, 1] = 1

    shifted, applied = circular_shift_to_rf_center(
        image,
        center_shift_mm=2.0,
        voxel_size_mm=1.0,
        logical_axis=0,
    )

    assert applied == -2
    assert shifted[3, 2, 1] == 1
    assert np.count_nonzero(shifted) == 1


def test_rf_center_shift_rejects_fractional_high_resolution_voxel() -> None:
    with pytest.raises(ValueError, match="integer number"):
        circular_shift_to_rf_center(
            np.zeros((3, 3, 3), dtype=np.float32),
            center_shift_mm=1.5,
            voxel_size_mm=1.0,
            logical_axis=0,
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


def test_physical_trajectory_uses_k_times_fov_and_arm_major_order() -> None:
    kx = np.array([[1, 2], [3, 4]], dtype=np.float64)
    ky = 10 * kx
    kz = -kx

    trajectory = prepare_physical_sigpy_trajectory(
        kx,
        ky,
        kz,
        fov_mm=(500, 200, 100),
        matrix_shape=(8, 16, 4),
    )

    np.testing.assert_allclose(
        trajectory.coordinates,
        np.column_stack(
            (
                np.array([1, 3, 2, 4]) * 0.5,
                np.array([10, 30, 20, 40]) * 0.2,
                -np.array([1, 3, 2, 4]) * 0.1,
            )
        ),
    )
    np.testing.assert_allclose(
        trajectory.maximum_absolute_coordinate, (2.0, 8.0, 0.4)
    )


def test_support_measurement_uses_centered_bounds_and_margin() -> None:
    image = np.zeros((20, 30, 40), dtype=np.float32)
    image[4:16, 10:21, 15:26] = 2

    support = measure_centered_signal_support(
        image,
        voxel_size_mm=(1, 1, 1),
        threshold_fraction=0.01,
        margin_mm=2,
        fov_rounding_mm=(2, 2, 2),
    )

    assert support.bbox_min_mm == (-6.0, -5.0, -5.0)
    assert support.bbox_max_mm == (5.0, 5.0, 5.0)
    assert support.extent_mm == (12.0, 11.0, 11.0)
    assert support.derived_fov_mm == (16.0, 14.0, 14.0)


def test_fov_matrix_is_nominal_and_physical_nyquist_safe() -> None:
    matrix = matrix_shape_for_fov(
        (500, 100, 250),
        (3.5, 3.5, 3.5),
        (143.96, 145.17, 145.15),
    )

    assert matrix == (144, 30, 73)


def test_encoding_grids_keep_padded_fov_at_lower_resolution() -> None:
    grids = prepare_encoding_grids(
        ground_truth_shape=(500, 500, 500),
        ground_truth_voxel_size_mm=(1.0, 1.0, 1.0),
        sequence_resolution_mm=np.array([3.5]),
    )

    assert grids.acquisition_fov_mm == (500.0, 500.0, 500.0)
    assert grids.acquisition_matrix_shape == (144, 144, 144)
    assert grids.reconstruction_fov_mm == (500.0, 500.0, 500.0)
    assert grids.reconstruction_matrix_shape == (144, 144, 144)


def test_trajectory_scale_is_set_by_requested_resolution() -> None:
    kx = np.array([[0.0], [3.0]])
    ky = np.array([[0.0], [4.0]])
    kz = np.zeros_like(kx)

    scaled, factor, target = scale_isotropic_trajectory_to_resolution(
        kx,
        ky,
        kz,
        resolution_mm=2.0,
    )

    assert target == 250.0
    assert factor == 50.0
    np.testing.assert_allclose(
        np.sqrt(scaled[0] ** 2 + scaled[1] ** 2 + scaled[2] ** 2).max(),
        250.0,
    )


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
    assert saved["adjoint_magnitude_center"].shape == (4, 4)
    assert saved["adjoint"].shape == (4, 4, 2)
    assert float(saved["nufft_oversampling"].item()) == 1.5
    assert float(saved["rf_center_shift_mm"].item()) == 0.0
    assert int(saved["applied_circular_shift_voxels"].item()) == 0
    np.testing.assert_array_equal(
        saved["acquisition_matrix_shape"].squeeze(), [4, 4, 2]
    )
    np.testing.assert_array_equal(
        saved["reconstruction_matrix_shape"].squeeze(), [4, 4, 2]
    )
    assert report.full_adjoint_saved


def test_multicoil_forward_saves_each_coil_and_rss_adjoint(
    tmp_path: Path,
) -> None:
    shape = (6, 6, 6)
    image = np.zeros(shape, dtype=np.float32)
    image[2:5, 2:5, 2:5] = 1
    coils = [
        np.ones(shape, dtype=np.complex64),
        np.full(shape, 1j, dtype=np.complex64),
    ]
    samples, arms = 4, 2
    kx = np.linspace(-10, 10, samples)[:, None] * np.ones((1, arms))
    ky = np.zeros_like(kx)
    kz = np.zeros_like(kx)
    grids = prepare_encoding_grids(
        ground_truth_shape=shape,
        ground_truth_voxel_size_mm=(1, 1, 1),
        sequence_resolution_mm=np.array([2]),
    )

    report = run_multicoil_nufft_debug(
        image,
        coil_count=2,
        coil_loader=lambda index: coils[index],
        kx_per_m=kx,
        ky_per_m=ky,
        kz_per_m=kz,
        density_compensation=np.ones_like(kx),
        encoding_grids=grids,
        output_path=tmp_path / "all_coils.mat",
        shifted_ground_truth_output_path=tmp_path / "shifted_gt.mat",
        rf_center_shift_mm=1.0,
        rf_axis_voxel_size_mm=1.0,
        rf_logical_axis=0,
        trajectory_scale_factor=1.0,
        target_kmax_per_m=250.0,
    )

    saved = loadmat(report.output_path)
    assert saved["kspace"].shape == (samples, arms, 2)
    assert saved["adjoint_coils"].shape == (4, 4, 4, 2)
    assert saved["adjoint_rss"].shape == (4, 4, 4)
    assert report.applied_circular_shift_voxels == -1
    assert report.nonfinite_value_count == 0


def test_reusable_multicoil_frame_can_skip_debug_adjoint() -> None:
    shape = (4, 4, 4)
    image = np.ones(shape, dtype=np.float32)
    samples, arms = 3, 2
    kx = np.linspace(-10, 10, samples)[:, None] * np.ones((1, arms))
    zeros = np.zeros_like(kx)
    grids = prepare_encoding_grids(
        ground_truth_shape=shape,
        ground_truth_voxel_size_mm=(1, 1, 1),
        sequence_resolution_mm=np.array([2]),
    )

    result = encode_multicoil_frame(
        image,
        coil_count=1,
        coil_loader=lambda _: np.ones(shape, dtype=np.complex64),
        kx_per_m=kx,
        ky_per_m=zeros,
        kz_per_m=zeros,
        density_compensation=np.ones_like(kx),
        encoding_grids=grids,
        rf_center_shift_mm=0.0,
        rf_axis_voxel_size_mm=1.0,
        rf_logical_axis=0,
        compute_adjoint=False,
    )

    assert result.kspace.shape == (samples, arms, 1)
    assert result.adjoint_coils is None
    assert result.adjoint_rss is None
