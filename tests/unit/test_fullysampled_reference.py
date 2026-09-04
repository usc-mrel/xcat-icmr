from __future__ import annotations

import numpy as np

from xcat_icmr.encoding.fullysampled_reference import (
    centered_resize,
    reconstruction_shape_for_trajectory,
)


def test_centered_resize_preserves_sampled_zero_for_crop_and_pad() -> None:
    source = np.zeros((6, 5, 4), dtype=np.float32)
    source[3, 2, 2] = 7.0

    cropped = centered_resize(source, (4, 3, 2))
    padded = centered_resize(source, (8, 7, 6))

    assert cropped[2, 1, 1] == 7.0
    assert padded[4, 3, 3] == 7.0


def test_requested_fov_matrix_contains_actual_trajectory_extent() -> None:
    shape = reconstruction_shape_for_trajectory(
        (500.0, 150.0, 250.0),
        3.5,
        (141.65221449, 142.84308594, 142.82072281),
    )

    assert shape == (143, 43, 72)


def test_normalized_sensitivity_combination_has_no_denominator() -> None:
    sensitivities = np.asarray(
        [1.0 / np.sqrt(2.0), 1j / np.sqrt(2.0)], dtype=np.complex64
    )[:, None, None, None]
    expected = np.full((2, 2, 2), 3.0 + 2.0j, dtype=np.complex64)
    adjoints = sensitivities * expected[None, ...]

    combined = np.sum(
        np.conj(sensitivities) * adjoints,
        axis=0,
        dtype=np.complex64,
    )

    np.testing.assert_allclose(combined, expected, rtol=1e-6, atol=1e-6)
