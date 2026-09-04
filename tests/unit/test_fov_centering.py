import numpy as np

from xcat_icmr.encoding.fov_centering import (
    center_kspace_on_rf_profile,
    rf_centering_phase_ramp,
)


def test_rf_centering_phase_uses_known_positive_shift() -> None:
    k = np.asarray([[0.0, 25.0], [-50.0, 100.0]])
    ramp = rf_centering_phase_ramp(k, rf_center_shift_mm=15.0)
    expected = np.exp(2j * np.pi * k * 0.015).astype(np.complex64)
    np.testing.assert_allclose(ramp, expected, atol=1e-7, rtol=1e-7)


def test_centering_applies_same_phase_to_every_coil() -> None:
    k = np.asarray([[0.0, 1.0], [2.0, 3.0]])
    source = np.ones((2, 2, 3), dtype=np.complex64)
    result = center_kspace_on_rf_profile(
        source, k, rf_center_shift_mm=15.0
    )
    expected = rf_centering_phase_ramp(k, rf_center_shift_mm=15.0)
    for coil in range(3):
        np.testing.assert_allclose(result[:, :, coil], expected)


def test_positive_rf_center_is_moved_to_image_origin() -> None:
    size = 8
    spacing_m = 1e-3
    positions_m = (np.arange(size) - size // 2) * spacing_m
    k_per_m = np.fft.fftfreq(size, d=spacing_m)
    source = np.zeros(size, dtype=np.complex64)
    source[size // 2 + 2] = 1.0  # object at the +2 mm RF center
    centered = np.roll(source, -2)
    forward = np.exp(
        -2j * np.pi * k_per_m[:, None] * positions_m[None, :]
    )
    original_kspace = forward @ source
    expected_centered_kspace = forward @ centered
    result = center_kspace_on_rf_profile(
        original_kspace[:, None],
        k_per_m[:, None],
        rf_center_shift_mm=2.0,
    )[:, 0]
    np.testing.assert_allclose(
        result, expected_centered_kspace, atol=1e-6, rtol=1e-6
    )
