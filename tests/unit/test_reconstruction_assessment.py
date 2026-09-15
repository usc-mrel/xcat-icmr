from __future__ import annotations

import numpy as np

from xcat_icmr.analysis.reconstruction_assessment import (
    _causal_arc_update,
    _curve_ssim,
    _directional_extrema,
    _local_peak_and_fwhm,
    _finite_fwhm_summary,
    _regular_disk_offsets,
)


def test_directional_extrema_ignore_missing_directions() -> None:
    values = np.asarray(((8.0, 10.0, np.nan), (np.nan, np.nan, np.nan)))
    minimum, maximum, ratio = _directional_extrema(values)
    np.testing.assert_allclose(minimum[0], 8.0)
    np.testing.assert_allclose(maximum[0], 10.0)
    np.testing.assert_allclose(ratio[0], 1.25)
    assert np.isnan(minimum[1])
    assert np.isnan(maximum[1])
    assert np.isnan(ratio[1])


def test_curve_ssim_is_one_for_identical_maps() -> None:
    values = np.arange(80, dtype=np.float64).reshape(8, 10)
    assert np.isclose(_curve_ssim(values, values), 1.0)


def test_local_fwhm_uses_interpolated_half_height_crossings() -> None:
    axis = np.arange(-10.0, 10.01, 0.25)
    sigma = 2.0
    profile = np.exp(-(axis**2) / (2.0 * sigma**2))
    peak, width = _local_peak_and_fwhm(profile, axis, 0.0, 8.0)
    expected = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma
    assert abs(peak) < 1e-12
    assert np.isclose(width, expected, atol=0.1)


def test_regular_disk_offsets_stay_inside_radius_and_include_center() -> None:
    offsets = _regular_disk_offsets(10.5, 1.75)
    assert np.any(np.all(np.isclose(offsets, 0.0), axis=1))
    assert np.all(np.linalg.norm(offsets, axis=1) <= 10.5 + 1e-9)


def test_causal_arc_tracker_uses_motion_to_reject_remote_peak() -> None:
    arc = np.arange(0.0, 41.0)
    first = np.exp(-0.5 * ((arc - 10.0) / 2.0) ** 2)
    first_s, _, first_status = _causal_arc_update(first, arc, None, [])
    second = np.exp(-0.5 * ((arc - 12.0) / 2.0) ** 2)
    second += 1.1 * np.exp(-0.5 * ((arc - 35.0) / 1.0) ** 2)
    second_s, _, second_status = _causal_arc_update(
        second, arc, first_s, [2.0]
    )
    assert first_status == "ok"
    assert second_status == "ok"
    assert abs(first_s - 10.0) <= 1.0
    assert abs(second_s - 12.0) <= 1.0


def test_finite_fwhm_summary_ignores_only_missing_widths() -> None:
    summary = _finite_fwhm_summary(
        np.asarray([10.0, np.nan, 12.0, 14.0])
    )

    assert summary["finite_frames"] == 3
    assert summary["median_mm"] == 12.0
    assert summary["mean_mm"] == 12.0
