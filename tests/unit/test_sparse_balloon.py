from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from xcat_icmr.intervention import (
    calculate_sparse_gd_bssfp_signal,
    gd_relaxation_times_ms,
    rasterize_sparse_balloon,
    sample_sparse_flip_angles,
)
from xcat_icmr.tissue.models import TissueProperties


BLOOD = TissueProperties(
    t1_ms=1122.0,
    t2_ms=263.0,
    proton_density_percent=95.0,
)


def test_rasterizes_ten_mm_sphere_as_sparse_voxels() -> None:
    support = rasterize_sparse_balloon(
        np.asarray([0.0, 0.0, 0.0]),
        volume_shape=(20, 20, 20),
        voxel_size_mm=(1.0, 1.0, 1.0),
        diameter_mm=(10.0, 10.0, 10.0),
        shape="sphere",
    )
    assert support.occupancy.ndim == 3
    assert support.voxel_count == np.count_nonzero(support.occupancy)
    assert support.occupied_indices_ijk().shape == (support.voxel_count, 3)
    assert support.occupancy.dtype == np.float32
    assert np.any((support.occupancy > 0.0) & (support.occupancy < 1.0))
    assert support.occupied_volume_mm3 == pytest.approx(
        4.0 / 3.0 * np.pi * 5.0**3,
        rel=2e-3,
    )


def test_subvoxel_translation_preserves_volume_and_moves_smoothly() -> None:
    centres = (
        np.asarray([0.0, 0.0, 0.0]),
        np.asarray([0.1, 0.0, 0.0]),
        np.asarray([0.25, 0.0, 0.0]),
        np.asarray([0.49, 0.0, 0.0]),
    )
    supports = [
        rasterize_sparse_balloon(
            centre,
            volume_shape=(24, 24, 24),
            voxel_size_mm=(1.0, 1.0, 1.0),
            diameter_mm=(10.0, 10.0, 10.0),
            shape="sphere",
        )
        for centre in centres
    ]
    volumes = np.asarray(
        [support.occupied_volume_mm3 for support in supports]
    )
    expected = 4.0 / 3.0 * np.pi * 5.0**3
    assert np.max(np.abs(volumes - expected)) / expected < 2e-3
    assert np.ptp(volumes) / expected < 2e-3

    measured_centres = []
    for support in supports:
        local = np.indices(support.occupancy.shape).reshape(3, -1).T
        indices = local + support.bounding_box_start_ijk[None, :]
        positions = support.origin_lps_mm + indices * np.asarray(
            support.voxel_size_mm
        )
        weights = support.occupancy.reshape(-1).astype(np.float64)
        measured_centres.append(
            np.sum(positions * weights[:, None], axis=0) / np.sum(weights)
        )
    np.testing.assert_allclose(measured_centres, centres, atol=0.015)


def test_gd_relaxation_matches_gen_cath_kspace_convention() -> None:
    t1_ms, t2_ms = gd_relaxation_times_ms(BLOOD, 1.1)
    expected_t1 = 1e3 / (1.0 / 1.122 + 5.2 * 1.1)
    expected_t2 = 1e3 / (1.0 / 0.263 + 7.0 * 1.1)
    assert t1_ms == pytest.approx(expected_t1)
    assert t2_ms == pytest.approx(expected_t2)


def test_samples_spatial_flip_and_calculates_only_sparse_signal(
    tmp_path: Path,
) -> None:
    support = rasterize_sparse_balloon(
        np.asarray([0.0, 0.0, 0.0]),
        volume_shape=(20, 20, 20),
        voxel_size_mm=(1.0, 1.0, 1.0),
        diameter_mm=(4.0, 4.0, 4.0),
        shape="sphere",
    )
    profile_path = tmp_path / "profile.mat"
    profile = np.linspace(10.0, 70.0, 20, dtype=np.float32)
    savemat(
        profile_path,
        {
            "applied_effective_flip_angle_deg": profile[None, :],
            "pcs_axis_zero_based": np.asarray([[0]], dtype=np.int32),
            "pcs_image_shape": np.asarray([[20, 20, 20]], dtype=np.int32),
        },
    )
    flip = sample_sparse_flip_angles(profile_path, support)
    result = calculate_sparse_gd_bssfp_signal(
        support,
        carrier=BLOOD,
        concentration_mM=1.1,
        flip_angle_deg=flip,
        te_ms=0.8,
        tr_ms=5.0,
    )
    assert result.values.shape == support.occupancy.shape
    assert result.values.dtype == np.float32
    assert np.all(np.isfinite(result.values))
    assert np.ptp(result.values) > 0.0
    start = support.bounding_box_start_ijk[0]
    expected = profile[start : start + support.occupancy.shape[0]]
    np.testing.assert_array_equal(flip[:, 0, 0], expected)


def test_gd_signal_uses_carrier_proton_density_scale(tmp_path: Path) -> None:
    support = rasterize_sparse_balloon(
        np.asarray([0.2, 0.0, 0.0]),
        volume_shape=(20, 20, 20),
        voxel_size_mm=(1.0, 1.0, 1.0),
        diameter_mm=(4.0, 4.0, 4.0),
        shape="sphere",
    )
    flip = np.full(support.occupancy.shape, 65.0)
    signal = calculate_sparse_gd_bssfp_signal(
        support,
        carrier=BLOOD,
        concentration_mM=1.1,
        flip_angle_deg=flip,
        te_ms=0.8,
        tr_ms=5.0,
    )
    unit_pd_carrier = TissueProperties(
        t1_ms=BLOOD.t1_ms,
        t2_ms=BLOOD.t2_ms,
        proton_density_percent=1.0,
    )
    unit_signal = calculate_sparse_gd_bssfp_signal(
        support,
        carrier=unit_pd_carrier,
        concentration_mM=1.1,
        flip_angle_deg=flip,
        te_ms=0.8,
        tr_ms=5.0,
    )
    np.testing.assert_allclose(signal.values, 95.0 * unit_signal.values)
