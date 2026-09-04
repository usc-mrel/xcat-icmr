from __future__ import annotations

import numpy as np

from xcat_icmr.analysis.curved_profile import (
    _disk_offsets,
    _sample_frame,
    map_lps_to_reconstruction_voxels,
)


def test_lps_mapping_includes_rf_recentering_shift() -> None:
    points = np.asarray([[-30.466, 18.681, -64.024]])
    pcs_to_logical = np.asarray(
        [
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )

    voxel, logical_mm = map_lps_to_reconstruction_voxels(
        points,
        pcs_to_logical=pcs_to_logical,
        rf_logical_axis=1,
        rf_center_shift_mm=15.0,
        target_fov_mm=(500.0, 150.0, 250.0),
        reconstruction_shape=(142, 44, 72),
    )

    np.testing.assert_allclose(logical_mm[0], (64.024, -45.466, -18.681))
    np.testing.assert_allclose(voxel[0], (89.182816, 8.663307, 30.619872))


def test_tube_profile_contains_centerline_and_maximum() -> None:
    image = np.zeros((16, 16, 16), dtype=np.float32)
    image[8, 8, 8] = 10.0
    curve = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    normal_one = np.asarray([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    normal_two = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    offsets = _disk_offsets(1.0, 1.0, 8)

    sampled = _sample_frame(
        image,
        curve,
        np.ones(3),
        normal_one,
        normal_two,
        offsets,
    )

    np.testing.assert_allclose(sampled[0], (10.0, 0.0))
    np.testing.assert_allclose(np.max(sampled, axis=0), (10.0, 0.0))
