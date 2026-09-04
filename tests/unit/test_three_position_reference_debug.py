from __future__ import annotations

import numpy as np

from xcat_icmr.intervention.reference_debug import (
    map_pcs_centers_to_reference_grids,
)


def test_center_mapping_follows_reorientation_crop_and_low_resolution() -> None:
    centers = np.asarray(
        [
            [218, 130, 235],
            [228, 140, 245],
        ],
        dtype=np.int32,
    )
    # logical = [-Tra, +Sag, -Cor]
    pcs_to_logical = np.asarray(
        ((0.0, 0.0, -1.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0))
    )

    high, low = map_pcs_centers_to_reference_grids(
        centers,
        pcs_shape=(436, 260, 470),
        pcs_to_logical=pcs_to_logical,
        high_resolution_shape=(500, 150, 250),
        high_resolution_voxel_mm=(1.0, 1.0, 1.0),
        reconstruction_shape=(143, 43, 72),
        reconstruction_voxel_mm=(500 / 143, 150 / 43, 250 / 72),
    )

    np.testing.assert_array_equal(high[0], np.asarray([250, 75, 125]))
    np.testing.assert_array_equal(low[0], np.asarray([71, 21, 36]))
    assert high.shape == (2, 3)
    assert low.shape == (2, 3)
