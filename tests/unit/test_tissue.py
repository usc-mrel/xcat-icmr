from __future__ import annotations

import numpy as np
import pytest

from xcat_icmr.tissue import (
    LEGACY_MATLAB_055T,
    LabelMappingError,
    XcatLabel,
    get_tissue_library,
    map_labels_to_tissue_properties,
)


def test_defines_complete_xcat_label_range() -> None:
    assert {int(label) for label in XcatLabel} == set(range(72))


def test_resolves_yaml_library_name() -> None:
    assert get_tissue_library("legacy-matlab-0p55t") is LEGACY_MATLAB_055T


def test_preserves_matlab_tissue_groups_and_properties() -> None:
    blood = LEGACY_MATLAB_055T.group_for_label(XcatLabel.ARTERY)
    kidney = LEGACY_MATLAB_055T.group_for_label(XcatLabel.PANCREAS)
    brain = LEGACY_MATLAB_055T.group_for_label(XcatLabel.BRAIN)

    assert blood is not None
    assert blood.name == "Blood"
    assert blood.properties.t1_ms == 1122.0
    assert blood.properties.t2_ms == 263.0
    assert blood.properties.proton_density_percent == 95.0
    assert kidney is not None and kidney.name == "Kidney"
    assert brain is None


def test_maps_label_volume_vectorially() -> None:
    labels = np.array(
        [
            [XcatLabel.OUTSIDE, XcatLabel.LEFT_VENTRICLE_MYOCARDIUM],
            [XcatLabel.LEFT_VENTRICLE_CHAMBER, XcatLabel.LIVER],
        ],
        dtype=np.float32,
    )

    result = map_labels_to_tissue_properties(labels, LEGACY_MATLAB_055T)

    assert result.t1_ms.shape == labels.shape
    assert result.t1_ms.dtype == np.float32
    np.testing.assert_array_equal(result.t1_ms, [[0, 701], [1122, 339]])
    np.testing.assert_array_equal(result.t2_ms, [[0, 58], [263, 66]])
    np.testing.assert_array_equal(
        result.proton_density_percent, [[0, 80], [95, 90]]
    )
    np.testing.assert_array_equal(result.mapped, np.ones_like(labels, dtype=bool))


def test_known_unassigned_label_is_explicit() -> None:
    result = map_labels_to_tissue_properties(
        [XcatLabel.BRAIN], LEGACY_MATLAB_055T
    )

    assert result.t1_ms[0] == 0
    assert result.t2_ms[0] == 0
    assert result.proton_density_percent[0] == 0
    assert not result.mapped[0]


def test_rejects_out_of_range_labels_by_default() -> None:
    with pytest.raises(LabelMappingError, match="outside the supported"):
        map_labels_to_tissue_properties([0, 72], LEGACY_MATLAB_055T)


def test_can_zero_fill_out_of_range_labels_explicitly() -> None:
    result = map_labels_to_tissue_properties(
        [72], LEGACY_MATLAB_055T, unknown="zero"
    )

    assert result.t1_ms[0] == 0
    assert not result.mapped[0]


@pytest.mark.parametrize("labels", ([1.5], [np.nan], ["muscle"]))
def test_rejects_malformed_labels(labels: list[object]) -> None:
    with pytest.raises(LabelMappingError):
        map_labels_to_tissue_properties(labels, LEGACY_MATLAB_055T)
