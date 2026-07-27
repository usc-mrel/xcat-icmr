from __future__ import annotations

import numpy as np
import pytest

from xcat_icmr.signal import (
    BssfpSignalError,
    bssfp_signal,
    bssfp_signal_from_tissue_properties,
)
from xcat_icmr.tissue import (
    LEGACY_MATLAB_055T,
    XcatLabel,
    map_labels_to_tissue_properties,
)


SEQUENCE = {
    "flip_angle_deg": 65.0,
    "te_ms": 0.7695,
    "tr_ms": 5.17,
}


def test_matches_reference_tissue_values_from_matlab_equation() -> None:
    signal = bssfp_signal(
        t1_ms=[701.0, 1122.0, 339.0, 187.0],
        t2_ms=[58.0, 263.0, 66.0, 93.0],
        proton_density_percent=[80.0, 95.0, 90.0, 70.0],
        dtype=np.float64,
        **SEQUENCE,
    )

    np.testing.assert_allclose(
        signal,
        [
            8.90094593259982,
            22.310118814985476,
            19.097806014625746,
            25.032157683238484,
        ],
        rtol=1e-13,
        atol=0.0,
    )


def test_converts_tissue_property_volumes_to_image() -> None:
    labels = np.array(
        [
            [XcatLabel.OUTSIDE, XcatLabel.LEFT_VENTRICLE_MYOCARDIUM],
            [XcatLabel.LEFT_VENTRICLE_CHAMBER, XcatLabel.BRAIN],
        ],
        dtype=np.float32,
    )
    properties = map_labels_to_tissue_properties(labels, LEGACY_MATLAB_055T)

    image = bssfp_signal_from_tissue_properties(properties, **SEQUENCE)

    assert image.shape == labels.shape
    assert image.dtype == np.float32
    assert image[0, 0] == 0.0
    assert image[1, 1] == 0.0
    np.testing.assert_allclose(image[0, 1], 8.900946, rtol=1e-6)
    np.testing.assert_allclose(image[1, 0], 22.31012, rtol=1e-6)


def test_broadcasts_scalar_and_array_properties() -> None:
    signal = bssfp_signal(
        t1_ms=701.0,
        t2_ms=58.0,
        proton_density_percent=np.array([80.0, 40.0]),
        **SEQUENCE,
    )

    assert signal.shape == (2,)
    np.testing.assert_allclose(signal[1], signal[0] / 2.0, rtol=1e-6)


def test_broadcasts_spatially_varying_flip_angle() -> None:
    flips = np.array([0.0, 30.0, 65.0])
    signal = bssfp_signal(
        t1_ms=701.0,
        t2_ms=58.0,
        proton_density_percent=80.0,
        flip_angle_deg=flips,
        te_ms=SEQUENCE["te_ms"],
        tr_ms=SEQUENCE["tr_ms"],
    )

    assert signal.shape == flips.shape
    assert signal[0] == 0
    assert signal[1] > 0
    np.testing.assert_allclose(signal[2], 8.900946, rtol=1e-6)


def test_undefined_zero_relaxation_signal_becomes_zero() -> None:
    signal = bssfp_signal(
        t1_ms=0.0,
        t2_ms=0.0,
        proton_density_percent=0.0,
        **SEQUENCE,
    )

    assert signal.shape == ()
    assert signal == 0.0


def test_off_resonance_is_explicitly_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="off-resonance"):
        bssfp_signal(
            701.0,
            58.0,
            80.0,
            off_resonance_enabled=True,
            **SEQUENCE,
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"flip_angle_deg": -1.0}, "flip_angle_deg"),
        ({"te_ms": -1.0}, "te_ms"),
        ({"tr_ms": 0.0}, "tr_ms"),
        ({"te_ms": 6.0}, "greater than"),
    ),
)
def test_rejects_invalid_sequence_parameters(
    changes: dict[str, float],
    message: str,
) -> None:
    sequence = SEQUENCE | changes

    with pytest.raises(BssfpSignalError, match=message):
        bssfp_signal(701.0, 58.0, 80.0, **sequence)


def test_rejects_negative_or_nonfinite_tissue_properties() -> None:
    with pytest.raises(BssfpSignalError, match="non-negative"):
        bssfp_signal(-1.0, 58.0, 80.0, **SEQUENCE)

    with pytest.raises(BssfpSignalError, match="non-finite"):
        bssfp_signal(701.0, np.nan, 80.0, **SEQUENCE)


def test_rejects_incompatible_property_shapes() -> None:
    with pytest.raises(BssfpSignalError, match="broadcast-compatible"):
        bssfp_signal(
            np.ones((2, 2)),
            np.ones(3),
            80.0,
            **SEQUENCE,
        )
