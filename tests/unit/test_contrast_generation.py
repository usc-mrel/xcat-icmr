from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat, savemat

from xcat_icmr.signal import bssfp_signal_from_tissue_properties
from xcat_icmr.signal.generation import (
    ContrastGenerationError,
    generate_bssfp_contrast,
)
from xcat_icmr.tissue import (
    LEGACY_MATLAB_055T,
    XcatLabel,
    map_labels_to_tissue_properties,
)


SEQUENCE = {
    "flip_angle_deg": 65.0,
    "te_ms": 0.7395,
    "tr_ms": 4.97,
}


def write_labels(path: Path) -> np.ndarray:
    labels = np.array(
        [
            [[XcatLabel.OUTSIDE, XcatLabel.LEFT_VENTRICLE_MYOCARDIUM]],
            [[XcatLabel.LEFT_VENTRICLE_CHAMBER, XcatLabel.LIVER]],
            [[XcatLabel.BRAIN, XcatLabel.BONE_MARROW]],
        ],
        dtype=np.float32,
    )
    savemat(path, {"P": labels})
    return labels


def test_generates_and_reopens_exact_float32_bssfp_image(
    tmp_path: Path,
) -> None:
    label_path = tmp_path / "labels.mat"
    image_path = tmp_path / "contrast" / "image.mat"
    labels = write_labels(label_path)
    expected = bssfp_signal_from_tissue_properties(
        map_labels_to_tissue_properties(labels, LEGACY_MATLAB_055T),
        **SEQUENCE,
    )

    report = generate_bssfp_contrast(
        label_path,
        image_path,
        LEGACY_MATLAB_055T,
        expected_shape=labels.shape,
        chunk_slices=1,
        **SEQUENCE,
    )

    saved = loadmat(image_path, variable_names=["image"])["image"]
    assert report.logical_shape == labels.shape
    assert report.dtype == "float32"
    np.testing.assert_array_equal(saved, expected)


def test_rejects_shape_that_differs_from_configuration(
    tmp_path: Path,
) -> None:
    label_path = tmp_path / "labels.mat"
    labels = write_labels(label_path)

    with pytest.raises(ContrastGenerationError, match="configured"):
        generate_bssfp_contrast(
            label_path,
            tmp_path / "image.mat",
            LEGACY_MATLAB_055T,
            expected_shape=(1, 2, 3),
            **SEQUENCE,
        )


def test_does_not_overwrite_without_permission(tmp_path: Path) -> None:
    label_path = tmp_path / "labels.mat"
    image_path = tmp_path / "image.mat"
    labels = write_labels(label_path)
    image_path.write_bytes(b"keep me")

    with pytest.raises(ContrastGenerationError, match="--overwrite"):
        generate_bssfp_contrast(
            label_path,
            image_path,
            LEGACY_MATLAB_055T,
            expected_shape=labels.shape,
            **SEQUENCE,
        )

    assert image_path.read_bytes() == b"keep me"


def test_off_resonance_remains_explicitly_unimplemented(
    tmp_path: Path,
) -> None:
    label_path = tmp_path / "labels.mat"
    labels = write_labels(label_path)

    with pytest.raises(NotImplementedError, match="off-resonance"):
        generate_bssfp_contrast(
            label_path,
            tmp_path / "image.mat",
            LEGACY_MATLAB_055T,
            expected_shape=labels.shape,
            off_resonance_enabled=True,
            **SEQUENCE,
        )
