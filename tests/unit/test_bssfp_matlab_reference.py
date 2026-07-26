from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from xcat_icmr.signal import bssfp_signal_from_tissue_properties
from xcat_icmr.signal.matlab_reference import (
    compare_bssfp_to_matlab,
    format_bssfp_matlab_comparison,
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


def write_reference_pair(
    tmp_path: Path,
    *,
    perturb: bool = False,
) -> tuple[Path, Path]:
    labels = np.array(
        [
            [[XcatLabel.OUTSIDE, XcatLabel.LEFT_VENTRICLE_MYOCARDIUM]],
            [[XcatLabel.LEFT_VENTRICLE_CHAMBER, XcatLabel.LIVER]],
            [[XcatLabel.BRAIN, XcatLabel.BONE_MARROW]],
        ],
        dtype=np.float32,
    )
    properties = map_labels_to_tissue_properties(labels, LEGACY_MATLAB_055T)
    image = bssfp_signal_from_tissue_properties(properties, **SEQUENCE)
    if perturb:
        image[1, 0, 0] += 0.1

    label_path = tmp_path / "labels.mat"
    image_path = tmp_path / "image.mat"
    with h5py.File(label_path, "w") as handle:
        handle.create_dataset("P", data=labels)
    with h5py.File(image_path, "w") as handle:
        handle.create_dataset("image", data=image)
    return label_path, image_path


def test_exact_chunked_comparison_passes(tmp_path: Path) -> None:
    labels, image = write_reference_pair(tmp_path)

    report = compare_bssfp_to_matlab(
        labels,
        image,
        LEGACY_MATLAB_055T,
        chunk_slices=2,
        atol=0.0,
        rtol=0.0,
        **SEQUENCE,
    )

    assert report.passed
    assert report.voxel_count == 6
    assert report.max_abs_error == 0.0
    assert {item.tissue for item in report.tissues} == {
        "Air",
        "Blood",
        "BoneMarrow",
        "Liver",
        "Muscle",
        "Unassigned",
    }
    assert "Overall:         PASS" in format_bssfp_matlab_comparison(report)


def test_detects_and_localizes_difference(tmp_path: Path) -> None:
    labels, image = write_reference_pair(tmp_path, perturb=True)

    report = compare_bssfp_to_matlab(
        labels,
        image,
        LEGACY_MATLAB_055T,
        atol=1e-5,
        rtol=1e-6,
        **SEQUENCE,
    )

    assert not report.passed
    assert report.mismatch_count == 1
    blood = next(item for item in report.tissues if item.tissue == "Blood")
    assert blood.mismatch_count == 1
