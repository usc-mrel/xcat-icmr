from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy.io import savemat

from xcat_icmr.config.models import ContrastConfig, SequenceConfig
from xcat_icmr.sequence.matlab_reference import compare_to_matlab
from xcat_icmr.sequence.reader import (
    SequenceReadError,
    read_pulseq_signature,
    read_sequence,
)


SIGNATURE = "0123456789abcdef0123456789abcdef"


def write_sequence(path: Path, signature: str = SIGNATURE) -> None:
    path.write_text(
        "\n".join(
            (
                "# Pulseq sequence file",
                "[VERSION]",
                "major 1",
                "[SIGNATURE]",
                "Type md5",
                f"Hash {signature}",
                "",
            )
        ),
        encoding="utf-8",
    )


def make_config(tmp_path: Path, orientation: str = "COR") -> SequenceConfig:
    sequence_dir = tmp_path / "sequences"
    metadata_dir = tmp_path / "metadata"
    sequence_dir.mkdir()
    metadata_dir.mkdir()
    write_sequence(sequence_dir / "test.seq")

    param = {
        "fov": np.array([0.28, 0.20, 0.16]),
        "spatial_resolution": 0.0025,
        "FA": 65,
        "TE": 0.8,
        "TR": 5.2,
        "interleaves": 2,
        "planes": 3,
        "pre_discard": 1,
        "dt": 2e-6,
    }
    base = np.arange(12, dtype=np.float64).reshape(3, 4)
    savemat(
        metadata_dir / f"{SIGNATURE}.mat",
        {
            "param": param,
            "kx": base,
            "ky": base + 100,
            "kz": base + 200,
            "w": base + 300,
        },
    )
    return SequenceConfig(
        folder=sequence_dir,
        file=Path("test.seq"),
        metadata_directory=metadata_dir,
        coordinate_mode="XYZ-in-TRA",
        orientation=orientation,
        rf_direction="LR",
        contrast=ContrastConfig(model="bssfp", tissue_library="test"),
    )


def test_reads_signature(tmp_path: Path) -> None:
    path = tmp_path / "signature.seq"
    write_sequence(path)
    assert read_pulseq_signature(path) == ("md5", SIGNATURE)


def test_rejects_invalid_signature(tmp_path: Path) -> None:
    path = tmp_path / "invalid.seq"
    write_sequence(path, "not-an-md5")

    with pytest.raises(SequenceReadError, match="invalid Pulseq MD5"):
        read_pulseq_signature(path)


def test_preserves_logical_metadata_and_derives_cor_dcs_mapping(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path, orientation="COR")
    data = read_sequence(config)
    base = np.arange(12, dtype=np.float64).reshape(3, 4)

    np.testing.assert_array_equal(data.logical_kx, base)
    np.testing.assert_array_equal(data.logical_ky, base + 100)
    np.testing.assert_array_equal(data.logical_kz, base + 200)
    np.testing.assert_array_equal(data.dcs_kx, base + 100)
    np.testing.assert_array_equal(data.dcs_ky, base + 200)
    np.testing.assert_array_equal(data.dcs_kz, base)
    np.testing.assert_allclose(data.fov_mm, [280, 200, 160])
    np.testing.assert_allclose(data.resolution_mm, [2.5])
    assert data.trajectory_shape == (3, 4)
    assert data.flip_angle_deg == 65
    assert data.te_ms == 0.8
    assert data.tr_ms == 5.2


def test_matlab_reference_comparison_is_exact(tmp_path: Path) -> None:
    data = read_sequence(make_config(tmp_path, orientation="COR"))
    reference = tmp_path / "reference.mat"

    with h5py.File(reference, "w") as handle:
        seq = handle.create_group("par/seq_params")
        seq.create_dataset("FA", data=np.asarray([[data.flip_angle_deg]]))
        seq.create_dataset("TE", data=np.asarray([[data.te_ms]]))
        seq.create_dataset("TR", data=np.asarray([[data.tr_ms]]))
        seq.create_dataset("FOV", data=data.fov_mm[:, None])
        seq.create_dataset("res", data=data.resolution_mm[:, None])
        seq.create_dataset("kx", data=data.dcs_kx.T)
        seq.create_dataset("ky", data=data.dcs_ky.T)
        seq.create_dataset("kz", data=data.dcs_kz.T)
        metadata = seq.create_group("metadata")
        metadata.create_dataset("w", data=data.density_compensation.T)

    report = compare_to_matlab(data, reference)

    assert report.passed
    assert all(item.max_abs_error == 0 for item in report.items)
