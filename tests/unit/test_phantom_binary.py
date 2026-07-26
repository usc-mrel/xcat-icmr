from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from xcat_icmr.config.models import SimulationConfig
from xcat_icmr.phantom.binary import XcatBinaryReadError, open_xcat_binary
from xcat_icmr.phantom.matlab_labels import (
    XcatLabelComparisonError,
    compare_xcat_labels_to_matlab,
)


FIXTURE = Path(__file__).parents[1] / "fixtures" / "valid_simulation.yaml"


def make_config() -> SimulationConfig:
    return SimulationConfig.model_validate(
        yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    )


def write_binary(path: Path, logical: np.ndarray) -> None:
    logical.astype("<f4").ravel(order="F").tofile(path)


def make_logical(config: SimulationConfig) -> np.ndarray:
    matrix = config.phantom.matrix_size_xy
    slices = config.phantom.slice_range.end - config.phantom.slice_range.start + 1
    values = np.arange(matrix * matrix * slices, dtype=np.int64) % 72
    return values.reshape((matrix, matrix, slices), order="F").astype(np.float32)


def test_memory_maps_matlab_column_major_shape_and_crop(tmp_path: Path) -> None:
    config = make_config()
    logical = make_logical(config)
    path = tmp_path / "frame.bin"
    write_binary(path, logical)

    volume = open_xcat_binary(config, path)

    assert isinstance(volume.raw, np.memmap)
    assert volume.raw_shape == logical.shape
    assert volume.cropped_shape == (64, 64, 11)
    np.testing.assert_array_equal(volume.raw, logical)
    np.testing.assert_array_equal(volume.cropped, logical)


def test_rejects_binary_with_wrong_size(tmp_path: Path) -> None:
    path = tmp_path / "short.bin"
    path.write_bytes(b"\0" * 4)

    with pytest.raises(XcatBinaryReadError, match="expected"):
        open_xcat_binary(make_config(), path)


def test_compares_reversed_matlab_hdf5_axes_exactly(tmp_path: Path) -> None:
    config = make_config()
    logical = make_logical(config)
    binary = tmp_path / "frame.bin"
    reference = tmp_path / "reference.mat"
    write_binary(binary, logical)
    with h5py.File(reference, "w") as handle:
        handle.create_dataset("P", data=logical.transpose(2, 1, 0))

    report = compare_xcat_labels_to_matlab(
        open_xcat_binary(config, binary),
        reference,
        chunk_slices=3,
    )

    assert report.passed
    assert report.mismatch_count == 0
    assert report.unique_labels == tuple(range(72))


def test_detects_matlab_label_difference(tmp_path: Path) -> None:
    config = make_config()
    logical = make_logical(config)
    stored = logical.transpose(2, 1, 0).copy()
    stored[0, 0, 0] += 1
    binary = tmp_path / "frame.bin"
    reference = tmp_path / "reference.mat"
    write_binary(binary, logical)
    with h5py.File(reference, "w") as handle:
        handle.create_dataset("P", data=stored)

    report = compare_xcat_labels_to_matlab(
        open_xcat_binary(config, binary), reference
    )

    assert not report.passed
    assert report.mismatch_count == 1
    assert report.max_abs_error == 1


def test_rejects_invalid_xcat_label_during_comparison(tmp_path: Path) -> None:
    config = make_config()
    logical = make_logical(config)
    logical[0, 0, 0] = 72
    binary = tmp_path / "frame.bin"
    reference = tmp_path / "reference.mat"
    write_binary(binary, logical)
    with h5py.File(reference, "w") as handle:
        handle.create_dataset("P", data=logical.transpose(2, 1, 0))

    with pytest.raises(XcatLabelComparisonError, match="outside"):
        compare_xcat_labels_to_matlab(
            open_xcat_binary(config, binary), reference
        )
