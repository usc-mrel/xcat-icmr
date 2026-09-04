from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.io import loadmat

from xcat_icmr.config.models import SimulationConfig
from xcat_icmr.phantom.binary import open_xcat_binary
from xcat_icmr.phantom.conversion import (
    XcatLabelConversionError,
    convert_xcat_labels_to_mat,
)


FIXTURE = Path(__file__).parents[1] / "fixtures" / "valid_simulation.yaml"


def make_config() -> SimulationConfig:
    return SimulationConfig.model_validate(
        yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    )


def make_binary(path: Path, config: SimulationConfig) -> np.ndarray:
    shape = (
        config.phantom.matrix_size_xy,
        config.phantom.matrix_size_xy,
        config.phantom.head_foot_slice_range.end
        - config.phantom.head_foot_slice_range.start
        + 1,
    )
    values = np.arange(np.prod(shape), dtype=np.int64) % 72
    labels = values.reshape(shape, order="F").astype(np.float32)
    labels.astype("<f4").ravel(order="F").tofile(path)
    return labels


def test_writes_matlab_p_with_exact_shape_dtype_and_values(
    tmp_path: Path,
) -> None:
    config = make_config()
    binary = tmp_path / "frame.bin"
    expected = make_binary(binary, config)
    destination = tmp_path / "labels" / "frame.mat"

    report = convert_xcat_labels_to_mat(
        open_xcat_binary(config, binary),
        destination,
        chunk_slices=3,
    )

    saved = loadmat(destination, variable_names=["P"])["P"]
    assert report.label_path == destination
    assert report.logical_shape == expected.shape
    assert report.dtype == "uint16"
    assert report.unique_labels == tuple(range(72))
    assert saved.dtype == np.dtype(np.uint16)
    np.testing.assert_array_equal(saved, expected)


def test_does_not_overwrite_without_explicit_permission(
    tmp_path: Path,
) -> None:
    config = make_config()
    binary = tmp_path / "frame.bin"
    make_binary(binary, config)
    destination = tmp_path / "frame.mat"
    destination.write_bytes(b"keep me")

    with pytest.raises(XcatLabelConversionError, match="--overwrite"):
        convert_xcat_labels_to_mat(
            open_xcat_binary(config, binary), destination
        )

    assert destination.read_bytes() == b"keep me"


def test_rejects_invalid_labels_before_writing(tmp_path: Path) -> None:
    config = make_config()
    binary = tmp_path / "frame.bin"
    labels = make_binary(binary, config)
    labels[0, 0, 0] = 72
    labels.astype("<f4").ravel(order="F").tofile(binary)
    destination = tmp_path / "frame.mat"

    with pytest.raises(XcatLabelConversionError, match="outside"):
        convert_xcat_labels_to_mat(
            open_xcat_binary(config, binary), destination
        )

    assert not destination.exists()
