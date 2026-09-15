from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from xcat_icmr.acquisition.schema import (
    ACQUISITION_SCHEMA_NAME,
    ACQUISITION_SCHEMA_VERSION,
    AcquisitionSchemaError,
    inspect_acquisition,
)


def _write_valid_acquisition(path: Path) -> None:
    samples, arms, trs, coils = 3, 4, 4, 2
    image_shape = (2, 2, 2)
    with h5py.File(path, "w") as handle:
        handle.attrs["xcat_icmr_schema"] = ACQUISITION_SCHEMA_NAME
        handle.attrs["xcat_icmr_schema_version"] = ACQUISITION_SCHEMA_VERSION
        handle.attrs["xcat_icmr_schema_status"] = "complete"
        handle.attrs["reconstruction_shape"] = image_shape
        handle.attrs["target_fov_mm"] = (8.0, 8.0, 8.0)
        handle.attrs["reconstruction_voxel_size_mm"] = (4.0, 4.0, 4.0)
        handle.attrs["trs_per_frame"] = 2
        handle.attrs["frame_duration_s"] = 0.01
        handle.attrs["effective_tr_s"] = 0.005
        handle.attrs["pulseq_sequence_filename"] = "test.seq"
        handle.attrs["pulseq_signature"] = "0" * 32
        handle.attrs["nufft_backend"] = "sigpy"
        handle.attrs["nufft_oversampling"] = 1.5
        handle.attrs["nufft_kernel_width"] = 4.0
        handle.create_dataset(
            "kspace", data=np.zeros((samples, trs, coils), dtype=np.complex64)
        )
        handle.create_dataset(
            "trajectory_tr_index_zero_based",
            data=np.arange(trs, dtype=np.int64),
        )
        handle.create_dataset(
            "frame_start_tr_zero_based", data=np.asarray((0, 2), dtype=np.int64)
        )
        handle.create_dataset(
            "frame_stop_tr_exclusive", data=np.asarray((2, 4), dtype=np.int64)
        )
        trajectory = handle.create_group("trajectory")
        trajectory.create_dataset(
            "coordinates", data=np.zeros((samples, arms, 3), dtype=np.float32)
        )
        trajectory.create_dataset(
            "density_compensation", data=np.ones((samples, arms), dtype=np.float32)
        )
        encoding = handle.create_group("encoding")
        encoding.create_dataset(
            "sensitivity_maps",
            data=np.zeros(image_shape + (coils,), dtype=np.complex64),
        )
        encoding.create_dataset(
            "sensitivity_complete", data=np.ones(coils, dtype=np.uint8)
        )


def test_inspect_acquisition_reports_reconstruction_contract(tmp_path: Path) -> None:
    path = tmp_path / "acquisition.h5"
    _write_valid_acquisition(path)
    report = inspect_acquisition(path)
    assert report.schema_name == ACQUISITION_SCHEMA_NAME
    assert report.storage == "materialized"
    assert report.kspace_shape == (3, 4, 2)
    assert report.trajectory_shape == (3, 4, 3)
    assert report.sensitivity_shape == (2, 2, 2, 2)
    assert report.frame_count == 2
    assert report.trs_per_frame == 2


def test_inspect_acquisition_rejects_out_of_range_arm(tmp_path: Path) -> None:
    path = tmp_path / "acquisition.h5"
    _write_valid_acquisition(path)
    with h5py.File(path, "r+") as handle:
        handle["trajectory_tr_index_zero_based"][3] = 4
    with pytest.raises(AcquisitionSchemaError, match="out of range"):
        inspect_acquisition(path)


def test_inspect_acquisition_rejects_incomplete_coils(tmp_path: Path) -> None:
    path = tmp_path / "acquisition.h5"
    _write_valid_acquisition(path)
    with h5py.File(path, "r+") as handle:
        handle["encoding/sensitivity_complete"][1] = 0
    with pytest.raises(AcquisitionSchemaError, match="incomplete"):
        inspect_acquisition(path)
