from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from pydantic import ValidationError
import yaml

from xcat_icmr.acquisition.schema import (
    ACQUISITION_SCHEMA_NAME,
    ACQUISITION_SCHEMA_VERSION,
)
from xcat_icmr.reconstruction import (
    ReconstructionPlanError,
    load_reconstruction_config,
    plan_reconstructions,
    run_reconstruction_plan,
    upsert_reconstruction_indexes,
)
from xcat_icmr.reconstruction.causal_irls import (
    estimate_pca_basis,
    prepare_frame_arrays,
)


def _write_acquisition(path: Path, *, coils: int = 4) -> None:
    experiment = path.parent / "human_experiment"
    with h5py.File(path, "w") as handle:
        handle.attrs["xcat_icmr_schema"] = ACQUISITION_SCHEMA_NAME
        handle.attrs["xcat_icmr_schema_version"] = ACQUISITION_SCHEMA_VERSION
        handle.attrs["xcat_icmr_schema_status"] = "complete"
        handle.attrs["acquisition_id"] = "a" * 16
        handle.attrs["control_points_filename"] = "branch.json"
        handle.attrs["velocity_cm_per_s"] = 0.5
        handle.attrs["experiment_directory"] = str(experiment)
        handle.attrs["run_output_root"] = str(path.parent)
        handle.attrs["reconstruction_shape"] = (2, 2, 2)
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
            "kspace", data=np.zeros((3, 4, coils), dtype=np.complex64)
        )
        handle.create_dataset(
            "trajectory_tr_index_zero_based", data=np.arange(4, dtype=np.int64)
        )
        handle.create_dataset(
            "frame_start_tr_zero_based", data=np.asarray((0, 2), dtype=np.int64)
        )
        handle.create_dataset(
            "frame_stop_tr_exclusive", data=np.asarray((2, 4), dtype=np.int64)
        )
        trajectory = handle.create_group("trajectory")
        trajectory.create_dataset(
            "coordinates", data=np.zeros((3, 4, 3), dtype=np.float32)
        )
        trajectory.create_dataset(
            "density_compensation", data=np.ones((3, 4), dtype=np.float32)
        )
        encoding = handle.create_group("encoding")
        encoding.create_dataset(
            "sensitivity_maps",
            data=np.zeros((2, 2, 2, coils), dtype=np.complex64),
        )
        encoding.create_dataset(
            "sensitivity_complete", data=np.ones(coils, dtype=np.uint8)
        )


def _write_config(
    path: Path,
    acquisition: Path,
    *,
    compression: bool = False,
    virtual_coils: int = 2,
) -> None:
    content = {
        "schema_version": 1,
        "inputs": {"acquisitions": [str(acquisition)]},
        "preprocessing": {
            "coil_compression": {
                "enabled": compression,
                "method": "pca",
                "virtual_coils": virtual_coils,
                "weights_file": None,
            }
        },
        "reconstructions": [
            {
                "method": "causal-irls",
                "parameters": {
                    "outer_iterations": 3,
                    "cg_iterations": 3,
                    "spatial_regularization": 0.0005,
                    "temporal_regularization": 0.005,
                },
            }
        ],
        "compute": {"device_id": -1, "precision": "single"},
        "output": {
            "checkpoint": True,
            "save_complex_image": True,
            "save_magnitude_image": False,
        },
    }
    path.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")


def test_template_loads_without_manual_names() -> None:
    path = Path(__file__).parents[2] / "configs/recon_config.template.yaml"
    config = load_reconstruction_config(path)
    assert not config.inputs.acquisitions
    assert len(config.inputs.simulations) == 1
    assert config.inputs.simulations[0].frame_duration_s == pytest.approx(0.330)
    assert config.inputs.simulations[0].config.is_absolute()
    assert len(config.reconstructions) == 1
    assert config.preprocessing.coil_compression.enabled is False


def test_planner_generates_readable_identity_and_path(tmp_path: Path) -> None:
    acquisition = tmp_path / "acquisition.h5"
    configuration = tmp_path / "recon.yaml"
    _write_acquisition(acquisition)
    _write_config(configuration, acquisition, compression=True)
    config = load_reconstruction_config(configuration)
    plan = plan_reconstructions(config, configuration_path=configuration)
    assert len(plan.jobs) == 1
    job = plan.jobs[0]
    assert job.acquisition.display_label == "branch | 0.5 cm/s | 10 ms | aaaaaaaa"
    assert job.readable_recipe == "o3_cg3_lt0p005_ls0p0005_pca2"
    assert job.recipe_id[:8] in job.output_directory.name
    assert job.output_directory.parent.name == "causal-irls"
    assert job.status == "not-started"


def test_disabled_compression_ignores_inactive_values_in_identity(
    tmp_path: Path,
) -> None:
    acquisition = tmp_path / "acquisition.h5"
    _write_acquisition(acquisition)
    first_path = tmp_path / "first.yaml"
    second_path = tmp_path / "second.yaml"
    _write_config(first_path, acquisition, compression=False, virtual_coils=2)
    _write_config(second_path, acquisition, compression=False, virtual_coils=3)
    first = plan_reconstructions(
        load_reconstruction_config(first_path), configuration_path=first_path
    ).jobs[0]
    second = plan_reconstructions(
        load_reconstruction_config(second_path), configuration_path=second_path
    ).jobs[0]
    assert first.recipe_id == second.recipe_id


def test_compression_cannot_request_more_than_acquired_coils(
    tmp_path: Path,
) -> None:
    acquisition = tmp_path / "acquisition.h5"
    configuration = tmp_path / "recon.yaml"
    _write_acquisition(acquisition, coils=4)
    _write_config(configuration, acquisition, compression=True, virtual_coils=5)
    with pytest.raises(ReconstructionPlanError, match="only 4"):
        plan_reconstructions(
            load_reconstruction_config(configuration),
            configuration_path=configuration,
        )


def test_duplicate_recipes_are_rejected(tmp_path: Path) -> None:
    acquisition = tmp_path / "acquisition.h5"
    configuration = tmp_path / "recon.yaml"
    _write_acquisition(acquisition)
    _write_config(configuration, acquisition)
    content = yaml.safe_load(configuration.read_text(encoding="utf-8"))
    content["reconstructions"].append(content["reconstructions"][0].copy())
    configuration.write_text(
        yaml.safe_dump(content, sort_keys=False), encoding="utf-8"
    )
    with pytest.raises(ValidationError, match="duplicate settings"):
        load_reconstruction_config(configuration)


def test_indexes_are_readable_and_upsert_by_identity(tmp_path: Path) -> None:
    first = {
        "reconstruction_id": "id-1",
        "acquisition_id": "acq",
        "control_points_filename": "branch.json",
        "velocity_cm_per_s": 0.5,
        "temporal_resolution_ms": 275.0,
        "method": "causal-irls",
        "readable_recipe": "o3_cg3",
        "status": "running",
        "latency_s": None,
        "reconstruction_file": None,
        "result_file": "result.json",
    }
    csv_path, json_path = upsert_reconstruction_indexes(tmp_path, first)
    first["status"] = "complete"
    upsert_reconstruction_indexes(tmp_path, first)
    records = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(records) == 1
    assert records[0]["status"] == "complete"
    assert "readable_recipe" in csv_path.read_text(encoding="utf-8")


def test_frame_adapter_preserves_tr_then_sample_order(tmp_path: Path) -> None:
    acquisition = tmp_path / "acquisition.h5"
    _write_acquisition(acquisition, coils=2)
    with h5py.File(acquisition, "r+") as handle:
        values = np.arange(3 * 4 * 2).reshape(3, 4, 2).astype(np.complex64)
        handle["kspace"][:] = values
        coordinates = np.arange(3 * 4 * 3).reshape(3, 4, 3).astype(np.float32)
        handle["trajectory/coordinates"][:] = coordinates
        handle["trajectory_tr_index_zero_based"][:2] = (2, 0)
    with h5py.File(acquisition, "r") as handle:
        kspace, coord, sqrt_dcf = prepare_frame_arrays(handle, 0, None)
    np.testing.assert_array_equal(kspace[0], values[:, :2, 0].T.reshape(-1))
    np.testing.assert_array_equal(
        coord, coordinates[:, (2, 0), :].transpose(1, 0, 2).reshape(-1, 3)
    )
    np.testing.assert_array_equal(sqrt_dcf, np.ones(6, dtype=np.float32))


def test_frame_adapter_supports_repeated_trajectory_arms(tmp_path: Path) -> None:
    acquisition = tmp_path / "acquisition.h5"
    _write_acquisition(acquisition, coils=1)
    with h5py.File(acquisition, "r+") as handle:
        coordinates = np.arange(3 * 4 * 3).reshape(3, 4, 3).astype(np.float32)
        handle["trajectory/coordinates"][:] = coordinates
        handle["trajectory_tr_index_zero_based"][:2] = (3, 3)
    with h5py.File(acquisition, "r") as handle:
        _kspace, coord, _sqrt_dcf = prepare_frame_arrays(handle, 0, None)
    expected = coordinates[:, (3, 3), :].transpose(1, 0, 2).reshape(-1, 3)
    np.testing.assert_array_equal(coord, expected)


def test_pca_basis_is_fixed_and_orthonormal(tmp_path: Path) -> None:
    path = tmp_path / "pca.h5"
    with h5py.File(path, "w") as handle:
        values = np.zeros((5, 7, 3), dtype=np.complex64)
        values[..., 0] = 3
        values[..., 1] = 1
        dataset = handle.create_dataset("kspace", data=values)
        basis = estimate_pca_basis(dataset, 2, tr_chunk=2)
    assert basis.shape == (3, 2)
    np.testing.assert_allclose(
        basis.conj().T @ basis, np.eye(2), atol=1e-6
    )


def test_causal_irls_backend_writes_resumable_result(tmp_path: Path) -> None:
    acquisition = tmp_path / "acquisition.h5"
    configuration = tmp_path / "recon.yaml"
    _write_acquisition(acquisition, coils=1)
    with h5py.File(acquisition, "r+") as handle:
        handle["encoding/sensitivity_maps"][:] = 1
    _write_config(configuration, acquisition)
    content = yaml.safe_load(configuration.read_text(encoding="utf-8"))
    content["reconstructions"][0]["parameters"].update(
        outer_iterations=1, cg_iterations=1
    )
    configuration.write_text(
        yaml.safe_dump(content, sort_keys=False), encoding="utf-8"
    )
    config = load_reconstruction_config(configuration)
    plan = plan_reconstructions(config, configuration_path=configuration)
    outputs = run_reconstruction_plan(plan, config)
    reconstruction = outputs[0] / "reconstruction.h5"
    result = json.loads((outputs[0] / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "complete"
    with h5py.File(reconstruction, "r") as handle:
        assert handle["image_complex"].shape == (2, 2, 2, 2)
        np.testing.assert_array_equal(handle["frame_complete"][:], (1, 1))
