from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from xcat_icmr.cli import main
from xcat_icmr.config.loader import load_config
from xcat_icmr.config.models import SimulationConfig
from xcat_icmr.config.validation import format_summary, validate_paths


FIXTURE = Path(__file__).parents[1] / "fixtures" / "valid_simulation.yaml"


def fixture_data() -> dict:
    return yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))


def write_config(tmp_path: Path, data: dict) -> Path:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    path = config_dir / "simulation.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def create_required_resources(config_path: Path) -> None:
    base = config_path.parent
    xcat_dir = base / "resources" / "xcat"
    sequence_dir = base / "resources" / "sequence"
    metadata_dir = base / "resources" / "metadata"
    xcat_dir.mkdir(parents=True)
    sequence_dir.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)

    executable = xcat_dir / "dxcat"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    (xcat_dir / "template.par").write_text("template\n", encoding="utf-8")
    (sequence_dir / "test.seq").write_text("sequence\n", encoding="utf-8")


def assert_invalid(data: dict, expected_text: str) -> None:
    with pytest.raises(ValidationError) as exc_info:
        SimulationConfig.model_validate(data)
    assert expected_text in str(exc_info.value)


def test_loads_valid_configuration_and_resolves_paths(tmp_path: Path) -> None:
    path = write_config(tmp_path, fixture_data())
    config = load_config(path)

    assert config.run.output_root == path.parent / "outputs" / "test-run"
    assert config.sequence.folder == path.parent / "resources" / "sequence"
    assert config.sequence.resolved_file == (
        path.parent / "resources" / "sequence" / "test.seq"
    )
    assert config.timeline.xcat_frames_per_kspace_frame == 10


def test_expands_environment_variables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = fixture_data()
    monkeypatch.setenv("XCAT_ICMR_TEST_ROOT", str(tmp_path / "external"))
    data["sequence"]["folder"] = "$XCAT_ICMR_TEST_ROOT/sequence"
    path = write_config(tmp_path, data)

    config = load_config(path)

    assert config.sequence.folder == tmp_path / "external" / "sequence"


def test_resolves_covariance_file_relative_to_yaml(tmp_path: Path) -> None:
    data = fixture_data()
    data["noise"]["coil_covariance"] = "resources/noise/covariance.npy"
    path = write_config(tmp_path, data)

    config = load_config(path)

    assert config.noise.coil_covariance == (
        path.parent / "resources" / "noise" / "covariance.npy"
    )


def test_rejects_unknown_fields() -> None:
    data = fixture_data()
    data["run"]["identifier"] = "typo"
    assert_invalid(data, "Extra inputs are not permitted")


def test_free_breathing_requires_frequency() -> None:
    data = fixture_data()
    data["phantom"]["motion"]["mode"] = "free-breathing"
    data["phantom"]["motion"]["respiratory"]["breaths_per_minute"] = None
    assert_invalid(data, "breaths_per_minute is required")


def test_kspace_step_must_be_integer_multiple() -> None:
    data = fixture_data()
    data["timeline"]["kspace_time_step_s"] = 0.047
    assert_invalid(data, "integer multiple")


def test_enabled_balloon_requires_control_points() -> None:
    data = fixture_data()
    data["intervention"]["gd_balloon"]["enabled"] = True
    data["timeline"]["duration_s"] = "auto"
    assert_invalid(data, "control_points_file is required")


def test_auto_duration_requires_balloon() -> None:
    data = fixture_data()
    data["timeline"]["duration_s"] = "auto"
    assert_invalid(data, "requires the Gd balloon to be enabled")


def test_enabled_coils_require_sensitivity_map() -> None:
    data = fixture_data()
    data["coils"]["enabled"] = True
    assert_invalid(data, "sensitivity_map is required")


def test_enabled_off_resonance_requires_field_map() -> None:
    data = fixture_data()
    data["scanner"]["effects"]["off_resonance"]["enabled"] = True
    assert_invalid(data, "field_map is required")


def test_path_validation_passes_for_existing_resources(tmp_path: Path) -> None:
    path = write_config(tmp_path, fixture_data())
    create_required_resources(path)
    config = load_config(path)

    assert validate_paths(config) == []


def test_path_validation_reports_missing_sequence(tmp_path: Path) -> None:
    path = write_config(tmp_path, fixture_data())
    create_required_resources(path)
    (path.parent / "resources" / "sequence" / "test.seq").unlink()
    config = load_config(path)

    issues = validate_paths(config)
    assert any(issue.field == "sequence.file" for issue in issues)


def test_summary_contains_resolved_timing() -> None:
    config = SimulationConfig.model_validate(fixture_data())
    summary = format_summary(config)

    assert "XCAT time step:" in summary
    assert "5 ms" in summary
    assert "XCAT frames/k-space frame:" in summary
    assert "10" in summary


def test_validate_cli_success(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = write_config(tmp_path, fixture_data())
    create_required_resources(path)

    assert main(["validate", str(path)]) == 0
    captured = capsys.readouterr()
    assert "Configuration is valid." in captured.out
    assert captured.err == ""


def test_validate_cli_failure(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data = deepcopy(fixture_data())
    data["compute"]["device_id"] = -2
    path = write_config(tmp_path, data)

    assert main(["validate", str(path)]) == 2
    captured = capsys.readouterr()
    assert "compute.device_id" in captured.err
