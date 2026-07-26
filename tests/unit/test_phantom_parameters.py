from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from xcat_icmr.config.models import SimulationConfig
from xcat_icmr.phantom import (
    XcatParameterError,
    build_xcat_parameter_values,
    plan_xcat_motion_cycle,
    prepare_xcat_parameter_file,
    render_xcat_parameter_file,
)


FIXTURE = Path(__file__).parents[1] / "fixtures" / "valid_simulation.yaml"


def make_config() -> SimulationConfig:
    return SimulationConfig.model_validate(
        yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    )


def test_translates_matlab_equivalent_phantom_values() -> None:
    values = build_xcat_parameter_values(
        make_config(), debug_one_frame=False
    )

    assert (values["phan_rotx"], values["phan_roty"], values["phan_rotz"]) == (
        0,
        90,
        0,
    )
    assert values["pixel_width"] == 0.1
    assert values["slice_width"] == 0.1
    assert values["array_size"] == 64
    assert values["startslice"] == 10
    assert values["endslice"] == 20
    assert values["gender"] == 0
    assert values["heart_base"] == "vmale50_heart.nrb"
    assert values["organ_file"] == "vmale50.nrb"
    assert values["papillary_flag"] == 0
    assert values["motion_option"] == 0
    assert values["time_per_frame"] == 0.005
    assert values["out_frames"] == 200
    assert values["hrt_period"] == 1.0
    assert values["resp_period"] == 5.0


def test_free_breathing_uses_both_motion_sources() -> None:
    data = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    data["phantom"]["motion"]["mode"] = "free-breathing"
    data["phantom"]["motion"]["respiratory"]["breaths_per_minute"] = 12
    config = SimulationConfig.model_validate(data)
    plan = plan_xcat_motion_cycle(config, debug_one_frame=False)
    values = build_xcat_parameter_values(
        config, debug_one_frame=False
    )

    assert values["motion_option"] == 2
    assert values["resp_period"] == 5.0
    assert values["max_diaphragm_motion"] == 1.0
    assert plan.cycle_duration_s == 5.0
    assert plan.cycle_frame_count == 1000
    assert plan.generated_frame_count == 1000


def test_no_motion_disables_heart_beating_and_respiratory_amplitudes() -> None:
    data = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    data["phantom"]["motion"]["mode"] = "no-motion"
    config = SimulationConfig.model_validate(data)
    plan = plan_xcat_motion_cycle(config, debug_one_frame=False)
    values = build_xcat_parameter_values(config, debug_one_frame=False)

    assert values["motion_option"] == 1
    assert values["max_diaphragm_motion"] == 0.0
    assert values["max_AP_exp"] == 0.0
    assert plan.cycle_duration_s is None
    assert plan.cycle_frame_count == 1
    assert plan.generated_frame_count == 1


def test_debug_mode_preserves_derived_cycle_but_writes_one_frame() -> None:
    plan = plan_xcat_motion_cycle(make_config(), debug_one_frame=True)

    assert plan.cycle_duration_s == 1.0
    assert plan.cycle_frame_count == 200
    assert plan.generated_frame_count == 1


def test_render_replaces_values_and_appends_missing_parameters() -> None:
    rendered, appended = render_xcat_parameter_file(
        "pixel_width = 0.2 # original\nout_frames = 3\n",
        {"pixel_width": 0.1, "out_frames": 5, "phan_rotx": 90},
    )

    assert "pixel_width = 0.1\t# original" in rendered
    assert "out_frames = 5" in rendered
    assert "phan_rotx = 90" in rendered
    assert appended == ("phan_rotx",)


def test_prepare_preserves_template_and_writes_run_copy(tmp_path: Path) -> None:
    template = tmp_path / "template.par"
    template.write_text(
        "pixel_width = 0.2 # original\nout_frames = 3\n",
        encoding="utf-8",
    )
    output = tmp_path / "run" / "parameters.par"
    config = make_config()
    config.resources.xcat.parameter_template = template

    result = prepare_xcat_parameter_file(
        config,
        output_path=output,
        debug_one_frame=True,
    )

    assert result.output_path == output
    assert result.motion_plan.cycle_frame_count == 200
    assert result.motion_plan.generated_frame_count == 1
    assert result.command_line_parameters == {
        "phan_rotx": 0,
        "phan_roty": 90,
        "phan_rotz": 0,
    }
    assert output.is_file()
    assert "pixel_width = 0.1" in output.read_text(encoding="utf-8")
    assert "out_frames = 1" in output.read_text(encoding="utf-8")
    assert "phan_rotx =" not in output.read_text(encoding="utf-8")
    assert template.read_text(encoding="utf-8").startswith("pixel_width = 0.2")


def test_rejects_non_square_in_plane_voxels() -> None:
    data = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    data["phantom"]["voxel_size_mm"] = [1.0, 1.5, 1.0]

    with pytest.raises(XcatParameterError, match="x and y sizes must match"):
        build_xcat_parameter_values(SimulationConfig.model_validate(data))


def test_rejects_motion_period_off_xcat_time_grid() -> None:
    data = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    data["phantom"]["motion"]["cardiac"]["heart_rate_bpm"] = 61

    with pytest.raises(XcatParameterError, match="integer multiple"):
        plan_xcat_motion_cycle(SimulationConfig.model_validate(data))
