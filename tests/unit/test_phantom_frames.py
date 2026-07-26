from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from xcat_icmr.config.models import SimulationConfig
from xcat_icmr.phantom.frames import (
    XcatFramePlanError,
    format_xcat_frame_plan,
    plan_xcat_frames,
)


FIXTURE = Path(__file__).parents[1] / "fixtures" / "valid_simulation.yaml"


def make_config() -> SimulationConfig:
    return SimulationConfig.model_validate(
        yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    )


def test_debug_plan_contains_one_frame_at_zero() -> None:
    plan = plan_xcat_frames(make_config(), debug_one_frame=True)

    assert plan.motion.cycle_frame_count == 200
    assert len(plan.frames) == 1
    assert plan.time_axis_s == (0.0,)
    assert plan.frames[0].index == 1
    assert plan.frames[0].binary_path.name == "phantom_test-run_act_1.bin"
    assert plan.frames[0].label_path is not None
    assert plan.frames[0].label_path.name == "phantom_test-run_act_1.mat"
    assert plan.output_prefix.name == "phantom_test-run"


def test_full_breath_hold_plan_excludes_repeated_endpoint() -> None:
    plan = plan_xcat_frames(make_config(), debug_one_frame=False)

    assert len(plan.frames) == 200
    assert plan.frames[0].time_s == 0.0
    assert plan.frames[-1].index == 200
    assert plan.frames[-1].time_s == pytest.approx(0.995)
    assert 1.0 not in plan.time_axis_s


def test_free_breathing_plan_covers_common_five_second_cycle() -> None:
    data = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    data["phantom"]["motion"]["mode"] = "free-breathing"
    data["phantom"]["motion"]["respiratory"]["breaths_per_minute"] = 12
    plan = plan_xcat_frames(
        SimulationConfig.model_validate(data),
        debug_one_frame=False,
    )

    assert len(plan.frames) == 1000
    assert plan.frames[-1].time_s == pytest.approx(4.995)
    assert 5.0 not in plan.time_axis_s


def test_no_motion_plan_is_one_frame_even_without_debug() -> None:
    data = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    data["phantom"]["motion"]["mode"] = "no-motion"
    plan = plan_xcat_frames(
        SimulationConfig.model_validate(data),
        debug_one_frame=False,
    )

    assert len(plan.frames) == 1
    assert plan.time_axis_s == (0.0,)


def test_plan_reports_preexisting_files_without_creating_directories(
    tmp_path: Path,
) -> None:
    config = make_config()
    config.run.output_root = tmp_path / "outputs"
    binary = (
        config.run.output_root
        / "xcat"
        / "raw"
        / "phantom_test-run_act_1.bin"
    )
    label = (
        config.run.output_root
        / "xcat"
        / "labels"
        / "phantom_test-run_act_1.mat"
    )
    binary.parent.mkdir(parents=True)
    label.parent.mkdir(parents=True)
    binary.touch()
    label.touch()

    plan = plan_xcat_frames(config)

    assert plan.frames[0].binary_exists
    assert plan.frames[0].label_exists
    assert "[exists]" in format_xcat_frame_plan(plan)


def test_omits_label_path_when_labels_are_disabled() -> None:
    config = make_config()
    config.outputs.save_tissue_labels = False

    plan = plan_xcat_frames(config)

    assert plan.frames[0].label_path is None
    assert "label:  not retained" in format_xcat_frame_plan(plan)


def test_rejects_unsafe_run_identifier() -> None:
    config = make_config()
    config.run.id = "../other-run"

    with pytest.raises(XcatFramePlanError, match="run.id"):
        plan_xcat_frames(config)
