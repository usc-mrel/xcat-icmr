from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from xcat_icmr.cli import _prepare_resumed_xcat_invocation, build_parser
from xcat_icmr.config.models import SimulationConfig
from xcat_icmr.phantom import (
    XcatExecutionError,
    execute_streaming_xcat_invocation,
    execute_xcat_invocation,
    expected_xcat_binary_bytes,
    plan_xcat_frames,
    prepare_xcat_parameter_file,
)
from xcat_icmr.phantom.runner import (
    build_xcat_invocation,
    format_xcat_preflight,
    preflight_xcat_invocation,
)


FIXTURE = Path(__file__).parents[1] / "fixtures" / "valid_simulation.yaml"


def make_runtime(tmp_path: Path) -> SimulationConfig:
    data = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    config = SimulationConfig.model_validate(data)
    config.run.output_root = tmp_path / "outputs"

    runtime = tmp_path / "xcat-runtime"
    runtime.mkdir()
    executable = runtime / "dxcat"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    for name in (
        "vmale50_heart.nrb",
        "vmale50.nrb",
        "heart_curve.txt",
        "diaphragm_curve.dat",
        "ap_curve.dat",
    ):
        (runtime / name).touch()
    template = runtime / "template.par"
    template.write_text(
        "\n".join(
            (
                "heart_base = vmale50_heart.nrb",
                "organ_file = vmale50.nrb",
                "heart_curve_file = heart_curve.txt",
                "dia_filename = diaphragm_curve.dat",
                "ap_filename = ap_curve.dat",
                "out_frames = 1",
                "",
            )
        ),
        encoding="utf-8",
    )
    config.resources.xcat.executable = executable
    config.resources.xcat.parameter_template = template
    return config


def prepare(
    config: SimulationConfig,
    *,
    debug_one_frame: bool = True,
):
    parameters = prepare_xcat_parameter_file(
        config, debug_one_frame=debug_one_frame
    )
    frames = plan_xcat_frames(
        config, debug_one_frame=debug_one_frame
    )
    return parameters, frames


def test_builds_non_shell_command_with_xcat_working_directory(
    tmp_path: Path,
) -> None:
    config = make_runtime(tmp_path)
    parameters, frames = prepare(config)

    invocation = build_xcat_invocation(config, parameters, frames)

    assert invocation.working_directory == config.resources.xcat.executable.parent
    assert invocation.command == (
        str(config.resources.xcat.executable),
        str(parameters.output_path),
        "--phan_rotx",
        "0",
        "--phan_roty",
        "0",
        "--phan_rotz",
        "0",
        str(frames.output_prefix),
    )


def test_preflight_passes_for_new_output(tmp_path: Path) -> None:
    config = make_runtime(tmp_path)
    parameters, frames = prepare(config)

    report = preflight_xcat_invocation(config, parameters, frames)

    assert report.passed
    assert report.output_state == "new"
    assert "Would execute: no (--dry-run)" in format_xcat_preflight(report)


def test_preflight_treats_complete_labels_as_reusable(tmp_path: Path) -> None:
    config = make_runtime(tmp_path)
    parameters, frames = prepare(config)
    label = frames.frames[0].label_path
    assert label is not None
    label.parent.mkdir(parents=True)
    label.touch()
    frames = plan_xcat_frames(config)

    report = preflight_xcat_invocation(config, parameters, frames)

    assert report.passed
    assert report.output_state == "complete"


def test_preflight_rejects_partial_full_cycle(tmp_path: Path) -> None:
    config = make_runtime(tmp_path)
    parameters, frames = prepare(config, debug_one_frame=False)
    first_binary = frames.frames[0].binary_path
    first_binary.parent.mkdir(parents=True)
    first_binary.touch()
    frames = plan_xcat_frames(config, debug_one_frame=False)

    report = preflight_xcat_invocation(config, parameters, frames)

    assert not report.passed
    assert report.output_state == "partial"


def test_streaming_preflight_accepts_resumable_partial_cycle(
    tmp_path: Path,
) -> None:
    config = make_runtime(tmp_path)
    parameters, frames = prepare(config, debug_one_frame=False)
    label = frames.frames[0].label_path
    assert label is not None
    label.parent.mkdir(parents=True)
    label.touch()
    frames = plan_xcat_frames(config, debug_one_frame=False)

    report = preflight_xcat_invocation(
        config,
        parameters,
        frames,
        allow_partial_outputs=True,
    )

    assert report.passed
    assert report.output_state == "partial"


def test_preflight_reports_missing_runtime_asset(tmp_path: Path) -> None:
    config = make_runtime(tmp_path)
    parameters, frames = prepare(config)
    (config.resources.xcat.executable.parent / "heart_curve.txt").unlink()

    report = preflight_xcat_invocation(config, parameters, frames)

    assert not report.passed
    missing = next(
        check
        for check in report.checks
        if check.name == "runtime asset: heart_curve_file"
    )
    assert not missing.passed


def set_fake_executable(config: SimulationConfig, script: str) -> None:
    executable = config.resources.xcat.executable
    executable.write_text(script, encoding="utf-8")
    executable.chmod(0o755)


def test_execution_writes_logs_record_and_exact_binary(tmp_path: Path) -> None:
    config = make_runtime(tmp_path)
    expected = expected_xcat_binary_bytes(config)
    set_fake_executable(
        config,
        (
            "#!/bin/sh\n"
            "for arg in \"$@\"; do prefix=\"$arg\"; done\n"
            f"dd if=/dev/zero of=\"${{prefix}}_act_1.bin\" "
            f"bs={expected} count=1 status=none\n"
            "echo generated\n"
        ),
    )
    parameters, frames = prepare(config)
    report = preflight_xcat_invocation(config, parameters, frames)

    result = execute_xcat_invocation(config, frames, report)

    assert result.status == "executed"
    assert result.return_code == 0
    assert result.binary_sizes == ((frames.frames[0].binary_path, expected),)
    assert result.stdout_log is not None and result.stdout_log.is_file()
    assert result.stderr_log is not None and result.stderr_log.is_file()
    assert (
        result.invocation_record is not None
        and result.invocation_record.is_file()
    )


def test_execution_rejects_truncated_binary(tmp_path: Path) -> None:
    config = make_runtime(tmp_path)
    set_fake_executable(
        config,
        (
            "#!/bin/sh\n"
            "for arg in \"$@\"; do prefix=\"$arg\"; done\n"
            "printf bad > \"${prefix}_act_1.bin\"\n"
        ),
    )
    parameters, frames = prepare(config)
    report = preflight_xcat_invocation(config, parameters, frames)

    with pytest.raises(XcatExecutionError, match="wrong size"):
        execute_xcat_invocation(config, frames, report)


def test_execution_reports_nonzero_exit(tmp_path: Path) -> None:
    config = make_runtime(tmp_path)
    set_fake_executable(config, "#!/bin/sh\nexit 7\n")
    parameters, frames = prepare(config)
    report = preflight_xcat_invocation(config, parameters, frames)

    with pytest.raises(XcatExecutionError, match="return code 7"):
        execute_xcat_invocation(config, frames, report)


def test_execution_reuses_complete_binary_without_subprocess(
    tmp_path: Path,
) -> None:
    config = make_runtime(tmp_path)
    parameters, frames = prepare(config)
    binary = frames.frames[0].binary_path
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"\0" * expected_xcat_binary_bytes(config))
    frames = plan_xcat_frames(config)
    report = preflight_xcat_invocation(config, parameters, frames)

    result = execute_xcat_invocation(config, frames, report)

    assert result.status == "reused"
    assert result.return_code is None
    assert result.stdout_log is None


def test_streaming_execution_consumes_and_removes_each_frame(
    tmp_path: Path,
) -> None:
    config = make_runtime(tmp_path)
    config.timeline.xcat_time_step_s = 0.5
    config.timeline.kspace_time_step_s = 0.5
    expected = expected_xcat_binary_bytes(config)
    set_fake_executable(
        config,
        (
            "#!/bin/sh\n"
            "for arg in \"$@\"; do prefix=\"$arg\"; done\n"
            f"dd if=/dev/zero of=\"${{prefix}}_act_1.bin\" "
            f"bs={expected} count=1 status=none\n"
            "sleep 0.1\n"
            f"dd if=/dev/zero of=\"${{prefix}}_act_2.bin\" "
            f"bs={expected} count=1 status=none\n"
        ),
    )
    parameters, frames = prepare(config, debug_one_frame=False)
    report = preflight_xcat_invocation(
        config,
        parameters,
        frames,
        allow_partial_outputs=True,
    )
    consumed = []

    def consume(frame) -> None:
        assert frame.binary_path.stat().st_size == expected
        assert frame.label_path is not None
        frame.label_path.parent.mkdir(parents=True, exist_ok=True)
        frame.label_path.touch()
        frame.binary_path.unlink()
        consumed.append(frame.index)

    result = execute_streaming_xcat_invocation(
        config,
        frames,
        report,
        consume,
        poll_interval_s=0.01,
    )

    assert consumed == [1, 2]
    assert result.consumed_frame_count == 2
    assert not any(frame.binary_path.exists() for frame in frames.frames)
    assert all(
        frame.label_path is not None and frame.label_path.exists()
        for frame in frames.frames
    )


def test_streaming_execution_refuses_partial_leftover(
    tmp_path: Path,
) -> None:
    config = make_runtime(tmp_path)
    parameters, frames = prepare(config)
    binary = frames.frames[0].binary_path
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"partial")
    frames = plan_xcat_frames(config)
    report = preflight_xcat_invocation(
        config,
        parameters,
        frames,
        allow_partial_outputs=True,
    )

    with pytest.raises(XcatExecutionError, match="partial raw frame"):
        execute_streaming_xcat_invocation(
            config,
            frames,
            report,
            lambda frame: None,
        )


def test_streaming_force_generate_bypasses_existing_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_runtime(tmp_path)
    parameters, frames = prepare(config)
    assert frames.frames[0].label_path is not None
    frames.frames[0].label_path.parent.mkdir(parents=True, exist_ok=True)
    frames.frames[0].label_path.touch()
    report = preflight_xcat_invocation(
        config,
        parameters,
        frames,
        allow_partial_outputs=True,
    )

    def launched(*args, **kwargs):
        raise RuntimeError("XCAT launch attempted")

    monkeypatch.setattr("xcat_icmr.phantom.runner.subprocess.Popen", launched)

    with pytest.raises(RuntimeError, match="XCAT launch attempted"):
        execute_streaming_xcat_invocation(
            config,
            frames,
            report,
            lambda frame: None,
            force_generate=True,
        )


def test_resume_plan_starts_at_missing_global_phase(
    tmp_path: Path,
) -> None:
    config = make_runtime(tmp_path)
    parameters, frames = prepare(config, debug_one_frame=False)

    resumed_parameters, resumed_frames = _prepare_resumed_xcat_invocation(
        config,
        parameters,
        frames,
        first_missing_zero_based=100,
    )

    assert resumed_parameters.parameters["out_frames"] == 100
    assert resumed_parameters.parameters["hrt_start_ph_index"] == 0.5
    assert resumed_parameters.parameters["resp_start_ph_index"] == pytest.approx(0.4)
    assert len(resumed_frames.frames) == 100
    assert resumed_frames.frames[0].index == 101
    assert resumed_frames.frames[-1].index == 200
    assert resumed_frames.frames[0].binary_path.name.endswith("_act_1.bin")
    assert resumed_frames.frames[-1].binary_path.name.endswith("_act_100.bin")


def test_free_breathing_resume_compensates_xcat_heart_time_offset(
    tmp_path: Path,
) -> None:
    config = make_runtime(tmp_path)
    config.phantom.motion.mode = "free-breathing"
    config.phantom.motion.respiratory.breaths_per_minute = 12
    parameters, frames = prepare(config, debug_one_frame=False)

    resumed_parameters, resumed_frames = _prepare_resumed_xcat_invocation(
        config,
        parameters,
        frames,
        first_missing_zero_based=100,
    )

    assert resumed_parameters.parameters["out_frames"] == 900
    assert resumed_parameters.parameters["hrt_start_ph_index"] == 0.5
    assert resumed_parameters.parameters["resp_start_ph_index"] == pytest.approx(0.4)
    assert resumed_frames.frames[0].index == 101
    assert resumed_frames.frames[-1].index == 1000


def test_dynamic_cycle_cli_accepts_regeneration_start_frame() -> None:
    args = build_parser().parse_args(
        [
            "generate-dynamic-cycle",
            "simulation.yaml",
            "--regenerate-from-frame",
            "102",
        ]
    )

    assert args.regenerate_from_frame == 102
