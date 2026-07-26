"""Safe construction and preflight validation of XCAT invocations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
from typing import Literal

from xcat_icmr.config.models import SimulationConfig
from xcat_icmr.phantom.frames import XcatFramePlan
from xcat_icmr.phantom.parameters import XcatParameterFile


OutputState = Literal["new", "complete", "partial"]
ExecutionStatus = Literal["executed", "reused"]


class XcatExecutionError(Exception):
    """Raised when XCAT fails or its binary outputs are invalid."""


@dataclass(frozen=True)
class XcatInvocation:
    """An XCAT command represented without shell interpretation."""

    executable: Path
    working_directory: Path
    parameter_file: Path
    output_prefix: Path
    command: tuple[str, ...]


@dataclass(frozen=True)
class PreflightCheck:
    """One independently reportable invocation precondition."""

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class XcatPreflightReport:
    """Dry-run result for a planned XCAT invocation."""

    invocation: XcatInvocation
    output_state: OutputState
    expected_frame_count: int
    existing_binary_count: int
    existing_label_count: int
    checks: tuple[PreflightCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)


@dataclass(frozen=True)
class XcatExecutionResult:
    """Verified outcome of one XCAT subprocess or output reuse decision."""

    status: ExecutionStatus
    return_code: int | None
    expected_binary_bytes: int
    binary_sizes: tuple[tuple[Path, int], ...]
    stdout_log: Path | None
    stderr_log: Path | None
    invocation_record: Path | None


def build_xcat_invocation(
    config: SimulationConfig,
    parameters: XcatParameterFile,
    frames: XcatFramePlan,
) -> XcatInvocation:
    """Build the argument vector and working directory used by XCAT."""

    executable = config.resources.xcat.executable
    command_parts = [str(executable), str(parameters.output_path)]
    for name, value in parameters.command_line_parameters.items():
        command_parts.extend((f"--{name}", str(value)))
    command_parts.append(str(frames.output_prefix))
    command = tuple(command_parts)
    return XcatInvocation(
        executable=executable,
        working_directory=executable.parent,
        parameter_file=parameters.output_path,
        output_prefix=frames.output_prefix,
        command=command,
    )


def _read_parameter_assignments(path: Path) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        content = raw_line.partition("#")[0].strip()
        if not content or "=" not in content:
            continue
        name, value = content.split("=", maxsplit=1)
        assignments[name.strip()] = value.strip()
    return assignments


def _runtime_asset_checks(invocation: XcatInvocation) -> list[PreflightCheck]:
    try:
        parameters = _read_parameter_assignments(invocation.parameter_file)
    except OSError as exc:
        return [
            PreflightCheck(
                "runtime assets",
                False,
                f"could not inspect parameter file: {exc}",
            )
        ]

    asset_parameters = (
        "heart_base",
        "organ_file",
        "heart_curve_file",
        "dia_filename",
        "ap_filename",
    )
    checks = []
    for parameter in asset_parameters:
        value = parameters.get(parameter)
        if value is None:
            checks.append(
                PreflightCheck(
                    f"runtime asset: {parameter}",
                    False,
                    "parameter is missing",
                )
            )
            continue
        path = Path(value)
        if not path.is_absolute():
            path = invocation.working_directory / path
        checks.append(
            PreflightCheck(
                f"runtime asset: {parameter}",
                path.is_file(),
                str(path),
            )
        )
    return checks


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _classify_outputs(frames: XcatFramePlan) -> tuple[OutputState, int, int]:
    binary_count = sum(frame.binary_exists for frame in frames.frames)
    label_count = sum(frame.label_exists for frame in frames.frames)
    expected = len(frames.frames)

    complete_labels = frames.save_tissue_labels and label_count == expected
    complete_binaries = binary_count == expected
    if complete_labels or complete_binaries:
        return "complete", binary_count, label_count
    if binary_count == 0 and label_count == 0:
        return "new", binary_count, label_count
    return "partial", binary_count, label_count


def preflight_xcat_invocation(
    config: SimulationConfig,
    parameters: XcatParameterFile,
    frames: XcatFramePlan,
) -> XcatPreflightReport:
    """Validate an XCAT command without creating output directories or running it."""

    invocation = build_xcat_invocation(config, parameters, frames)
    output_state, binary_count, label_count = _classify_outputs(frames)
    expected = len(frames.frames)
    output_root = config.run.output_root.resolve(strict=False)
    output_prefix = invocation.output_prefix.resolve(strict=False)
    output_parent = _nearest_existing_parent(invocation.output_prefix.parent)
    expected_binaries = {frame.binary_path for frame in frames.frames}
    matching_binaries = set(
        frames.raw_directory.glob(f"{frames.output_prefix.name}_act_*.bin")
    )
    unexpected_binaries = sorted(matching_binaries - expected_binaries)

    checks = [
        PreflightCheck(
            "executable exists",
            invocation.executable.is_file(),
            str(invocation.executable),
        ),
        PreflightCheck(
            "executable permission",
            invocation.executable.is_file()
            and os.access(invocation.executable, os.X_OK),
            str(invocation.executable),
        ),
        PreflightCheck(
            "working directory",
            invocation.working_directory.is_dir(),
            str(invocation.working_directory),
        ),
        PreflightCheck(
            "parameter file",
            invocation.parameter_file.is_file(),
            str(invocation.parameter_file),
        ),
        PreflightCheck(
            "frame-count agreement",
            parameters.motion_plan.generated_frame_count == expected
            and parameters.parameters.get("out_frames") == expected,
            (
                f"parameters={parameters.parameters.get('out_frames')}, "
                f"plan={expected}"
            ),
        ),
        PreflightCheck(
            "output prefix scope",
            output_prefix.is_relative_to(output_root),
            f"{output_prefix} inside {output_root}",
        ),
        PreflightCheck(
            "output parent writable",
            output_parent.is_dir() and os.access(output_parent, os.W_OK),
            str(output_parent),
        ),
        PreflightCheck(
            "existing outputs",
            output_state != "partial",
            (
                f"state={output_state}, binaries={binary_count}/{expected}, "
                f"labels={label_count}/{expected}"
            ),
        ),
        PreflightCheck(
            "unexpected binary outputs",
            not unexpected_binaries,
            (
                "none"
                if not unexpected_binaries
                else ", ".join(str(path) for path in unexpected_binaries[:8])
            ),
        ),
    ]
    checks.extend(_runtime_asset_checks(invocation))
    return XcatPreflightReport(
        invocation=invocation,
        output_state=output_state,
        expected_frame_count=expected,
        existing_binary_count=binary_count,
        existing_label_count=label_count,
        checks=tuple(checks),
    )


def expected_xcat_binary_bytes(config: SimulationConfig) -> int:
    """Return the exact byte count of one raw float32 XCAT label volume."""

    slice_count = (
        config.phantom.slice_range.end
        - config.phantom.slice_range.start
        + 1
    )
    matrix = config.phantom.matrix_size_xy
    return matrix * matrix * slice_count * 4


def verify_xcat_binary_outputs(
    config: SimulationConfig,
    frames: XcatFramePlan,
) -> tuple[tuple[Path, int], ...]:
    """Require every planned XCAT binary to exist at the exact expected size."""

    expected_bytes = expected_xcat_binary_bytes(config)
    failures: list[str] = []
    sizes = []
    for frame in frames.frames:
        path = frame.binary_path
        if not path.is_file():
            failures.append(f"missing: {path}")
            continue
        size = path.stat().st_size
        sizes.append((path, size))
        if size != expected_bytes:
            failures.append(
                f"wrong size: {path} has {size:,} bytes; "
                f"expected {expected_bytes:,}"
            )

    expected_paths = {frame.binary_path for frame in frames.frames}
    produced_paths = set(
        frames.raw_directory.glob(f"{frames.output_prefix.name}_act_*.bin")
    )
    for path in sorted(produced_paths - expected_paths):
        failures.append(f"unexpected: {path}")

    if failures:
        raise XcatExecutionError(
            "XCAT binary verification failed:\n  " + "\n  ".join(failures)
        )
    return tuple(sizes)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _write_invocation_record(
    path: Path,
    *,
    report: XcatPreflightReport,
    started: datetime,
    finished: datetime,
    return_code: int,
    expected_bytes: int,
    binary_sizes: tuple[tuple[Path, int], ...],
    stdout_log: Path,
    stderr_log: Path,
    verification_error: str | None,
) -> None:
    data = {
        "command": list(report.invocation.command),
        "working_directory": str(report.invocation.working_directory),
        "parameter_file": str(report.invocation.parameter_file),
        "output_prefix": str(report.invocation.output_prefix),
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "return_code": return_code,
        "expected_frame_count": report.expected_frame_count,
        "expected_binary_bytes_per_frame": expected_bytes,
        "binary_sizes": {
            str(binary_path): size for binary_path, size in binary_sizes
        },
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "verification_error": verification_error,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def execute_xcat_invocation(
    config: SimulationConfig,
    frames: XcatFramePlan,
    report: XcatPreflightReport,
) -> XcatExecutionResult:
    """Run one preflighted XCAT command and verify all planned binaries."""

    if not report.passed:
        raise XcatExecutionError("XCAT preflight did not pass")

    expected_bytes = expected_xcat_binary_bytes(config)
    if report.output_state == "complete":
        existing = tuple(
            (frame.binary_path, frame.binary_path.stat().st_size)
            for frame in frames.frames
            if frame.binary_path.is_file()
        )
        return XcatExecutionResult(
            status="reused",
            return_code=None,
            expected_binary_bytes=expected_bytes,
            binary_sizes=existing,
            stdout_log=None,
            stderr_log=None,
            invocation_record=None,
        )
    if report.output_state != "new":
        raise XcatExecutionError(
            f"cannot execute with output state {report.output_state!r}"
        )

    frames.raw_directory.mkdir(parents=True, exist_ok=True)
    logs_directory = config.run.output_root / "xcat" / "logs"
    logs_directory.mkdir(parents=True, exist_ok=True)
    attempt = _utc_now().strftime("%Y%m%dT%H%M%S.%fZ")
    stdout_log = logs_directory / f"xcat_{attempt}.stdout.log"
    stderr_log = logs_directory / f"xcat_{attempt}.stderr.log"
    record_path = logs_directory / f"xcat_{attempt}.invocation.json"

    started = _utc_now()
    with (
        stdout_log.open("w", encoding="utf-8") as stdout_handle,
        stderr_log.open("w", encoding="utf-8") as stderr_handle,
    ):
        completed = subprocess.run(
            report.invocation.command,
            cwd=report.invocation.working_directory,
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            check=False,
        )
    finished = _utc_now()

    binary_sizes: tuple[tuple[Path, int], ...] = ()
    verification_error = None
    if completed.returncode == 0:
        try:
            binary_sizes = verify_xcat_binary_outputs(config, frames)
        except XcatExecutionError as exc:
            verification_error = str(exc)
    else:
        verification_error = (
            f"XCAT exited with return code {completed.returncode}"
        )

    _write_invocation_record(
        record_path,
        report=report,
        started=started,
        finished=finished,
        return_code=completed.returncode,
        expected_bytes=expected_bytes,
        binary_sizes=binary_sizes,
        stdout_log=stdout_log,
        stderr_log=stderr_log,
        verification_error=verification_error,
    )
    if verification_error is not None:
        raise XcatExecutionError(
            f"{verification_error}\n"
            f"stdout: {stdout_log}\n"
            f"stderr: {stderr_log}\n"
            f"record: {record_path}"
        )

    return XcatExecutionResult(
        status="executed",
        return_code=completed.returncode,
        expected_binary_bytes=expected_bytes,
        binary_sizes=binary_sizes,
        stdout_log=stdout_log,
        stderr_log=stderr_log,
        invocation_record=record_path,
    )


def format_xcat_execution(result: XcatExecutionResult) -> str:
    """Format the verified result of an actual XCAT invocation."""

    lines = [
        "XCAT execution result",
        f"Status:                    {result.status}",
        f"Return code:               {result.return_code}",
        f"Expected bytes per binary: {result.expected_binary_bytes:,}",
        f"Verified binaries:         {len(result.binary_sizes)}",
    ]
    for path, size in result.binary_sizes:
        lines.append(f"  {path} ({size:,} bytes)")
    if result.stdout_log is not None:
        lines.append(f"stdout log:                {result.stdout_log}")
    if result.stderr_log is not None:
        lines.append(f"stderr log:                {result.stderr_log}")
    if result.invocation_record is not None:
        lines.append(f"Invocation record:         {result.invocation_record}")
    if result.status == "reused":
        lines.append("No subprocess was started because outputs were complete.")
    return "\n".join(lines)


def format_xcat_preflight(
    report: XcatPreflightReport,
    *,
    dry_run: bool = True,
) -> str:
    """Format a dry-run report with the exact future command."""

    lines = [
        "XCAT invocation dry run" if dry_run else "XCAT invocation preflight",
        f"Preflight:         {'PASS' if report.passed else 'FAIL'}",
        f"Working directory: {report.invocation.working_directory}",
        f"Parameter file:    {report.invocation.parameter_file}",
        f"Output prefix:     {report.invocation.output_prefix}",
        f"Output state:      {report.output_state}",
        (
            f"Existing outputs:  binaries "
            f"{report.existing_binary_count}/{report.expected_frame_count}, "
            f"labels {report.existing_label_count}/"
            f"{report.expected_frame_count}"
        ),
        "",
        "Command (display only; no shell is used):",
        f"  {shlex.join(report.invocation.command)}",
        "",
        "Preflight checks:",
    ]
    for check in report.checks:
        status = "PASS" if check.passed else "FAIL"
        lines.append(f"  {status:<4} {check.name}: {check.detail}")

    if report.passed and report.output_state == "new":
        action = "ready to generate planned frames"
    elif report.passed and report.output_state == "complete":
        action = "outputs are complete; a real run would reuse and skip"
    else:
        action = "not ready; resolve failed checks before execution"
    would_execute = "no (--dry-run)" if dry_run else "yes, after preflight"
    lines.extend(
        ("", f"Would execute: {would_execute}", f"Decision:      {action}")
    )
    return "\n".join(lines)
