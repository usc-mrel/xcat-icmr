"""Stage fingerprints for safe reuse of expensive simulation products."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Literal

from xcat_icmr.config import SimulationConfig
from xcat_icmr.encoding.sigpy_backend import (
    DEFAULT_NUFFT_KERNEL_WIDTH,
    DEFAULT_NUFFT_OVERSAMPLING,
)


StageName = Literal["labels", "contrast", "fullysampled_kspace"]


@dataclass(frozen=True)
class StageReuseStatus:
    """Whether one stage manifest and all declared outputs remain valid."""

    stage: StageName
    manifest_path: Path
    expected_digest: str
    recorded_digest: str | None
    reusable: bool
    reason: str


def _file_signature(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve(strict=False)
    if not resolved.is_file():
        return {"path": str(resolved), "exists": False}
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "exists": True,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _digest(payload: dict[str, object]) -> str:
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def stage_payloads(config: SimulationConfig) -> dict[StageName, dict[str, object]]:
    """Build dependency-separated payloads for the three implemented stages."""

    labels: dict[str, object] = {
        "schema": 1,
        "stage": "labels",
        "resume_phase_algorithm": 3,
        "phantom": config.phantom.model_dump(mode="json"),
        "timeline": {
            "duration_s": config.timeline.duration_s,
            "xcat_time_step_s": config.timeline.xcat_time_step_s,
        },
        "xcat_executable": _file_signature(config.resources.xcat.executable),
        "xcat_parameter_template": _file_signature(
            config.resources.xcat.parameter_template
        ),
    }
    labels_digest = _digest(labels)
    contrast: dict[str, object] = {
        "schema": 1,
        "stage": "contrast",
        "labels_digest": labels_digest,
        "scanner": config.scanner.model_dump(mode="json"),
        "contrast": config.sequence.contrast.model_dump(mode="json"),
        "rf_profile": config.sequence.rf_profile.model_dump(mode="json"),
        "sequence_geometry": {
            "coordinate_mode": config.sequence.coordinate_mode,
            "orientation": config.sequence.orientation,
        },
        "sequence_file": _file_signature(config.sequence.resolved_file),
    }
    contrast_digest = _digest(contrast)
    kspace: dict[str, object] = {
        "schema": 1,
        "stage": "fullysampled_kspace",
        "contrast_digest": contrast_digest,
        "coils": config.coils.model_dump(mode="json"),
        "sensitivity_map": _file_signature(config.coils.sensitivity_map),
        "sequence_file": _file_signature(config.sequence.resolved_file),
        "encoding": {
            "fov_mm": [500.0, 500.0, 500.0],
            "resolution_mm": 3.5,
            "matrix_shape": [144, 144, 144],
            "trajectory_scale": "isotropic-radius-to-resolution",
            "backend": "sigpy",
            "oversampling": DEFAULT_NUFFT_OVERSAMPLING,
            "kernel_width": DEFAULT_NUFFT_KERNEL_WIDTH,
            "rf_center_shift_mm": config.sequence.rf_profile.center_shift_mm,
        },
    }
    return {
        "labels": labels,
        "contrast": contrast,
        "fullysampled_kspace": kspace,
    }


def stage_digest(config: SimulationConfig, stage: StageName) -> str:
    """Return the canonical dependency digest for one stage."""

    return _digest(stage_payloads(config)[stage])


def stage_manifest_path(config: SimulationConfig, stage: StageName) -> Path:
    """Return the manifest path underneath the run output root."""

    return config.run.output_root / "manifests" / f"{stage}.json"


def write_stage_manifest(
    config: SimulationConfig,
    stage: StageName,
    output_paths: list[str | Path],
) -> Path:
    """Atomically record one completed stage and its concrete output files."""

    outputs = [_file_signature(Path(path)) for path in output_paths]
    if not outputs or any(not item or not item.get("exists") for item in outputs):
        raise ValueError(f"cannot record {stage}: one or more outputs are missing")
    payload = stage_payloads(config)[stage]
    content = {
        "manifest_schema": 1,
        "stage": stage,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "digest": _digest(payload),
        "payload": payload,
        "outputs": outputs,
    }
    destination = stage_manifest_path(config, stage)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            mode="w",
            encoding="utf-8",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(content, temporary, indent=2, sort_keys=True)
            temporary.write("\n")
        temporary_path.chmod(0o644)
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination


def stage_reuse_status(
    config: SimulationConfig,
    stage: StageName,
) -> StageReuseStatus:
    """Check digest equality and every output signature without loading arrays."""

    manifest = stage_manifest_path(config, stage)
    expected = stage_digest(config, stage)
    if not manifest.is_file():
        return StageReuseStatus(
            stage, manifest, expected, None, False, "manifest is missing"
        )
    try:
        content = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return StageReuseStatus(
            stage, manifest, expected, None, False, f"manifest is unreadable: {exc}"
        )
    recorded = content.get("digest")
    if recorded != expected:
        return StageReuseStatus(
            stage, manifest, expected, str(recorded), False, "inputs changed"
        )
    outputs = content.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        return StageReuseStatus(
            stage, manifest, expected, str(recorded), False, "outputs are missing"
        )
    for saved in outputs:
        if not isinstance(saved, dict) or "path" not in saved:
            return StageReuseStatus(
                stage, manifest, expected, str(recorded), False, "invalid output record"
            )
        current = _file_signature(Path(str(saved["path"])))
        if current != saved:
            return StageReuseStatus(
                stage,
                manifest,
                expected,
                str(recorded),
                False,
                f"output changed: {saved['path']}",
            )
    return StageReuseStatus(
        stage, manifest, expected, str(recorded), True, "inputs and outputs match"
    )
