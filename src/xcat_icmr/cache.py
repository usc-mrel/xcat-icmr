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


StageName = Literal[
    "labels",
    "contrast",
    "fullysampled_kspace",
    "fullysampled_reference",
]
CacheKind = Literal[
    "labels",
    "contrast",
    "tissue_kspace",
    "fullysampled_reference",
    "gd_kspace",
]
CACHE_SCHEMA_VERSION = 1
CACHE_ID_LENGTH = 16


@dataclass(frozen=True)
class StageReuseStatus:
    """Whether one stage manifest and all declared outputs remain valid."""

    stage: StageName
    manifest_path: Path
    expected_digest: str
    recorded_digest: str | None
    reusable: bool
    reason: str


@dataclass(frozen=True)
class ArtifactCacheEntry:
    """Content-addressed directory for one reusable simulation artifact."""

    kind: CacheKind
    cache_id: str
    full_digest: str
    directory: Path
    manifest_path: Path
    payload: dict[str, object]


@dataclass(frozen=True)
class ArtifactCacheStatus:
    """Fast manifest-based state for one content-addressed cache entry."""

    entry: ArtifactCacheEntry
    state: Literal["HIT", "PARTIAL", "MISS"]
    reason: str


def cache_root(config: SimulationConfig) -> Path:
    """Return the project-level cache shared by all run output directories."""

    return config.run.output_root.parent / "cache"


def _checksum_index_path(root: Path) -> Path:
    return root / "checksums.json"


def _read_checksum_index(root: Path) -> dict[str, dict[str, object]]:
    path = _checksum_index_path(root)
    if not path.is_file():
        return {}
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return content if isinstance(content, dict) else {}


def _write_checksum_index(
    root: Path, content: dict[str, dict[str, object]]
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    destination = _checksum_index_path(root)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=root,
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


def _content_file_signature(
    path: Path | None,
    *,
    checksum_root: Path,
) -> dict[str, object] | None:
    """Hash file contents, reusing a stat-keyed checksum when unchanged."""

    if path is None:
        return None
    resolved = path.expanduser().resolve(strict=False)
    if not resolved.is_file():
        return {"exists": False}
    stat = resolved.stat()
    key = str(resolved)
    index = _read_checksum_index(checksum_root)
    saved = index.get(key)
    if (
        isinstance(saved, dict)
        and saved.get("size_bytes") == stat.st_size
        and saved.get("mtime_ns") == stat.st_mtime_ns
        and isinstance(saved.get("sha256"), str)
    ):
        digest = str(saved["sha256"])
    else:
        hasher = hashlib.sha256()
        with resolved.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        index[key] = {
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": digest,
        }
        _write_checksum_index(checksum_root, index)
    return {
        "exists": True,
        "size_bytes": stat.st_size,
        "sha256": digest,
    }


def _artifact_entry(
    config: SimulationConfig,
    kind: CacheKind,
    payload: dict[str, object],
) -> ArtifactCacheEntry:
    full_digest = _digest(payload)
    cache_id = full_digest[:CACHE_ID_LENGTH]
    directory = cache_root(config) / kind / cache_id
    return ArtifactCacheEntry(
        kind=kind,
        cache_id=cache_id,
        full_digest=full_digest,
        directory=directory,
        manifest_path=directory / "manifest.json",
        payload=payload,
    )


def _sequence_metadata_signature(
    config: SimulationConfig,
    *,
    checksum_root: Path,
) -> dict[str, object] | None:
    """Hash the signature-keyed trajectory metadata used by the sequence."""

    from xcat_icmr.sequence.reader import read_pulseq_signature

    sequence_path = config.sequence.resolved_file
    if not sequence_path.is_file():
        return {"exists": False}
    _, signature = read_pulseq_signature(sequence_path)
    return _content_file_signature(
        config.sequence.metadata_directory / f"{signature}.mat",
        checksum_root=checksum_root,
    )


def label_cache_entry(config: SimulationConfig) -> ArtifactCacheEntry:
    """Resolve anatomy-generation inputs to one stable label cache ID."""

    checksum_root = cache_root(config)
    phantom = config.phantom.model_dump(
        mode="json", exclude={"patient_position"}
    )
    payload: dict[str, object] = {
        "cache_schema": CACHE_SCHEMA_VERSION,
        "kind": "labels",
        "storage_dtype": "uint16",
        "generation_algorithm": 4,
        "phantom": phantom,
        "xcat_time_step_s": config.timeline.xcat_time_step_s,
        "xcat_executable": _content_file_signature(
            config.resources.xcat.executable, checksum_root=checksum_root
        ),
        "xcat_parameter_template": _content_file_signature(
            config.resources.xcat.parameter_template,
            checksum_root=checksum_root,
        ),
    }
    return _artifact_entry(config, "labels", payload)


def contrast_cache_entry(config: SimulationConfig) -> ArtifactCacheEntry:
    """Resolve label and MR-signal inputs to one contrast cache ID."""

    labels = label_cache_entry(config)
    checksum_root = cache_root(config)
    payload: dict[str, object] = {
        "cache_schema": CACHE_SCHEMA_VERSION,
        "kind": "contrast",
        "storage_dtype": "float32",
        "generation_algorithm": 3,
        "label_cache_id": labels.cache_id,
        "scanner": config.scanner.model_dump(mode="json"),
        "contrast": config.sequence.contrast.model_dump(mode="json"),
        "rf_profile": config.sequence.rf_profile.model_dump(mode="json"),
        "patient_position": config.phantom.patient_position,
        "coordinate_mode": config.sequence.coordinate_mode,
        "orientation": config.sequence.orientation,
        "sequence_file": _content_file_signature(
            config.sequence.resolved_file, checksum_root=checksum_root
        ),
        "sequence_metadata": _sequence_metadata_signature(
            config, checksum_root=checksum_root
        ),
    }
    return _artifact_entry(config, "contrast", payload)


def tissue_kspace_cache_entry(config: SimulationConfig) -> ArtifactCacheEntry:
    """Resolve contrast and encoding inputs to one tissue k-space cache ID."""

    contrast = contrast_cache_entry(config)
    checksum_root = cache_root(config)
    payload: dict[str, object] = {
        "cache_schema": CACHE_SCHEMA_VERSION,
        "kind": "tissue_kspace",
        "storage_dtype": "complex64",
        "reference": {
            "format": "hdf5",
            "layout": "logical_x,logical_y,logical_z,time",
            "resampling": "linear-image-space",
            "dtype": "complex64",
        },
        "encoding_algorithm": 5,
        "contrast_cache_id": contrast.cache_id,
        "sequence_file": _content_file_signature(
            config.sequence.resolved_file, checksum_root=checksum_root
        ),
        "sequence_metadata": _sequence_metadata_signature(
            config, checksum_root=checksum_root
        ),
        "coordinate_mode": config.sequence.coordinate_mode,
        "orientation": config.sequence.orientation,
        "patient_position": config.phantom.patient_position,
        "temporal_aggregation": {
            "xcat_time_step_s": config.timeline.xcat_time_step_s,
            "kspace_time_step_s": config.timeline.kspace_time_step_s,
            "xcat_frames_per_kspace_frame": (
                config.timeline.xcat_frames_per_kspace_frame
            ),
            "method": config.timeline.xcat_to_kspace,
        },
        "coils": config.coils.model_dump(
            mode="json", exclude={"sensitivity_map"}
        ),
        "sensitivity_map": _content_file_signature(
            config.coils.sensitivity_map, checksum_root=checksum_root
        ),
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
    return _artifact_entry(config, "tissue_kspace", payload)


def fullysampled_reference_cache_entry(
    config: SimulationConfig,
) -> ArtifactCacheEntry:
    """Resolve inputs to the image-only fully sampled reference cache."""

    contrast = contrast_cache_entry(config)
    checksum_root = cache_root(config)
    payload: dict[str, object] = {
        "cache_schema": CACHE_SCHEMA_VERSION,
        "kind": "fullysampled_reference",
        "storage_dtype": "complex64",
        "generation_algorithm": 2,
        "contrast_cache_id": contrast.cache_id,
        "sequence_file": _content_file_signature(
            config.sequence.resolved_file, checksum_root=checksum_root
        ),
        "sequence_metadata": _sequence_metadata_signature(
            config, checksum_root=checksum_root
        ),
        "coordinate_mode": config.sequence.coordinate_mode,
        "orientation": config.sequence.orientation,
        "patient_position": config.phantom.patient_position,
        "target_fov_mm": list(config.encoding.target_fov_mm),
        "resolution_source": "pulseq-sequence-metadata",
        "trajectory_scale": "isotropic-radius-to-resolution",
        "backend": "sigpy",
        "oversampling": DEFAULT_NUFFT_OVERSAMPLING,
        "kernel_width": DEFAULT_NUFFT_KERNEL_WIDTH,
        "coil_combination": "sum-conj-sensitivity-times-adjoint",
        "rf_shift_application": "already-in-high-resolution-contrast",
        "temporal_aggregation": {
            "xcat_time_step_s": config.timeline.xcat_time_step_s,
            "reference_time_step_s": config.timeline.kspace_time_step_s,
            "xcat_frames_per_reference_frame": (
                config.timeline.xcat_frames_per_kspace_frame
            ),
            "method": config.timeline.xcat_to_kspace,
        },
        "coils": config.coils.model_dump(
            mode="json", exclude={"sensitivity_map"}
        ),
        "sensitivity_map": _content_file_signature(
            config.coils.sensitivity_map, checksum_root=checksum_root
        ),
    }
    return _artifact_entry(config, "fullysampled_reference", payload)


def gd_kspace_cache_entry(config: SimulationConfig) -> ArtifactCacheEntry:
    """Resolve intervention and encoding inputs to one Gd-delta cache ID."""

    tissue = tissue_kspace_cache_entry(config)
    checksum_root = cache_root(config)
    balloon = config.intervention.gd_balloon.model_dump(mode="json")
    balloon["path"]["control_points_file"] = _content_file_signature(
        config.intervention.gd_balloon.path.control_points_file,
        checksum_root=checksum_root,
    )
    payload: dict[str, object] = {
        "cache_schema": CACHE_SCHEMA_VERSION,
        "kind": "gd_kspace",
        "storage_dtype": "complex64",
        "generation_algorithm": 1,
        "tissue_kspace_cache_id": tissue.cache_id,
        "balloon": balloon,
        "simulation_duration": config.timeline.duration_s,
        "sparse_geometry": {
            "boundary_samples_per_axis": 8,
            "logical_mapping": "centered-zero-preserving",
        },
        "roi_encoding": {
            "minimum_shape": 32,
            "margin_voxels": 4,
            "global_phase_correction": True,
            "rf_shift_phase_correction": True,
        },
    }
    return _artifact_entry(config, "gd_kspace", payload)


def label_frame_path(config: SimulationConfig, frame_index: int) -> Path:
    return label_cache_entry(config).directory / "frames" / (
        f"label_frame_{frame_index:04d}.mat"
    )


def contrast_frame_path(config: SimulationConfig, frame_index: int) -> Path:
    return contrast_cache_entry(config).directory / "frames" / (
        f"contrast_frame_{frame_index:04d}.mat"
    )


def contrast_profile_path(config: SimulationConfig) -> Path:
    return contrast_cache_entry(config).directory / "rf_profile.mat"


def gd_kspace_frame_path(config: SimulationConfig, frame_index: int) -> Path:
    return gd_kspace_cache_entry(config).directory / "kspace" / (
        f"gd_delta_kspace_frame_{frame_index:06d}.mat"
    )


def write_artifact_manifest(
    entry: ArtifactCacheEntry,
    *,
    status: Literal["partial", "complete"],
    frame_count: int,
    completed_frame_indices: list[int],
    outputs: list[str | Path],
) -> Path:
    """Atomically write a cache-entry manifest without deleting old entries."""

    entry.directory.mkdir(parents=True, exist_ok=True)
    content = {
        "cache_schema": CACHE_SCHEMA_VERSION,
        "kind": entry.kind,
        "cache_id": entry.cache_id,
        "full_digest": entry.full_digest,
        "status": status,
        "frame_count": frame_count,
        "completed_frame_indices": completed_frame_indices,
        "payload": entry.payload,
        "outputs": [str(Path(path).resolve(strict=False)) for path in outputs],
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{entry.manifest_path.name}.",
            suffix=".tmp",
            dir=entry.directory,
            mode="w",
            encoding="utf-8",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(content, temporary, indent=2, sort_keys=True)
            temporary.write("\n")
        temporary_path.chmod(0o644)
        os.replace(temporary_path, entry.manifest_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return entry.manifest_path


def artifact_cache_status(entry: ArtifactCacheEntry) -> ArtifactCacheStatus:
    """Inspect one cache manifest without loading large array contents."""

    if not entry.manifest_path.is_file():
        state = "PARTIAL" if entry.directory.exists() else "MISS"
        return ArtifactCacheStatus(entry, state, "manifest is missing")
    try:
        content = json.loads(entry.manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return ArtifactCacheStatus(entry, "PARTIAL", f"manifest is unreadable: {exc}")
    if content.get("full_digest") != entry.full_digest:
        return ArtifactCacheStatus(entry, "MISS", "manifest digest differs")
    outputs = content.get("outputs", [])
    if not isinstance(outputs, list) or not outputs or any(
        not Path(str(path)).is_file() for path in outputs
    ):
        return ArtifactCacheStatus(entry, "PARTIAL", "one or more outputs are missing")
    if content.get("status") != "complete":
        return ArtifactCacheStatus(entry, "PARTIAL", "cache is incomplete")
    return ArtifactCacheStatus(entry, "HIT", "complete cache entry")


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
    """Build dependency-separated payloads for implemented stages."""

    labels = dict(label_cache_entry(config).payload)
    labels["stage"] = "labels"
    contrast = dict(contrast_cache_entry(config).payload)
    contrast["stage"] = "contrast"
    kspace = dict(tissue_kspace_cache_entry(config).payload)
    kspace["stage"] = "fullysampled_kspace"
    reference = dict(fullysampled_reference_cache_entry(config).payload)
    reference["stage"] = "fullysampled_reference"
    return {
        "labels": labels,
        "contrast": contrast,
        "fullysampled_kspace": kspace,
        "fullysampled_reference": reference,
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
