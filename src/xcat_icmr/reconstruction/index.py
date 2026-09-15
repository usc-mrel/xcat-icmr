"""Atomic human-readable indexes for reconstruction results."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping


INDEX_FIELDS = (
    "reconstruction_id",
    "acquisition_id",
    "control_points_filename",
    "velocity_cm_per_s",
    "temporal_resolution_ms",
    "method",
    "readable_recipe",
    "status",
    "latency_s",
    "reconstruction_file",
    "result_file",
)


def _read_records(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(content, list):
        return []
    return [dict(item) for item in content if isinstance(item, dict)]


def _atomic_text(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            writer(stream)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def upsert_reconstruction_indexes(
    directory: str | Path,
    record: Mapping[str, object],
) -> tuple[Path, Path]:
    """Update colocated JSON and CSV indexes keyed by reconstruction ID."""

    root = Path(directory).expanduser().resolve(strict=False)
    json_path = root / "index.json"
    csv_path = root / "index.csv"
    records = _read_records(json_path)
    identity = str(record["reconstruction_id"])
    updated = [
        item
        for item in records
        if str(item.get("reconstruction_id")) != identity
    ]
    updated.append({field: record.get(field) for field in INDEX_FIELDS})
    updated.sort(
        key=lambda item: (
            str(item.get("control_points_filename", "")),
            float(item.get("velocity_cm_per_s") or 0.0),
            float(item.get("temporal_resolution_ms") or 0.0),
            str(item.get("method", "")),
            str(item.get("readable_recipe", "")),
        )
    )
    _atomic_text(
        json_path,
        lambda stream: json.dump(updated, stream, indent=2, sort_keys=True),
    )

    def write_csv(stream) -> None:
        writer = csv.DictWriter(stream, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(updated)

    _atomic_text(csv_path, write_csv)
    return csv_path, json_path
