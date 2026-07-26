"""YAML loading and path resolution for simulation configurations."""

from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from xcat_icmr.config.models import SimulationConfig


class ConfigurationLoadError(Exception):
    """Raised when a configuration file cannot be read or parsed."""


_PATH_LOCATIONS: tuple[tuple[str, ...], ...] = (
    ("run", "output_root"),
    ("resources", "xcat", "executable"),
    ("resources", "xcat", "parameter_template"),
    ("scanner", "effects", "off_resonance", "field_map"),
    ("sequence", "folder"),
    ("sequence", "metadata_directory"),
    (
        "intervention",
        "gd_balloon",
        "path",
        "control_points_file",
    ),
    ("coils", "sensitivity_map"),
)


def _resolve_path(value: Any, base_directory: Path) -> Any:
    if value is None or not isinstance(value, str):
        return value
    path = Path(os.path.expandvars(value)).expanduser()
    if not path.is_absolute():
        path = base_directory / path
    return path.resolve(strict=False)


def _resolve_config_paths(data: dict[str, Any], base_directory: Path) -> dict[str, Any]:
    resolved = deepcopy(data)
    for location in _PATH_LOCATIONS:
        parent: Any = resolved
        for key in location[:-1]:
            if not isinstance(parent, dict) or key not in parent:
                parent = None
                break
            parent = parent[key]
        if isinstance(parent, dict) and location[-1] in parent:
            key = location[-1]
            parent[key] = _resolve_path(parent[key], base_directory)

    noise = resolved.get("noise")
    if isinstance(noise, dict):
        covariance = noise.get("coil_covariance")
        if isinstance(covariance, str) and covariance != "identity":
            noise["coil_covariance"] = _resolve_path(covariance, base_directory)
    return resolved


def load_config(path: str | Path) -> SimulationConfig:
    """Read, resolve, and validate a simulation YAML file."""

    config_path = Path(path).expanduser().resolve(strict=False)
    if not config_path.is_file():
        raise ConfigurationLoadError(
            f"configuration file does not exist: {config_path}"
        )

    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigurationLoadError(
            f"could not read configuration file {config_path}: {exc}"
        ) from exc

    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigurationLoadError(
            f"invalid YAML in {config_path}: {exc}"
        ) from exc

    if not isinstance(raw, dict):
        raise ConfigurationLoadError(
            f"configuration root must be a mapping: {config_path}"
        )

    resolved = _resolve_config_paths(raw, config_path.parent)
    return SimulationConfig.model_validate(resolved)


def format_validation_error(error: ValidationError) -> str:
    """Format Pydantic errors with YAML-style dotted field paths."""

    lines = ["Configuration errors:"]
    for item in error.errors(include_url=False):
        location = ".".join(str(part) for part in item["loc"])
        if not location:
            location = "<configuration>"
        lines.append(f"  - {location}: {item['msg']}")
    return "\n".join(lines)
