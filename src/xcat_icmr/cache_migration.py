"""Adopt legacy run-scoped labels and contrasts into artifact caches."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import tempfile
from typing import TYPE_CHECKING, Callable

import numpy as np
from scipy.io import loadmat, savemat, whosmat

from xcat_icmr.cache import (
    contrast_cache_entry,
    contrast_frame_path,
    contrast_profile_path,
    label_cache_entry,
    label_frame_path,
    write_artifact_manifest,
)
from xcat_icmr.phantom import plan_xcat_frames, xcat_label_shape

if TYPE_CHECKING:
    from xcat_icmr.config.models import SimulationConfig


class CacheMigrationError(ValueError):
    """Raised when existing products cannot be adopted safely."""


@dataclass(frozen=True)
class CacheMigrationResult:
    label_cache_id: str
    contrast_cache_id: str | None
    converted_label_count: int
    reused_label_count: int
    adopted_contrast_count: int
    reused_contrast_count: int
    label_directory: Path
    contrast_directory: Path | None


def _atomic_savemat(path: Path, variables: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
        savemat(temporary_path, variables, appendmat=False, do_compression=False)
        temporary_path.chmod(0o644)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _valid_mat(path: Path, variable: str, shape: tuple[int, ...], dtype: str) -> bool:
    if not path.is_file():
        return False
    try:
        entries = {
            name: (saved_shape, saved_dtype)
            for name, saved_shape, saved_dtype in whosmat(path)
        }
    except (OSError, ValueError):
        return False
    return entries == {variable: (shape, dtype)}


def _adopt_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.adopting"
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copy2(source, temporary)
    temporary.chmod(0o644)
    os.replace(temporary, destination)


def adopt_legacy_cache(
    config: "SimulationConfig",
    *,
    include_contrast: bool = True,
    overwrite: bool = False,
    progress: Callable[[str], None] | None = None,
) -> CacheMigrationResult:
    """Convert legacy float labels and link compatible contrast frames."""

    frame_count = len(plan_xcat_frames(config, debug_one_frame=False).frames)
    shape = xcat_label_shape(config)
    label_entry = label_cache_entry(config)
    converted_labels = 0
    reused_labels = 0
    label_outputs: list[Path] = []
    for index in range(1, frame_count + 1):
        source = (
            config.run.output_root
            / "xcat"
            / "labels"
            / f"phantom_{config.run.id}_act_{index}.mat"
        )
        destination = label_frame_path(config, index)
        label_outputs.append(destination)
        if _valid_mat(destination, "P", shape, "uint16") and not overwrite:
            reused_labels += 1
            continue
        if destination.exists() and not overwrite:
            raise CacheMigrationError(
                f"existing cached label is invalid: {destination}; "
                "pass --overwrite"
            )
        if not source.is_file():
            raise CacheMigrationError(f"legacy label frame is missing: {source}")
        try:
            labels = np.asarray(loadmat(source, variable_names=["P"])["P"])
        except (OSError, ValueError, KeyError, NotImplementedError) as exc:
            raise CacheMigrationError(f"could not read {source}: {exc}") from exc
        if labels.shape != shape or not np.all(np.isfinite(labels)):
            raise CacheMigrationError(f"legacy label frame is invalid: {source}")
        rounded = np.rint(labels)
        if (
            not np.array_equal(labels, rounded)
            or float(rounded.min()) < 0
            or float(rounded.max()) > np.iinfo(np.uint16).max
        ):
            raise CacheMigrationError(
                f"legacy labels cannot be represented as uint16: {source}"
            )
        _atomic_savemat(destination, {"P": rounded.astype(np.uint16)})
        if not _valid_mat(destination, "P", shape, "uint16"):
            raise CacheMigrationError(f"adopted label failed verification: {destination}")
        converted_labels += 1
        if progress is not None:
            progress(f"Labels {index}/{frame_count}: {destination}")
    write_artifact_manifest(
        label_entry,
        status="complete",
        frame_count=frame_count,
        completed_frame_indices=list(range(1, frame_count + 1)),
        outputs=label_outputs,
    )

    contrast_id = None
    contrast_directory = None
    adopted_contrasts = 0
    reused_contrasts = 0
    if include_contrast:
        contrast_entry = contrast_cache_entry(config)
        contrast_id = contrast_entry.cache_id
        contrast_directory = contrast_entry.directory
        contrast_outputs: list[Path] = []
        legacy_profile = (
            config.run.output_root
            / "contrast"
            / f"phantom_{config.run.id}_rf_slice_profile.mat"
        )
        profile_destination = contrast_profile_path(config)
        if not legacy_profile.is_file():
            raise CacheMigrationError(
                f"legacy RF profile is missing: {legacy_profile}"
            )
        if profile_destination.exists() and not overwrite:
            reused_profile = True
        else:
            _adopt_file(legacy_profile, profile_destination)
            reused_profile = False
        if not profile_destination.is_file():
            raise CacheMigrationError(
                f"adopted RF profile is missing: {profile_destination}"
            )
        contrast_outputs.append(profile_destination)
        for index in range(1, frame_count + 1):
            source = (
                config.run.output_root
                / "contrast"
                / (
                    f"phantom_{config.run.id}_act_{index}_"
                    f"{config.sequence.contrast.model}.mat"
                )
            )
            destination = contrast_frame_path(config, index)
            contrast_outputs.append(destination)
            if _valid_mat(destination, "image", shape, "single") and not overwrite:
                reused_contrasts += 1
                continue
            if destination.exists() and not overwrite:
                raise CacheMigrationError(
                    f"existing cached contrast is invalid: {destination}; "
                    "pass --overwrite"
                )
            if not _valid_mat(source, "image", shape, "single"):
                raise CacheMigrationError(
                    f"legacy contrast frame is missing or invalid: {source}"
                )
            _adopt_file(source, destination)
            adopted_contrasts += 1
            if progress is not None:
                progress(f"Contrast {index}/{frame_count}: {destination}")
        if progress is not None:
            profile_state = "reused" if reused_profile else "adopted"
            progress(f"RF profile: {profile_state} {profile_destination}")
        write_artifact_manifest(
            contrast_entry,
            status="complete",
            frame_count=frame_count,
            completed_frame_indices=list(range(1, frame_count + 1)),
            outputs=contrast_outputs,
        )

    return CacheMigrationResult(
        label_cache_id=label_entry.cache_id,
        contrast_cache_id=contrast_id,
        converted_label_count=converted_labels,
        reused_label_count=reused_labels,
        adopted_contrast_count=adopted_contrasts,
        reused_contrast_count=reused_contrasts,
        label_directory=label_entry.directory,
        contrast_directory=contrast_directory,
    )


def format_cache_migration(result: CacheMigrationResult) -> str:
    return "\n".join(
        (
            "Legacy artifact-cache adoption",
            f"Label cache ID:       {result.label_cache_id}",
            f"Labels converted:     {result.converted_label_count}",
            f"Labels reused:        {result.reused_label_count}",
            f"Label directory:      {result.label_directory}",
            f"Contrast cache ID:    {result.contrast_cache_id or 'not requested'}",
            f"Contrasts adopted:    {result.adopted_contrast_count}",
            f"Contrasts reused:     {result.reused_contrast_count}",
            f"Contrast directory:   {result.contrast_directory or 'not requested'}",
        )
    )
