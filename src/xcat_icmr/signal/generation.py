"""Generation and verified persistence of MRI contrast volumes."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile

import numpy as np
from scipy.io import loadmat, savemat

from xcat_icmr.signal.bssfp import bssfp_signal_from_tissue_properties
from xcat_icmr.tissue import TissueLibrary, map_labels_to_tissue_properties


class ContrastGenerationError(Exception):
    """Raised when a contrast image cannot be generated or saved safely."""


@dataclass(frozen=True)
class ContrastGeneration:
    """Summary of one verified tissue-label to contrast conversion."""

    label_path: Path
    image_path: Path
    model: str
    logical_shape: tuple[int, ...]
    dtype: str
    flip_angle_deg: float
    te_ms: float
    tr_ms: float
    file_size_bytes: int
    signal_min: float
    signal_max: float


def _load_label_volume(
    path: str | Path,
    *,
    variable_name: str = "P",
) -> tuple[Path, np.ndarray]:
    label_path = Path(path).expanduser().resolve(strict=False)
    if not label_path.is_file():
        raise ContrastGenerationError(
            f"tissue-label file does not exist: {label_path}"
        )
    try:
        content = loadmat(
            label_path,
            variable_names=[variable_name],
            squeeze_me=False,
        )
    except (OSError, ValueError, NotImplementedError) as exc:
        raise ContrastGenerationError(
            f"could not read MATLAB tissue labels {label_path}: {exc}"
        ) from exc
    if variable_name not in content:
        raise ContrastGenerationError(
            f"MATLAB tissue-label file is missing variable {variable_name!r}"
        )
    labels = np.asarray(content[variable_name])
    if labels.ndim != 3:
        raise ContrastGenerationError(
            f"tissue-label volume must be three-dimensional; got {labels.shape}"
        )
    return label_path, labels


def generate_bssfp_contrast(
    label_path: str | Path,
    output_path: str | Path,
    library: TissueLibrary,
    *,
    expected_shape: tuple[int, int, int],
    flip_angle_deg: float,
    te_ms: float,
    tr_ms: float,
    off_resonance_enabled: bool = False,
    chunk_slices: int = 8,
    overwrite: bool = False,
) -> ContrastGeneration:
    """Generate, save, and reopen a float32 bSSFP ``image`` volume."""

    if off_resonance_enabled:
        raise NotImplementedError(
            "off-resonance bSSFP signal simulation is not implemented"
        )
    if chunk_slices <= 0:
        raise ValueError("chunk_slices must be positive")

    source, labels = _load_label_volume(label_path)
    if labels.shape != expected_shape:
        raise ContrastGenerationError(
            f"tissue-label shape {labels.shape} does not match the configured "
            f"phantom shape {expected_shape}"
        )

    destination = Path(output_path).expanduser().resolve(strict=False)
    if destination.exists() and not overwrite:
        raise ContrastGenerationError(
            f"contrast image already exists: {destination}; "
            "pass --overwrite to replace it"
        )

    image = np.empty(labels.shape, dtype=np.float32)
    try:
        for start in range(0, labels.shape[2], chunk_slices):
            stop = min(start + chunk_slices, labels.shape[2])
            properties = map_labels_to_tissue_properties(
                labels[:, :, start:stop],
                library,
                dtype=np.float32,
            )
            image[:, :, start:stop] = bssfp_signal_from_tissue_properties(
                properties,
                flip_angle_deg=flip_angle_deg,
                te_ms=te_ms,
                tr_ms=tr_ms,
                off_resonance_enabled=False,
                dtype=np.float32,
            )
    except ValueError as exc:
        raise ContrastGenerationError(str(exc)) from exc

    if not np.all(np.isfinite(image)):
        raise ContrastGenerationError(
            "generated bSSFP image contains non-finite values"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        savemat(
            temporary_path,
            {"image": image},
            appendmat=False,
            do_compression=False,
        )
        reopened = loadmat(
            temporary_path,
            variable_names=["image"],
            squeeze_me=False,
        )
        if "image" not in reopened:
            raise ContrastGenerationError(
                "saved MATLAB file does not contain variable 'image'"
            )
        saved = reopened["image"]
        if saved.shape != image.shape:
            raise ContrastGenerationError(
                "saved contrast shape changed during writing: "
                f"{saved.shape} != {image.shape}"
            )
        if saved.dtype != np.dtype(np.float32):
            raise ContrastGenerationError(
                "saved contrast dtype changed during writing: "
                f"{saved.dtype} != float32"
            )
        temporary_path.chmod(0o644)
        os.replace(temporary_path, destination)
        temporary_path = None
    except ContrastGenerationError:
        raise
    except (OSError, ValueError) as exc:
        raise ContrastGenerationError(
            f"could not write MATLAB contrast image {destination}: {exc}"
        ) from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return ContrastGeneration(
        label_path=source,
        image_path=destination,
        model="bssfp",
        logical_shape=image.shape,
        dtype="float32",
        flip_angle_deg=flip_angle_deg,
        te_ms=te_ms,
        tr_ms=tr_ms,
        file_size_bytes=destination.stat().st_size,
        signal_min=float(np.min(image)),
        signal_max=float(np.max(image)),
    )


def format_contrast_generation(report: ContrastGeneration) -> str:
    """Format one verified contrast-generation result."""

    return "\n".join(
        (
            f"Tissue labels: {report.label_path}",
            f"Contrast image: {report.image_path}",
            f"Model:         {report.model}",
            f"Variable:      image",
            f"Shape:         {report.logical_shape}",
            f"Data type:     {report.dtype}",
            (
                f"Sequence:      FA={report.flip_angle_deg:g} deg, "
                f"TE={report.te_ms:g} ms, TR={report.tr_ms:g} ms"
            ),
            (
                f"Signal range:  "
                f"{report.signal_min:g} to {report.signal_max:g}"
            ),
            f"File size:     {report.file_size_bytes:,} bytes",
            "Verification:  PASS",
        )
    )
