"""Conversion of HF-limited, full-RL/AP XCAT label binaries to MATLAB files."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile

import numpy as np
from scipy.io import loadmat, savemat

from xcat_icmr.phantom.binary import XcatBinaryVolume
from xcat_icmr.phantom.matlab_labels import (
    XcatLabelComparisonError,
    validate_xcat_labels,
)


class XcatLabelConversionError(Exception):
    """Raised when an XCAT label volume cannot be saved safely."""


@dataclass(frozen=True)
class XcatLabelConversion:
    """Summary of one verified XCAT-binary to MATLAB conversion."""

    binary_path: Path
    label_path: Path
    logical_shape: tuple[int, int, int]
    dtype: str
    unique_labels: tuple[int, ...]
    file_size_bytes: int


def convert_xcat_labels_to_mat(
    volume: XcatBinaryVolume,
    output_path: str | Path,
    *,
    chunk_slices: int = 8,
    overwrite: bool = False,
) -> XcatLabelConversion:
    """Validate, save, and reopen one XCAT label volume as MATLAB ``P``."""

    destination = Path(output_path).expanduser().resolve(strict=False)
    if destination.exists() and not overwrite:
        raise XcatLabelConversionError(
            f"label file already exists: {destination}; "
            "pass --overwrite to replace it"
        )

    try:
        unique_labels = validate_xcat_labels(
            volume, chunk_slices=chunk_slices
        )
    except (ValueError, XcatLabelComparisonError) as exc:
        raise XcatLabelConversionError(str(exc)) from exc

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

        # MATLAB v5 files are directly loadable by MATLAB and are sufficient
        # here while a float32 frame remains below the MATLAB v5 2 GB limit.
        labels = np.asarray(volume.cropped, dtype=np.float32)
        savemat(
            temporary_path,
            {"P": labels},
            appendmat=False,
            do_compression=False,
        )

        reopened = loadmat(
            temporary_path,
            variable_names=["P"],
            squeeze_me=False,
        )
        if "P" not in reopened:
            raise XcatLabelConversionError(
                "saved MATLAB file does not contain variable 'P'"
            )
        saved_labels = reopened["P"]
        if saved_labels.shape != volume.cropped_shape:
            raise XcatLabelConversionError(
                "saved MATLAB label shape changed during writing: "
                f"{saved_labels.shape} != {volume.cropped_shape}"
            )
        if saved_labels.dtype != np.dtype(np.float32):
            raise XcatLabelConversionError(
                "saved MATLAB label dtype changed during writing: "
                f"{saved_labels.dtype} != float32"
            )

        temporary_path.chmod(0o644)
        os.replace(temporary_path, destination)
        temporary_path = None
    except XcatLabelConversionError:
        raise
    except (OSError, ValueError) as exc:
        raise XcatLabelConversionError(
            f"could not write MATLAB label file {destination}: {exc}"
        ) from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return XcatLabelConversion(
        binary_path=volume.path,
        label_path=destination,
        logical_shape=volume.cropped_shape,
        dtype="float32",
        unique_labels=unique_labels,
        file_size_bytes=destination.stat().st_size,
    )


def format_xcat_label_conversion(report: XcatLabelConversion) -> str:
    """Format one verified label conversion."""

    return "\n".join(
        (
            f"Binary:           {report.binary_path}",
            f"MATLAB labels:    {report.label_path}",
            f"Variable:         P",
            f"Shape:            {report.logical_shape}",
            f"Data type:        {report.dtype}",
            f"File size:        {report.file_size_bytes:,} bytes",
            (
                "Validated labels: "
                + ", ".join(str(label) for label in report.unique_labels)
            ),
            "Verification:     PASS",
        )
    )
