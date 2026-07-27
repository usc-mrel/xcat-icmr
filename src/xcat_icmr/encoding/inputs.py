"""Preparation of image and coil inputs for non-Cartesian encoding."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile

import numpy as np
from scipy.io import loadmat, savemat

from xcat_icmr.sequence.orientation import (
    CoordinateTransforms,
    reorient_spatial_array,
)


class EncodingInputError(Exception):
    """Raised when an image cannot be prepared for the encoding grid."""


@dataclass(frozen=True)
class PreparedContrast:
    """One finite contrast image centered on the sensitivity-map grid."""

    path: Path
    source_frame: str
    target_frame: str
    source_shape: tuple[int, int, int]
    oriented_shape: tuple[int, int, int]
    target_shape: tuple[int, int, int]
    padding: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    target_axis_patient_directions: tuple[str, str, str] | None
    image: np.ndarray


@dataclass(frozen=True)
class LogicalInputPreview:
    """Inspectable center planes after Step 5, before any NUFFT."""

    output_path: Path
    contrast_shape: tuple[int, int, int]
    coil_shape: tuple[int, int, int]
    coil_index: int
    axis_patient_directions: tuple[str, str, str]
    file_size_bytes: int


def center_padding(
    source_shape: tuple[int, int, int],
    target_shape: tuple[int, int, int],
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Return deterministic before/after padding for a centered 3-D array."""

    if len(source_shape) != 3 or len(target_shape) != 3:
        raise ValueError("source_shape and target_shape must be 3-D")
    padding = []
    for source, target in zip(source_shape, target_shape, strict=True):
        if source <= 0 or target <= 0:
            raise ValueError("source and target dimensions must be positive")
        if target < source:
            raise EncodingInputError(
                f"encoding-grid shape {target_shape} is smaller than "
                f"contrast shape {source_shape}"
            )
        difference = target - source
        before = difference // 2
        padding.append((before, difference - before))
    return tuple(padding)  # type: ignore[return-value]


def prepare_contrast_for_encoding(
    path: str | Path,
    target_shape: tuple[int, int, int],
    *,
    variable_name: str = "image",
    source_to_target: np.ndarray | None = None,
    source_frame: str = "unspecified",
    target_frame: str = "unspecified",
    target_axis_patient_directions: tuple[str, str, str] | None = None,
) -> PreparedContrast:
    """Load, optionally reorient, and center-pad a MATLAB contrast image."""

    resolved = Path(path).expanduser().resolve(strict=False)
    if not resolved.is_file():
        raise EncodingInputError(
            f"contrast image does not exist: {resolved}"
        )
    try:
        content = loadmat(
            resolved,
            variable_names=[variable_name],
            squeeze_me=False,
        )
    except (OSError, ValueError, NotImplementedError) as exc:
        raise EncodingInputError(
            f"could not load contrast image {resolved}: {exc}"
        ) from exc
    if variable_name not in content:
        raise EncodingInputError(
            f"contrast file is missing variable {variable_name!r}"
        )
    source = np.asarray(content[variable_name])
    if source.ndim != 3:
        raise EncodingInputError(
            f"contrast image must be three-dimensional; got {source.shape}"
        )
    if not np.issubdtype(source.dtype, np.number):
        raise EncodingInputError("contrast image must be numeric")
    if not np.all(np.isfinite(source)):
        raise EncodingInputError("contrast image contains non-finite values")

    oriented = (
        np.asarray(source, dtype=np.float32)
        if source_to_target is None
        else reorient_spatial_array(
            np.asarray(source, dtype=np.float32), source_to_target
        )
    )
    padding = center_padding(oriented.shape, target_shape)
    padded = np.pad(
        oriented,
        padding,
        mode="constant",
        constant_values=0,
    )
    return PreparedContrast(
        path=resolved,
        source_frame=source_frame,
        target_frame=target_frame,
        source_shape=source.shape,
        oriented_shape=oriented.shape,
        target_shape=target_shape,
        padding=padding,
        target_axis_patient_directions=target_axis_patient_directions,
        image=padded,
    )


def _center_planes(array: np.ndarray) -> dict[str, np.ndarray]:
    x_center, y_center, z_center = (
        size // 2 for size in array.shape[:3]
    )
    return {
        "xy": np.asarray(array[:, :, z_center]),
        "xz": np.asarray(array[:, y_center, :]),
        "yz": np.asarray(array[x_center, :, :]),
    }


def save_logical_input_preview(
    prepared: PreparedContrast,
    logical_coil: np.ndarray,
    transforms: CoordinateTransforms,
    output_path: str | Path,
    *,
    coil_index: int,
) -> LogicalInputPreview:
    """Save center planes that verify logical image/coil alignment pre-NUFFT."""

    coil = np.asarray(logical_coil, dtype=np.complex64)
    if coil.shape != prepared.image.shape:
        raise EncodingInputError(
            f"logical contrast and coil shapes differ: "
            f"{prepared.image.shape} != {coil.shape}"
        )
    if not np.all(np.isfinite(coil)):
        raise EncodingInputError("logical sensitivity map is not finite")

    contrast_planes = _center_planes(prepared.image)
    coil_planes = _center_planes(coil)
    weighted_planes = {
        plane: contrast_planes[plane] * coil_planes[plane]
        for plane in contrast_planes
    }
    variables: dict[str, object] = {
        "pcs_to_dcs": transforms.pcs_to_dcs,
        "logical_to_dcs": transforms.logical_to_dcs,
        "dcs_to_logical": transforms.dcs_to_logical,
        "pcs_to_logical": transforms.pcs_to_logical,
        "contrast_logical_shape": np.asarray(
            prepared.image.shape, dtype=np.int32
        ),
        "coil_logical_shape": np.asarray(coil.shape, dtype=np.int32),
        "logical_axis_patient_directions": np.asarray(
            transforms.logical_axis_patient_directions, dtype=object
        ),
        "coordinate_frame": "sequence-logical",
        "coil_index_zero_based": np.asarray([[coil_index]], dtype=np.int32),
    }
    for plane, values in contrast_planes.items():
        variables[f"contrast_logical_{plane}"] = values
        variables[f"coil_logical_{plane}"] = coil_planes[plane]
        variables[f"coil_weighted_logical_{plane}"] = weighted_planes[plane]

    destination = Path(output_path).expanduser().resolve(strict=False)
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
            variables,
            appendmat=False,
            do_compression=True,
        )
        reopened = loadmat(
            temporary_path,
            variable_names=(
                "pcs_to_logical",
                "contrast_logical_shape",
                "contrast_logical_xy",
                "coil_logical_xy",
            ),
        )
        if reopened["pcs_to_logical"].shape != (3, 3):
            raise EncodingInputError(
                "logical preview matrix changed during MATLAB writing"
            )
        if tuple(
            int(value)
            for value in reopened["contrast_logical_shape"].reshape(-1)
        ) != prepared.image.shape:
            raise EncodingInputError(
                "logical preview shape changed during MATLAB writing"
            )
        temporary_path.chmod(0o644)
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return LogicalInputPreview(
        output_path=destination,
        contrast_shape=prepared.image.shape,
        coil_shape=coil.shape,
        coil_index=coil_index,
        axis_patient_directions=(
            transforms.logical_axis_patient_directions
        ),
        file_size_bytes=destination.stat().st_size,
    )


def format_prepared_contrast(prepared: PreparedContrast) -> str:
    """Format contrast-grid preparation and validation."""

    return "\n".join(
        (
            "Encoding image preparation",
            f"Source:                {prepared.path}",
            f"Source frame:          {prepared.source_frame}",
            f"Source shape:          {prepared.source_shape}",
            f"Oriented shape:        {prepared.oriented_shape}",
            f"Target frame:          {prepared.target_frame}",
            f"Target shape:          {prepared.target_shape}",
            (
                "Target axis directions: "
                + (
                    "not declared"
                    if prepared.target_axis_patient_directions is None
                    else ", ".join(
                        prepared.target_axis_patient_directions
                    )
                )
            ),
            f"Padding:               {prepared.padding}",
            f"Data type:             {prepared.image.dtype}",
            (
                f"Finite voxels:         "
                f"{np.count_nonzero(np.isfinite(prepared.image)):,} / "
                f"{prepared.image.size:,}"
            ),
            f"Signal range:          {prepared.image.min():g} to "
            f"{prepared.image.max():g}",
        )
    )


def format_logical_input_preview(report: LogicalInputPreview) -> str:
    """Format one pre-NUFFT logical-frame inspection artifact."""

    return "\n".join(
        (
            "Logical input preview (Step 5)",
            "NUFFT executed:          no",
            f"Coordinate frame:       sequence-logical",
            (
                "Patient directions:    "
                + ", ".join(report.axis_patient_directions)
            ),
            f"Contrast shape:         {report.contrast_shape}",
            f"Coil shape:             {report.coil_shape}",
            f"Preview coil:           {report.coil_index} (zero-based)",
            f"MATLAB output:          {report.output_path}",
            f"File size:              {report.file_size_bytes:,} bytes",
            "Verification:           PASS",
        )
    )
