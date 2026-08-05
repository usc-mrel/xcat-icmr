"""Streaming four-dimensional NRRD export for dynamic contrast volumes."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from scipy.io import loadmat


class NrrdExportError(Exception):
    """Raised when a dynamic NRRD cannot be exported safely."""


@dataclass(frozen=True)
class NrrdExport:
    """Verified metadata for one attached-data 4-D NRRD."""

    output_path: Path
    spatial_shape: tuple[int, int, int]
    frame_count: int
    voxel_size_mm: tuple[float, float, float]
    time_step_s: float
    dtype: str
    coordinate_frame: str
    file_size_bytes: int
    data_size_bytes: int


ProgressCallback = Callable[[int, int], None]


def _load_contrast(path: Path, variable_name: str) -> np.ndarray:
    if not path.is_file():
        raise NrrdExportError(f"contrast frame does not exist: {path}")
    try:
        content = loadmat(
            path,
            variable_names=[variable_name],
            squeeze_me=False,
        )
    except (OSError, ValueError, NotImplementedError) as exc:
        raise NrrdExportError(
            f"could not read contrast frame {path}: {exc}"
        ) from exc
    if variable_name not in content:
        raise NrrdExportError(
            f"contrast frame {path} is missing {variable_name!r}"
        )
    image = np.asarray(content[variable_name])
    if image.ndim != 3:
        raise NrrdExportError(
            f"contrast frame must be 3-D; got {image.shape} in {path}"
        )
    if not np.issubdtype(image.dtype, np.floating):
        raise NrrdExportError(
            f"contrast frame must be floating point; got {image.dtype}"
        )
    if not np.all(np.isfinite(image)):
        raise NrrdExportError(f"contrast frame is not finite: {path}")
    return image


def _header(
    *,
    shape: tuple[int, int, int],
    frame_count: int,
    voxel_size_mm: tuple[float, float, float],
    time_step_s: float,
    nrrd_type: str = "float",
    content_description: str = "bSSFP motion cycle",
    modality: str = "MRI",
) -> bytes:
    origin = tuple(
        -(size // 2) * spacing
        for size, spacing in zip(shape, voxel_size_mm, strict=True)
    )
    directions = (
        f"({voxel_size_mm[0]:g},0,0) "
        f"(0,{voxel_size_mm[1]:g},0) "
        f"(0,0,{voxel_size_mm[2]:g}) none"
    )
    return (
        "\n".join(
            (
                "NRRD0005",
                f"# Complete XCAT-iCMR {content_description}",
                f"type: {nrrd_type}",
                "dimension: 4",
                f"sizes: {shape[0]} {shape[1]} {shape[2]} {frame_count}",
                "encoding: raw",
                "endian: little",
                "space: left-posterior-superior",
                f"space directions: {directions}",
                (
                    "space origin: "
                    f"({origin[0]:g},{origin[1]:g},{origin[2]:g})"
                ),
                "kinds: domain domain domain list",
                'space units: "mm" "mm" "mm"',
                'labels: "Sag (L+)" "Cor (P+)" "Tra (S+)" "time"',
                "centerings: cell cell cell cell",
                "measurement frame: (1,0,0) (0,1,0) (0,0,1)",
                f"modality:={modality}",
                "xcat_icmr_coordinate_frame:=XCAT PCS [Sag, Cor, Tra]",
                f"xcat_icmr_time_step_s:={time_step_s:g}",
                "xcat_icmr_time_origin_s:=0",
                "",
                "",
            )
        )
    ).encode("ascii")


def _load_labels(path: Path, variable_name: str) -> np.ndarray:
    if not path.is_file():
        raise NrrdExportError(f"label frame does not exist: {path}")
    try:
        content = loadmat(
            path,
            variable_names=[variable_name],
            squeeze_me=False,
        )
    except (OSError, ValueError, NotImplementedError) as exc:
        raise NrrdExportError(
            f"could not read label frame {path}: {exc}"
        ) from exc
    if variable_name not in content:
        raise NrrdExportError(
            f"label frame {path} is missing {variable_name!r}"
        )
    labels = np.asarray(content[variable_name])
    if labels.ndim != 3 or not np.issubdtype(labels.dtype, np.number):
        raise NrrdExportError(
            f"label frame must be a numeric 3-D array; got "
            f"{labels.shape} {labels.dtype}"
        )
    if (
        not np.all(np.isfinite(labels))
        or np.any(labels < 0)
        or np.any(labels > np.iinfo(np.uint16).max)
        or not np.all(labels == np.rint(labels))
    ):
        raise NrrdExportError(
            f"label values cannot be represented exactly as uint16: {path}"
        )
    return labels


def export_label_series_nrrd(
    frame_paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    voxel_size_mm: tuple[float, float, float],
    time_step_s: float,
    variable_name: str = "P",
    overwrite: bool = False,
    progress: ProgressCallback | None = None,
) -> NrrdExport:
    """Stream PCS tissue labels into one uint16 attached-data 4-D NRRD."""

    paths = tuple(
        Path(path).expanduser().resolve(strict=False)
        for path in frame_paths
    )
    if not paths:
        raise NrrdExportError("at least one label frame is required")
    voxel = tuple(float(value) for value in voxel_size_mm)
    if len(voxel) != 3 or not np.all(np.isfinite(voxel)) or any(
        value <= 0 for value in voxel
    ):
        raise NrrdExportError(
            "voxel_size_mm must contain three positive values"
        )
    if not np.isfinite(time_step_s) or time_step_s <= 0:
        raise NrrdExportError("time_step_s must be positive and finite")

    first = _load_labels(paths[0], variable_name)
    shape = tuple(int(value) for value in first.shape)
    destination = Path(output_path).expanduser().resolve(strict=False)
    if destination.exists() and not overwrite:
        raise NrrdExportError(
            f"NRRD output already exists: {destination}; pass --overwrite"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.partial")
    temporary.unlink(missing_ok=True)
    header = _header(
        shape=shape,
        frame_count=len(paths),
        voxel_size_mm=voxel,
        time_step_s=float(time_step_s),
        nrrd_type="ushort",
        content_description="tissue-label motion cycle",
        modality="XCAT tissue labels",
    )
    data_bytes = int(np.prod(shape, dtype=np.int64)) * len(paths) * 2

    try:
        with temporary.open("wb") as handle:
            handle.write(header)
            for index, path in enumerate(paths):
                labels = (
                    first
                    if index == 0
                    else _load_labels(path, variable_name)
                )
                if labels.shape != shape:
                    raise NrrdExportError(
                        f"label shape changed at frame {index + 1}: "
                        f"{labels.shape} != {shape}"
                    )
                frame = np.asfortranarray(labels, dtype="<u2")
                frame.ravel(order="F").tofile(handle)
                if progress is not None:
                    progress(index + 1, len(paths))
            handle.flush()
            os.fsync(handle.fileno())
        expected_size = len(header) + data_bytes
        actual_size = temporary.stat().st_size
        if actual_size != expected_size:
            raise NrrdExportError(
                f"NRRD size verification failed: {actual_size} != "
                f"{expected_size}"
            )
        temporary.chmod(0o644)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    return NrrdExport(
        output_path=destination,
        spatial_shape=shape,
        frame_count=len(paths),
        voxel_size_mm=voxel,
        time_step_s=float(time_step_s),
        dtype="uint16",
        coordinate_frame="XCAT PCS [Sag, Cor, Tra] / LPS",
        file_size_bytes=destination.stat().st_size,
        data_size_bytes=data_bytes,
    )


def export_contrast_series_nrrd(
    frame_paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    voxel_size_mm: tuple[float, float, float],
    time_step_s: float,
    variable_name: str = "image",
    overwrite: bool = False,
    progress: ProgressCallback | None = None,
) -> NrrdExport:
    """Stream PCS contrast frames into one raw, attached-data 4-D NRRD."""

    paths = tuple(Path(path).expanduser().resolve(strict=False) for path in frame_paths)
    if not paths:
        raise NrrdExportError("at least one contrast frame is required")
    voxel = tuple(float(value) for value in voxel_size_mm)
    if len(voxel) != 3 or not np.all(np.isfinite(voxel)) or any(
        value <= 0 for value in voxel
    ):
        raise NrrdExportError("voxel_size_mm must contain three positive values")
    if not np.isfinite(time_step_s) or time_step_s <= 0:
        raise NrrdExportError("time_step_s must be positive and finite")

    first = _load_contrast(paths[0], variable_name)
    shape = tuple(int(value) for value in first.shape)
    destination = Path(output_path).expanduser().resolve(strict=False)
    if destination.exists() and not overwrite:
        raise NrrdExportError(
            f"NRRD output already exists: {destination}; pass --overwrite"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.partial")
    temporary.unlink(missing_ok=True)
    header = _header(
        shape=shape,
        frame_count=len(paths),
        voxel_size_mm=voxel,
        time_step_s=float(time_step_s),
    )
    data_bytes = int(np.prod(shape, dtype=np.int64)) * len(paths) * 4

    try:
        with temporary.open("wb") as handle:
            handle.write(header)
            for index, path in enumerate(paths):
                image = first if index == 0 else _load_contrast(
                    path, variable_name
                )
                if image.shape != shape:
                    raise NrrdExportError(
                        f"contrast shape changed at frame {index + 1}: "
                        f"{image.shape} != {shape}"
                    )
                # NRRD axis zero is fastest. A Fortran-ordered 3-D frame
                # therefore stores Sag, then Cor, then Tra, with time slowest.
                frame = np.asfortranarray(image, dtype="<f4")
                frame.ravel(order="F").tofile(handle)
                if progress is not None:
                    progress(index + 1, len(paths))
            handle.flush()
            os.fsync(handle.fileno())
        expected_size = len(header) + data_bytes
        actual_size = temporary.stat().st_size
        if actual_size != expected_size:
            raise NrrdExportError(
                f"NRRD size verification failed: {actual_size} != "
                f"{expected_size}"
            )
        temporary.chmod(0o644)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    return NrrdExport(
        output_path=destination,
        spatial_shape=shape,
        frame_count=len(paths),
        voxel_size_mm=voxel,
        time_step_s=float(time_step_s),
        dtype="float32",
        coordinate_frame="XCAT PCS [Sag, Cor, Tra] / LPS",
        file_size_bytes=destination.stat().st_size,
        data_size_bytes=data_bytes,
    )


def format_nrrd_export(report: NrrdExport) -> str:
    """Format a verified dynamic NRRD export."""

    return "\n".join(
        (
            "Dynamic NRRD export",
            f"Output:           {report.output_path}",
            f"Spatial shape:    {report.spatial_shape}",
            f"Frames:           {report.frame_count}",
            f"4-D sizes:        {report.spatial_shape + (report.frame_count,)}",
            f"Voxel size:       {report.voxel_size_mm} mm",
            f"Time step:        {report.time_step_s:g} s",
            f"Data type:        {report.dtype}",
            f"Coordinate frame: {report.coordinate_frame}",
            f"Data bytes:       {report.data_size_bytes:,}",
            f"File bytes:       {report.file_size_bytes:,}",
            "Encoding:         raw attached data",
            "Verification:     PASS",
        )
    )
