"""Pulseq signature and MATLAB trajectory-metadata reader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np
from scipy.io import loadmat

from xcat_icmr.config.models import SequenceConfig
from xcat_icmr.sequence.orientation import (
    logical_to_dcs_matrix,
    transform_vector_components,
)


class SequenceReadError(Exception):
    """Raised when sequence metadata cannot be resolved or validated."""


@dataclass(frozen=True)
class SequenceData:
    """Resolved sequence parameters with logical and derived DCS trajectories."""

    sequence_path: Path
    metadata_path: Path
    signature_type: str
    signature: str
    coordinate_mode: str
    orientation: str
    logical_to_dcs: np.ndarray
    fov_mm: np.ndarray
    resolution_mm: np.ndarray
    flip_angle_deg: float
    te_ms: float
    tr_ms: float
    logical_kx: np.ndarray
    logical_ky: np.ndarray
    logical_kz: np.ndarray
    dcs_kx: np.ndarray
    dcs_ky: np.ndarray
    dcs_kz: np.ndarray
    density_compensation: np.ndarray
    interleaves: int | None
    planes: int | None
    pre_discard: int | None
    sample_time_s: float | None

    @property
    def kx(self) -> np.ndarray:
        """Pulseq logical kx; retained for encoding compatibility."""

        return self.logical_kx

    @property
    def ky(self) -> np.ndarray:
        """Pulseq logical ky; retained for encoding compatibility."""

        return self.logical_ky

    @property
    def kz(self) -> np.ndarray:
        """Pulseq logical kz; retained for encoding compatibility."""

        return self.logical_kz

    @property
    def trajectory_shape(self) -> tuple[int, int]:
        return self.kx.shape

    @property
    def sample_count(self) -> int:
        return self.trajectory_shape[0]

    @property
    def arm_count(self) -> int:
        return self.trajectory_shape[1]


_HASH_PATTERN = re.compile(r"^[0-9a-fA-F]{32}$")


def read_pulseq_signature(path: str | Path) -> tuple[str, str]:
    """Read the Type and Hash entries from a Pulseq SIGNATURE section."""

    sequence_path = Path(path)
    try:
        lines = sequence_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise SequenceReadError(
            f"could not read Pulseq file {sequence_path}: {exc}"
        ) from exc

    in_signature = False
    values: dict[str, str] = {}
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            in_signature = line.upper() == "[SIGNATURE]"
            continue
        if in_signature:
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                values[parts[0].lower()] = parts[1].strip()

    signature_type = values.get("type")
    signature = values.get("hash")
    if not signature_type or not signature:
        raise SequenceReadError(
            f"Pulseq SIGNATURE section is missing Type or Hash: {sequence_path}"
        )
    if signature_type.lower() != "md5":
        raise SequenceReadError(
            f"unsupported Pulseq signature type {signature_type!r}; expected 'md5'"
        )
    if not _HASH_PATTERN.fullmatch(signature):
        raise SequenceReadError(
            f"invalid Pulseq MD5 signature {signature!r}: {sequence_path}"
        )
    return signature_type.lower(), signature.lower()


def _required_param(param: Any, name: str) -> Any:
    if not hasattr(param, name):
        raise SequenceReadError(f"metadata parameter is missing: param.{name}")
    return getattr(param, name)


def _optional_int(param: Any, name: str) -> int | None:
    if not hasattr(param, name):
        return None
    return int(np.asarray(getattr(param, name)).squeeze())


def _optional_float(param: Any, name: str) -> float | None:
    if not hasattr(param, name):
        return None
    return float(np.asarray(getattr(param, name)).squeeze())


def _as_trajectory(data: dict[str, Any], name: str) -> np.ndarray:
    if name not in data:
        raise SequenceReadError(f"metadata variable is missing: {name}")
    array = np.asarray(data[name], dtype=np.float64)
    if array.ndim != 2:
        raise SequenceReadError(
            f"metadata {name} must be two-dimensional; got shape {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise SequenceReadError(f"metadata {name} contains non-finite values")
    return array


def read_sequence(config: SequenceConfig) -> SequenceData:
    """Resolve a Pulseq sequence and its signature-keyed metadata file."""

    sequence_path = config.resolved_file
    signature_type, signature = read_pulseq_signature(sequence_path)
    metadata_path = config.metadata_directory / f"{signature}.mat"
    if not metadata_path.is_file():
        raise SequenceReadError(
            f"sequence metadata file does not exist: {metadata_path}"
        )

    variable_names = ("param", "kx", "ky", "kz", "w")
    try:
        metadata = loadmat(
            metadata_path,
            squeeze_me=True,
            struct_as_record=False,
            variable_names=variable_names,
        )
    except (OSError, ValueError, NotImplementedError) as exc:
        raise SequenceReadError(
            f"could not load sequence metadata {metadata_path}: {exc}"
        ) from exc

    if "param" not in metadata:
        raise SequenceReadError(f"metadata variable is missing: param")
    param = metadata["param"]

    raw_kx = _as_trajectory(metadata, "kx")
    raw_ky = _as_trajectory(metadata, "ky")
    raw_kz = _as_trajectory(metadata, "kz")
    dcf = _as_trajectory(metadata, "w")
    shapes = {raw_kx.shape, raw_ky.shape, raw_kz.shape, dcf.shape}
    if len(shapes) != 1:
        raise SequenceReadError(
            "kx, ky, kz, and density-compensation shapes must match; "
            f"got {sorted(shapes)}"
        )

    logical_to_dcs = logical_to_dcs_matrix(
        config.coordinate_mode, config.orientation
    )
    dcs_kx, dcs_ky, dcs_kz = transform_vector_components(
        logical_to_dcs, raw_kx, raw_ky, raw_kz
    )
    fov_mm = np.atleast_1d(
        np.asarray(_required_param(param, "fov"), dtype=np.float64).squeeze()
    ) * 1e3
    resolution_mm = np.atleast_1d(
        np.asarray(
            _required_param(param, "spatial_resolution"), dtype=np.float64
        ).squeeze()
    ) * 1e3

    return SequenceData(
        sequence_path=sequence_path,
        metadata_path=metadata_path,
        signature_type=signature_type,
        signature=signature,
        coordinate_mode=config.coordinate_mode,
        orientation=config.orientation,
        logical_to_dcs=logical_to_dcs,
        fov_mm=fov_mm,
        resolution_mm=resolution_mm,
        flip_angle_deg=float(np.asarray(_required_param(param, "FA")).squeeze()),
        te_ms=float(np.asarray(_required_param(param, "TE")).squeeze()),
        tr_ms=float(np.asarray(_required_param(param, "TR")).squeeze()),
        logical_kx=raw_kx,
        logical_ky=raw_ky,
        logical_kz=raw_kz,
        dcs_kx=dcs_kx,
        dcs_ky=dcs_ky,
        dcs_kz=dcs_kz,
        density_compensation=dcf,
        interleaves=_optional_int(param, "interleaves"),
        planes=_optional_int(param, "planes"),
        pre_discard=_optional_int(param, "pre_discard"),
        sample_time_s=_optional_float(param, "dt"),
    )


def _format_vector(values: np.ndarray) -> str:
    return " × ".join(f"{float(value):g}" for value in values)


def format_sequence_summary(data: SequenceData) -> str:
    """Format the sequence parameters used by downstream simulation stages."""

    rows = (
        ("Sequence", data.sequence_path.name),
        ("Signature", data.signature),
        ("Metadata", str(data.metadata_path)),
        ("Coordinate mode", data.coordinate_mode),
        ("Orientation", data.orientation),
        ("Stored trajectory", "Pulseq logical [kx, ky, kz]"),
        (
            "Derived DCS mapping",
            "[X, Y, Z] = "
            + {
                "TRA": "[ky, kx, kz]",
                "COR": "[ky, kz, kx]",
                "SAG": "[kz, ky, kx]",
            }[data.orientation],
        ),
        ("TE", f"{data.te_ms:g} ms"),
        ("TR", f"{data.tr_ms:g} ms"),
        ("Flip angle", f"{data.flip_angle_deg:g} deg"),
        ("FOV", f"{_format_vector(data.fov_mm)} mm"),
        ("Resolution", f"{_format_vector(data.resolution_mm)} mm"),
        (
            "Trajectory shape",
            f"{data.sample_count} samples × {data.arm_count} arms",
        ),
        ("Interleaves", str(data.interleaves or "not provided")),
        ("Planes", str(data.planes or "not provided")),
        (
            "Density compensation",
            f"available, shape {data.density_compensation.shape}",
        ),
    )
    width = max(len(label) for label, _ in rows)
    return "\n".join(f"{label + ':':<{width + 2}} {value}" for label, value in rows)
