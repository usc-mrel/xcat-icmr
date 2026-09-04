"""Vectorized conversion from XCAT labels to quantitative MR properties."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from xcat_icmr.tissue.models import TissueLibrary, XcatLabel


class LabelMappingError(ValueError):
    """Raised when a label volume cannot be mapped safely."""


@dataclass(frozen=True)
class TissueParameterVolumes:
    """Voxel-aligned tissue properties produced from an XCAT label volume."""

    t1_ms: npt.NDArray[np.floating]
    t2_ms: npt.NDArray[np.floating]
    proton_density_percent: npt.NDArray[np.floating]
    mapped: npt.NDArray[np.bool_]


def _property_tables(
    library: TissueLibrary,
    dtype: npt.DTypeLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    size = max(int(label) for label in XcatLabel) + 1
    t1 = np.zeros(size, dtype=dtype)
    t2 = np.zeros(size, dtype=dtype)
    pd = np.zeros(size, dtype=dtype)
    mapped = np.zeros(size, dtype=np.bool_)

    for group in library.groups:
        indices = np.fromiter((int(label) for label in group.labels), dtype=np.uint8)
        t1[indices] = group.properties.t1_ms
        t2[indices] = group.properties.t2_ms
        pd[indices] = group.properties.proton_density_percent
        mapped[indices] = True
    return t1, t2, pd, mapped


def map_labels_to_tissue_properties(
    labels: npt.ArrayLike,
    library: TissueLibrary,
    *,
    dtype: npt.DTypeLike = np.float32,
    unknown: Literal["error", "zero"] = "error",
) -> TissueParameterVolumes:
    """Map an XCAT label array to T1, T2, and proton-density arrays.

    Known but unassigned XCAT labels preserve MATLAB behavior: their properties
    are zero and ``mapped`` is False. Values outside XCAT's 0--71 label range
    raise by default; ``unknown="zero"`` maps them to zero with ``mapped=False``.
    """

    if unknown not in ("error", "zero"):
        raise ValueError("unknown must be either 'error' or 'zero'")

    label_array = np.asarray(labels)
    if not np.issubdtype(label_array.dtype, np.number):
        raise LabelMappingError("XCAT labels must be numeric")
    if not np.all(np.isfinite(label_array)):
        raise LabelMappingError("XCAT labels contain non-finite values")
    if not np.all(label_array == np.floor(label_array)):
        raise LabelMappingError("XCAT labels must be integer-valued")

    labels_int = label_array.astype(np.int64, copy=False)
    max_known = max(int(label) for label in XcatLabel)
    valid = (labels_int >= 0) & (labels_int <= max_known)
    if unknown == "error" and not np.all(valid):
        bad = np.unique(labels_int[~valid])
        preview = ", ".join(str(value) for value in bad[:8])
        suffix = " ..." if bad.size > 8 else ""
        raise LabelMappingError(
            f"XCAT labels outside the supported 0--{max_known} range: "
            f"{preview}{suffix}"
        )

    t1_table, t2_table, pd_table, mapped_table = _property_tables(library, dtype)
    safe_labels = np.where(valid, labels_int, 0)
    t1 = t1_table[safe_labels]
    t2 = t2_table[safe_labels]
    pd = pd_table[safe_labels]
    mapped = mapped_table[safe_labels]

    if not np.all(valid):
        t1 = np.where(valid, t1, 0)
        t2 = np.where(valid, t2, 0)
        pd = np.where(valid, pd, 0)
        mapped = mapped & valid

    return TissueParameterVolumes(
        t1_ms=t1,
        t2_ms=t2,
        proton_density_percent=pd,
        mapped=mapped,
    )
