"""MATLAB-equivalent steady-state balanced SSFP signal model."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from xcat_icmr.tissue.mapping import TissueParameterVolumes


class BssfpSignalError(ValueError):
    """Raised when bSSFP inputs are not physically or numerically valid."""


def _validate_sequence_parameters(
    flip_angle_deg: npt.ArrayLike,
    te_ms: float,
    tr_ms: float,
) -> np.ndarray:
    flip = np.asarray(flip_angle_deg, dtype=np.float64)
    if not np.all(np.isfinite(flip)):
        raise BssfpSignalError("flip angle must be finite")
    if not math.isfinite(te_ms) or not math.isfinite(tr_ms):
        raise BssfpSignalError("TE and TR must be finite")
    if np.any(flip < 0.0) or np.any(flip > 180.0):
        raise BssfpSignalError("flip_angle_deg must be between 0 and 180")
    if te_ms < 0.0:
        raise BssfpSignalError("te_ms must be non-negative")
    if tr_ms <= 0.0:
        raise BssfpSignalError("tr_ms must be positive")
    if te_ms > tr_ms:
        raise BssfpSignalError("te_ms cannot be greater than tr_ms")
    return flip


def _as_nonnegative_finite_array(
    values: npt.ArrayLike,
    name: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise BssfpSignalError(f"{name} contains non-finite values")
    if np.any(array < 0):
        raise BssfpSignalError(f"{name} must be non-negative")
    return array


def bssfp_signal(
    t1_ms: npt.ArrayLike,
    t2_ms: npt.ArrayLike,
    proton_density_percent: npt.ArrayLike,
    *,
    flip_angle_deg: npt.ArrayLike,
    te_ms: float,
    tr_ms: float,
    off_resonance_enabled: bool = False,
    dtype: npt.DTypeLike = np.float32,
) -> npt.NDArray[np.floating]:
    """Calculate the steady-state bSSFP signal used by the MATLAB simulator.

    The implementation is the vectorized equivalent of ``s_bssfp.m`` and
    ``bssfp_ss_on_resonance_`` in ``tissue2signal.m``. T1, T2, TE, and TR are
    all expressed in milliseconds; proton density uses the MATLAB percentage
    scale (for example, blood is 95 rather than 0.95).
    """

    if off_resonance_enabled:
        raise NotImplementedError(
            "off-resonance bSSFP signal simulation is not implemented"
        )

    flip_deg = _validate_sequence_parameters(flip_angle_deg, te_ms, tr_ms)
    t1 = _as_nonnegative_finite_array(t1_ms, "t1_ms")
    t2 = _as_nonnegative_finite_array(t2_ms, "t2_ms")
    pd = _as_nonnegative_finite_array(
        proton_density_percent, "proton_density_percent"
    )

    try:
        t1, t2, pd, flip_deg = np.broadcast_arrays(t1, t2, pd, flip_deg)
    except ValueError as exc:
        raise BssfpSignalError(
            "T1, T2, and proton-density arrays are not broadcast-compatible"
        ) from exc

    flip_rad = np.deg2rad(flip_deg)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        e1 = np.exp(-float(tr_ms) / t1)
        e2 = np.exp(-float(tr_ms) / t2)
        numerator = (
            pd
            * np.sin(flip_rad)
            * (1.0 - e1)
            * np.exp(-float(te_ms) / t2)
        )
        denominator = (
            1.0
            - np.cos(flip_rad) * (e1 - e2)
            - e1 * e2
        )
        signal = numerator / denominator

    # MATLAB explicitly converts NaN tissue signals to zero. Infinite values
    # are also made zero here so an invalid singularity cannot enter k-space.
    signal = np.where(np.isfinite(signal), signal, 0.0)
    return np.asarray(signal, dtype=dtype)


def bssfp_signal_from_tissue_properties(
    properties: TissueParameterVolumes,
    *,
    flip_angle_deg: npt.ArrayLike,
    te_ms: float,
    tr_ms: float,
    off_resonance_enabled: bool = False,
    dtype: npt.DTypeLike = np.float32,
) -> npt.NDArray[np.floating]:
    """Calculate a bSSFP image from voxel-aligned tissue-property volumes."""

    return bssfp_signal(
        properties.t1_ms,
        properties.t2_ms,
        properties.proton_density_percent,
        flip_angle_deg=flip_angle_deg,
        te_ms=te_ms,
        tr_ms=tr_ms,
        off_resonance_enabled=off_resonance_enabled,
        dtype=dtype,
    )
