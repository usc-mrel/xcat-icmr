"""Gadolinium relaxation and sparse bSSFP signal calculation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from xcat_icmr.intervention.balloon import SparseBalloonSupport
from xcat_icmr.signal.bssfp import bssfp_signal
from xcat_icmr.tissue.models import TissueProperties


class GdSignalError(ValueError):
    """Raised when a Gd signal or local flip angles cannot be calculated."""


@dataclass(frozen=True)
class GdRelaxivity:
    name: str
    r1_per_s_per_mM: float
    r2_per_s_per_mM: float


@dataclass(frozen=True)
class GdSignal:
    concentration_mM: float
    t1_ms: float
    t2_ms: float
    flip_angle_deg: np.ndarray
    values: np.ndarray


GD_DEFAULT = GdRelaxivity(
    name="gd-default",
    r1_per_s_per_mM=5.2,
    r2_per_s_per_mM=7.0,
)


def gd_relaxation_times_ms(
    carrier: TissueProperties,
    concentration_mM: float,
    *,
    library: str = "gd-default",
) -> tuple[float, float]:
    """Apply the relaxivity convention used by ``gen_cath_kspace.py``."""

    if library != GD_DEFAULT.name:
        raise GdSignalError(f"unknown Gd relaxivity library: {library}")
    if not np.isfinite(concentration_mM) or concentration_mM <= 0.0:
        raise GdSignalError("concentration_mM must be positive")
    if carrier.t1_ms <= 0.0 or carrier.t2_ms <= 0.0:
        raise GdSignalError("carrier T1 and T2 must be positive")
    t1_s = 1.0 / (
        1.0 / (carrier.t1_ms * 1e-3)
        + GD_DEFAULT.r1_per_s_per_mM * concentration_mM
    )
    t2_s = 1.0 / (
        1.0 / (carrier.t2_ms * 1e-3)
        + GD_DEFAULT.r2_per_s_per_mM * concentration_mM
    )
    return t1_s * 1e3, t2_s * 1e3


def sample_sparse_flip_angles(
    profile_path: str | Path,
    support: SparseBalloonSupport,
) -> np.ndarray:
    """Broadcast the applied 1-D RF profile over the local balloon box."""

    path = Path(profile_path).expanduser().resolve(strict=False)
    try:
        content = loadmat(
            path,
            variable_names=[
                "applied_effective_flip_angle_deg",
                "pcs_axis_zero_based",
                "pcs_image_shape",
            ],
            squeeze_me=False,
        )
    except (OSError, ValueError, NotImplementedError) as exc:
        raise GdSignalError(f"could not read RF profile {path}: {exc}") from exc
    required = {
        "applied_effective_flip_angle_deg",
        "pcs_axis_zero_based",
        "pcs_image_shape",
    }
    if not required.issubset(content):
        raise GdSignalError(f"RF profile is missing variables: {required - content.keys()}")
    image_shape = tuple(
        int(value) for value in np.asarray(content["pcs_image_shape"]).reshape(-1)
    )
    if image_shape != support.volume_shape:
        raise GdSignalError(
            f"RF profile shape {image_shape} does not match {support.volume_shape}"
        )
    axis = int(np.asarray(content["pcs_axis_zero_based"]).squeeze())
    if axis not in {0, 1, 2}:
        raise GdSignalError(f"invalid PCS RF axis: {axis}")
    profile = np.asarray(
        content["applied_effective_flip_angle_deg"], dtype=np.float64
    ).reshape(-1)
    if len(profile) != support.volume_shape[axis]:
        raise GdSignalError("applied RF profile has the wrong length")
    start = int(support.bounding_box_start_ijk[axis])
    stop = start + support.occupancy.shape[axis]
    local_profile = profile[start:stop]
    broadcast_shape = [1, 1, 1]
    broadcast_shape[axis] = len(local_profile)
    return np.asarray(
        np.broadcast_to(
            local_profile.reshape(broadcast_shape), support.occupancy.shape
        ),
        dtype=np.float64,
    ).copy()


def sample_sparse_flip_angles_from_profile(
    applied_effective_flip_angle_deg: np.ndarray,
    *,
    pcs_axis: int,
    pcs_image_shape: tuple[int, int, int],
    support: SparseBalloonSupport,
) -> np.ndarray:
    """Broadcast an in-memory applied RF profile over a balloon support."""

    if tuple(pcs_image_shape) != support.volume_shape or pcs_axis not in {0, 1, 2}:
        raise GdSignalError("in-memory RF profile geometry is incompatible")
    profile = np.asarray(applied_effective_flip_angle_deg, dtype=np.float64).reshape(-1)
    if profile.size != support.volume_shape[pcs_axis]:
        raise GdSignalError("in-memory applied RF profile has the wrong length")
    start = int(support.bounding_box_start_ijk[pcs_axis])
    stop = start + support.occupancy.shape[pcs_axis]
    shape = [1, 1, 1]
    shape[pcs_axis] = stop - start
    return np.asarray(
        np.broadcast_to(profile[start:stop].reshape(shape), support.occupancy.shape),
        dtype=np.float64,
    ).copy()


def calculate_gd_bssfp_signal(
    *,
    carrier: TissueProperties,
    concentration_mM: float,
    flip_angle_deg: np.ndarray,
    te_ms: float,
    tr_ms: float,
    relaxivity_library: str = "gd-default",
) -> GdSignal:
    """Calculate Gd bSSFP values for a reusable flip-angle array."""

    flip = np.asarray(flip_angle_deg, dtype=np.float64)
    if not np.all(np.isfinite(flip)):
        raise GdSignalError("flip-angle array contains non-finite values")
    t1_ms, t2_ms = gd_relaxation_times_ms(
        carrier,
        concentration_mM,
        library=relaxivity_library,
    )
    # Gd changes relaxation, not the signal-unit convention. Use the same
    # carrier proton-density scale as the tissue contrast so local replacement
    # can be formed as occupancy * (Gd - tissue).
    values = bssfp_signal(
        t1_ms,
        t2_ms,
        carrier.proton_density_percent,
        flip_angle_deg=flip,
        te_ms=te_ms,
        tr_ms=tr_ms,
        dtype=np.float32,
    )
    values = np.asarray(values, dtype=np.float32)
    return GdSignal(
        concentration_mM=float(concentration_mM),
        t1_ms=float(t1_ms),
        t2_ms=float(t2_ms),
        flip_angle_deg=flip,
        values=values,
    )


def calculate_sparse_gd_bssfp_signal(
    support: SparseBalloonSupport,
    *,
    carrier: TissueProperties,
    concentration_mM: float,
    flip_angle_deg: np.ndarray,
    te_ms: float,
    tr_ms: float,
    relaxivity_library: str = "gd-default",
) -> GdSignal:
    """Calculate Gd signal on the small local balloon box."""

    flip = np.asarray(flip_angle_deg, dtype=np.float64)
    if flip.shape != support.occupancy.shape:
        raise GdSignalError(
            f"flip-angle shape {flip.shape} does not match "
            f"local balloon shape {support.occupancy.shape}"
        )
    return calculate_gd_bssfp_signal(
        carrier=carrier,
        concentration_mM=concentration_mM,
        flip_angle_deg=flip,
        te_ms=te_ms,
        tr_ms=tr_ms,
        relaxivity_library=relaxivity_library,
    )
