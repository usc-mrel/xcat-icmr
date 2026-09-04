"""K-space coordinate-origin changes for RF-centered imaging FOVs."""

from __future__ import annotations

import numpy as np


class FovCenteringError(ValueError):
    """Raised when an encoding FOV shift is not well defined."""


def rf_centering_phase_ramp(
    k_axis_per_m: np.ndarray,
    *,
    rf_center_shift_mm: float,
) -> np.ndarray:
    """Return the phase ramp that places the shifted RF center at image zero.

    For the forward convention ``exp(-i 2 pi k r)``, expressing the image in
    coordinates ``r' = r - r0`` gives ``K'(k) = exp(+i 2 pi k r0) K(k)``.
    """

    coordinates = np.asarray(k_axis_per_m, dtype=np.float64)
    if coordinates.ndim != 2 or not np.all(np.isfinite(coordinates)):
        raise FovCenteringError(
            "RF-axis trajectory must be finite with shape [sample, arm]"
        )
    if not np.isfinite(rf_center_shift_mm):
        raise FovCenteringError("RF center shift must be finite")
    shift_m = float(rf_center_shift_mm) * 1e-3
    return np.asarray(
        np.exp(2j * np.pi * coordinates * shift_m), dtype=np.complex64
    )


def center_kspace_on_rf_profile(
    kspace: np.ndarray,
    k_axis_per_m: np.ndarray,
    *,
    rf_center_shift_mm: float,
) -> np.ndarray:
    """Apply RF-centered coordinates to arm-ordered k-space in-place safely."""

    values = np.asarray(kspace, dtype=np.complex64)
    coordinates = np.asarray(k_axis_per_m)
    if values.shape[:2] != coordinates.shape:
        raise FovCenteringError(
            f"k-space leading shape {values.shape[:2]} does not match "
            f"trajectory shape {coordinates.shape}"
        )
    ramp = rf_centering_phase_ramp(
        coordinates, rf_center_shift_mm=rf_center_shift_mm
    )
    if values.ndim == 2:
        return np.asarray(values * ramp, dtype=np.complex64)
    if values.ndim == 3:
        return np.asarray(values * ramp[:, :, None], dtype=np.complex64)
    raise FovCenteringError("k-space must have shape [sample, arm] or [sample, arm, coil]")
