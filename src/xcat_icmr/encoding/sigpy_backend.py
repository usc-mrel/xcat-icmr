"""SigPy implementation of paired forward and adjoint 3-D NUFFT."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import tempfile

import numpy as np

from xcat_icmr.encoding.trajectory import EncodingTrajectory


class NufftBackendError(Exception):
    """Raised when a NUFFT backend cannot execute safely."""


def _sigpy():
    cache = Path(tempfile.gettempdir()) / "xcat-icmr-numba-cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache))
    try:
        import sigpy as sp
    except (ImportError, RuntimeError) as exc:
        raise NufftBackendError(f"could not import SigPy: {exc}") from exc
    return sp


class SigpyNufftBackend:
    """Three-dimensional SigPy NUFFT with explicit device and kernel settings."""

    def __init__(
        self,
        *,
        device_id: int,
        oversampling: float = 1.25,
        kernel_width: float = 4.0,
    ) -> None:
        if device_id < -1:
            raise ValueError("device_id must be -1 or a non-negative GPU ID")
        if oversampling < 1:
            raise ValueError("oversampling must be at least 1")
        if kernel_width <= 0:
            raise ValueError("kernel_width must be positive")
        if device_id >= 0 and importlib.util.find_spec("cupy") is None:
            raise NufftBackendError(
                f"GPU {device_id} was requested but CuPy is not installed"
            )
        self.device_id = device_id
        self.oversampling = float(oversampling)
        self.kernel_width = float(kernel_width)

    def forward(
        self,
        image: np.ndarray,
        trajectory: EncodingTrajectory,
    ) -> np.ndarray:
        """Apply the unitary-centered SigPy forward NUFFT."""

        array = np.asarray(image)
        if array.ndim != 3:
            raise NufftBackendError(
                f"NUFFT image must be three-dimensional; got {array.shape}"
            )
        for axis, (maximum, size) in enumerate(
            zip(
                trajectory.maximum_absolute_coordinate,
                array.shape,
                strict=True,
            )
        ):
            if maximum > size / 2 + 1e-5:
                raise NufftBackendError(
                    f"trajectory axis {axis} exceeds forward image Nyquist "
                    f"range: {maximum:g} > {size / 2:g}"
                )
        if not np.all(np.isfinite(array)):
            raise NufftBackendError("NUFFT image contains non-finite values")
        sp = _sigpy()
        device = sp.Device(self.device_id)
        try:
            with device:
                device_image = sp.to_device(
                    np.asarray(array, dtype=np.complex64), device
                )
                device_coordinates = sp.to_device(
                    trajectory.coordinates, device
                )
                result = sp.nufft(
                    device_image,
                    device_coordinates,
                    oversamp=self.oversampling,
                    width=self.kernel_width,
                )
                output = sp.to_device(result, sp.cpu_device)
        except Exception as exc:
            raise NufftBackendError(f"SigPy forward NUFFT failed: {exc}") from exc
        return np.asarray(output, dtype=np.complex64)

    def adjoint(
        self,
        kspace: np.ndarray,
        trajectory: EncodingTrajectory,
    ) -> np.ndarray:
        """Apply a SigPy adjoint on the trajectory's matrix."""

        values = np.asarray(kspace)
        if values.shape != (trajectory.point_count,):
            raise NufftBackendError(
                f"k-space shape {values.shape} does not match "
                f"{trajectory.point_count} trajectory points"
            )
        if not np.all(np.isfinite(values)):
            raise NufftBackendError("NUFFT k-space contains non-finite values")
        sp = _sigpy()
        device = sp.Device(self.device_id)
        try:
            with device:
                device_values = sp.to_device(
                    np.asarray(values, dtype=np.complex64), device
                )
                device_coordinates = sp.to_device(
                    trajectory.coordinates, device
                )
                result = sp.nufft_adjoint(
                    device_values,
                    device_coordinates,
                    oshape=trajectory.matrix_shape,
                    oversamp=self.oversampling,
                    width=self.kernel_width,
                )
                output = sp.to_device(result, sp.cpu_device)
        except Exception as exc:
            raise NufftBackendError(f"SigPy adjoint NUFFT failed: {exc}") from exc
        return np.asarray(output, dtype=np.complex64)
