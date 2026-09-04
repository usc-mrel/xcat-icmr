"""SigPy implementation of paired forward and adjoint 3-D NUFFT."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import tempfile

import numpy as np

from xcat_icmr.encoding.trajectory import EncodingTrajectory


DEFAULT_NUFFT_OVERSAMPLING = 1.5
DEFAULT_NUFFT_KERNEL_WIDTH = 4.0


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
        oversampling: float = DEFAULT_NUFFT_OVERSAMPLING,
        kernel_width: float = DEFAULT_NUFFT_KERNEL_WIDTH,
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


class SigpyNufftSession:
    """Persistent-device batched NUFFT session.

    The trajectory is uploaded once. Callers may retain other static arrays,
    such as sensitivity maps, DCF, and phase ramps, on ``device`` for the
    lifetime of the session and download only final products.
    """

    def __init__(
        self,
        trajectory: EncodingTrajectory,
        *,
        device_id: int,
        oversampling: float = DEFAULT_NUFFT_OVERSAMPLING,
        kernel_width: float = DEFAULT_NUFFT_KERNEL_WIDTH,
    ) -> None:
        if device_id < -1:
            raise ValueError("device_id must be -1 or a non-negative GPU ID")
        if oversampling < 1 or kernel_width <= 0:
            raise ValueError("invalid NUFFT kernel settings")
        if device_id >= 0 and importlib.util.find_spec("cupy") is None:
            raise NufftBackendError(
                f"GPU {device_id} was requested but CuPy is not installed"
            )
        self.trajectory = trajectory
        self.device_id = device_id
        self.oversampling = float(oversampling)
        self.kernel_width = float(kernel_width)
        self.sp = _sigpy()
        self.device = self.sp.Device(device_id)
        self.xp = self.device.xp
        with self.device:
            self.coordinates = self.sp.to_device(
                trajectory.coordinates, self.device
            )

    @property
    def device_name(self) -> str:
        return "CPU" if self.device_id < 0 else f"GPU {self.device_id}"

    def upload(self, values: np.ndarray, *, dtype=None):
        array = np.asarray(values, dtype=dtype)
        with self.device:
            return self.sp.to_device(array, self.device)

    def empty(self, shape: tuple[int, ...], *, dtype=np.complex64):
        with self.device:
            return self.xp.empty(shape, dtype=dtype)

    def download(self, values) -> np.ndarray:
        return np.asarray(
            self.sp.to_device(values, self.sp.cpu_device)
        )

    def forward_device(self, images):
        """Forward NUFFT for one image or a leading batch of images."""

        if images.ndim < 3:
            raise NufftBackendError("forward input must have at least 3 axes")
        spatial_shape = tuple(int(value) for value in images.shape[-3:])
        for axis, (maximum, size) in enumerate(
            zip(
                self.trajectory.maximum_absolute_coordinate,
                spatial_shape,
                strict=True,
            )
        ):
            if maximum > size / 2 + 1e-5:
                raise NufftBackendError(
                    f"trajectory axis {axis} exceeds forward image Nyquist "
                    f"range: {maximum:g} > {size / 2:g}"
                )
        try:
            with self.device:
                return self.sp.nufft(
                    images,
                    self.coordinates,
                    oversamp=self.oversampling,
                    width=self.kernel_width,
                )
        except Exception as exc:
            raise NufftBackendError(
                f"SigPy batched forward NUFFT failed: {exc}"
            ) from exc

    def adjoint_device(self, kspace):
        """Adjoint NUFFT for one vector or a leading batch of vectors."""

        if kspace.ndim < 1 or int(kspace.shape[-1]) != self.trajectory.point_count:
            raise NufftBackendError(
                "adjoint input must end with the flattened trajectory length"
            )
        batch_shape = tuple(int(value) for value in kspace.shape[:-1])
        output_shape = batch_shape + self.trajectory.matrix_shape
        try:
            with self.device:
                return self.sp.nufft_adjoint(
                    kspace,
                    self.coordinates,
                    oshape=output_shape,
                    oversamp=self.oversampling,
                    width=self.kernel_width,
                )
        except Exception as exc:
            raise NufftBackendError(
                f"SigPy batched adjoint NUFFT failed: {exc}"
            ) from exc
