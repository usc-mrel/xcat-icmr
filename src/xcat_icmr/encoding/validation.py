"""Reference checks and reduced real-data validation for SigPy NUFFT."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
import time
from typing import Callable

import numpy as np
from scipy.io import savemat

from xcat_icmr.encoding.sigpy_backend import SigpyNufftBackend
from xcat_icmr.encoding.trajectory import (
    EncodingGrids,
    EncodingTrajectory,
    prepare_physical_sigpy_trajectory,
    prepare_sigpy_trajectory,
)


@dataclass(frozen=True)
class SigpyReferenceValidation:
    """Small-array direct-DFT and paired-adjoint error measurements."""

    direct_dft_max_abs_error: float
    direct_dft_relative_error: float
    impulse_magnitude_spread: float
    adjoint_relative_error: float

    @property
    def passed(self) -> bool:
        return (
            self.direct_dft_relative_error < 1e-2
            and self.impulse_magnitude_spread < 1e-3
            and self.adjoint_relative_error < 1e-5
        )


@dataclass(frozen=True)
class ReducedNufftValidation:
    """Result of one-coil, reduced-arm forward and debug-adjoint encoding."""

    output_path: Path
    image_shape: tuple[int, int, int]
    acquisition_matrix_shape: tuple[int, int, int]
    reconstruction_matrix_shape: tuple[int, int, int]
    coil_index: int
    sample_count: int
    arm_count: int
    kspace_shape: tuple[int, int]
    kspace_min_magnitude: float
    kspace_max_magnitude: float
    adjoint_shape: tuple[int, int, int]
    adjoint_min_magnitude: float
    adjoint_max_magnitude: float
    nonfinite_value_count: int
    full_adjoint_saved: bool
    rf_center_shift_mm: float
    rf_logical_axis: int
    applied_circular_shift_voxels: int
    shifted_ground_truth_path: Path | None
    device_id: int
    elapsed_s: float


@dataclass(frozen=True)
class MulticoilNufftDebug:
    """Saved full-trajectory forward and adjoint result for every coil."""

    output_path: Path
    shifted_ground_truth_path: Path
    coil_count: int
    image_shape: tuple[int, int, int]
    matrix_shape: tuple[int, int, int]
    kspace_shape: tuple[int, int, int]
    adjoint_coils_shape: tuple[int, int, int, int]
    rf_center_shift_mm: float
    rf_logical_axis: int
    applied_circular_shift_voxels: int
    trajectory_scale_factor: float
    target_kmax_per_m: float
    nonfinite_value_count: int
    device_id: int
    elapsed_s: float


@dataclass(frozen=True)
class MulticoilFrameEncoding:
    """In-memory result of one reusable fully sampled multicoil frame."""

    shifted_ground_truth: np.ndarray
    kspace: np.ndarray
    adjoint_coils: np.ndarray | None
    adjoint_rss: np.ndarray | None
    trajectory: EncodingTrajectory
    normalized_dcf: np.ndarray
    applied_circular_shift_voxels: int
    nonfinite_value_count: int
    device_id: int


def _direct_unitary_dft(
    image: np.ndarray,
    coordinates: np.ndarray,
) -> np.ndarray:
    shape = image.shape
    axes = [
        np.arange(size, dtype=np.float64) - size // 2
        for size in shape
    ]
    x, y, z = np.meshgrid(*axes, indexing="ij")
    output = []
    scale = np.sqrt(np.prod(shape))
    for kx, ky, kz in coordinates:
        phase = -2j * np.pi * (
            kx * x / shape[0]
            + ky * y / shape[1]
            + kz * z / shape[2]
        )
        output.append(np.sum(image * np.exp(phase)) / scale)
    return np.asarray(output, dtype=np.complex64)


def validate_sigpy_reference() -> SigpyReferenceValidation:
    """Validate signs, axes, scaling, impulse response, and adjointness."""

    rng = np.random.default_rng(20260726)
    shape = (5, 6, 4)
    coordinates = np.array(
        (
            (0.0, 0.0, 0.0),
            (0.2, -0.4, 0.7),
            (-1.1, 0.3, 0.25),
            (1.3, -1.2, -0.8),
        ),
        dtype=np.float32,
    )
    trajectory = EncodingTrajectory(
        coordinates=coordinates,
        sample_count=coordinates.shape[0],
        arm_count=1,
        matrix_shape=shape,
        source_maximum_absolute_k_per_m=tuple(
            float(np.max(np.abs(coordinates[:, axis])))
            for axis in range(3)
        ),
        maximum_absolute_coordinate=tuple(
            float(np.max(np.abs(coordinates[:, axis])))
            for axis in range(3)
        ),
    )
    backend = SigpyNufftBackend(device_id=-1)
    image = (
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    ).astype(np.complex64)
    forward = backend.forward(image, trajectory)
    direct = _direct_unitary_dft(image, coordinates)
    difference = np.abs(
        forward.astype(np.complex128) - direct.astype(np.complex128)
    )
    max_abs = float(np.max(difference))
    relative = max_abs / max(float(np.max(np.abs(direct))), 1e-12)

    impulse = np.zeros(shape, dtype=np.complex64)
    impulse[tuple(size // 2 for size in shape)] = 1
    impulse_kspace = backend.forward(impulse, trajectory)
    impulse_spread = float(
        np.max(np.abs(impulse_kspace))
        - np.min(np.abs(impulse_kspace))
    )

    test_kspace = (
        rng.standard_normal(trajectory.point_count)
        + 1j * rng.standard_normal(trajectory.point_count)
    ).astype(np.complex64)
    adjoint = backend.adjoint(test_kspace, trajectory)
    left = np.vdot(forward, test_kspace)
    right = np.vdot(image, adjoint)
    adjoint_error = float(
        np.abs(left - right) / max(np.abs(left), np.abs(right), 1e-12)
    )
    return SigpyReferenceValidation(
        direct_dft_max_abs_error=max_abs,
        direct_dft_relative_error=relative,
        impulse_magnitude_spread=impulse_spread,
        adjoint_relative_error=adjoint_error,
    )


def _normalized_magnitude(values: np.ndarray) -> np.ndarray:
    magnitude = np.abs(values).astype(np.float32)
    maximum = float(np.max(magnitude))
    if maximum > 0:
        magnitude /= maximum
    return magnitude


def circular_shift_to_rf_center(
    array: np.ndarray,
    *,
    center_shift_mm: float,
    voxel_size_mm: float,
    logical_axis: int,
) -> tuple[np.ndarray, int]:
    """Move physical signal into coordinates centered on the shifted RF box."""

    values = np.asarray(array)
    if values.ndim != 3:
        raise ValueError(
            f"RF-center shift expects a 3-D array; got {values.shape}"
        )
    if logical_axis not in (0, 1, 2):
        raise ValueError("RF logical axis must be 0, 1, or 2")
    if not np.isfinite(center_shift_mm):
        raise ValueError("RF center shift must be finite")
    if not np.isfinite(voxel_size_mm) or voxel_size_mm <= 0:
        raise ValueError("RF-axis voxel size must be positive and finite")

    center_shift_voxels = center_shift_mm / voxel_size_mm
    rounded_center_shift = int(np.rint(center_shift_voxels))
    if not np.isclose(center_shift_voxels, rounded_center_shift, atol=1e-6):
        raise ValueError(
            "RF center shift must be an integer number of high-resolution "
            f"voxels for circular shifting; got {center_shift_voxels:g} voxels"
        )

    # A box center displaced in the positive physical direction makes the
    # stationary anatomy appear displaced negatively in box-local coordinates.
    applied_shift = -rounded_center_shift
    if applied_shift == 0:
        return values, 0
    return np.roll(values, shift=applied_shift, axis=logical_axis), applied_shift


def _save_shifted_ground_truth(
    image: np.ndarray,
    output_path: str | Path,
    *,
    center_shift_mm: float,
    logical_axis: int,
    applied_shift_voxels: int,
) -> Path:
    """Atomically save the full high-resolution GT used by shifted encoding."""

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
            {
                "image": np.asarray(image, dtype=np.float32),
                "rf_center_shift_mm": np.asarray(
                    [[center_shift_mm]], dtype=np.float32
                ),
                "rf_logical_axis_zero_based": np.asarray(
                    [[logical_axis]], dtype=np.int32
                ),
                "applied_circular_shift_voxels": np.asarray(
                    [[applied_shift_voxels]], dtype=np.int32
                ),
            },
            appendmat=False,
            do_compression=False,
        )
        temporary_path.chmod(0o644)
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination


def run_reduced_nufft_validation(
    image: np.ndarray,
    normalized_coil: np.ndarray,
    *,
    kx_per_m: np.ndarray,
    ky_per_m: np.ndarray,
    kz_per_m: np.ndarray,
    density_compensation: np.ndarray,
    encoding_grids: EncodingGrids,
    coil_index: int,
    arm_count: int,
    output_path: str | Path,
    save_full_adjoint: bool = False,
    axis_patient_directions: tuple[str, str, str] | None = None,
    pcs_to_logical: np.ndarray | None = None,
    rf_center_shift_mm: float = 0.0,
    rf_axis_voxel_size_mm: float = 1.0,
    rf_logical_axis: int = 0,
    shifted_ground_truth_output_path: str | Path | None = None,
    device_id: int = -1,
) -> ReducedNufftValidation:
    """Run a 3-D SigPy forward and DCF-weighted adjoint for one coil."""

    if image.shape != normalized_coil.shape:
        raise ValueError(
            f"image and sensitivity shapes differ: "
            f"{image.shape} != {normalized_coil.shape}"
        )
    if arm_count <= 0 or arm_count > kx_per_m.shape[1]:
        raise ValueError(
            f"arm_count must be between 1 and {kx_per_m.shape[1]}"
        )
    arms = np.arange(arm_count, dtype=np.intp)
    acquisition_trajectory = prepare_physical_sigpy_trajectory(
        kx_per_m,
        ky_per_m,
        kz_per_m,
        fov_mm=encoding_grids.acquisition_fov_mm,
        matrix_shape=encoding_grids.acquisition_matrix_shape,
        arm_indices=arms,
    )
    # Forward encoding and debug reconstruction share the padded physical FOV,
    # so they use the same matrix-scaled coordinate array.
    reconstruction_trajectory = acquisition_trajectory
    backend = SigpyNufftBackend(device_id=device_id)
    unshifted_coil_image = (
        np.asarray(image, dtype=np.float32)
        * np.asarray(normalized_coil, dtype=np.complex64)
    ).astype(np.complex64)
    shifted_ground_truth, applied_shift_voxels = circular_shift_to_rf_center(
        np.asarray(image, dtype=np.float32),
        center_shift_mm=rf_center_shift_mm,
        voxel_size_mm=rf_axis_voxel_size_mm,
        logical_axis=rf_logical_axis,
    )
    coil_image, coil_shift_voxels = circular_shift_to_rf_center(
        unshifted_coil_image,
        center_shift_mm=rf_center_shift_mm,
        voxel_size_mm=rf_axis_voxel_size_mm,
        logical_axis=rf_logical_axis,
    )
    if coil_shift_voxels != applied_shift_voxels:
        raise ValueError("ground truth and coil-weighted image shifts differ")
    started = time.perf_counter()
    flattened_kspace = backend.forward(coil_image, acquisition_trajectory)
    kspace = acquisition_trajectory.reshape_kspace(flattened_kspace)

    dcf = np.asarray(density_compensation, dtype=np.float32)
    if dcf.shape != kx_per_m.shape:
        raise ValueError(
            f"DCF shape {dcf.shape} does not match trajectory "
            f"{kx_per_m.shape}"
        )
    selected_dcf = dcf[:, arms].T.reshape(-1)
    maximum_dcf = float(np.max(selected_dcf))
    if not np.isfinite(maximum_dcf) or maximum_dcf <= 0:
        raise ValueError("selected density compensation must have a positive max")
    selected_dcf = selected_dcf / maximum_dcf
    adjoint = backend.adjoint(
        flattened_kspace * selected_dcf,
        reconstruction_trajectory,
    )
    elapsed_s = time.perf_counter() - started

    nonfinite = int(np.count_nonzero(~np.isfinite(flattened_kspace)))
    nonfinite += int(np.count_nonzero(~np.isfinite(adjoint)))
    if nonfinite:
        raise ValueError(
            f"reduced NUFFT produced {nonfinite} non-finite values"
        )

    input_center = image.shape[2] // 2
    adjoint_center = adjoint.shape[2] // 2
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
        saved_variables = {
            "kspace": kspace,
            "dcf": selected_dcf.reshape(arm_count, -1).T,
            "input_magnitude_center": _normalized_magnitude(
                coil_image[:, :, input_center]
            ),
            "adjoint_magnitude_center": _normalized_magnitude(
                adjoint[:, :, adjoint_center]
            ),
            "acquisition_coordinates": acquisition_trajectory.coordinates,
            "reconstruction_coordinates": reconstruction_trajectory.coordinates,
            "acquisition_matrix_shape": np.asarray(
                encoding_grids.acquisition_matrix_shape, dtype=np.int32
            ),
            "reconstruction_matrix_shape": np.asarray(
                encoding_grids.reconstruction_matrix_shape, dtype=np.int32
            ),
            "nufft_oversampling": np.asarray(
                [[backend.oversampling]], dtype=np.float64
            ),
            "nufft_kernel_width": np.asarray(
                [[backend.kernel_width]], dtype=np.float64
            ),
            "rf_center_shift_mm": np.asarray(
                [[rf_center_shift_mm]], dtype=np.float32
            ),
            "rf_logical_axis_zero_based": np.asarray(
                [[rf_logical_axis]], dtype=np.int32
            ),
            "applied_circular_shift_voxels": np.asarray(
                [[applied_shift_voxels]], dtype=np.int32
            ),
            "elapsed_s": np.asarray([[elapsed_s]], dtype=np.float64),
        }
        if axis_patient_directions is not None:
            saved_variables["logical_axis_patient_directions"] = np.asarray(
                axis_patient_directions, dtype=object
            )
        if pcs_to_logical is not None:
            matrix = np.asarray(pcs_to_logical, dtype=np.float64)
            if matrix.shape != (3, 3):
                raise ValueError("pcs_to_logical must have shape (3, 3)")
            saved_variables["pcs_to_logical"] = matrix
        if save_full_adjoint:
            saved_variables["adjoint"] = adjoint
        savemat(
            temporary_path,
            saved_variables,
            appendmat=False,
            do_compression=not save_full_adjoint,
        )
        temporary_path.chmod(0o644)
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    shifted_ground_truth_path = (
        _save_shifted_ground_truth(
            shifted_ground_truth,
            shifted_ground_truth_output_path,
            center_shift_mm=rf_center_shift_mm,
            logical_axis=rf_logical_axis,
            applied_shift_voxels=applied_shift_voxels,
        )
        if shifted_ground_truth_output_path is not None
        else None
    )

    return ReducedNufftValidation(
        output_path=destination,
        image_shape=image.shape,
        acquisition_matrix_shape=encoding_grids.acquisition_matrix_shape,
        reconstruction_matrix_shape=encoding_grids.reconstruction_matrix_shape,
        coil_index=coil_index,
        sample_count=acquisition_trajectory.sample_count,
        arm_count=acquisition_trajectory.arm_count,
        kspace_shape=kspace.shape,
        kspace_min_magnitude=float(np.min(np.abs(kspace))),
        kspace_max_magnitude=float(np.max(np.abs(kspace))),
        adjoint_shape=adjoint.shape,
        adjoint_min_magnitude=float(np.min(np.abs(adjoint))),
        adjoint_max_magnitude=float(np.max(np.abs(adjoint))),
        nonfinite_value_count=nonfinite,
        full_adjoint_saved=save_full_adjoint,
        rf_center_shift_mm=float(rf_center_shift_mm),
        rf_logical_axis=rf_logical_axis,
        applied_circular_shift_voxels=applied_shift_voxels,
        shifted_ground_truth_path=shifted_ground_truth_path,
        device_id=device_id,
        elapsed_s=elapsed_s,
    )


def encode_multicoil_frame(
    image: np.ndarray,
    *,
    coil_count: int,
    coil_loader: Callable[[int], np.ndarray],
    kx_per_m: np.ndarray,
    ky_per_m: np.ndarray,
    kz_per_m: np.ndarray,
    density_compensation: np.ndarray,
    encoding_grids: EncodingGrids,
    rf_center_shift_mm: float,
    rf_axis_voxel_size_mm: float,
    rf_logical_axis: int,
    device_id: int = -1,
    compute_adjoint: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> MulticoilFrameEncoding:
    """Encode one frame for every coil, optionally computing debug adjoints."""

    ground_truth = np.asarray(image, dtype=np.float32)
    if ground_truth.ndim != 3 or not np.all(np.isfinite(ground_truth)):
        raise ValueError("multicoil ground truth must be one finite 3-D image")
    if coil_count <= 0:
        raise ValueError("coil_count must be positive")
    trajectory = prepare_physical_sigpy_trajectory(
        kx_per_m,
        ky_per_m,
        kz_per_m,
        fov_mm=encoding_grids.acquisition_fov_mm,
        matrix_shape=encoding_grids.acquisition_matrix_shape,
    )
    dcf = np.asarray(density_compensation, dtype=np.float32)
    if dcf.shape != np.asarray(kx_per_m).shape:
        raise ValueError("DCF and trajectory shapes differ")
    selected_dcf = dcf.T.reshape(-1)
    maximum_dcf = float(np.max(selected_dcf))
    if not np.isfinite(maximum_dcf) or maximum_dcf <= 0:
        raise ValueError("density compensation must have a positive finite max")
    selected_dcf = np.asarray(selected_dcf / maximum_dcf, dtype=np.float32)
    shifted_ground_truth, applied_shift = circular_shift_to_rf_center(
        ground_truth,
        center_shift_mm=rf_center_shift_mm,
        voxel_size_mm=rf_axis_voxel_size_mm,
        logical_axis=rf_logical_axis,
    )
    kspace = np.empty(
        (trajectory.sample_count, trajectory.arm_count, coil_count),
        dtype=np.complex64,
    )
    adjoint_coils = (
        np.empty(trajectory.matrix_shape + (coil_count,), dtype=np.complex64)
        if compute_adjoint
        else None
    )
    backend = SigpyNufftBackend(device_id=device_id)
    nonfinite = 0
    for coil_index in range(coil_count):
        coil = np.asarray(coil_loader(coil_index), dtype=np.complex64)
        if coil.shape != ground_truth.shape:
            raise ValueError(
                f"coil {coil_index} shape {coil.shape} does not match "
                f"ground truth {ground_truth.shape}"
            )
        coil_image = np.asarray(ground_truth * coil, dtype=np.complex64)
        coil_image, coil_shift = circular_shift_to_rf_center(
            coil_image,
            center_shift_mm=rf_center_shift_mm,
            voxel_size_mm=rf_axis_voxel_size_mm,
            logical_axis=rf_logical_axis,
        )
        if coil_shift != applied_shift:
            raise ValueError("coil and ground-truth circular shifts differ")
        flattened = backend.forward(coil_image, trajectory)
        coil_kspace = trajectory.reshape_kspace(flattened)
        nonfinite += int(np.count_nonzero(~np.isfinite(coil_kspace)))
        kspace[:, :, coil_index] = coil_kspace
        if adjoint_coils is not None:
            coil_adjoint = backend.adjoint(
                flattened * selected_dcf, trajectory
            )
            nonfinite += int(np.count_nonzero(~np.isfinite(coil_adjoint)))
            adjoint_coils[:, :, :, coil_index] = coil_adjoint
            del coil_adjoint
        del coil, coil_image, flattened, coil_kspace
        if progress is not None:
            progress(coil_index + 1, coil_count)
    if nonfinite:
        raise ValueError(
            f"multicoil NUFFT produced {nonfinite} non-finite values"
        )
    adjoint_rss = (
        np.sqrt(
            np.sum(np.abs(adjoint_coils) ** 2, axis=3, dtype=np.float64)
        ).astype(np.float32)
        if adjoint_coils is not None
        else None
    )
    return MulticoilFrameEncoding(
        shifted_ground_truth=shifted_ground_truth,
        kspace=kspace,
        adjoint_coils=adjoint_coils,
        adjoint_rss=adjoint_rss,
        trajectory=trajectory,
        normalized_dcf=selected_dcf.reshape(
            trajectory.arm_count, trajectory.sample_count
        ).T,
        applied_circular_shift_voxels=applied_shift,
        nonfinite_value_count=nonfinite,
        device_id=device_id,
    )


def run_multicoil_nufft_debug(
    image: np.ndarray,
    *,
    coil_count: int,
    coil_loader: Callable[[int], np.ndarray],
    kx_per_m: np.ndarray,
    ky_per_m: np.ndarray,
    kz_per_m: np.ndarray,
    density_compensation: np.ndarray,
    encoding_grids: EncodingGrids,
    output_path: str | Path,
    shifted_ground_truth_output_path: str | Path,
    rf_center_shift_mm: float,
    rf_axis_voxel_size_mm: float,
    rf_logical_axis: int,
    trajectory_scale_factor: float,
    target_kmax_per_m: float,
    device_id: int = -1,
    progress: Callable[[int, int], None] | None = None,
) -> MulticoilNufftDebug:
    """Encode all coils and save individual adjoints plus an RSS adjoint."""

    started = time.perf_counter()
    encoding = encode_multicoil_frame(
        image,
        coil_count=coil_count,
        coil_loader=coil_loader,
        kx_per_m=kx_per_m,
        ky_per_m=ky_per_m,
        kz_per_m=kz_per_m,
        density_compensation=density_compensation,
        encoding_grids=encoding_grids,
        rf_center_shift_mm=rf_center_shift_mm,
        rf_axis_voxel_size_mm=rf_axis_voxel_size_mm,
        rf_logical_axis=rf_logical_axis,
        device_id=device_id,
        compute_adjoint=True,
        progress=progress,
    )
    elapsed_s = time.perf_counter() - started
    if encoding.adjoint_coils is None or encoding.adjoint_rss is None:
        raise ValueError("debug encoding did not produce requested adjoints")
    ground_truth = np.asarray(image, dtype=np.float32)
    shifted_ground_truth = encoding.shifted_ground_truth
    applied_shift = encoding.applied_circular_shift_voxels
    trajectory = encoding.trajectory
    dcf = np.asarray(density_compensation, dtype=np.float32)
    kspace = encoding.kspace
    adjoint_coils = encoding.adjoint_coils
    adjoint_rss = encoding.adjoint_rss
    nonfinite = encoding.nonfinite_value_count
    shifted_gt_path = _save_shifted_ground_truth(
        shifted_ground_truth,
        shifted_ground_truth_output_path,
        center_shift_mm=rf_center_shift_mm,
        logical_axis=rf_logical_axis,
        applied_shift_voxels=applied_shift,
    )

    backend = SigpyNufftBackend(device_id=device_id)

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
            {
                "kspace": kspace,
                "dcf": dcf,
                "adjoint_coils": adjoint_coils,
                "adjoint_rss": adjoint_rss,
                "adjoint_rss_center": _normalized_magnitude(
                    adjoint_rss[:, :, adjoint_rss.shape[2] // 2]
                ),
                "coordinates": trajectory.coordinates,
                "fov_mm": np.asarray(
                    [encoding_grids.acquisition_fov_mm], dtype=np.float64
                ),
                "matrix_shape": np.asarray(
                    [trajectory.matrix_shape], dtype=np.int32
                ),
                "resolution_mm": np.asarray(
                    [encoding_grids.resolution_mm], dtype=np.float64
                ),
                "coil_count": np.asarray([[coil_count]], dtype=np.int32),
                "rf_center_shift_mm": np.asarray(
                    [[rf_center_shift_mm]], dtype=np.float32
                ),
                "rf_logical_axis_zero_based": np.asarray(
                    [[rf_logical_axis]], dtype=np.int32
                ),
                "applied_circular_shift_voxels": np.asarray(
                    [[applied_shift]], dtype=np.int32
                ),
                "trajectory_scale_factor": np.asarray(
                    [[trajectory_scale_factor]], dtype=np.float64
                ),
                "target_kmax_per_m": np.asarray(
                    [[target_kmax_per_m]], dtype=np.float64
                ),
                "nufft_oversampling": np.asarray(
                    [[backend.oversampling]], dtype=np.float64
                ),
                "nufft_kernel_width": np.asarray(
                    [[backend.kernel_width]], dtype=np.float64
                ),
                "elapsed_s": np.asarray([[elapsed_s]], dtype=np.float64),
            },
            appendmat=False,
            do_compression=False,
        )
        temporary_path.chmod(0o644)
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return MulticoilNufftDebug(
        output_path=destination,
        shifted_ground_truth_path=shifted_gt_path,
        coil_count=coil_count,
        image_shape=ground_truth.shape,
        matrix_shape=trajectory.matrix_shape,
        kspace_shape=kspace.shape,
        adjoint_coils_shape=adjoint_coils.shape,
        rf_center_shift_mm=float(rf_center_shift_mm),
        rf_logical_axis=rf_logical_axis,
        applied_circular_shift_voxels=applied_shift,
        trajectory_scale_factor=float(trajectory_scale_factor),
        target_kmax_per_m=float(target_kmax_per_m),
        nonfinite_value_count=nonfinite,
        device_id=device_id,
        elapsed_s=elapsed_s,
    )


def format_multicoil_nufft_debug(report: MulticoilNufftDebug) -> str:
    """Format a completed all-coil forward/adjoint debug run."""

    return "\n".join(
        (
            "All-coil 3-D SigPy forward and adjoint",
            (
                "Device:                "
                + ("CPU" if report.device_id == -1 else f"GPU {report.device_id}")
            ),
            f"Coils:                 {report.coil_count}",
            f"High-resolution input: {report.image_shape}",
            f"Encoding matrix:       {report.matrix_shape}",
            f"K-space shape:         {report.kspace_shape}",
            f"Per-coil adjoints:     {report.adjoint_coils_shape}",
            (
                f"RF/image shift:        {report.rf_center_shift_mm:g} mm; "
                f"axis {report.rf_logical_axis}; "
                f"{report.applied_circular_shift_voxels:+d} voxels"
            ),
            f"Target kmax:           {report.target_kmax_per_m:g} cycles/m",
            f"Trajectory scale:      {report.trajectory_scale_factor:g}",
            f"Non-finite values:     {report.nonfinite_value_count}",
            f"Encoding time:         {report.elapsed_s:.3f} s",
            f"Shifted GT:            {report.shifted_ground_truth_path}",
            f"Output:                {report.output_path}",
            "Overall:               PASS",
        )
    )


def format_sigpy_reference_validation(
    report: SigpyReferenceValidation,
) -> str:
    """Format small exact-DFT and adjoint checks."""

    return "\n".join(
        (
            "SigPy mathematical validation",
            (
                f"Direct DFT maximum error: {report.direct_dft_max_abs_error:g}"
            ),
            (
                f"Direct DFT relative error: "
                f"{report.direct_dft_relative_error:g}"
            ),
            (
                f"Impulse magnitude spread:  "
                f"{report.impulse_magnitude_spread:g}"
            ),
            (
                f"Adjoint relative error:    "
                f"{report.adjoint_relative_error:g}"
            ),
            f"Overall:                   {'PASS' if report.passed else 'FAIL'}",
        )
    )


def format_reduced_nufft_validation(
    report: ReducedNufftValidation,
) -> str:
    """Format one-coil, reduced-arm real-data validation."""

    return "\n".join(
        (
            "One-coil 3-D SigPy forward and debug adjoint",
            (
                "Device:                "
                + ("CPU" if report.device_id == -1 else f"GPU {report.device_id}")
            ),
            f"Ground-truth shape:    {report.image_shape}",
            f"Acquisition matrix:    {report.acquisition_matrix_shape}",
            f"Reconstruction matrix: {report.reconstruction_matrix_shape}",
            f"Coil:                  {report.coil_index}",
            (
                f"RF/image center:       {report.rf_center_shift_mm:g} mm; "
                f"logical axis {report.rf_logical_axis}; array shift "
                f"{report.applied_circular_shift_voxels:+d} voxels"
            ),
            (
                f"Trajectory:            {report.sample_count} samples × "
                f"{report.arm_count} arms"
            ),
            f"K-space shape:         {report.kspace_shape}",
            (
                f"K-space |signal|:      "
                f"{report.kspace_min_magnitude:g} to "
                f"{report.kspace_max_magnitude:g}"
            ),
            f"Adjoint shape:         {report.adjoint_shape}",
            (
                f"Adjoint |signal|:      "
                f"{report.adjoint_min_magnitude:g} to "
                f"{report.adjoint_max_magnitude:g}"
            ),
            f"Non-finite values:     {report.nonfinite_value_count}",
            f"Encoding time:         {report.elapsed_s:.3f} s",
            f"Debug output:          {report.output_path}",
            (
                "Full complex adjoint: "
                + ("saved" if report.full_adjoint_saved else "not saved")
            ),
            (
                "Shifted high-res GT:    "
                + (
                    str(report.shifted_ground_truth_path)
                    if report.shifted_ground_truth_path is not None
                    else "not saved"
                )
            ),
            "Overall:               PASS",
        )
    )
