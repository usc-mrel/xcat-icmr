"""Reference checks and reduced real-data validation for SigPy NUFFT."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile

import numpy as np
from scipy.io import savemat

from xcat_icmr.encoding.sigpy_backend import SigpyNufftBackend
from xcat_icmr.encoding.trajectory import (
    EncodingGrids,
    EncodingTrajectory,
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
    acquisition_trajectory = prepare_sigpy_trajectory(
        kx_per_m,
        ky_per_m,
        kz_per_m,
        matrix_shape=encoding_grids.acquisition_matrix_shape,
        arm_indices=arms,
    )
    # Forward encoding and debug reconstruction share the padded physical FOV,
    # so they use the same matrix-scaled coordinate array.
    reconstruction_trajectory = acquisition_trajectory
    backend = SigpyNufftBackend(device_id=-1)
    coil_image = (
        np.asarray(image, dtype=np.float32)
        * np.asarray(normalized_coil, dtype=np.complex64)
    ).astype(np.complex64)
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
            "Device:                CPU reference",
            f"Ground-truth shape:    {report.image_shape}",
            f"Acquisition matrix:    {report.acquisition_matrix_shape}",
            f"Reconstruction matrix: {report.reconstruction_matrix_shape}",
            f"Coil:                  {report.coil_index}",
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
            f"Debug output:          {report.output_path}",
            (
                "Full complex adjoint: "
                + ("saved" if report.full_adjoint_saved else "not saved")
            ),
            "Overall:               PASS",
        )
    )
