"""Causal Fair-L1 IRLS reconstruction adapted from the NIH SPI pipeline."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import tempfile
import time

import h5py
import numpy as np

from xcat_icmr.reconstruction.planning import ReconstructionJob


class CausalIrlsError(RuntimeError):
    """Raised when a causal IRLS reconstruction cannot be completed."""


def _sigpy():
    cache = Path(tempfile.gettempdir()) / "xcat-icmr-numba-cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache))
    try:
        import sigpy as sp
    except (ImportError, RuntimeError) as exc:
        raise CausalIrlsError(f"could not import SigPy: {exc}") from exc
    return sp


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _device_scalar(sp, value) -> float:
    return float(np.asarray(sp.to_device(value, sp.cpu_device)))


def fair_l1(xp, magnitude, delta: float):
    """Fair-L1 penalty used by the reference causal IRLS implementation."""

    delta32 = np.float32(delta)
    return magnitude - delta32 * xp.log1p(magnitude / delta32)


def make_weighted_sense(
    sp,
    mps,
    coord,
    sqrt_dcf,
    *,
    oversampling: float,
    kernel_width: float,
    coil_batch_size: int,
):
    """Return ``sqrt(DCF) * NUFFT * sensitivity`` with explicit kernel settings."""

    image_shape = tuple(int(value) for value in mps.shape[1:])
    coil_count = int(mps.shape[0])
    batch = coil_count if coil_batch_size <= 0 else coil_batch_size
    if batch < coil_count:
        operators = [
            make_weighted_sense(
                sp,
                mps[start : start + batch],
                coord,
                sqrt_dcf,
                oversampling=oversampling,
                kernel_width=kernel_width,
                coil_batch_size=0,
            )
            for start in range(0, coil_count, batch)
        ]
        return sp.linop.Vstack(operators, axis=0)
    sensitivity = sp.linop.Multiply(image_shape, mps)
    nufft = sp.linop.NUFFT(
        sensitivity.oshape,
        coord,
        oversamp=oversampling,
        width=kernel_width,
    )
    unweighted = nufft * sensitivity
    weighting = sp.linop.Multiply(unweighted.oshape, sqrt_dcf[None, :])
    return weighting * unweighted


def cost_terms(
    sp,
    xp,
    operator,
    gradient,
    x,
    y_weighted,
    previous,
    lambda_s: float,
    lambda_t: float,
    delta: float,
) -> dict[str, float]:
    residual = operator * x - y_weighted
    coefficients = gradient * x
    spatial_magnitude = xp.sqrt(xp.sum(xp.abs(coefficients) ** 2, axis=0))
    fidelity = 0.5 * _device_scalar(sp, xp.real(xp.vdot(residual, residual)))
    spatial = lambda_s * _device_scalar(
        sp, xp.sum(fair_l1(xp, spatial_magnitude, delta))
    )
    temporal = 0.0
    if previous is not None and lambda_t > 0:
        temporal = lambda_t * _device_scalar(
            sp, xp.sum(fair_l1(xp, xp.abs(x - previous), delta))
        )
    return {
        "fidelity": fidelity,
        "spatial": spatial,
        "temporal": temporal,
        "total": fidelity + spatial + temporal,
    }


def solve_frame(
    sp,
    xp,
    operator,
    data_normal,
    gradient,
    y_weighted,
    previous,
    x0,
    *,
    lambda_s: float,
    lambda_t: float,
    delta: float,
    outer_iterations: int,
    cg_iterations: int,
):
    """Run the reference IRLS/CG update for one causal temporal frame."""

    x = x0
    rhs_data = operator.H * y_weighted
    rows: list[dict[str, object]] = []
    for outer in range(1, outer_iterations + 1):
        started = time.perf_counter()
        coefficients = gradient * x
        magnitude = xp.sqrt(xp.sum(xp.abs(coefficients) ** 2, axis=0))
        spatial_weight = (1.0 / (magnitude + np.float32(delta))).astype(
            xp.float32
        )
        spatial_multiply = sp.linop.Multiply(
            gradient.oshape, spatial_weight[None, ...]
        )
        normal = data_normal + lambda_s * (
            gradient.H * spatial_multiply * gradient
        )
        rhs = rhs_data.copy()
        if previous is not None and lambda_t > 0:
            temporal_weight = (
                1.0 / (xp.abs(x - previous) + np.float32(delta))
            ).astype(xp.float32)
            temporal_multiply = sp.linop.Multiply(operator.ishape, temporal_weight)
            normal = normal + lambda_t * temporal_multiply
            rhs += lambda_t * temporal_weight * previous
        algorithm = sp.alg.ConjugateGradient(
            normal, rhs, x, max_iter=cg_iterations, tol=0
        )
        while not algorithm.done():
            algorithm.update()
        terms = cost_terms(
            sp,
            xp,
            operator,
            gradient,
            x,
            y_weighted,
            previous,
            lambda_s,
            lambda_t if previous is not None else 0.0,
            delta,
        )
        rows.append(
            {
                "outer_iteration": outer,
                "cg_iterations": int(algorithm.iter),
                "cg_residual": _device_scalar(sp, algorithm.resid),
                "cg_not_positive_definite": bool(
                    algorithm.not_positive_definite
                ),
                "outer_seconds": time.perf_counter() - started,
                **terms,
            }
        )
    return x, rows


def _load_matrix(path: Path) -> np.ndarray:
    if not path.is_file():
        raise CausalIrlsError(f"coil-compression weights do not exist: {path}")
    if path.suffix.lower() == ".npy":
        value = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".npz":
        archive = np.load(path, allow_pickle=False)
        if "weights" not in archive:
            raise CausalIrlsError(f"{path} must contain an array named 'weights'")
        value = archive["weights"]
    else:
        raise CausalIrlsError("compression weights must be a .npy or .npz file")
    return np.asarray(value, dtype=np.complex64)


def estimate_pca_basis(
    kspace: h5py.Dataset,
    virtual_coils: int,
    *,
    tr_chunk: int = 16,
) -> np.ndarray:
    """Accumulate full-acquisition coil covariance using constant host memory."""

    coil_count = int(kspace.shape[2])
    covariance = np.zeros((coil_count, coil_count), dtype=np.complex128)
    for start in range(0, int(kspace.shape[1]), tr_chunk):
        stop = min(start + tr_chunk, int(kspace.shape[1]))
        block = np.asarray(kspace[:, start:stop, :], dtype=np.complex64).reshape(
            -1, coil_count
        )
        covariance += block.conj().T @ block
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1][:virtual_coils]
    return np.asarray(eigenvectors[:, order], dtype=np.complex64)


def prepare_frame_arrays(
    handle: h5py.File,
    frame: int,
    basis: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read one frame and flatten samples in TR-major order for SigPy."""

    start = int(handle["frame_start_tr_zero_based"][frame])
    stop = int(handle["frame_stop_tr_exclusive"][frame])
    indices_dataset = handle.get("trajectory_tr_index_zero_based")
    if not isinstance(indices_dataset, h5py.Dataset):
        indices_dataset = handle["trajectory/acquired_arm_index_zero_based"]
    arm_indices = np.asarray(indices_dataset[start:stop], dtype=np.int64)
    # h5py fancy indices must be strictly increasing. View orders are allowed
    # to move backwards and repeat arms, so read the sorted unique arms once
    # and restore the exact acquisition order in NumPy.
    unique_arms, restore_order = np.unique(arm_indices, return_inverse=True)
    coordinates_unique = np.asarray(
        handle["trajectory/coordinates"][:, unique_arms, :], dtype=np.float32
    )
    coordinates = coordinates_unique[:, restore_order, :].transpose(
        1, 0, 2
    ).reshape(-1, 3)
    dcf_unique = np.asarray(
        handle["trajectory/density_compensation"][:, unique_arms],
        dtype=np.float32,
    )
    dcf = dcf_unique[:, restore_order].T.reshape(-1)
    if np.any(dcf < 0) or not np.all(np.isfinite(dcf)):
        raise CausalIrlsError(f"frame {frame} has invalid DCF values")
    values = np.asarray(
        handle["kspace"][:, start:stop, :], dtype=np.complex64
    ).transpose(2, 1, 0).reshape(int(handle["kspace"].shape[2]), -1)
    if basis is not None:
        values = basis.conj().T @ values
    return (
        np.asarray(values, dtype=np.complex64),
        coordinates,
        np.sqrt(dcf).astype(np.float32, copy=False),
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_causal_irls(
    job: ReconstructionJob,
    *,
    preprocessing,
    compute,
    output,
    overwrite: bool = False,
) -> Path:
    """Reconstruct one self-describing acquisition with causal IRLS."""

    sp = _sigpy()
    destination = job.output_directory
    result_path = destination / "result.json"
    reconstruction_path = destination / "reconstruction.h5"
    if result_path.is_file() and not overwrite:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") == "complete":
            print(f"REUSE {destination}", flush=True)
            return destination
    try:
        device = sp.Device(compute.device_id)
        with device:
            device.xp.empty(1, dtype=device.xp.float32)
    except Exception as exc:
        requested = "CPU" if compute.device_id < 0 else f"GPU {compute.device_id}"
        raise CausalIrlsError(
            f"requested reconstruction device {requested} is unavailable: {exc}"
        ) from exc
    xp = device.xp
    if overwrite:
        result_path.unlink(missing_ok=True)
        reconstruction_path.unlink(missing_ok=True)
        (destination / "coil_compression_weights.npy").unlink(missing_ok=True)
    destination.mkdir(parents=True, exist_ok=True)

    parameters = job.specification.parameters
    lambda_s = parameters.spatial_regularization * parameters.regularization_scale
    lambda_t = parameters.temporal_regularization * parameters.regularization_scale
    acquisition_path = job.acquisition.inspection.path
    compression = preprocessing.coil_compression
    saved_basis_path = destination / "coil_compression_weights.npy"
    with h5py.File(acquisition_path, "r") as source:
        acquired_coils = int(source["kspace"].shape[2])
        basis = None
        if compression.enabled:
            if compression.weights_file is not None:
                basis = _load_matrix(compression.weights_file)
            elif saved_basis_path.is_file():
                basis = _load_matrix(saved_basis_path)
                print(f"REUSE {saved_basis_path}", flush=True)
            elif compression.method == "pca":
                print("Estimating one acquisition-wide PCA basis...", flush=True)
                basis = estimate_pca_basis(source["kspace"], compression.virtual_coils)
            if basis is None or basis.shape != (
                acquired_coils,
                compression.virtual_coils,
            ):
                raise CausalIrlsError(
                    "compression weights must have shape "
                    f"[{acquired_coils}, {compression.virtual_coils}]"
                )
        maps = np.asarray(
            source["encoding/sensitivity_maps"], dtype=np.complex64
        ).transpose(3, 0, 1, 2)
        if basis is not None:
            maps = np.tensordot(basis.conj().T, maps, axes=(1, 0)).astype(
                np.complex64, copy=False
            )
            np.save(saved_basis_path, basis)
        frame_count = int(source["frame_start_tr_zero_based"].shape[0])
        image_shape = tuple(int(value) for value in maps.shape[1:])

    result = {
        "status": "running",
        "reconstruction_id": job.reconstruction_id,
        "recipe_id": job.recipe_id,
        "acquisition_id": job.acquisition.acquisition_id,
        "acquisition_file": str(acquisition_path),
        "method": "causal-irls",
        "readable_recipe": job.readable_recipe,
        "frame_count": frame_count,
        "completed_frames": 0,
        "image_shape": list(image_shape),
        "device_id": compute.device_id,
        "nufft_oversampling": job.acquisition.inspection.nufft_oversampling,
        "nufft_kernel_width": job.acquisition.inspection.nufft_kernel_width,
        "coil_compression": job.coil_compression,
        "reconstruction_file": str(reconstruction_path),
        "started_unix": time.time(),
    }
    _atomic_json(result_path, result)

    file_mode = "r+" if reconstruction_path.exists() else "w"
    with h5py.File(reconstruction_path, file_mode) as reconstructed:
        if "frame_complete" not in reconstructed:
            reconstructed.attrs["dimension_order"] = "time,logical_x,logical_y,logical_z"
            reconstructed.attrs["source_acquisition"] = str(acquisition_path)
            reconstructed.attrs["reconstruction_id"] = job.reconstruction_id
            reconstructed.create_dataset(
                "frame_complete", data=np.zeros(frame_count, dtype=np.uint8)
            )
            if output.save_complex_image:
                reconstructed.create_dataset(
                    "image_complex",
                    shape=(frame_count, *image_shape),
                    dtype=np.complex64,
                    chunks=(1, *image_shape),
                )
            if output.save_magnitude_image:
                reconstructed.create_dataset(
                    "image_magnitude",
                    shape=(frame_count, *image_shape),
                    dtype=np.float32,
                    chunks=(1, *image_shape),
                )
        complete = np.asarray(reconstructed["frame_complete"][:], dtype=bool)
        completed = int(np.sum(complete))
        if completed and not np.all(complete[:completed]):
            raise CausalIrlsError("checkpoint has non-contiguous completed frames")
        if completed and not output.save_complex_image:
            raise CausalIrlsError(
                "cannot resume causal reconstruction without saved complex images"
            )

        with device:
            mps = sp.to_device(maps, device)
            gradient = sp.linop.FiniteDifference(image_shape)
            previous = (
                sp.to_device(reconstructed["image_complex"][completed - 1], device)
                if completed
                else None
            )
        costs: list[dict[str, object]] = []
        timings: list[dict[str, object]] = []
        with h5py.File(acquisition_path, "r") as source:
            for frame in range(completed, frame_count):
                frame_started = time.perf_counter()
                y_host, coord_host, sqrt_dcf_host = prepare_frame_arrays(
                    source, frame, basis
                )
                with device:
                    y = sp.to_device(y_host, device)
                    coord = sp.to_device(coord_host, device)
                    sqrt_dcf = sp.to_device(sqrt_dcf_host, device)
                    operator = make_weighted_sense(
                        sp,
                        mps,
                        coord,
                        sqrt_dcf,
                        oversampling=job.acquisition.inspection.nufft_oversampling,
                        kernel_width=job.acquisition.inspection.nufft_kernel_width,
                        coil_batch_size=compute.coil_batch_size,
                    )
                    y_weighted = y * sqrt_dcf[None, :]
                    x0 = operator.H * y_weighted if previous is None else previous.copy()
                    x, frame_costs = solve_frame(
                        sp,
                        xp,
                        operator,
                        operator.H * operator,
                        gradient,
                        y_weighted,
                        previous,
                        x0,
                        lambda_s=lambda_s,
                        lambda_t=lambda_t if previous is not None else 0.0,
                        delta=parameters.fair_l1_delta,
                        outer_iterations=parameters.outer_iterations,
                        cg_iterations=parameters.cg_iterations,
                    )
                    host_image = np.asarray(
                        sp.to_device(x, sp.cpu_device), dtype=np.complex64
                    )
                    previous = x.copy()
                if output.save_complex_image:
                    reconstructed["image_complex"][frame] = host_image
                if output.save_magnitude_image:
                    reconstructed["image_magnitude"][frame] = np.abs(host_image)
                reconstructed["frame_complete"][frame] = 1
                if output.checkpoint:
                    reconstructed.flush()
                elapsed = time.perf_counter() - frame_started
                for row in frame_costs:
                    costs.append({"frame": frame, **row})
                timings.append({"frame": frame, "total_seconds": elapsed})
                result["completed_frames"] = frame + 1
                result["last_frame_seconds"] = elapsed
                _atomic_json(result_path, result)
                print(
                    f"FRAME {frame + 1:04d}/{frame_count:04d} {elapsed:.3f} s",
                    flush=True,
                )
        reconstructed.flush()

    _write_csv(destination / "costs.csv", costs)
    _write_csv(destination / "timings.csv", timings)
    result.update(
        status="complete",
        completed_frames=frame_count,
        finished_unix=time.time(),
        latency_s=(
            float(np.mean([row["total_seconds"] for row in timings]))
            if timings
            else None
        ),
    )
    _atomic_json(result_path, result)
    return destination
