"""Automatic identities and human-readable paths for reconstruction jobs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import h5py

from xcat_icmr import __version__
from xcat_icmr.acquisition import AcquisitionInspection, inspect_acquisition
from xcat_icmr.cache import undersampled_acquisition_cache_entry
from xcat_icmr.config import load_config
from xcat_icmr.reconstruction.config import (
    CoilCompressionConfig,
    ReconstructionConfig,
    ReconstructionSpec,
    SimulationAcquisitionInput,
)
from xcat_icmr.undersampling import resolve_bracketing_multiples


class ReconstructionPlanError(ValueError):
    """Raised when requested reconstruction jobs cannot be planned."""


@dataclass(frozen=True)
class ReconstructionAcquisition:
    inspection: AcquisitionInspection
    acquisition_id: str
    control_points_filename: str
    velocity_cm_per_s: float
    experiment_directory: Path
    run_output_root: Path

    @property
    def display_label(self) -> str:
        return (
            f"{Path(self.control_points_filename).stem} | "
            f"{self.velocity_cm_per_s:g} cm/s | "
            f"{self.inspection.frame_duration_s * 1e3:g} ms | "
            f"{self.acquisition_id[:8]}"
        )


@dataclass(frozen=True)
class ReconstructionJob:
    acquisition: ReconstructionAcquisition
    specification: ReconstructionSpec
    recipe_id: str
    reconstruction_id: str
    readable_recipe: str
    coil_compression: str
    output_directory: Path
    status: str


@dataclass(frozen=True)
class ReconstructionPlan:
    configuration_path: Path
    jobs: tuple[ReconstructionJob, ...]


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _number_slug(value: float) -> str:
    return (
        f"{value:.6g}"
        .replace("-", "m")
        .replace(".", "p")
        .replace("+", "")
    )


def automatic_recipe_label(
    specification: ReconstructionSpec,
    coil_compression: CoilCompressionConfig | None = None,
) -> str:
    """Build a readable label exclusively from validated parameters."""

    parameters = specification.parameters
    label = (
        f"o{parameters.outer_iterations}_cg{parameters.cg_iterations}_"
        f"lt{_number_slug(parameters.temporal_regularization)}_"
        f"ls{_number_slug(parameters.spatial_regularization)}"
    )
    if coil_compression is not None and coil_compression.enabled:
        label += f"_{coil_compression.method}{coil_compression.virtual_coils}"
    return label


def load_reconstruction_acquisition(
    path: str | Path,
) -> ReconstructionAcquisition:
    """Read the self-describing acquisition without a simulation config."""

    inspection = inspect_acquisition(path)
    try:
        with h5py.File(inspection.path, "r") as handle:
            acquisition_id = str(handle.attrs.get("acquisition_id", "")).strip()
            control_points = str(
                handle.attrs.get(
                    "control_points_filename", "unknown_path.json"
                )
            )
            velocity = float(
                handle.attrs.get("velocity_cm_per_s", float("nan"))
            )
            experiment_value = str(
                handle.attrs.get("experiment_directory", "")
            )
            run_root_value = str(handle.attrs.get("run_output_root", ""))
    except OSError as exc:
        raise ReconstructionPlanError(
            f"could not read acquisition identity from {inspection.path}: {exc}"
        ) from exc
    if not acquisition_id:
        raise ReconstructionPlanError(
            f"acquisition_id is missing from {inspection.path}; "
            "regenerate its descriptor"
        )
    if not math.isfinite(velocity) or velocity <= 0:
        raise ReconstructionPlanError(
            f"velocity_cm_per_s is missing or invalid in {inspection.path}; "
            "regenerate its descriptor"
        )
    experiment_directory = (
        Path(experiment_value).expanduser().resolve(strict=False)
        if experiment_value
        else inspection.path.parent
    )
    run_output_root = (
        Path(run_root_value).expanduser().resolve(strict=False)
        if run_root_value
        else experiment_directory
    )
    return ReconstructionAcquisition(
        inspection=inspection,
        acquisition_id=acquisition_id,
        control_points_filename=control_points,
        velocity_cm_per_s=velocity,
        experiment_directory=experiment_directory,
        run_output_root=run_output_root,
    )


def load_simulation_acquisition(
    selection: SimulationAcquisitionInput,
) -> ReconstructionAcquisition:
    """Resolve a readable simulation selection to its cached acquisition."""

    simulation = load_config(selection.config)
    if not simulation.undersampling.enabled:
        raise ReconstructionPlanError(
            f"undersampling is disabled in {selection.config}"
        )
    source_duration = simulation.acquisition.frame_duration_s
    ratio = selection.frame_duration_s / source_duration
    factor = int(round(ratio))
    if factor <= 0 or not math.isclose(
        selection.frame_duration_s,
        factor * source_duration,
        rel_tol=1e-7,
        abs_tol=1e-9,
    ):
        raise ReconstructionPlanError(
            f"requested frame duration {selection.frame_duration_s * 1e3:g} ms "
            f"is not an integer multiple of the canonical "
            f"{source_duration * 1e3:g} ms frame in {selection.config}"
        )
    generated_factors = resolve_bracketing_multiples(
        target_frame_duration_s=(
            simulation.undersampling.target_frame_duration_s
        ),
        source_frame_duration_s=source_duration,
    )
    if factor not in generated_factors:
        generated = ", ".join(
            f"{item * source_duration * 1e3:g} ms" for item in generated_factors
        )
        raise ReconstructionPlanError(
            f"{selection.config} is configured to generate {generated}; "
            f"it does not generate the requested "
            f"{selection.frame_duration_s * 1e3:g} ms grouping"
        )
    entry = undersampled_acquisition_cache_entry(
        simulation,
        source_frames_per_output_frame=factor,
        view_order_cycles=None,
    )
    acquisition = entry.directory / "grouped_multicoil_kspace.h5"
    if not acquisition.is_file():
        raise ReconstructionPlanError(
            f"the {selection.frame_duration_s * 1e3:g} ms undersampled "
            f"acquisition for {selection.config} has not been generated. Run:\n"
            f"  xcat-icmr generate-undersampled-acquisition {selection.config}"
        )
    return load_reconstruction_acquisition(acquisition)


def _job_status(output_directory: Path, reconstruction_id: str) -> str:
    result = output_directory / "result.json"
    if not result.is_file():
        return "not-started"
    try:
        content = json.loads(result.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "invalid-result"
    if content.get("reconstruction_id") != reconstruction_id:
        return "identity-mismatch"
    status = str(content.get("status", "unknown"))
    output = content.get("reconstruction_file")
    if status == "complete" and (not output or not Path(output).is_file()):
        return "missing-output"
    return status


def plan_reconstructions(
    config: ReconstructionConfig,
    *,
    configuration_path: str | Path,
) -> ReconstructionPlan:
    """Expand every validated recipe over every validated acquisition."""

    config_path = Path(configuration_path).expanduser().resolve(strict=False)
    direct_acquisitions = tuple(
        load_reconstruction_acquisition(path)
        for path in config.inputs.acquisitions
    )
    simulation_acquisitions = tuple(
        load_simulation_acquisition(selection)
        for selection in config.inputs.simulations
    )
    acquisitions = direct_acquisitions + simulation_acquisitions
    identities = [item.acquisition_id for item in acquisitions]
    if len(set(identities)) != len(identities):
        raise ReconstructionPlanError(
            "reconstruction inputs resolve to duplicate acquisitions"
        )
    compression = config.preprocessing.coil_compression
    if compression.enabled:
        for acquisition in acquisitions:
            available = acquisition.inspection.kspace_shape[2]
            if compression.virtual_coils > available:
                raise ReconstructionPlanError(
                    f"coil compression requests {compression.virtual_coils} "
                    f"virtual coils but {acquisition.display_label} has only "
                    f"{available}"
                )

    jobs: list[ReconstructionJob] = []
    preprocessing_payload = (
        config.preprocessing.model_dump(mode="json")
        if compression.enabled
        else {"coil_compression": {"enabled": False}}
    )
    for acquisition in acquisitions:
        for specification in config.reconstructions:
            recipe_payload = {
                "reconstruction": specification.model_dump(mode="json"),
                "preprocessing": preprocessing_payload,
                "precision": config.compute.precision,
            }
            recipe_digest = _canonical_digest(recipe_payload)
            reconstruction_digest = _canonical_digest(
                {
                    "acquisition_id": acquisition.acquisition_id,
                    "recipe": recipe_payload,
                    "package_version": __version__,
                }
            )
            readable = automatic_recipe_label(specification, compression)
            output_directory = (
                acquisition.experiment_directory
                / "reconstructions"
                / specification.method
                / f"{readable}__{recipe_digest[:8]}"
            )
            compression_description = (
                f"{compression.method}: "
                f"{acquisition.inspection.kspace_shape[2]} -> "
                f"{compression.virtual_coils} coils "
                + (
                    f"using {compression.weights_file}"
                    if compression.weights_file is not None
                    else "using an acquisition-wide fixed PCA basis"
                )
                if compression.enabled
                else "disabled"
            )
            jobs.append(
                ReconstructionJob(
                    acquisition=acquisition,
                    specification=specification,
                    recipe_id=recipe_digest,
                    reconstruction_id=reconstruction_digest,
                    readable_recipe=readable,
                    coil_compression=compression_description,
                    output_directory=output_directory,
                    status=_job_status(
                        output_directory, reconstruction_digest
                    ),
                )
            )
    return ReconstructionPlan(config_path, tuple(jobs))


def format_reconstruction_plan(plan: ReconstructionPlan) -> str:
    lines = [
        "Reconstruction plan",
        f"Configuration: {plan.configuration_path}",
        f"Jobs:          {len(plan.jobs)}",
    ]
    for index, job in enumerate(plan.jobs, start=1):
        lines.extend(
            (
                "",
                f"[{index}] {job.acquisition.display_label}",
                f"    Method: {job.specification.method}",
                f"    Recipe: {job.readable_recipe} [{job.recipe_id[:8]}]",
                f"    Status: {job.status}",
                f"    Acquired coils:    {job.acquisition.inspection.kspace_shape[2]}",
                f"    Coil compression:  {job.coil_compression}",
                f"    Output: {job.output_directory}",
            )
        )
    return "\n".join(lines)
