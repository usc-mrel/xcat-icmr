"""Backend dispatch for modular reconstruction jobs."""

from __future__ import annotations

import json
from pathlib import Path

from xcat_icmr.reconstruction.causal_irls import run_causal_irls
from xcat_icmr.reconstruction.config import ReconstructionConfig
from xcat_icmr.reconstruction.index import upsert_reconstruction_indexes
from xcat_icmr.reconstruction.planning import ReconstructionPlan


def run_reconstruction_plan(
    plan: ReconstructionPlan,
    config: ReconstructionConfig,
    *,
    overwrite: bool = False,
) -> tuple[Path, ...]:
    """Dispatch every planned job to its named reconstruction backend."""

    backends = {"causal-irls": run_causal_irls}
    outputs: list[Path] = []
    for job in plan.jobs:
        backend = backends[job.specification.method]
        output = backend(
            job,
            preprocessing=config.preprocessing,
            compute=config.compute,
            output=config.output,
            overwrite=overwrite,
        )
        outputs.append(output)
        result_path = output / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        record = {
            "reconstruction_id": job.reconstruction_id,
            "acquisition_id": job.acquisition.acquisition_id,
            "control_points_filename": job.acquisition.control_points_filename,
            "velocity_cm_per_s": job.acquisition.velocity_cm_per_s,
            "temporal_resolution_ms": (
                job.acquisition.inspection.frame_duration_s * 1e3
            ),
            "method": job.specification.method,
            "readable_recipe": job.readable_recipe,
            "status": result["status"],
            "latency_s": result.get("latency_s"),
            "reconstruction_file": result.get("reconstruction_file"),
            "result_file": str(result_path),
        }
        upsert_reconstruction_indexes(
            job.acquisition.experiment_directory / "reconstructions", record
        )
    return tuple(outputs)
