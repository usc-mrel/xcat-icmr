"""Typed, reusable reconstruction configuration independent of simulation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator


class ReconstructionConfigError(ValueError):
    """Raised when a reconstruction configuration cannot be loaded."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SimulationAcquisitionInput(_Model):
    """User-facing reference to one generated temporal grouping."""

    config: Path
    frame_duration_s: float = Field(gt=0.0, allow_inf_nan=False)


class ReconstructionInputs(_Model):
    acquisitions: tuple[Path, ...] = ()
    simulations: tuple[SimulationAcquisitionInput, ...] = ()

    @model_validator(mode="after")
    def reject_duplicates(self) -> "ReconstructionInputs":
        if not self.acquisitions and not self.simulations:
            raise ValueError(
                "inputs must contain at least one acquisition or simulation"
            )
        normalized = [str(path) for path in self.acquisitions]
        if len(set(normalized)) != len(normalized):
            raise ValueError("inputs.acquisitions contains duplicate paths")
        simulation_keys = [
            (str(item.config), item.frame_duration_s) for item in self.simulations
        ]
        if len(set(simulation_keys)) != len(simulation_keys):
            raise ValueError("inputs.simulations contains duplicate selections")
        return self


class CoilCompressionConfig(_Model):
    """Optional fixed linear virtual-coil compression."""

    enabled: bool = False
    method: Literal["pca", "rovir"] = "pca"
    virtual_coils: PositiveInt = 8
    weights_file: Path | None = None

    @model_validator(mode="after")
    def require_rovir_weights(self) -> "CoilCompressionConfig":
        if self.enabled and self.method == "rovir" and self.weights_file is None:
            raise ValueError(
                "ROVIR compression requires coil_compression.weights_file"
            )
        return self


class ReconstructionPreprocessing(_Model):
    coil_compression: CoilCompressionConfig = Field(
        default_factory=CoilCompressionConfig
    )


class CausalIrlsParameters(_Model):
    outer_iterations: PositiveInt
    cg_iterations: PositiveInt
    spatial_regularization: float = Field(ge=0.0, allow_inf_nan=False)
    temporal_regularization: float = Field(ge=0.0, allow_inf_nan=False)
    fair_l1_delta: float = Field(default=1e-3, gt=0.0, allow_inf_nan=False)
    regularization_scale: float = Field(default=1.0, gt=0.0, allow_inf_nan=False)


class ReconstructionSpec(_Model):
    method: Literal["causal-irls"]
    parameters: CausalIrlsParameters


class ReconstructionCompute(_Model):
    device_id: int = Field(ge=-1)
    precision: Literal["single"] = "single"
    coil_batch_size: int = Field(default=0, ge=0)


class ReconstructionOutput(_Model):
    checkpoint: bool = True
    save_complex_image: bool = True
    save_magnitude_image: bool = False

    @model_validator(mode="after")
    def require_an_image(self) -> "ReconstructionOutput":
        if not self.save_complex_image and not self.save_magnitude_image:
            raise ValueError(
                "output must save at least one of the complex or magnitude images"
            )
        return self


class ReconstructionConfig(_Model):
    schema_version: Literal[1]
    inputs: ReconstructionInputs
    preprocessing: ReconstructionPreprocessing = Field(
        default_factory=ReconstructionPreprocessing
    )
    reconstructions: tuple[ReconstructionSpec, ...] = Field(min_length=1)
    compute: ReconstructionCompute
    output: ReconstructionOutput = Field(default_factory=ReconstructionOutput)

    @model_validator(mode="after")
    def reject_duplicate_recipes(self) -> "ReconstructionConfig":
        serialized = [
            json.dumps(
                item.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
            )
            for item in self.reconstructions
        ]
        if len(set(serialized)) != len(serialized):
            raise ValueError("reconstructions contains duplicate settings")
        return self


def load_reconstruction_config(path: str | Path) -> ReconstructionConfig:
    """Load reconstruction YAML and resolve acquisition paths."""

    import yaml

    resolved = Path(path).expanduser().resolve(strict=False)
    if not resolved.is_file():
        raise ReconstructionConfigError(
            f"reconstruction configuration does not exist: {resolved}"
        )
    try:
        content = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ReconstructionConfigError(
            f"could not read reconstruction configuration {resolved}: {exc}"
        ) from exc
    if not isinstance(content, dict):
        raise ReconstructionConfigError(
            "reconstruction configuration must be a mapping"
        )
    inputs = content.get("inputs")
    if isinstance(inputs, dict) and isinstance(inputs.get("acquisitions"), list):
        inputs["acquisitions"] = [
            str(
                (
                    Path(value).expanduser()
                    if Path(value).expanduser().is_absolute()
                    else resolved.parent / Path(value).expanduser()
                ).resolve(strict=False)
            )
            for value in inputs["acquisitions"]
        ]
    if isinstance(inputs, dict) and isinstance(inputs.get("simulations"), list):
        for selection in inputs["simulations"]:
            if not isinstance(selection, dict) or not selection.get("config"):
                continue
            value = Path(selection["config"]).expanduser()
            selection["config"] = str(
                (value if value.is_absolute() else resolved.parent / value).resolve(
                    strict=False
                )
            )
    preprocessing = content.get("preprocessing")
    if isinstance(preprocessing, dict):
        compression = preprocessing.get("coil_compression")
        if isinstance(compression, dict) and compression.get("weights_file"):
            value = Path(compression["weights_file"]).expanduser()
            compression["weights_file"] = str(
                (value if value.is_absolute() else resolved.parent / value).resolve(
                    strict=False
                )
            )
    return ReconstructionConfig.model_validate(content)
