"""Typed models for the XCAT-iCMR simulation configuration."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, model_validator


class ConfigModel(BaseModel):
    """Base model shared by all configuration sections."""

    model_config = ConfigDict(extra="forbid")


class RunConfig(ConfigModel):
    id: str = Field(min_length=1)
    output_root: Path


class ComputeConfig(ConfigModel):
    device_id: int = Field(ge=-1)


class XcatResourceConfig(ConfigModel):
    executable: Path
    parameter_template: Path


class ResourcesConfig(ConfigModel):
    xcat: XcatResourceConfig


class OutputsConfig(ConfigModel):
    save_tissue_labels: bool
    save_contrast_images: bool
    save_fully_sampled_kspace: bool
    retain_xcat_binary_files: bool


class OffResonanceConfig(ConfigModel):
    enabled: bool
    field_map: Path | None


class ToggleEffectConfig(ConfigModel):
    enabled: bool


class ScannerEffectsConfig(ConfigModel):
    off_resonance: OffResonanceConfig
    concomitant_fields: ToggleEffectConfig
    gradient_nonlinearity: ToggleEffectConfig


class ScannerConfig(ConfigModel):
    field_strength_t: PositiveFloat
    effects: ScannerEffectsConfig


class ContrastConfig(ConfigModel):
    model: Literal["bssfp"]
    tissue_library: str = Field(min_length=1)


class SequenceConfig(ConfigModel):
    folder: Path
    file: Path
    metadata_directory: Path
    orientation: Literal["COR", "SAG", "TRA", "2D"]
    rf_direction: Literal["LR", "AP", "SI"]
    contrast: ContrastConfig

    @property
    def resolved_file(self) -> Path:
        if self.file.is_absolute():
            return self.file
        return self.folder / self.file


Duration = Literal["auto"] | PositiveFloat


class TimelineConfig(ConfigModel):
    duration_s: Duration
    xcat_time_step_s: PositiveFloat
    kspace_time_step_s: PositiveFloat
    xcat_to_kspace: Literal["average", "center", "trajectory-aware"]

    @model_validator(mode="after")
    def validate_time_steps(self) -> "TimelineConfig":
        if self.kspace_time_step_s < self.xcat_time_step_s:
            raise ValueError(
                "kspace_time_step_s must be greater than or equal to "
                "xcat_time_step_s"
            )

        ratio = self.kspace_time_step_s / self.xcat_time_step_s
        if not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                "kspace_time_step_s must be an integer multiple of "
                "xcat_time_step_s"
            )
        return self

    @property
    def xcat_frames_per_kspace_frame(self) -> int:
        return round(self.kspace_time_step_s / self.xcat_time_step_s)


class AnatomyConfig(ConfigModel):
    sex: Literal["male", "female"]
    papillary_muscles: bool


class SliceRangeConfig(ConfigModel):
    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_order(self) -> "SliceRangeConfig":
        if self.end < self.start:
            raise ValueError("end must be greater than or equal to start")
        return self


class TransformConfig(ConfigModel):
    additional_rotation_deg_xyz: tuple[float, float, float]
    translation_mm_xyz: tuple[float, float, float]


class CropConfig(ConfigModel):
    rows: tuple[int, int]
    columns: tuple[int, int]

    @model_validator(mode="after")
    def validate_ranges(self) -> "CropConfig":
        for name, values in (("rows", self.rows), ("columns", self.columns)):
            start, stop = values
            if start < 0:
                raise ValueError(f"{name} start must be non-negative")
            if stop <= start:
                raise ValueError(f"{name} stop must be greater than start")
        return self


class CardiacMotionConfig(ConfigModel):
    heart_rate_bpm: PositiveFloat
    start_phase: float = Field(ge=0.0, le=1.0)


class RespiratoryMotionConfig(ConfigModel):
    breaths_per_minute: PositiveFloat | None
    start_phase: float = Field(ge=0.0, le=1.0)
    diaphragm_motion_cm: float = Field(ge=0.0)
    anterior_posterior_expansion_cm: float = Field(ge=0.0)


class MotionConfig(ConfigModel):
    mode: Literal["no-motion", "breath-hold", "free-breathing"]
    cardiac: CardiacMotionConfig
    respiratory: RespiratoryMotionConfig

    @model_validator(mode="after")
    def validate_respiratory_frequency(self) -> "MotionConfig":
        if (
            self.mode == "free-breathing"
            and self.respiratory.breaths_per_minute is None
        ):
            raise ValueError(
                "respiratory.breaths_per_minute is required in "
                "free-breathing mode"
            )
        return self


class PhantomConfig(ConfigModel):
    orientation: Literal["COR", "SAG", "TRA", "SAX"]
    anatomy: AnatomyConfig
    voxel_size_mm: tuple[PositiveFloat, PositiveFloat, PositiveFloat]
    matrix_size_xy: int = Field(gt=0)
    slice_range: SliceRangeConfig
    transform: TransformConfig
    crop: CropConfig
    motion: MotionConfig


class BalloonPathConfig(ConfigModel):
    control_points_file: Path | None
    format: Literal["3d-slicer-markups"]
    coordinate_system: Literal["auto", "RAS", "LPS"]
    interpolation: Literal["cubic-arc-length"]


class BalloonMovementConfig(ConfigModel):
    velocity_cm_per_s: PositiveFloat
    start_time_s: float = Field(ge=0.0)


class BalloonGeometryConfig(ConfigModel):
    shape: Literal["sphere", "ellipsoid"]
    diameter_mm: tuple[PositiveFloat, PositiveFloat, PositiveFloat]


class ConcentrationConfig(ConfigModel):
    type: Literal["final-concentration"]
    value_mM: PositiveFloat


class ContrastAgentConfig(ConfigModel):
    agent: Literal["gadolinium"]
    carrier_tissue: Literal["blood"]
    concentration: ConcentrationConfig
    relaxivity_library: str = Field(min_length=1)


class CompositionConfig(ConfigModel):
    mode: Literal["additive", "replace-background", "difference"]


class GdBalloonConfig(ConfigModel):
    enabled: bool
    path: BalloonPathConfig
    movement: BalloonMovementConfig
    geometry: BalloonGeometryConfig
    contrast_agent: ContrastAgentConfig
    composition: CompositionConfig


class InterventionConfig(ConfigModel):
    gd_balloon: GdBalloonConfig


class CoilsConfig(ConfigModel):
    enabled: bool
    sensitivity_map: Path | None
    normalize: bool


class UndersamplingConfig(ConfigModel):
    enabled: bool
    frame_duration_s: PositiveFloat


class NoiseConfig(ConfigModel):
    enabled: bool
    snr_db: PositiveFloat
    coil_covariance: Literal["identity"] | Path
    seed: int = Field(ge=0)


class SimulationConfig(ConfigModel):
    schema_version: Literal[1]
    run: RunConfig
    compute: ComputeConfig
    resources: ResourcesConfig
    outputs: OutputsConfig
    scanner: ScannerConfig
    sequence: SequenceConfig
    timeline: TimelineConfig
    phantom: PhantomConfig
    intervention: InterventionConfig
    coils: CoilsConfig
    undersampling: UndersamplingConfig
    noise: NoiseConfig

    @model_validator(mode="after")
    def validate_cross_section_rules(self) -> "SimulationConfig":
        balloon = self.intervention.gd_balloon

        if balloon.enabled and balloon.path.control_points_file is None:
            raise ValueError(
                "intervention.gd_balloon.path.control_points_file is required "
                "when the Gd balloon is enabled"
            )

        if self.timeline.duration_s == "auto" and not balloon.enabled:
            raise ValueError(
                "timeline.duration_s='auto' requires the Gd balloon to be enabled"
            )

        if self.coils.enabled and self.coils.sensitivity_map is None:
            raise ValueError(
                "coils.sensitivity_map is required when coils are enabled"
            )

        off_resonance = self.scanner.effects.off_resonance
        if off_resonance.enabled and off_resonance.field_map is None:
            raise ValueError(
                "scanner.effects.off_resonance.field_map is required when "
                "off-resonance is enabled"
            )

        return self
