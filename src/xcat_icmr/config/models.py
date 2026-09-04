"""Typed models for the XCAT-iCMR simulation configuration."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    PositiveInt,
    model_validator,
)


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
    save_tissue_labels_nrrd: bool = False
    tissue_labels_nrrd_time_step_s: PositiveFloat = 0.050
    save_gt_contrast: bool
    save_fullysampled_contrast: bool
    save_fully_sampled_kspace: bool
    retain_xcat_binary_files: bool
    cache_full_tissue_kspace_library: bool = False
    save_debug_contrast_frame: bool = False
    debug_contrast_frame: PositiveInt = 1


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


class RfProfileConfig(ConfigModel):
    center_shift_mm: float = Field(default=0.0, allow_inf_nan=False)


class SequenceConfig(ConfigModel):
    folder: Path
    file: Path
    metadata_directory: Path
    coordinate_mode: Literal["XYZ-in-TRA"]
    orientation: Literal["COR", "SAG", "TRA"]
    rf_profile: RfProfileConfig = Field(default_factory=RfProfileConfig)
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
    reference_time_step_s: PositiveFloat
    xcat_to_reference: Literal["average", "center", "trajectory-aware"]

    @model_validator(mode="after")
    def validate_time_steps(self) -> "TimelineConfig":
        if self.reference_time_step_s < self.xcat_time_step_s:
            raise ValueError(
                "reference_time_step_s must be greater than or equal to "
                "xcat_time_step_s"
            )

        ratio = self.reference_time_step_s / self.xcat_time_step_s
        if not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                "reference_time_step_s must be an integer multiple of "
                "xcat_time_step_s"
            )
        return self

    @property
    def xcat_frames_per_reference_frame(self) -> int:
        return round(self.reference_time_step_s / self.xcat_time_step_s)


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


class InPlaneCropConfig(ConfigModel):
    """Optional 1-based, inclusive MATLAB index ranges in patient axes."""

    right_left: tuple[int, int] | None
    anterior_posterior: tuple[int, int] | None

    @model_validator(mode="after")
    def validate_ranges(self) -> "InPlaneCropConfig":
        for name, values in (
            ("right_left", self.right_left),
            ("anterior_posterior", self.anterior_posterior),
        ):
            if values is None:
                continue
            start, end = values
            if start < 1:
                raise ValueError(
                    f"{name} start must be at least 1 for MATLAB indexing"
                )
            if end < start:
                raise ValueError(
                    f"{name} end must be greater than or equal to start"
                )
        return self


class PhantomConfig(ConfigModel):
    patient_position: Literal["HFS"]
    anatomy: AnatomyConfig
    voxel_size_mm: tuple[PositiveFloat, PositiveFloat, PositiveFloat]
    matrix_size_xy: int = Field(gt=0)
    head_foot_slice_range: SliceRangeConfig
    in_plane_crop: InPlaneCropConfig
    transform: TransformConfig
    motion: MotionConfig


class BalloonPathConfig(ConfigModel):
    control_points_file: Path | None
    format: Literal["3d-slicer-markups"]
    coordinate_system: Literal["auto", "RAS", "LPS"]
    interpolation: Literal["cubic-arc-length"]


class BalloonMovementConfig(ConfigModel):
    velocity_cm_per_s: PositiveFloat
    start_time_s: float = Field(ge=0.0)
    traversal: Literal["one-way", "round-trip"] = "one-way"


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
    coordinate_frame: Literal["DCS"]
    axis_order: tuple[
        Literal["X", "Y", "Z"],
        Literal["X", "Y", "Z"],
        Literal["X", "Y", "Z"],
    ]
    normalize: bool

    @model_validator(mode="after")
    def validate_axis_order(self) -> "CoilsConfig":
        if set(self.axis_order) != {"X", "Y", "Z"}:
            raise ValueError("axis_order must contain X, Y, and Z exactly once")
        return self


class EncodingConfig(ConfigModel):
    """Target physical grid for fully sampled forward/adjoint encoding."""

    target_fov_mm: tuple[PositiveFloat, PositiveFloat, PositiveFloat]


class ViewOrderConfig(ConfigModel):
    """One repeatable acquisition cycle of trajectory-TR indices."""

    file: Path
    variable: str = Field(min_length=1)
    # Kept only so existing YAML files with `repeat: true` still load. New
    # configurations omit it because cycling is always implicit.
    repeat: Literal[True] = Field(default=True, exclude=True)


FrameCount = Literal["auto"] | PositiveInt


class AcquisitionConfig(ConfigModel):
    """Dynamic acquisition timing and generic trajectory-TR ordering."""

    frame_duration_s: PositiveFloat
    tr_snap_tolerance_percent: float = Field(ge=0.0, allow_inf_nan=False)
    frame_count: FrameCount = "auto"
    incomplete_final_frame: Literal["drop"] = "drop"
    view_order: ViewOrderConfig


class UndersamplingConfig(ConfigModel):
    enabled: bool
    frame_duration_s: PositiveFloat


class NoiseConfig(ConfigModel):
    enabled: bool
    snr_db: PositiveFloat
    coil_covariance: Literal["identity"] | Path
    seed: int = Field(ge=0)


class CurvedLineProfileConfig(ConfigModel):
    """Derived time-distance profile along the configured balloon path."""

    enabled: bool = False
    sample_step_mm: PositiveFloat = 0.5
    tube_radius_mm: PositiveFloat = 7.0
    angular_samples: int = Field(default=16, ge=4)


class AnalysisConfig(ConfigModel):
    curved_line_profile: CurvedLineProfileConfig = Field(
        default_factory=CurvedLineProfileConfig
    )


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
    encoding: EncodingConfig
    acquisition: AcquisitionConfig
    undersampling: UndersamplingConfig
    noise: NoiseConfig
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)

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

        if (
            self.outputs.cache_full_tissue_kspace_library
            and not self.coils.enabled
        ):
            raise ValueError(
                "outputs.cache_full_tissue_kspace_library requires coils.enabled"
            )

        if balloon.enabled and balloon.composition.mode != "additive":
            raise ValueError(
                "the production acquisition currently requires "
                "intervention.gd_balloon.composition.mode='additive'"
            )

        if self.outputs.save_tissue_labels_nrrd:
            ratio = (
                self.outputs.tissue_labels_nrrd_time_step_s
                / self.timeline.xcat_time_step_s
            )
            if ratio < 1 or not math.isclose(
                ratio, round(ratio), rel_tol=0.0, abs_tol=1e-9
            ):
                raise ValueError(
                    "outputs.tissue_labels_nrrd_time_step_s must be an "
                    "integer multiple of timeline.xcat_time_step_s"
                )

        off_resonance = self.scanner.effects.off_resonance
        if off_resonance.enabled and off_resonance.field_map is None:
            raise ValueError(
                "scanner.effects.off_resonance.field_map is required when "
                "off-resonance is enabled"
            )

        return self
