"""Coil-sensitivity loading and normalization."""

from xcat_icmr.coils.sensitivity import (
    RssNormalization,
    SensitivityMapError,
    SensitivityMapInfo,
    format_sensitivity_preparation,
    inspect_sensitivity_map,
    load_normalized_coil,
    load_normalized_coil_in_logical_frame,
    prepare_rss_normalization,
    sensitivity_shape_in_logical_frame,
)

__all__ = [
    "RssNormalization",
    "SensitivityMapError",
    "SensitivityMapInfo",
    "format_sensitivity_preparation",
    "inspect_sensitivity_map",
    "load_normalized_coil",
    "load_normalized_coil_in_logical_frame",
    "prepare_rss_normalization",
    "sensitivity_shape_in_logical_frame",
]
