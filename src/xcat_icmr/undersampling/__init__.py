"""Temporal k-space undersampling."""

from xcat_icmr.undersampling.grouped import (
    GroupedAcquisitionError,
    GroupedAcquisitionResult,
    TemporalGrouping,
    format_grouped_acquisition,
    generate_bracketed_acquisitions,
    generate_grouped_acquisition,
    resolve_bracketing_multiples,
    resolve_temporal_grouping,
)

__all__ = [
    "GroupedAcquisitionError",
    "GroupedAcquisitionResult",
    "TemporalGrouping",
    "format_grouped_acquisition",
    "generate_bracketed_acquisitions",
    "generate_grouped_acquisition",
    "resolve_bracketing_multiples",
    "resolve_temporal_grouping",
]
