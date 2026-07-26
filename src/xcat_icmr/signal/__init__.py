"""MR signal models used to convert tissue properties into image contrast."""

from xcat_icmr.signal.bssfp import (
    BssfpSignalError,
    bssfp_signal,
    bssfp_signal_from_tissue_properties,
)
from xcat_icmr.signal.generation import (
    ContrastGeneration,
    ContrastGenerationError,
    format_contrast_generation,
    generate_bssfp_contrast,
)
from xcat_icmr.signal.matlab_reference import (
    BssfpMatlabComparison,
    MatlabSignalReferenceError,
    TissueSignalComparison,
    compare_bssfp_to_matlab,
    format_bssfp_matlab_comparison,
)

__all__ = [
    "BssfpMatlabComparison",
    "BssfpSignalError",
    "ContrastGeneration",
    "ContrastGenerationError",
    "MatlabSignalReferenceError",
    "TissueSignalComparison",
    "bssfp_signal",
    "bssfp_signal_from_tissue_properties",
    "compare_bssfp_to_matlab",
    "format_bssfp_matlab_comparison",
    "format_contrast_generation",
    "generate_bssfp_contrast",
]
