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
from xcat_icmr.signal.rf_contrast import (
    calculate_rf_profile_bssfp_contrast,
    RfProfileContrastError,
    RfProfileContrastGeneration,
    format_rf_profile_contrast_generation,
    generate_rf_profile_bssfp_contrast,
)
from xcat_icmr.signal.slice_profile import (
    PulseqExcitation,
    SliceProfile,
    SliceProfileError,
    generate_slice_profile,
    read_pulseq_excitation,
    simulate_bloch_profile,
)

__all__ = [
    "BssfpMatlabComparison",
    "BssfpSignalError",
    "ContrastGeneration",
    "ContrastGenerationError",
    "MatlabSignalReferenceError",
    "PulseqExcitation",
    "RfProfileContrastError",
    "RfProfileContrastGeneration",
    "SliceProfile",
    "SliceProfileError",
    "TissueSignalComparison",
    "bssfp_signal",
    "bssfp_signal_from_tissue_properties",
    "compare_bssfp_to_matlab",
    "format_bssfp_matlab_comparison",
    "format_contrast_generation",
    "format_rf_profile_contrast_generation",
    "generate_rf_profile_bssfp_contrast",
    "calculate_rf_profile_bssfp_contrast",
    "generate_slice_profile",
    "generate_bssfp_contrast",
    "read_pulseq_excitation",
    "simulate_bloch_profile",
]
