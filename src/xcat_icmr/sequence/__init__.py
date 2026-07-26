"""Sequence and trajectory metadata handling."""

from xcat_icmr.sequence.matlab_reference import (
    MatlabComparison,
    MatlabReferenceError,
    compare_to_matlab,
    format_matlab_comparison,
)
from xcat_icmr.sequence.reader import (
    SequenceData,
    SequenceReadError,
    format_sequence_summary,
    read_pulseq_signature,
    read_sequence,
)

__all__ = [
    "MatlabComparison",
    "MatlabReferenceError",
    "SequenceData",
    "SequenceReadError",
    "compare_to_matlab",
    "format_matlab_comparison",
    "format_sequence_summary",
    "read_pulseq_signature",
    "read_sequence",
]
