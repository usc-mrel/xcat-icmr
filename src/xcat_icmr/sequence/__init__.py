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
from xcat_icmr.sequence.orientation import (
    CoordinateTransforms,
    OrientationTransformError,
    build_coordinate_transforms,
    logical_to_dcs_matrix,
    pcs_to_dcs_matrix,
    reorient_spatial_array,
    reoriented_spatial_shape,
    transform_vector_components,
)

__all__ = [
    "MatlabComparison",
    "MatlabReferenceError",
    "CoordinateTransforms",
    "OrientationTransformError",
    "SequenceData",
    "SequenceReadError",
    "compare_to_matlab",
    "build_coordinate_transforms",
    "format_matlab_comparison",
    "format_sequence_summary",
    "read_pulseq_signature",
    "read_sequence",
    "logical_to_dcs_matrix",
    "pcs_to_dcs_matrix",
    "reorient_spatial_array",
    "reoriented_spatial_shape",
    "transform_vector_components",
]
