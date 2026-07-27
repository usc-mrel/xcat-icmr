"""Non-Cartesian encoding preparation and, later, NUFFT backends."""

from xcat_icmr.encoding.inputs import (
    EncodingInputError,
    LogicalInputPreview,
    PreparedContrast,
    center_padding,
    format_prepared_contrast,
    format_logical_input_preview,
    prepare_contrast_for_encoding,
    save_logical_input_preview,
)
from xcat_icmr.encoding.sigpy_backend import (
    NufftBackendError,
    SigpyNufftBackend,
)
from xcat_icmr.encoding.trajectory import (
    EncodingGrids,
    EncodingTrajectory,
    TrajectoryPreparationError,
    format_encoding_trajectory,
    prepare_encoding_grids,
    prepare_sigpy_trajectory,
)
from xcat_icmr.encoding.validation import (
    ReducedNufftValidation,
    SigpyReferenceValidation,
    format_reduced_nufft_validation,
    format_sigpy_reference_validation,
    run_reduced_nufft_validation,
    validate_sigpy_reference,
)

__all__ = [
    "EncodingInputError",
    "LogicalInputPreview",
    "EncodingGrids",
    "EncodingTrajectory",
    "NufftBackendError",
    "PreparedContrast",
    "ReducedNufftValidation",
    "SigpyNufftBackend",
    "SigpyReferenceValidation",
    "TrajectoryPreparationError",
    "center_padding",
    "format_prepared_contrast",
    "format_logical_input_preview",
    "format_encoding_trajectory",
    "format_reduced_nufft_validation",
    "format_sigpy_reference_validation",
    "prepare_contrast_for_encoding",
    "save_logical_input_preview",
    "prepare_encoding_grids",
    "prepare_sigpy_trajectory",
    "run_reduced_nufft_validation",
    "validate_sigpy_reference",
]
