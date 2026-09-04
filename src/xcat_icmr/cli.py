"""Command-line entry point for XCAT-iCMR."""

from __future__ import annotations

import argparse
from dataclasses import replace
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from pydantic import ValidationError
from scipy.io import loadmat, whosmat

from xcat_icmr.acquisition import (
    AcquisitionScheduleError,
    build_acquisition_schedule,
    estimate_dynamic_acquisition_storage,
)
from xcat_icmr.acquisition.dynamic import (
    DynamicAcquisitionError,
    format_dynamic_acquisition,
    generate_dynamic_acquisition,
)
from xcat_icmr.acquisition.reference import (
    generate_dynamic_fullysampled_reference,
)
from xcat_icmr.analysis import (
    CurvedLineProfileError,
    format_curved_line_profile,
    generate_curved_line_profile,
)

from xcat_icmr.cache import (
    artifact_cache_status,
    contrast_cache_entry,
    contrast_frame_path,
    contrast_profile_path,
    fullysampled_reference_cache_entry,
    label_cache_entry,
    stage_reuse_status,
    tissue_kspace_cache_entry,
    write_artifact_manifest,
    write_stage_manifest,
)
from xcat_icmr.cache_migration import (
    CacheMigrationError,
    adopt_legacy_cache,
    format_cache_migration,
)

from xcat_icmr.coils import (
    SensitivityMapError,
    format_sensitivity_preparation,
    inspect_sensitivity_map,
    load_normalized_coil,
    load_normalized_coil_in_logical_frame,
    prepare_rss_normalization,
    sensitivity_shape_in_logical_frame,
)
from xcat_icmr.config import (
    ConfigurationLoadError,
    format_summary,
    load_config,
    validate_paths,
)
from xcat_icmr.config.loader import format_validation_error
from xcat_icmr.encoding import (
    EncodingInputError,
    NufftBackendError,
    TrajectoryPreparationError,
    format_fov_psf_diagnostic,
    format_multicoil_nufft_debug,
    format_reduced_nufft_validation,
    format_logical_input_preview,
    format_sigpy_reference_validation,
    format_prepared_contrast,
    compare_device_references,
    prepare_contrast_for_encoding,
    prepare_encoding_grids,
    measure_centered_signal_support,
    run_fov_psf_diagnostic,
    run_multicoil_nufft_debug,
    run_reduced_nufft_validation,
    scale_isotropic_trajectory_to_resolution,
    save_logical_input_preview,
    validate_sigpy_reference,
    validate_image_reference,
)
from xcat_icmr.encoding.tissue_library import (
    TissueKspaceLibraryError,
    format_tissue_kspace_library,
    generate_tissue_kspace_library,
)
from xcat_icmr.encoding.tissue_reference import (
    TissueAdjointReferenceError,
    format_tissue_adjoint_reference,
    generate_tissue_adjoint_reference,
)
from xcat_icmr.encoding.fullysampled_reference import (
    FullysampledReferenceError,
    format_fullysampled_reference,
    generate_fullysampled_reference,
)
from xcat_icmr.exporting import (
    NrrdExportError,
    export_contrast_series_nrrd,
    export_label_series_nrrd,
    format_nrrd_export,
)
from xcat_icmr.intervention import (
    BalloonPathError,
    GdSignalError,
    SparseBalloonError,
)
from xcat_icmr.intervention.debug import (
    BalloonDebugError,
    format_balloon_debug,
    format_balloon_path_debug,
    generate_balloon_debug_frames,
    generate_balloon_path_debug,
)
from xcat_icmr.intervention.encoding_debug import (
    BalloonEncodingDebugError,
    format_balloon_encoding_debug,
    validate_balloon_kspace_linearity,
)
from xcat_icmr.intervention.reference_debug import (
    ThreePositionReferenceDebugError,
    format_three_position_reference_debug,
    generate_three_position_reference_debug,
)
from xcat_icmr import __version__
from xcat_icmr.phantom import (
    XcatBinaryReadError,
    XcatFramePlanError,
    XcatLabelComparisonError,
    XcatLabelConversionError,
    XcatExecutionError,
    XcatParameterError,
    compare_xcat_labels_to_matlab,
    convert_xcat_labels_to_mat,
    format_xcat_frame_plan,
    format_xcat_execution,
    format_xcat_label_comparison,
    format_xcat_label_conversion,
    format_xcat_parameter_summary,
    format_xcat_preflight,
    open_xcat_binary,
    xcat_label_shape,
    plan_xcat_frames,
    prepare_xcat_parameter_file,
    preflight_xcat_invocation,
    execute_xcat_invocation,
    execute_streaming_xcat_invocation,
    render_xcat_parameter_file,
)
from xcat_icmr.phantom.frames import XcatFramePlan
from xcat_icmr.phantom.parameters import XcatParameterFile
from xcat_icmr.sequence import (
    MatlabReferenceError,
    OrientationTransformError,
    SequenceReadError,
    compare_to_matlab,
    build_coordinate_transforms,
    format_matlab_comparison,
    format_sequence_summary,
    read_sequence,
)
from xcat_icmr.signal import (
    ContrastGenerationError,
    MatlabSignalReferenceError,
    RfProfileContrastError,
    SliceProfileError,
    compare_bssfp_to_matlab,
    format_bssfp_matlab_comparison,
    format_contrast_generation,
    format_rf_profile_contrast_generation,
    generate_rf_profile_bssfp_contrast,
    generate_slice_profile,
    generate_bssfp_contrast,
    read_pulseq_excitation,
)
from xcat_icmr.tissue import get_tissue_library


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xcat-icmr",
        description="Dynamic interventional cardiovascular MRI simulation",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command")

    validate_parser = subparsers.add_parser(
        "validate",
        help="validate a simulation YAML file without running the simulation",
    )
    validate_parser.add_argument(
        "configuration",
        type=Path,
        help="path to a simulation YAML file",
    )

    reuse_parser = subparsers.add_parser(
        "inspect-reuse",
        help="report whether labels, contrast, and k-space can be reused",
    )
    reuse_parser.add_argument(
        "configuration",
        type=Path,
        help="path to a simulation YAML file",
    )

    cache_parser = subparsers.add_parser(
        "inspect-cache",
        help="show content-addressed label, contrast, and k-space cache IDs",
    )
    cache_parser.add_argument(
        "configuration",
        type=Path,
        help="path to a simulation YAML file",
    )

    adopt_parser = subparsers.add_parser(
        "adopt-legacy-cache",
        help="convert existing labels to uint16 and adopt existing contrasts",
    )
    adopt_parser.add_argument(
        "configuration",
        type=Path,
        help="path to a simulation YAML file",
    )
    adopt_parser.add_argument(
        "--labels-only",
        action="store_true",
        help="adopt labels without adopting the current contrast series",
    )
    adopt_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace files inside the resolved cache IDs",
    )

    inspect_parser = subparsers.add_parser(
        "inspect-sequence",
        help="resolve sequence metadata and report trajectory parameters",
    )
    inspect_parser.add_argument(
        "configuration",
        type=Path,
        help="path to a simulation YAML file",
    )
    inspect_parser.add_argument(
        "--matlab-reference",
        type=Path,
        help="optional MATLAB v7.3 par file for exact comparison",
    )

    compare_bssfp_parser = subparsers.add_parser(
        "compare-bssfp",
        help="compare Python bSSFP contrast with MATLAB v7.3 volumes",
    )
    compare_bssfp_parser.add_argument(
        "configuration",
        type=Path,
        help="path to the simulation YAML used for sequence parameters",
    )
    compare_bssfp_parser.add_argument(
        "labels",
        type=Path,
        help="MATLAB v7.3 tissue-label file containing dataset P",
    )
    compare_bssfp_parser.add_argument(
        "matlab_image",
        type=Path,
        help="MATLAB v7.3 contrast file containing dataset image",
    )
    compare_bssfp_parser.add_argument(
        "--chunk-slices",
        type=int,
        default=8,
        help="number of slices read per chunk (default: 8)",
    )
    compare_bssfp_parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="absolute comparison tolerance (default: 1e-5)",
    )
    compare_bssfp_parser.add_argument(
        "--rtol",
        type=float,
        default=1e-6,
        help="relative comparison tolerance (default: 1e-6)",
    )

    prepare_xcat_parser = subparsers.add_parser(
        "prepare-xcat",
        help="write a run-specific XCAT parameter file without launching XCAT",
    )
    prepare_xcat_parser.add_argument(
        "configuration",
        type=Path,
        help="path to a simulation YAML file",
    )
    prepare_xcat_parser.add_argument(
        "--output",
        type=Path,
        help="output .par path (default: run.output_root/xcat/parameters.par)",
    )
    xcat_frame_mode = prepare_xcat_parser.add_mutually_exclusive_group()
    xcat_frame_mode.add_argument(
        "--debug-one-frame",
        dest="debug_one_frame",
        action="store_true",
        help="write one frame while still reporting the full motion cycle",
    )
    xcat_frame_mode.add_argument(
        "--full-motion-cycle",
        dest="debug_one_frame",
        action="store_false",
        help="write the complete non-repeating motion cycle",
    )
    prepare_xcat_parser.set_defaults(debug_one_frame=True)

    plan_xcat_parser = subparsers.add_parser(
        "plan-xcat",
        help="show XCAT frame times and expected paths without writing files",
    )
    plan_xcat_parser.add_argument(
        "configuration",
        type=Path,
        help="path to a simulation YAML file",
    )
    xcat_plan_mode = plan_xcat_parser.add_mutually_exclusive_group()
    xcat_plan_mode.add_argument(
        "--debug-one-frame",
        dest="debug_one_frame",
        action="store_true",
        help="plan one frame while still reporting the full motion cycle",
    )
    xcat_plan_mode.add_argument(
        "--full-motion-cycle",
        dest="debug_one_frame",
        action="store_false",
        help="plan the complete non-repeating motion cycle",
    )
    plan_xcat_parser.set_defaults(debug_one_frame=True)

    run_xcat_parser = subparsers.add_parser(
        "run-xcat",
        help="run XCAT after preflight, or inspect it with --dry-run",
    )
    run_xcat_parser.add_argument(
        "configuration",
        type=Path,
        help="path to a simulation YAML file",
    )
    run_xcat_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="prepare parameters and validate the command without launching XCAT",
    )
    xcat_run_mode = run_xcat_parser.add_mutually_exclusive_group()
    xcat_run_mode.add_argument(
        "--debug-one-frame",
        dest="debug_one_frame",
        action="store_true",
        help="preflight one frame while reporting the full motion cycle",
    )
    xcat_run_mode.add_argument(
        "--full-motion-cycle",
        dest="debug_one_frame",
        action="store_false",
        help="preflight the complete non-repeating motion cycle",
    )
    run_xcat_parser.set_defaults(debug_one_frame=True)

    dynamic_cycle_parser = subparsers.add_parser(
        "generate-dynamic-cycle",
        help="stream XCAT frames into verified labels and bSSFP contrasts",
    )
    dynamic_cycle_parser.add_argument(
        "configuration",
        type=Path,
        help="path to a simulation YAML file",
    )
    dynamic_cycle_parser.add_argument(
        "--chunk-slices",
        type=int,
        default=8,
        help="number of label/contrast slices processed per chunk",
    )
    dynamic_cycle_parser.add_argument(
        "--regenerate-from-frame",
        type=int,
        metavar="N",
        help=(
            "regenerate and replace labels and contrasts from one-based "
            "frame N through the end of the motion cycle"
        ),
    )

    compare_xcat_labels_parser = subparsers.add_parser(
        "compare-xcat-labels",
        help="decode an XCAT binary and compare it with a MATLAB label volume",
    )
    compare_xcat_labels_parser.add_argument(
        "configuration",
        type=Path,
        help="path to a simulation YAML file",
    )
    compare_xcat_labels_parser.add_argument(
        "matlab_reference",
        type=Path,
        help="MATLAB v7.3 file containing the P label volume",
    )
    compare_xcat_labels_parser.add_argument(
        "--binary",
        type=Path,
        help="raw XCAT binary (default: planned debug frame 1)",
    )
    compare_xcat_labels_parser.add_argument(
        "--chunk-slices",
        type=int,
        default=8,
        help="number of logical slices compared per chunk (default: 8)",
    )

    convert_xcat_labels_parser = subparsers.add_parser(
        "convert-xcat-labels",
        help="convert XCAT binaries to verified MATLAB label files",
    )
    convert_xcat_labels_parser.add_argument(
        "configuration",
        type=Path,
        help="path to a simulation YAML file",
    )
    convert_xcat_labels_parser.add_argument(
        "--chunk-slices",
        type=int,
        default=8,
        help="number of slices validated per chunk (default: 8)",
    )
    convert_xcat_labels_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace label files that already exist",
    )
    xcat_convert_mode = convert_xcat_labels_parser.add_mutually_exclusive_group()
    xcat_convert_mode.add_argument(
        "--debug-one-frame",
        dest="debug_one_frame",
        action="store_true",
        help="convert the first planned frame only",
    )
    xcat_convert_mode.add_argument(
        "--full-motion-cycle",
        dest="debug_one_frame",
        action="store_false",
        help="convert every frame in the complete motion cycle",
    )
    convert_xcat_labels_parser.set_defaults(debug_one_frame=True)

    generate_contrast_parser = subparsers.add_parser(
        "generate-contrast",
        help="generate verified MRI contrast images from tissue-label files",
    )
    generate_contrast_parser.add_argument(
        "configuration",
        type=Path,
        help="path to a simulation YAML file",
    )
    generate_contrast_parser.add_argument(
        "--chunk-slices",
        type=int,
        default=8,
        help="number of slices processed per chunk (default: 8)",
    )
    generate_contrast_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace contrast files that already exist",
    )
    contrast_frame_mode = generate_contrast_parser.add_mutually_exclusive_group()
    contrast_frame_mode.add_argument(
        "--debug-one-frame",
        dest="debug_one_frame",
        action="store_true",
        help="generate contrast for the first planned frame only",
    )
    contrast_frame_mode.add_argument(
        "--full-motion-cycle",
        dest="debug_one_frame",
        action="store_false",
        help="generate contrast for every frame in the motion cycle",
    )
    generate_contrast_parser.set_defaults(debug_one_frame=True)

    export_nrrd_parser = subparsers.add_parser(
        "export-contrast-nrrd",
        help="stream the complete contrast motion cycle into one 4-D NRRD",
    )
    export_nrrd_parser.add_argument(
        "configuration",
        type=Path,
        help="path to a simulation YAML file",
    )
    export_nrrd_parser.add_argument(
        "--output",
        type=Path,
        help="output NRRD path (default: run exports directory)",
    )
    export_nrrd_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing 4-D NRRD",
    )

    export_label_nrrd_parser = subparsers.add_parser(
        "export-labels-nrrd",
        help="stream the complete tissue-label cycle into one uint16 4-D NRRD",
    )
    export_label_nrrd_parser.add_argument(
        "configuration",
        type=Path,
        help="path to a simulation YAML file",
    )
    export_label_nrrd_parser.add_argument(
        "--output",
        type=Path,
        help="output NRRD path (default: run exports directory)",
    )
    export_label_nrrd_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing tissue-label NRRD",
    )

    balloon_debug_parser = subparsers.add_parser(
        "generate-balloon-debug",
        help="save representative high-resolution tissue/Gd replacement frames",
    )
    balloon_debug_parser.add_argument(
        "configuration",
        type=Path,
        help="path to a simulation YAML file",
    )
    balloon_debug_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing balloon debug frames",
    )

    balloon_path_debug_parser = subparsers.add_parser(
        "generate-balloon-path-debug",
        help="save the complete swept A-to-B balloon path in one anatomy frame",
    )
    balloon_path_debug_parser.add_argument(
        "configuration",
        type=Path,
        help="path to a simulation YAML file",
    )
    balloon_path_debug_parser.add_argument(
        "--center-spacing-mm",
        type=float,
        default=0.5,
        help="distance between swept-path balloon centres (default: 0.5 mm)",
    )
    balloon_path_debug_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing complete-path debug frame",
    )

    prepare_kspace_inputs_parser = subparsers.add_parser(
        "prepare-kspace-inputs",
        help="validate and normalize coil/image inputs before NUFFT",
    )
    prepare_kspace_inputs_parser.add_argument(
        "configuration",
        type=Path,
        help="path to a simulation YAML file",
    )
    prepare_kspace_inputs_parser.add_argument(
        "--rebuild-rss-cache",
        action="store_true",
        help="recompute the sensitivity-map RSS denominator",
    )

    validate_sigpy_parser = subparsers.add_parser(
        "generate-kspace-debug",
        help="run the full 3-D trajectory for one coil and save its adjoint",
    )
    validate_sigpy_parser.add_argument(
        "configuration",
        type=Path,
        help="path to a simulation YAML file",
    )
    validate_sigpy_parser.add_argument(
        "--coil",
        type=int,
        default=0,
        help="coil index in the prepared file (default: 0)",
    )
    validate_sigpy_parser.add_argument(
        "--device-id",
        type=int,
        help="override YAML device ID (-1 CPU; 0, 1, ... GPU)",
    )

    multicoil_debug_parser = subparsers.add_parser(
        "generate-kspace-all-coils-debug",
        help="run the full 3-D trajectory for every coil and save RSS adjoint",
    )
    multicoil_debug_parser.add_argument(
        "configuration",
        type=Path,
        help="path to a simulation YAML file",
    )

    balloon_kspace_debug_parser = subparsers.add_parser(
        "generate-balloon-kspace-debug",
        help=(
            "encode the frame-1 tissue/Gd balloon image with all coils and "
            "save its RSS adjoint"
        ),
    )
    balloon_kspace_debug_parser.add_argument(
        "configuration",
        type=Path,
        help="path to a simulation YAML file",
    )

    balloon_linearity_parser = subparsers.add_parser(
        "validate-balloon-kspace-linearity",
        help="compare full-image NUFFT with tissue plus sparse Gd-delta NUFFT",
    )
    balloon_linearity_parser.add_argument(
        "configuration",
        type=Path,
        help="path to a simulation YAML file",
    )

    reference_parser = subparsers.add_parser(
        "generate-fullysampled-reference",
        help=(
            "forward/adjoint encode tissue contrast and save only the "
            "coil-combined fully sampled image series"
        ),
    )
    reference_parser.add_argument(
        "configuration",
        type=Path,
        help="path to a simulation YAML file",
    )
    reference_parser.add_argument(
        "--start-frame",
        type=int,
        default=1,
        help="first one-based reference frame to encode (default: 1)",
    )
    reference_parser.add_argument(
        "--end-frame",
        type=int,
        help="last one-based reference frame (default: final frame)",
    )
    reference_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace selected completed reference frames",
    )

    acquisition_plan_parser = subparsers.add_parser(
        "plan-acquisition",
        help="validate TR snapping and the generic user view order",
    )
    acquisition_plan_parser.add_argument("configuration", type=Path)

    tissue_library_parser = subparsers.add_parser(
        "generate-tissue-kspace-library",
        help=(
            "transiently calculate each contrast phase and persist its full "
            "multicoil trajectory k-space"
        ),
    )
    tissue_library_parser.add_argument("configuration", type=Path)
    tissue_library_parser.add_argument("--start-frame", type=int, default=1)
    tissue_library_parser.add_argument("--end-frame", type=int)
    tissue_library_parser.add_argument("--overwrite", action="store_true")
    tissue_library_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show shapes and storage preflight without generating data",
    )

    tissue_adjoint_parser = subparsers.add_parser(
        "generate-tissue-adjoint-reference",
        help=(
            "reconstruct cached fully sampled tissue k-space into one "
            "coil-combined 4-D reference"
        ),
    )
    tissue_adjoint_parser.add_argument("configuration", type=Path)
    tissue_adjoint_parser.add_argument("--start-frame", type=int, default=1)
    tissue_adjoint_parser.add_argument("--end-frame", type=int)
    tissue_adjoint_parser.add_argument("--overwrite", action="store_true")
    tissue_adjoint_parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="reconstruct available selected phases and leave missing phases incomplete",
    )

    dynamic_acquisition_parser = subparsers.add_parser(
        "generate-dynamic-acquisition",
        help="gather tissue arms and add sparse moving-Gd k-space at every TR",
    )
    dynamic_acquisition_parser.add_argument("configuration", type=Path)
    dynamic_acquisition_parser.add_argument("--overwrite", action="store_true")
    dynamic_acquisition_parser.add_argument(
        "--view-order-cycles",
        type=int,
        help=(
            "generate exactly this many complete view-order cycles in a "
            "separate debug output"
        ),
    )
    dynamic_acquisition_parser.add_argument(
        "--save-adjoint-debug",
        action="store_true",
        help="save a coil-combined adjoint image for every output frame",
    )
    dynamic_acquisition_parser.add_argument(
        "--dry-run", action="store_true", help="validate and estimate storage only"
    )

    dynamic_reference_parser = subparsers.add_parser(
        "generate-dynamic-fullysampled-reference",
        help="save the motion-averaged fully sampled tissue-plus-Gd image series",
    )
    dynamic_reference_parser.add_argument("configuration", type=Path)
    dynamic_reference_parser.add_argument("--overwrite", action="store_true")

    curved_profile_parser = subparsers.add_parser(
        "generate-curved-line-profile",
        help="measure a curved-tube intensity profile in a fully sampled 4-D image",
    )
    curved_profile_parser.add_argument("configuration", type=Path)
    curved_profile_parser.add_argument(
        "--input",
        type=Path,
        help=(
            "optional fully sampled image HDF5; defaults to the dynamic-acquisition "
            "cache for this configuration"
        ),
    )
    curved_profile_parser.add_argument("--overwrite", action="store_true")

    three_position_parser = subparsers.add_parser(
        "generate-three-position-reference-debug",
        help=(
            "add first/middle/last catheter balloons to frame 1 and encode "
            "their all-coil fully sampled reference"
        ),
    )
    three_position_parser.add_argument(
        "configuration",
        type=Path,
        help="path to a simulation YAML file",
    )
    three_position_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace the existing three-position debug MAT file",
    )

    parity_parser = subparsers.add_parser(
        "compare-nufft-devices",
        help="compare saved CPU and GPU forward/adjoint references",
    )
    parity_parser.add_argument("cpu_reference", type=Path)
    parity_parser.add_argument("gpu_reference", type=Path)
    parity_parser.add_argument("--output", type=Path, required=True)

    image_validation_parser = subparsers.add_parser(
        "validate-kspace-reference",
        help="compare shifted high-resolution GT with an all-coil RSS adjoint",
    )
    image_validation_parser.add_argument("shifted_gt", type=Path)
    image_validation_parser.add_argument("multicoil_reference", type=Path)
    image_validation_parser.add_argument("--output", type=Path, required=True)

    fov_diagnostic_parser = subparsers.add_parser(
        "diagnose-kspace-fov",
        help="generate an impulse PSF and compare fixed reconstruction FOVs",
    )
    fov_diagnostic_parser.add_argument(
        "configuration",
        type=Path,
        help="path to a simulation YAML file",
    )
    fov_diagnostic_parser.add_argument(
        "--coil",
        type=int,
        default=0,
        help="coil index of the saved tissue k-space (default: 0)",
    )
    fov_diagnostic_parser.add_argument(
        "--kspace",
        type=Path,
        help="existing debug MAT file containing kspace",
    )
    fov_diagnostic_parser.add_argument(
        "--support-threshold",
        type=float,
        default=0.01,
        help="relative magnitude used to measure object support (default: 0.01)",
    )
    fov_diagnostic_parser.add_argument(
        "--support-margin-mm",
        type=float,
        default=10.0,
        help="margin added on both sides of measured support (default: 10)",
    )
    fov_diagnostic_parser.add_argument(
        "--device-id",
        type=int,
        default=-1,
        help="-1 for CPU or a non-negative CuPy GPU ID (default: -1)",
    )
    fov_diagnostic_parser.add_argument(
        "--output",
        type=Path,
        help="diagnostic MAT output path",
    )
    fov_diagnostic_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing diagnostic MAT file",
    )
    return parser


def _validate(configuration: Path) -> int:
    try:
        config = load_config(configuration)
    except ConfigurationLoadError as exc:
        print(f"Configuration error:\n  {exc}", file=sys.stderr)
        return 2
    except ValidationError as exc:
        print(format_validation_error(exc), file=sys.stderr)
        return 2

    issues = validate_paths(config)
    if issues:
        print("Configuration path errors:", file=sys.stderr)
        for issue in issues:
            print(f"  - {issue.format()}", file=sys.stderr)
        return 2

    print("Configuration is valid.\n")
    print(format_summary(config))
    return 0


def _inspect_sequence(
    configuration: Path, matlab_reference: Path | None
) -> int:
    try:
        config = load_config(configuration)
        sequence = read_sequence(config.sequence)
        print(format_sequence_summary(sequence))

        if matlab_reference is not None:
            comparison = compare_to_matlab(sequence, matlab_reference)
            print("\n" + format_matlab_comparison(comparison))
            return 0 if comparison.passed else 1
        return 0
    except ConfigurationLoadError as exc:
        print(f"Configuration error:\n  {exc}", file=sys.stderr)
    except ValidationError as exc:
        print(format_validation_error(exc), file=sys.stderr)
    except (SequenceReadError, MatlabReferenceError) as exc:
        print(f"Sequence error:\n  {exc}", file=sys.stderr)
    return 2


def _compare_bssfp(
    configuration: Path,
    labels: Path,
    matlab_image: Path,
    *,
    chunk_slices: int,
    atol: float,
    rtol: float,
) -> int:
    try:
        config = load_config(configuration)
        sequence = read_sequence(config.sequence)
        library = get_tissue_library(
            config.sequence.contrast.tissue_library
        )
        report = compare_bssfp_to_matlab(
            labels,
            matlab_image,
            library,
            flip_angle_deg=sequence.flip_angle_deg,
            te_ms=sequence.te_ms,
            tr_ms=sequence.tr_ms,
            off_resonance_enabled=(
                config.scanner.effects.off_resonance.enabled
            ),
            chunk_slices=chunk_slices,
            atol=atol,
            rtol=rtol,
        )
        print(format_bssfp_matlab_comparison(report))
        return 0 if report.passed else 1
    except ConfigurationLoadError as exc:
        print(f"Configuration error:\n  {exc}", file=sys.stderr)
    except ValidationError as exc:
        print(format_validation_error(exc), file=sys.stderr)
    except SequenceReadError as exc:
        print(f"Sequence error:\n  {exc}", file=sys.stderr)
    except KeyError as exc:
        print(f"Tissue-library error:\n  {exc}", file=sys.stderr)
    except MatlabSignalReferenceError as exc:
        print(f"MATLAB comparison error:\n  {exc}", file=sys.stderr)
    except (ValueError, NotImplementedError) as exc:
        print(f"Signal-model error:\n  {exc}", file=sys.stderr)
    return 2


def _prepare_xcat(
    configuration: Path,
    *,
    output: Path | None,
    debug_one_frame: bool,
) -> int:
    try:
        config = load_config(configuration)
        result = prepare_xcat_parameter_file(
            config,
            output_path=output,
            debug_one_frame=debug_one_frame,
        )
        print(format_xcat_parameter_summary(result))
        return 0
    except ConfigurationLoadError as exc:
        print(f"Configuration error:\n  {exc}", file=sys.stderr)
    except ValidationError as exc:
        print(format_validation_error(exc), file=sys.stderr)
    except XcatParameterError as exc:
        print(f"XCAT parameter error:\n  {exc}", file=sys.stderr)
    return 2


def _plan_xcat(configuration: Path, *, debug_one_frame: bool) -> int:
    try:
        config = load_config(configuration)
        plan = plan_xcat_frames(
            config, debug_one_frame=debug_one_frame
        )
        print(format_xcat_frame_plan(plan))
        return 0
    except ConfigurationLoadError as exc:
        print(f"Configuration error:\n  {exc}", file=sys.stderr)
    except ValidationError as exc:
        print(format_validation_error(exc), file=sys.stderr)
    except (XcatParameterError, XcatFramePlanError) as exc:
        print(f"XCAT frame-plan error:\n  {exc}", file=sys.stderr)
    return 2


def _run_xcat(
    configuration: Path,
    *,
    dry_run: bool,
    debug_one_frame: bool,
) -> int:
    try:
        config = load_config(configuration)
        parameters = prepare_xcat_parameter_file(
            config, debug_one_frame=debug_one_frame
        )
        frames = plan_xcat_frames(
            config, debug_one_frame=debug_one_frame
        )
        report = preflight_xcat_invocation(config, parameters, frames)
        print(format_xcat_preflight(report, dry_run=dry_run))
        if dry_run or not report.passed:
            return 0 if report.passed else 2

        result = execute_xcat_invocation(config, frames, report)
        print("\n" + format_xcat_execution(result))
        return 0
    except ConfigurationLoadError as exc:
        print(f"Configuration error:\n  {exc}", file=sys.stderr)
    except ValidationError as exc:
        print(format_validation_error(exc), file=sys.stderr)
    except (XcatParameterError, XcatFramePlanError) as exc:
        print(f"XCAT setup error:\n  {exc}", file=sys.stderr)
    except XcatExecutionError as exc:
        print(f"XCAT execution error:\n  {exc}", file=sys.stderr)
    return 2


def _compare_xcat_labels(
    configuration: Path,
    matlab_reference: Path,
    *,
    binary: Path | None,
    chunk_slices: int,
) -> int:
    try:
        config = load_config(configuration)
        if binary is None:
            frame_plan = plan_xcat_frames(config, debug_one_frame=True)
            binary = frame_plan.frames[0].binary_path
        volume = open_xcat_binary(config, binary)
        report = compare_xcat_labels_to_matlab(
            volume,
            matlab_reference,
            chunk_slices=chunk_slices,
        )
        print(format_xcat_label_comparison(report))
        return 0 if report.passed else 1
    except ConfigurationLoadError as exc:
        print(f"Configuration error:\n  {exc}", file=sys.stderr)
    except ValidationError as exc:
        print(format_validation_error(exc), file=sys.stderr)
    except (
        XcatParameterError,
        XcatFramePlanError,
        XcatBinaryReadError,
        XcatLabelComparisonError,
    ) as exc:
        print(f"XCAT label error:\n  {exc}", file=sys.stderr)
    return 2


def _convert_xcat_labels(
    configuration: Path,
    *,
    debug_one_frame: bool,
    chunk_slices: int,
    overwrite: bool,
) -> int:
    try:
        config = load_config(configuration)
        if not config.outputs.save_tissue_labels:
            raise XcatLabelConversionError(
                "outputs.save_tissue_labels is false; enable it to write "
                "MATLAB label files"
            )
        frame_plan = plan_xcat_frames(
            config, debug_one_frame=debug_one_frame
        )
        reports = []
        total = len(frame_plan.frames)
        for completed, frame in enumerate(frame_plan.frames, start=1):
            if frame.label_path is None:
                raise XcatLabelConversionError(
                    f"frame {frame.index} has no planned label path"
                )
            report = convert_xcat_labels_to_mat(
                open_xcat_binary(config, frame.binary_path),
                frame.label_path,
                chunk_slices=chunk_slices,
                overwrite=overwrite,
            )
            reports.append(report)
            if not config.outputs.retain_xcat_binary_files:
                frame.binary_path.unlink()
            if completed == 1 or completed % 10 == 0 or completed == total:
                print(
                    f"Label frame {completed}/{total}: {report.label_path}",
                    flush=True,
                )

        print(
            f"\nXCAT label conversion: {len(reports)} frame(s)\n"
            f"Shape:        {reports[0].logical_shape}\n"
            f"First label:  {reports[0].label_path}\n"
            f"Last label:   {reports[-1].label_path}\n"
            "Raw binaries: removed after verified conversion\n"
            "Verification: PASS"
        )
        if not debug_one_frame:
            label_entry = label_cache_entry(config)
            label_paths = [report.label_path for report in reports]
            write_artifact_manifest(
                label_entry,
                status="complete",
                frame_count=total,
                completed_frame_indices=list(range(1, total + 1)),
                outputs=label_paths,
            )
            manifest = write_stage_manifest(config, "labels", label_paths)
            print(f"Label manifest: {manifest}")
        return 0
    except ConfigurationLoadError as exc:
        print(f"Configuration error:\n  {exc}", file=sys.stderr)
    except ValidationError as exc:
        print(format_validation_error(exc), file=sys.stderr)
    except (
        XcatParameterError,
        XcatFramePlanError,
        XcatBinaryReadError,
        XcatLabelConversionError,
    ) as exc:
        print(f"XCAT label conversion error:\n  {exc}", file=sys.stderr)
    return 2


def _generate_contrast(
    configuration: Path,
    *,
    debug_one_frame: bool,
    chunk_slices: int,
    overwrite: bool,
) -> int:
    try:
        config = load_config(configuration)
        if not config.outputs.save_gt_contrast:
            raise ContrastGenerationError(
                "outputs.save_gt_contrast is false; enable it to write "
                "high-resolution XCAT contrast images"
            )
        sequence = read_sequence(config.sequence)
        library = get_tissue_library(
            config.sequence.contrast.tissue_library
        )
        frame_plan = plan_xcat_frames(
            config, debug_one_frame=debug_one_frame
        )
        expected_shape = xcat_label_shape(config)
        reports = []
        for frame in frame_plan.frames:
            if frame.label_path is None:
                raise ContrastGenerationError(
                    "contrast generation currently requires "
                    "outputs.save_tissue_labels to be true"
                )
            image_path = contrast_frame_path(config, frame.index)
            reports.append(
                generate_bssfp_contrast(
                    frame.label_path,
                    image_path,
                    library,
                    expected_shape=expected_shape,
                    flip_angle_deg=sequence.flip_angle_deg,
                    te_ms=sequence.te_ms,
                    tr_ms=sequence.tr_ms,
                    off_resonance_enabled=(
                        config.scanner.effects.off_resonance.enabled
                    ),
                    chunk_slices=chunk_slices,
                    overwrite=overwrite,
                )
            )

        print(f"Contrast generation ({len(reports)} frame(s))")
        for index, report in enumerate(reports):
            if index:
                print()
            print(format_contrast_generation(report))
        return 0
    except ConfigurationLoadError as exc:
        print(f"Configuration error:\n  {exc}", file=sys.stderr)
    except ValidationError as exc:
        print(format_validation_error(exc), file=sys.stderr)
    except (XcatParameterError, XcatFramePlanError) as exc:
        print(f"Contrast frame-plan error:\n  {exc}", file=sys.stderr)
    except SequenceReadError as exc:
        print(f"Sequence error:\n  {exc}", file=sys.stderr)
    except KeyError as exc:
        print(f"Tissue-library error:\n  {exc}", file=sys.stderr)
    except ContrastGenerationError as exc:
        print(f"Contrast generation error:\n  {exc}", file=sys.stderr)
    except (ValueError, NotImplementedError) as exc:
        print(f"Signal-model error:\n  {exc}", file=sys.stderr)
    return 2


def _prepare_kspace_inputs(
    configuration: Path,
    *,
    rebuild_rss_cache: bool,
) -> int:
    try:
        config = load_config(configuration)
        if not config.coils.enabled:
            raise SensitivityMapError(
                "coils.enabled is false; no sensitivity map is configured"
            )
        if config.coils.sensitivity_map is None:
            raise SensitivityMapError(
                "coils.sensitivity_map is required when coils are enabled"
            )
        if not config.coils.normalize:
            raise SensitivityMapError(
                "coils.normalize must be true for safe RSS normalization"
            )

        sequence = read_sequence(config.sequence)
        transforms = build_coordinate_transforms(
            patient_position=config.phantom.patient_position,
            coordinate_mode=config.sequence.coordinate_mode,
            sequence_orientation=config.sequence.orientation,
        )
        info = inspect_sensitivity_map(config.coils.sensitivity_map)
        cache_path = (
            config.run.output_root
            / "kspace"
            / "cache"
            / "sensitivity_rss.npy"
        )
        last_bucket = [-1]

        def show_progress(completed: int, total: int) -> None:
            percent = int(100 * completed / total)
            bucket = percent // 5
            if bucket > last_bucket[0]:
                last_bucket[0] = bucket
                print(
                    f"Computing sensitivity RSS: {percent}% "
                    f"({completed}/{total} blocks)",
                    flush=True,
                )

        normalization = prepare_rss_normalization(
            info,
            cache_path,
            relative_epsilon=1e-6,
            rebuild=rebuild_rss_cache,
            progress=show_progress,
        )

        image_path = contrast_frame_path(config, 1)
        logical_shape = sensitivity_shape_in_logical_frame(
            info,
            stored_axis_order=config.coils.axis_order,
            dcs_to_logical=transforms.dcs_to_logical,
        )
        prepared = prepare_contrast_for_encoding(
            image_path,
            logical_shape,
            source_to_target=transforms.pcs_to_logical,
            source_frame="XCAT-PCS [Sag, Cor, Tra]",
            target_frame="sequence-logical [x, y, z]",
            target_axis_patient_directions=(
                transforms.logical_axis_patient_directions
            ),
        )
        logical_coil = load_normalized_coil_in_logical_frame(
            info,
            0,
            normalization,
            stored_axis_order=config.coils.axis_order,
            dcs_to_logical=transforms.dcs_to_logical,
        )
        preview = save_logical_input_preview(
            prepared,
            logical_coil,
            transforms,
            config.run.output_root
            / "kspace"
            / "debug"
            / "logical_inputs_frame_0001_coil_00.mat",
            coil_index=0,
        )
        print(format_sensitivity_preparation(info, normalization))
        print("\n" + format_sequence_summary(sequence))
        print("\n" + format_prepared_contrast(prepared))
        print("\n" + format_logical_input_preview(preview))
        print("\nK-space inputs through Step 5: PASS")
        return 0
    except ConfigurationLoadError as exc:
        print(f"Configuration error:\n  {exc}", file=sys.stderr)
    except ValidationError as exc:
        print(format_validation_error(exc), file=sys.stderr)
    except SensitivityMapError as exc:
        print(f"Sensitivity-map error:\n  {exc}", file=sys.stderr)
    except EncodingInputError as exc:
        print(f"Encoding-input error:\n  {exc}", file=sys.stderr)
    except OrientationTransformError as exc:
        print(f"Orientation error:\n  {exc}", file=sys.stderr)
    except ValueError as exc:
        print(f"K-space input error:\n  {exc}", file=sys.stderr)
    return 2


def _generate_spatially_varying_fa_contrast(
    configuration: Path,
    *,
    debug_one_frame: bool,
    chunk_slices: int,
    overwrite: bool,
) -> int:
    try:
        config = load_config(configuration)
        if config.scanner.effects.off_resonance.enabled:
            raise NotImplementedError(
                "off-resonance bSSFP signal simulation is not implemented"
            )
        if not config.coils.enabled or config.coils.sensitivity_map is None:
            raise SliceProfileError(
                "an enabled sensitivity map is required to define the full "
                "logical matrix"
            )
        sequence = read_sequence(config.sequence)
        transforms = build_coordinate_transforms(
            patient_position=config.phantom.patient_position,
            coordinate_mode=config.sequence.coordinate_mode,
            sequence_orientation=config.sequence.orientation,
        )
        info = inspect_sensitivity_map(config.coils.sensitivity_map)
        logical_shape = sensitivity_shape_in_logical_frame(
            info,
            stored_axis_order=config.coils.axis_order,
            dcs_to_logical=transforms.dcs_to_logical,
        )
        excitation = read_pulseq_excitation(sequence.sequence_path)
        logical_axis = excitation.logical_axis
        if not np.isclose(
            excitation.nominal_flip_angle_deg,
            sequence.flip_angle_deg,
            atol=1e-3,
            rtol=0.0,
        ):
            raise SliceProfileError(
                "integrated Pulseq RF flip angle does not match sequence "
                f"metadata: {excitation.nominal_flip_angle_deg:g} vs "
                f"{sequence.flip_angle_deg:g} deg"
            )
        pcs_voxel = np.asarray(
            config.phantom.voxel_size_mm, dtype=np.float64
        )
        logical_voxel = np.abs(transforms.pcs_to_logical) @ pcs_voxel
        profile = generate_slice_profile(
            excitation,
            matrix_size=logical_shape[logical_axis],
            voxel_size_mm=float(logical_voxel[logical_axis]),
            center_shift_mm=config.sequence.rf_profile.center_shift_mm,
        )
        contrast_entry = contrast_cache_entry(config)
        frame_plan = plan_xcat_frames(
            config, debug_one_frame=debug_one_frame
        )
        profile_path = contrast_profile_path(config)
        library = get_tissue_library(
            config.sequence.contrast.tissue_library
        )
        reports = []
        total = len(frame_plan.frames)
        for completed, frame in enumerate(frame_plan.frames, start=1):
            if frame.label_path is None:
                raise RfProfileContrastError(
                    f"frame {frame.index} has no planned label path"
                )
            report = generate_rf_profile_bssfp_contrast(
                label_path=frame.label_path,
                profile=profile,
                transforms=transforms,
                pcs_voxel_size_mm=config.phantom.voxel_size_mm,
                library=library,
                te_ms=sequence.te_ms,
                tr_ms=sequence.tr_ms,
                profile_output_path=profile_path,
                image_output_path=contrast_frame_path(config, frame.index),
                chunk_slices=chunk_slices,
                overwrite=overwrite,
                write_profile=completed == 1,
            )
            reports.append(report)
            if completed == 1 or completed % 10 == 0 or completed == total:
                print(
                    f"Contrast frame {completed}/{total}: "
                    f"{report.image_path}",
                    flush=True,
                )
        print(
            f"\nSpatially varying-FA contrast generation: "
            f"{len(reports)} frame(s)\n"
            f"Shape:          {reports[0].image_shape}\n"
            f"RF shift:       {reports[0].center_shift_mm:g} mm\n"
            f"Profile:        {reports[0].profile_path}\n"
            f"First contrast: {reports[0].image_path}\n"
            f"Last contrast:  {reports[-1].image_path}\n"
            "Verification:   PASS"
        )
        if not debug_one_frame:
            write_artifact_manifest(
                contrast_entry,
                status="complete",
                frame_count=total,
                completed_frame_indices=list(range(1, total + 1)),
                outputs=[profile_path, *[report.image_path for report in reports]],
            )
            manifest = write_stage_manifest(
                config,
                "contrast",
                [profile_path, *[report.image_path for report in reports]],
            )
            print(f"Contrast manifest: {manifest}")
        return 0
    except ConfigurationLoadError as exc:
        print(f"Configuration error:\n  {exc}", file=sys.stderr)
    except ValidationError as exc:
        print(format_validation_error(exc), file=sys.stderr)
    except SequenceReadError as exc:
        print(f"Sequence error:\n  {exc}", file=sys.stderr)
    except SensitivityMapError as exc:
        print(f"Sensitivity-map error:\n  {exc}", file=sys.stderr)
    except OrientationTransformError as exc:
        print(f"Orientation error:\n  {exc}", file=sys.stderr)
    except (SliceProfileError, RfProfileContrastError) as exc:
        print(f"RF-profile error:\n  {exc}", file=sys.stderr)
    except KeyError as exc:
        print(f"Tissue-library error:\n  {exc}", file=sys.stderr)
    except (ValueError, NotImplementedError) as exc:
        print(f"Signal-model error:\n  {exc}", file=sys.stderr)
    return 2


def _mat_variable_matches(
    path: Path,
    variable_name: str,
    expected_shape: tuple[int, ...],
    expected_dtype: str = "single",
) -> bool:
    """Check a saved frame without loading its complete voxel array."""

    if not path.is_file():
        return False
    try:
        entries = {
            name: (shape, data_type)
            for name, shape, data_type in whosmat(path)
        }
    except (OSError, ValueError):
        return False
    return entries.get(variable_name) == (expected_shape, expected_dtype)


def _prepare_resumed_xcat_invocation(
    config,
    parameters: XcatParameterFile,
    frames: XcatFramePlan,
    *,
    first_missing_zero_based: int,
) -> tuple[XcatParameterFile, XcatFramePlan]:
    """Start XCAT at the first missing phase and map local outputs globally."""

    remaining = len(frames.frames) - first_missing_zero_based
    if remaining <= 0:
        raise XcatFramePlanError("resume plan has no missing frames")
    start_time_s = frames.frames[first_missing_zero_based].time_s
    values = dict(parameters.parameters)
    cardiac_period_s = 60.0 / config.phantom.motion.cardiac.heart_rate_bpm
    values["hrt_start_ph_index"] = (
        config.phantom.motion.cardiac.start_phase
        + start_time_s / cardiac_period_s
    ) % 1.0
    respiratory = config.phantom.motion.respiratory
    xcat_start_time_s = values["hrt_start_ph_index"] * cardiac_period_s
    respiratory_period_s = float(values["resp_period"])
    if config.phantom.motion.mode == "free-breathing":
        if respiratory.breaths_per_minute is None:
            raise XcatFramePlanError(
                "free-breathing resume requires breaths_per_minute"
            )
        desired_respiratory_phase = (
            respiratory.start_phase
            + start_time_s / respiratory_period_s
        ) % 1.0
        # With respiratory motion enabled, XCAT advances both clocks by the
        # heart start-time offset. Back out that offset so the resumed first
        # frame lands on the desired global respiratory phase.
        values["resp_start_ph_index"] = (
            desired_respiratory_phase
            - xcat_start_time_s / respiratory_period_s
        ) % 1.0
    else:
        # XCAT keeps the respiratory phase fixed for beating-heart-only
        # (breath-hold) runs, independent of hrt_start_ph_index.
        values["resp_start_ph_index"] = respiratory.start_phase
    values["out_frames"] = remaining

    file_values = {
        name: value
        for name, value in values.items()
        if name not in parameters.command_line_parameters
    }
    template_text = parameters.template_path.read_text(encoding="utf-8")
    rendered, appended = render_xcat_parameter_file(
        template_text, file_values
    )
    resume_directory = (
        config.run.output_root
        / "xcat"
        / "resume"
        / f"from_{first_missing_zero_based + 1:04d}"
    )
    resume_directory.mkdir(parents=True, exist_ok=True)
    parameter_path = resume_directory / "parameters.par"
    parameter_path.write_text(rendered, encoding="utf-8")
    motion = replace(
        parameters.motion_plan,
        generated_frame_count=remaining,
        debug_one_frame=False,
    )
    resumed_parameters = replace(
        parameters,
        output_path=parameter_path,
        motion_plan=motion,
        parameters=values,
        appended_parameters=appended,
    )

    output_prefix = (
        resume_directory
        / f"phantom_{config.run.id}_from_{first_missing_zero_based + 1:04d}"
    )
    resumed_frames = tuple(
        replace(
            global_frame,
            binary_path=(
                resume_directory
                / f"{output_prefix.name}_act_{local_index}.bin"
            ),
            binary_exists=False,
        )
        for local_index, global_frame in enumerate(
            frames.frames[first_missing_zero_based:], start=1
        )
    )
    resumed_plan = replace(
        frames,
        raw_directory=resume_directory,
        output_prefix=output_prefix,
        motion=motion,
        frames=resumed_frames,
    )
    return resumed_parameters, resumed_plan


def _write_dynamic_stage_manifests(config, frames: XcatFramePlan) -> None:
    """Record the verified label and tissue-contrast products independently."""

    label_paths = []
    contrast_paths = []
    for frame in frames.frames:
        if frame.label_path is None:
            raise XcatFramePlanError(
                f"frame {frame.index} has no label path for manifest"
            )
        label_paths.append(frame.label_path)
        contrast_paths.append(contrast_frame_path(config, frame.index))
    label_entry = label_cache_entry(config)
    contrast_entry = contrast_cache_entry(config)
    indices = [frame.index for frame in frames.frames]
    write_artifact_manifest(
        label_entry,
        status="complete",
        frame_count=len(frames.frames),
        completed_frame_indices=indices,
        outputs=label_paths,
    )
    write_artifact_manifest(
        contrast_entry,
        status="complete",
        frame_count=len(frames.frames),
        completed_frame_indices=indices,
        outputs=[contrast_profile_path(config), *contrast_paths],
    )
    write_stage_manifest(config, "labels", label_paths)
    write_stage_manifest(
        config,
        "contrast",
        [contrast_profile_path(config), *contrast_paths],
    )


def _inspect_reuse(configuration: Path) -> int:
    """Print stage-level cache validity without loading simulation arrays."""

    try:
        config = load_config(configuration)
        print("Simulation stage reuse")
        for stage in ("labels", "contrast", "fullysampled_reference"):
            status = stage_reuse_status(config, stage)
            state = "REUSE" if status.reusable else "REGENERATE"
            print(f"{stage:24s} {state:10s} {status.reason}")
            print(f"  manifest: {status.manifest_path}")
        return 0
    except ConfigurationLoadError as exc:
        print(f"Configuration error:\n  {exc}", file=sys.stderr)
    except ValidationError as exc:
        print(format_validation_error(exc), file=sys.stderr)
    return 2


def _inspect_artifact_cache(configuration: Path) -> int:
    """Print stable IDs and cache states without generating artifacts."""

    try:
        config = load_config(configuration)
        entries = (
            label_cache_entry(config),
            contrast_cache_entry(config),
            fullysampled_reference_cache_entry(config),
        )
        print("Content-addressed artifact cache")
        for entry in entries:
            status = artifact_cache_status(entry)
            print(f"{entry.kind:16s} {status.state:7s} {entry.cache_id}")
            print(f"  path:   {entry.directory}")
            print(f"  reason: {status.reason}")
        return 0
    except ConfigurationLoadError as exc:
        print(f"Configuration error:\n  {exc}", file=sys.stderr)
    except ValidationError as exc:
        print(format_validation_error(exc), file=sys.stderr)
    except OSError as exc:
        print(f"Cache inspection error:\n  {exc}", file=sys.stderr)
    return 2


def _adopt_legacy_cache(
    configuration: Path,
    *,
    labels_only: bool,
    overwrite: bool,
) -> int:
    """Adopt the current run-scoped outputs without rerunning simulation."""

    try:
        config = load_config(configuration)
        report = adopt_legacy_cache(
            config,
            include_contrast=not labels_only,
            overwrite=overwrite,
            progress=lambda message: print(message, flush=True),
        )
        print("\n" + format_cache_migration(report))
        return 0
    except ConfigurationLoadError as exc:
        print(f"Configuration error:\n  {exc}", file=sys.stderr)
    except ValidationError as exc:
        print(format_validation_error(exc), file=sys.stderr)
    except (CacheMigrationError, OSError, ValueError) as exc:
        print(f"Cache adoption error:\n  {exc}", file=sys.stderr)
    return 2


def _generate_dynamic_cycle(
    configuration: Path,
    *,
    chunk_slices: int,
    regenerate_from_frame: int | None = None,
) -> int:
    """Generate, consume, and clean one complete XCAT motion cycle."""

    try:
        config = load_config(configuration)
        if not config.outputs.save_tissue_labels:
            raise XcatLabelConversionError(
                "streaming generation requires save_tissue_labels: true"
            )
        if not config.outputs.save_gt_contrast:
            raise RfProfileContrastError(
                "streaming generation requires save_gt_contrast: true"
            )
        if config.scanner.effects.off_resonance.enabled:
            raise NotImplementedError(
                "off-resonance bSSFP signal simulation is not implemented"
            )
        if not config.coils.enabled or config.coils.sensitivity_map is None:
            raise SliceProfileError(
                "an enabled sensitivity map is required to define the "
                "RF-profile grid"
            )

        parameters = prepare_xcat_parameter_file(
            config, debug_one_frame=False
        )
        frames = plan_xcat_frames(config, debug_one_frame=False)
        total = len(frames.frames)
        if regenerate_from_frame is not None and not (
            1 <= regenerate_from_frame <= total
        ):
            raise XcatFramePlanError(
                "--regenerate-from-frame must be between 1 and "
                f"{total}; got {regenerate_from_frame}"
            )

        sequence = read_sequence(config.sequence)
        transforms = build_coordinate_transforms(
            patient_position=config.phantom.patient_position,
            coordinate_mode=config.sequence.coordinate_mode,
            sequence_orientation=config.sequence.orientation,
        )
        info = inspect_sensitivity_map(config.coils.sensitivity_map)
        logical_shape = sensitivity_shape_in_logical_frame(
            info,
            stored_axis_order=config.coils.axis_order,
            dcs_to_logical=transforms.dcs_to_logical,
        )
        excitation = read_pulseq_excitation(sequence.sequence_path)
        logical_axis = excitation.logical_axis
        pcs_voxel = np.asarray(
            config.phantom.voxel_size_mm, dtype=np.float64
        )
        logical_voxel = np.abs(transforms.pcs_to_logical) @ pcs_voxel
        profile = generate_slice_profile(
            excitation,
            matrix_size=logical_shape[logical_axis],
            voxel_size_mm=float(logical_voxel[logical_axis]),
            center_shift_mm=config.sequence.rf_profile.center_shift_mm,
        )
        expected_shape = xcat_label_shape(config)
        contrast_entry = contrast_cache_entry(config)
        profile_path = contrast_profile_path(config)
        library = get_tissue_library(
            config.sequence.contrast.tissue_library
        )
        profile_written = _mat_variable_matches(
            profile_path,
            "effective_flip_angle_deg",
            (1, logical_shape[logical_axis]),
        )
        reused_labels = 0
        reused_contrasts = 0

        def ensure_contrast(frame, *, label_existed: bool) -> bool:
            nonlocal profile_written, reused_contrasts
            if frame.label_path is None:
                raise XcatLabelConversionError(
                    f"frame {frame.index} has no label destination"
                )
            contrast_path = contrast_frame_path(config, frame.index)
            contrast_valid = _mat_variable_matches(
                contrast_path, "image", expected_shape
            )
            if label_existed and contrast_valid:
                reused_contrasts += 1
                return True
            generate_rf_profile_bssfp_contrast(
                label_path=frame.label_path,
                profile=profile,
                transforms=transforms,
                pcs_voxel_size_mm=config.phantom.voxel_size_mm,
                library=library,
                te_ms=sequence.te_ms,
                tr_ms=sequence.tr_ms,
                profile_output_path=profile_path,
                image_output_path=contrast_path,
                chunk_slices=chunk_slices,
                overwrite=contrast_path.exists(),
                write_profile=not profile_written,
            )
            profile_written = True
            if not _mat_variable_matches(
                contrast_path, "image", expected_shape
            ):
                raise RfProfileContrastError(
                    f"contrast verification failed: {contrast_path}"
                )
            return False

        missing_label_indices = []
        for zero_based, frame in enumerate(frames.frames):
            if frame.label_path is None:
                raise XcatLabelConversionError(
                    f"frame {frame.index} has no label destination"
                )
            label_valid = _mat_variable_matches(
                frame.label_path, "P", expected_shape, "uint16"
            )
            force_regeneration = (
                regenerate_from_frame is not None
                and frame.index >= regenerate_from_frame
            )
            if (
                frame.label_path.exists()
                and not label_valid
                and not force_regeneration
            ):
                raise XcatLabelConversionError(
                    f"existing label failed validation: {frame.label_path}"
                )
            if force_regeneration:
                missing_label_indices.append(zero_based)
            elif label_valid:
                ensure_contrast(frame, label_existed=True)
            else:
                missing_label_indices.append(zero_based)

        if not missing_label_indices:
            print(
                "Dynamic cycle generation\n"
                f"Frames:             {total}/{total}\n"
                "Missing labels:     0\n"
                "Raw files retained: 0\n"
                "Verification:       PASS"
            )
            _write_dynamic_stage_manifests(config, frames)
            if config.outputs.save_tissue_labels_nrrd:
                return _export_labels_nrrd(
                    configuration,
                    output=None,
                    overwrite=False,
                )
            return 0
        first_missing = missing_label_indices[0]
        parameters, producer_frames = _prepare_resumed_xcat_invocation(
            config,
            parameters,
            frames,
            first_missing_zero_based=first_missing,
        )
        preflight = preflight_xcat_invocation(
            config,
            parameters,
            producer_frames,
            allow_partial_outputs=True,
        )
        if not preflight.passed:
            print(format_xcat_preflight(preflight, dry_run=False))
            return 2
        print(
            f"Resuming at global frame {first_missing + 1}/{total}; "
            f"XCAT will generate {len(producer_frames.frames)} missing-tail "
            "phase(s).",
            flush=True,
        )

        def consume(frame) -> None:
            nonlocal profile_written, reused_labels, reused_contrasts
            if frame.label_path is None:
                raise XcatLabelConversionError(
                    f"frame {frame.index} has no label destination"
                )
            label_existed = _mat_variable_matches(
                frame.label_path, "P", expected_shape, "uint16"
            )
            force_regeneration = (
                regenerate_from_frame is not None
                and frame.index >= regenerate_from_frame
            )
            if (
                frame.label_path.exists()
                and not label_existed
                and not force_regeneration
            ):
                raise XcatLabelConversionError(
                    f"existing label failed validation: {frame.label_path}"
                )
            label_reused = label_existed and not force_regeneration
            if label_reused:
                reused_labels += 1
            else:
                convert_xcat_labels_to_mat(
                    open_xcat_binary(config, frame.binary_path),
                    frame.label_path,
                    chunk_slices=chunk_slices,
                    overwrite=force_regeneration,
                )
            if not _mat_variable_matches(
                frame.label_path, "P", expected_shape, "uint16"
            ):
                raise XcatLabelConversionError(
                    f"label verification failed: {frame.label_path}"
                )

            # The raw is redundant only after the label has passed reopening.
            frame.binary_path.unlink()

            contrast_reused = ensure_contrast(
                frame, label_existed=label_reused
            )
            print(
                f"Frame {frame.index}/{total}: label verified, raw removed, "
                f"contrast {'reused' if contrast_reused else 'generated'}",
                flush=True,
            )

        result = execute_streaming_xcat_invocation(
            config,
            producer_frames,
            preflight,
            consume,
            force_generate=regenerate_from_frame is not None,
        )
        print(
            "\nDynamic cycle generation\n"
            f"Frames:             {total}/{total}\n"
            f"Tail phases run:    {result.consumed_frame_count}\n"
            f"Labels reused:      {reused_labels}\n"
            f"Contrasts reused:   {reused_contrasts}\n"
            "Raw files retained: 0\n"
            f"stdout:             {result.stdout_log}\n"
            f"stderr:             {result.stderr_log}\n"
            "Verification:       PASS"
        )
        _write_dynamic_stage_manifests(config, frames)
        if config.outputs.save_tissue_labels_nrrd:
            return _export_labels_nrrd(
                configuration,
                output=None,
                overwrite=regenerate_from_frame is not None,
            )
        return 0
    except ConfigurationLoadError as exc:
        print(f"Configuration error:\n  {exc}", file=sys.stderr)
    except ValidationError as exc:
        print(format_validation_error(exc), file=sys.stderr)
    except (
        XcatParameterError,
        XcatFramePlanError,
        XcatExecutionError,
        XcatBinaryReadError,
        XcatLabelConversionError,
        SequenceReadError,
        SensitivityMapError,
        SliceProfileError,
        RfProfileContrastError,
        OrientationTransformError,
        ValueError,
        NotImplementedError,
    ) as exc:
        print(f"Dynamic-cycle error:\n  {exc}", file=sys.stderr)
    return 2


def _generate_balloon_debug(
    configuration: Path,
    *,
    overwrite: bool,
) -> int:
    try:
        config = load_config(configuration)
        report = generate_balloon_debug_frames(config, overwrite=overwrite)
        print(format_balloon_debug(report))
        return 0
    except ConfigurationLoadError as exc:
        print(f"Configuration error:\n  {exc}", file=sys.stderr)
    except ValidationError as exc:
        print(format_validation_error(exc), file=sys.stderr)
    except (
        BalloonDebugError,
        BalloonPathError,
        SparseBalloonError,
        GdSignalError,
        SequenceReadError,
        KeyError,
    ) as exc:
        print(f"Balloon debug error:\n  {exc}", file=sys.stderr)
    return 2


def _generate_balloon_path_debug(
    configuration: Path,
    *,
    center_spacing_mm: float,
    overwrite: bool,
) -> int:
    try:
        config = load_config(configuration)
        report = generate_balloon_path_debug(
            config,
            center_spacing_mm=center_spacing_mm,
            overwrite=overwrite,
        )
        print(format_balloon_path_debug(report))
        return 0
    except ConfigurationLoadError as exc:
        print(f"Configuration error:\n  {exc}", file=sys.stderr)
    except ValidationError as exc:
        print(format_validation_error(exc), file=sys.stderr)
    except (
        BalloonDebugError,
        BalloonPathError,
        SparseBalloonError,
        GdSignalError,
        SequenceReadError,
        KeyError,
    ) as exc:
        print(f"Balloon path debug error:\n  {exc}", file=sys.stderr)
    return 2


def _generate_kspace_debug(
    configuration: Path,
    *,
    coil: int,
    device_id: int | None,
) -> int:
    try:
        config = load_config(configuration)
        reference = validate_sigpy_reference()
        print(format_sigpy_reference_validation(reference))
        if not reference.passed:
            return 1

        if not config.coils.enabled or config.coils.sensitivity_map is None:
            raise SensitivityMapError(
                "an enabled sensitivity map is required for real-data validation"
            )
        if not config.coils.normalize:
            raise SensitivityMapError(
                "coils.normalize must be true for real-data validation"
            )
        info = inspect_sensitivity_map(config.coils.sensitivity_map)
        if not 0 <= coil < info.coil_count:
            raise SensitivityMapError(
                f"coil must be between 0 and {info.coil_count - 1}"
            )
        cache_path = (
            config.run.output_root
            / "kspace"
            / "cache"
            / "sensitivity_rss.npy"
        )
        normalization = prepare_rss_normalization(info, cache_path)
        transforms = build_coordinate_transforms(
            patient_position=config.phantom.patient_position,
            coordinate_mode=config.sequence.coordinate_mode,
            sequence_orientation=config.sequence.orientation,
        )
        logical_shape = sensitivity_shape_in_logical_frame(
            info,
            stored_axis_order=config.coils.axis_order,
            dcs_to_logical=transforms.dcs_to_logical,
        )
        normalized_coil = load_normalized_coil_in_logical_frame(
            info,
            coil,
            normalization,
            stored_axis_order=config.coils.axis_order,
            dcs_to_logical=transforms.dcs_to_logical,
        )

        image_path = contrast_frame_path(config, 1)
        prepared = prepare_contrast_for_encoding(
            image_path,
            logical_shape,
            source_to_target=transforms.pcs_to_logical,
            source_frame="XCAT PCS [Sag, Cor, Tra]",
            target_frame="Pulseq logical [x, y, z]",
            target_axis_patient_directions=(
                transforms.logical_axis_patient_directions
            ),
        )
        sequence = read_sequence(config.sequence)
        excitation = read_pulseq_excitation(sequence.sequence_path)
        pcs_voxel = np.asarray(
            config.phantom.voxel_size_mm, dtype=np.float64
        )
        logical_voxel = np.abs(transforms.pcs_to_logical) @ pcs_voxel
        resolution_values = np.asarray(
            sequence.resolution_mm, dtype=np.float64
        ).reshape(-1)
        if resolution_values.size != 1:
            raise TrajectoryPreparationError(
                "current trajectory scaling requires one isotropic resolution"
            )
        scaled_k, _, _ = scale_isotropic_trajectory_to_resolution(
            sequence.kx,
            sequence.ky,
            sequence.kz,
            resolution_mm=float(resolution_values[0]),
        )
        encoding_grids = prepare_encoding_grids(
            ground_truth_shape=prepared.image.shape,
            ground_truth_voxel_size_mm=config.phantom.voxel_size_mm,
            sequence_resolution_mm=sequence.resolution_mm,
        )
        selected_device = (
            config.compute.device_id if device_id is None else device_id
        )
        if selected_device < -1:
            raise ValueError("device ID must be -1 or a non-negative GPU ID")
        device_tag = (
            "cpu_reference"
            if selected_device == -1
            else f"gpu_{selected_device:02d}_reference"
        )
        output_path = (
            config.run.output_root
            / "kspace"
            / "debug"
            / (
                f"sigpy_3d_frame_0001_coil_{coil:02d}_{device_tag}_"
                "forward_adjoint.mat"
            )
        )
        shifted_gt_output_path = (
            config.run.output_root
            / "kspace"
            / "debug"
            / "shifted_high_resolution_gt_frame_0001.mat"
        )
        report = run_reduced_nufft_validation(
            prepared.image,
            normalized_coil,
            kx_per_m=scaled_k[0],
            ky_per_m=scaled_k[1],
            kz_per_m=scaled_k[2],
            density_compensation=sequence.density_compensation,
            encoding_grids=encoding_grids,
            coil_index=coil,
            arm_count=sequence.arm_count,
            output_path=output_path,
            save_full_adjoint=True,
            axis_patient_directions=(
                transforms.logical_axis_patient_directions
            ),
            pcs_to_logical=transforms.pcs_to_logical,
            rf_center_shift_mm=config.sequence.rf_profile.center_shift_mm,
            rf_axis_voxel_size_mm=float(
                logical_voxel[excitation.logical_axis]
            ),
            rf_logical_axis=excitation.logical_axis,
            shifted_ground_truth_output_path=shifted_gt_output_path,
            device_id=selected_device,
        )
        print("\n" + format_reduced_nufft_validation(report))
        return 0
    except ConfigurationLoadError as exc:
        print(f"Configuration error:\n  {exc}", file=sys.stderr)
    except ValidationError as exc:
        print(format_validation_error(exc), file=sys.stderr)
    except SequenceReadError as exc:
        print(f"Sequence error:\n  {exc}", file=sys.stderr)
    except SensitivityMapError as exc:
        print(f"Sensitivity-map error:\n  {exc}", file=sys.stderr)
    except EncodingInputError as exc:
        print(f"Encoding-input error:\n  {exc}", file=sys.stderr)
    except (TrajectoryPreparationError, NufftBackendError, ValueError) as exc:
        print(f"NUFFT validation error:\n  {exc}", file=sys.stderr)
    return 2


def _generate_kspace_all_coils_debug(
    configuration: Path,
    *,
    gd_balloon: bool = False,
) -> int:
    """Run one full-trajectory forward/adjoint for every normalized coil."""

    try:
        config = load_config(configuration)
        if not config.coils.enabled or config.coils.sensitivity_map is None:
            raise SensitivityMapError(
                "an enabled sensitivity map is required for multicoil encoding"
            )
        if not config.coils.normalize:
            raise SensitivityMapError(
                "coils.normalize must be true for multicoil encoding"
            )
        info = inspect_sensitivity_map(config.coils.sensitivity_map)
        cache_path = (
            config.run.output_root
            / "kspace"
            / "cache"
            / "sensitivity_rss.npy"
        )
        normalization = prepare_rss_normalization(info, cache_path)
        transforms = build_coordinate_transforms(
            patient_position=config.phantom.patient_position,
            coordinate_mode=config.sequence.coordinate_mode,
            sequence_orientation=config.sequence.orientation,
        )
        logical_shape = sensitivity_shape_in_logical_frame(
            info,
            stored_axis_order=config.coils.axis_order,
            dcs_to_logical=transforms.dcs_to_logical,
        )
        if gd_balloon:
            image_path = (
                config.run.output_root
                / "intervention"
                / "balloon_debug"
                / "balloon_debug_01_A.mat"
            )
            if not image_path.is_file():
                raise BalloonDebugError(
                    f"required frame-1 balloon image does not exist: "
                    f"{image_path}; run generate-balloon-debug first"
                )
        else:
            image_path = contrast_frame_path(config, 1)
        prepared = prepare_contrast_for_encoding(
            image_path,
            logical_shape,
            source_to_target=transforms.pcs_to_logical,
            source_frame="XCAT PCS [Sag, Cor, Tra]",
            target_frame="Pulseq logical [x, y, z]",
            target_axis_patient_directions=(
                transforms.logical_axis_patient_directions
            ),
        )
        sequence = read_sequence(config.sequence)
        excitation = read_pulseq_excitation(sequence.sequence_path)
        pcs_voxel = np.asarray(
            config.phantom.voxel_size_mm, dtype=np.float64
        )
        logical_voxel = np.abs(transforms.pcs_to_logical) @ pcs_voxel
        resolution_values = np.asarray(
            sequence.resolution_mm, dtype=np.float64
        ).reshape(-1)
        if resolution_values.size != 1:
            raise TrajectoryPreparationError(
                "current trajectory scaling requires one isotropic resolution"
            )
        resolution_mm = float(resolution_values[0])
        scaled_k, scale_factor, target_kmax = (
            scale_isotropic_trajectory_to_resolution(
                sequence.kx,
                sequence.ky,
                sequence.kz,
                resolution_mm=resolution_mm,
            )
        )
        encoding_grids = prepare_encoding_grids(
            ground_truth_shape=prepared.image.shape,
            ground_truth_voxel_size_mm=config.phantom.voxel_size_mm,
            sequence_resolution_mm=sequence.resolution_mm,
        )
        debug_directory = config.run.output_root / "kspace" / "debug"
        device_tag = (
            "cpu_reference"
            if config.compute.device_id == -1
            else f"gpu_{config.compute.device_id:02d}_reference"
        )
        signal_tag = "gd_balloon_A_" if gd_balloon else ""
        output_path = debug_directory / (
            f"sigpy_3d_frame_0001_{signal_tag}all_"
            f"{info.coil_count:02d}_coils_{device_tag}_forward_adjoint.mat"
        )
        shifted_gt_path = debug_directory / (
            f"shifted_high_resolution_gt_frame_0001_{signal_tag.rstrip('_')}.mat"
            if gd_balloon
            else "shifted_high_resolution_gt_frame_0001.mat"
        )

        def load_coil(coil_index: int) -> np.ndarray:
            return load_normalized_coil_in_logical_frame(
                info,
                coil_index,
                normalization,
                stored_axis_order=config.coils.axis_order,
                dcs_to_logical=transforms.dcs_to_logical,
            )

        def show_progress(completed: int, total: int) -> None:
            print(
                f"Completed coil {completed}/{total}", flush=True
            )

        report = run_multicoil_nufft_debug(
            prepared.image,
            coil_count=info.coil_count,
            coil_loader=load_coil,
            kx_per_m=scaled_k[0],
            ky_per_m=scaled_k[1],
            kz_per_m=scaled_k[2],
            density_compensation=sequence.density_compensation,
            encoding_grids=encoding_grids,
            output_path=output_path,
            shifted_ground_truth_output_path=shifted_gt_path,
            rf_center_shift_mm=config.sequence.rf_profile.center_shift_mm,
            rf_axis_voxel_size_mm=float(
                logical_voxel[excitation.logical_axis]
            ),
            rf_logical_axis=excitation.logical_axis,
            trajectory_scale_factor=scale_factor,
            target_kmax_per_m=target_kmax,
            device_id=config.compute.device_id,
            progress=show_progress,
        )
        print("\n" + format_multicoil_nufft_debug(report))
        if gd_balloon:
            print(f"Balloon input:         {image_path}")
        else:
            manifest = write_stage_manifest(
                config,
                "fullysampled_kspace",
                [report.output_path, report.shifted_ground_truth_path],
            )
            print(f"K-space manifest:      {manifest}")
        return 0
    except ConfigurationLoadError as exc:
        print(f"Configuration error:\n  {exc}", file=sys.stderr)
    except ValidationError as exc:
        print(format_validation_error(exc), file=sys.stderr)
    except SequenceReadError as exc:
        print(f"Sequence error:\n  {exc}", file=sys.stderr)
    except SensitivityMapError as exc:
        print(f"Sensitivity-map error:\n  {exc}", file=sys.stderr)
    except BalloonDebugError as exc:
        print(f"Balloon debug error:\n  {exc}", file=sys.stderr)
    except EncodingInputError as exc:
        print(f"Encoding-input error:\n  {exc}", file=sys.stderr)
    except (TrajectoryPreparationError, NufftBackendError, ValueError) as exc:
        print(f"Multicoil NUFFT error:\n  {exc}", file=sys.stderr)
    return 2


def _validate_balloon_kspace_linearity(configuration: Path) -> int:
    """Validate all-coil tissue plus sparse Gd encoding for frame one."""

    try:
        config = load_config(configuration)

        def show_progress(signal: str, completed: int, total: int) -> None:
            print(f"{signal}: completed coil {completed}/{total}", flush=True)

        report = validate_balloon_kspace_linearity(
            config,
            progress=show_progress,
        )
        print("\n" + format_balloon_encoding_debug(report))
        passed = (
            report.nonfinite_value_count == 0
            and report.forward_relative_error < 1e-5
            and report.adjoint_relative_error < 1e-5
        )
        return 0 if passed else 2
    except ConfigurationLoadError as exc:
        print(f"Configuration error:\n  {exc}", file=sys.stderr)
    except ValidationError as exc:
        print(format_validation_error(exc), file=sys.stderr)
    except (
        BalloonEncodingDebugError,
        BalloonDebugError,
        BalloonPathError,
        GdSignalError,
        SparseBalloonError,
        SequenceReadError,
        SensitivityMapError,
        EncodingInputError,
        TrajectoryPreparationError,
        NufftBackendError,
        RfProfileContrastError,
        SliceProfileError,
        ValueError,
    ) as exc:
        print(f"Balloon NUFFT validation error:\n  {exc}", file=sys.stderr)
    return 2


def _generate_fullysampled_reference(
    configuration: Path,
    *,
    start_frame: int,
    end_frame: int | None,
    overwrite: bool,
) -> int:
    """Generate or resume the fully sampled tissue image reference."""

    try:
        config = load_config(configuration)
        report = generate_fullysampled_reference(
            config,
            start_frame=start_frame,
            end_frame=end_frame,
            overwrite=overwrite,
            progress=lambda message: print(message, flush=True),
        )
        print("\n" + format_fullysampled_reference(report))
        return 0
    except ConfigurationLoadError as exc:
        print(f"Configuration error:\n  {exc}", file=sys.stderr)
    except ValidationError as exc:
        print(format_validation_error(exc), file=sys.stderr)
    except (
        FullysampledReferenceError,
        SequenceReadError,
        SensitivityMapError,
        EncodingInputError,
        TrajectoryPreparationError,
        NufftBackendError,
        NotImplementedError,
        OSError,
        ValueError,
    ) as exc:
        print(f"Fully sampled reference error:\n  {exc}", file=sys.stderr)
    return 2


def _plan_acquisition(configuration: Path) -> int:
    """Resolve the generic TR schedule without allocating simulation arrays."""

    try:
        config = load_config(configuration)
        sequence = read_sequence(config.sequence)
        phases = len(plan_xcat_frames(config, debug_one_frame=False).frames)
        schedule = build_acquisition_schedule(
            config,
            actual_tr_s=sequence.tr_ms * 1e-3,
            trajectory_tr_count=sequence.arm_count,
            cardiac_phase_count=phases,
        )
        estimate = estimate_dynamic_acquisition_storage(
            sequence.sample_count,
            schedule.acquisition_count,
            inspect_sensitivity_map(config.coils.sensitivity_map).coil_count,
        )
        print(
            "Acquisition schedule\n"
            f"Pulseq TR:          {schedule.actual_tr_s * 1e3:.6g} ms\n"
            f"Effective TR:       {schedule.effective_tr_s * 1e3:.6g} ms\n"
            f"TR mismatch:        {schedule.tr_mismatch_percent:.3f}%\n"
            f"TRs per frame:      {schedule.trs_per_frame}\n"
            f"Frame duration:     {schedule.frame_duration_s * 1e3:.6g} ms\n"
            f"Complete frames:    {schedule.frame_count}\n"
            f"Retained duration:  {schedule.retained_duration_s:.6g} s\n"
            f"Dropped tail:       {schedule.dropped_duration_s:.6g} s\n"
            f"Stored acquisition: {estimate.gib:.2f} GiB complex64\n"
            f"Trajectory TRs:     {schedule.trajectory_tr_count}\n"
            f"View-order cycle:   {schedule.view_order_cycle_length} TRs\n"
            f"Complete cycles:    {schedule.complete_view_order_cycles}\n"
            f"Partial cycle:      {schedule.partial_view_order_cycle_tr_count} TRs\n"
            "Plane metadata:     not required"
        )
        return 0
    except (ConfigurationLoadError, ValidationError, AcquisitionScheduleError, SequenceReadError, SensitivityMapError, OSError, ValueError) as exc:
        print(f"Acquisition planning error:\n  {exc}", file=sys.stderr)
        return 2


def _generate_tissue_kspace_library(
    configuration: Path,
    *,
    start_frame: int,
    end_frame: int | None,
    overwrite: bool,
    dry_run: bool,
) -> int:
    """Generate or preflight the resumable full-trajectory tissue library."""

    try:
        config = load_config(configuration)
        report = generate_tissue_kspace_library(
            config,
            start_frame=start_frame,
            end_frame=end_frame,
            overwrite=overwrite,
            dry_run=dry_run,
            progress=lambda message: print(message, flush=True),
        )
        print("\n" + format_tissue_kspace_library(report))
        if dry_run:
            print("Generation:          skipped (--dry-run)")
        return 0
    except (ConfigurationLoadError, ValidationError, TissueKspaceLibraryError, SequenceReadError, SensitivityMapError, NufftBackendError, OSError, ValueError) as exc:
        print(f"Tissue k-space library error:\n  {exc}", file=sys.stderr)
        return 2


def _generate_tissue_adjoint_reference(
    configuration: Path,
    *,
    start_frame: int,
    end_frame: int | None,
    overwrite: bool,
    allow_missing: bool,
) -> int:
    try:
        config = load_config(configuration)
        report = generate_tissue_adjoint_reference(
            config,
            start_frame=start_frame,
            end_frame=end_frame,
            overwrite=overwrite,
            allow_missing=allow_missing,
            progress=lambda message: print(message, flush=True),
        )
        print("\n" + format_tissue_adjoint_reference(report))
        return 0
    except (
        ConfigurationLoadError,
        ValidationError,
        TissueAdjointReferenceError,
        SequenceReadError,
        SensitivityMapError,
        NufftBackendError,
        OSError,
        ValueError,
    ) as exc:
        print(f"Tissue adjoint reference error:\n  {exc}", file=sys.stderr)
        return 2


def _generate_dynamic_acquisition(
    configuration: Path,
    *,
    overwrite: bool,
    dry_run: bool,
    view_order_cycles: int | None,
    save_adjoint_debug: bool,
) -> int:
    try:
        config = load_config(configuration)
        report = generate_dynamic_acquisition(
            config,
            overwrite=overwrite,
            dry_run=dry_run,
            view_order_cycles=view_order_cycles,
            save_adjoint_debug=save_adjoint_debug,
            progress=lambda message: print(message, flush=True),
        )
        print("\n" + format_dynamic_acquisition(report))
        if dry_run:
            print("Generation:          skipped (--dry-run)")
        return 0
    except (
        ConfigurationLoadError,
        ValidationError,
        DynamicAcquisitionError,
        AcquisitionScheduleError,
        SequenceReadError,
        SensitivityMapError,
        NufftBackendError,
        OSError,
        ValueError,
    ) as exc:
        print(f"Dynamic acquisition error:\n  {exc}", file=sys.stderr)
        return 2


def _generate_dynamic_reference(configuration: Path, *, overwrite: bool) -> int:
    try:
        config = load_config(configuration)
        result = generate_dynamic_fullysampled_reference(
            config,
            overwrite=overwrite,
            progress=lambda message: print(message, flush=True),
        )
        print(
            "\nFully sampled tissue-plus-Gd reference\n"
            f"Shape:            {result.shape} complex64\n"
            f"TRs averaged:     {result.trs_averaged_per_frame}\n"
            f"Generated/reused: {result.generated_frames}/{result.reused_frames}\n"
            f"Output:           {result.output_path}"
        )
        if config.analysis.curved_line_profile.enabled:
            profile = generate_curved_line_profile(config, overwrite=overwrite)
            print("\n" + format_curved_line_profile(profile))
        return 0
    except (
        ConfigurationLoadError,
        ValidationError,
        DynamicAcquisitionError,
        AcquisitionScheduleError,
        SequenceReadError,
        SensitivityMapError,
        NufftBackendError,
        OSError,
        ValueError,
        CurvedLineProfileError,
    ) as exc:
        print(f"Dynamic fully sampled reference error:\n  {exc}", file=sys.stderr)
        return 2


def _generate_curved_line_profile(
    configuration: Path,
    *,
    input_path: Path | None,
    overwrite: bool,
) -> int:
    try:
        config = load_config(configuration)
        result = generate_curved_line_profile(
            config,
            input_path=input_path,
            overwrite=overwrite,
        )
        print(format_curved_line_profile(result))
        return 0
    except (
        ConfigurationLoadError,
        ValidationError,
        CurvedLineProfileError,
        AcquisitionScheduleError,
        SequenceReadError,
        OSError,
        ValueError,
    ) as exc:
        print(f"Curved-line profile error:\n  {exc}", file=sys.stderr)
        return 2


def _generate_three_position_reference_debug(
    configuration: Path,
    *,
    overwrite: bool,
) -> int:
    """Generate the frame-1 A/middle/B catheter alignment diagnostic."""

    try:
        config = load_config(configuration)
        report = generate_three_position_reference_debug(
            config,
            overwrite=overwrite,
            progress=lambda message: print(message, flush=True),
        )
        print("\n" + format_three_position_reference_debug(report))
        return 0
    except ConfigurationLoadError as exc:
        print(f"Configuration error:\n  {exc}", file=sys.stderr)
    except ValidationError as exc:
        print(format_validation_error(exc), file=sys.stderr)
    except (
        ThreePositionReferenceDebugError,
        BalloonPathError,
        GdSignalError,
        SparseBalloonError,
        SequenceReadError,
        SensitivityMapError,
        EncodingInputError,
        TrajectoryPreparationError,
        NufftBackendError,
        OSError,
        ValueError,
    ) as exc:
        print(f"Three-position reference debug error:\n  {exc}", file=sys.stderr)
    return 2


def _compare_nufft_devices(
    cpu_reference: Path,
    gpu_reference: Path,
    output: Path,
) -> int:
    try:
        report = compare_device_references(
            cpu_reference, gpu_reference, output
        )
        print(
            "CPU/GPU NUFFT parity\n"
            f"K-space relative L2:  {report.kspace_relative_l2_error:g}\n"
            f"K-space maximum abs:  {report.kspace_maximum_absolute_error:g}\n"
            f"Adjoint relative L2:  {report.adjoint_relative_l2_error:g}\n"
            f"Adjoint maximum abs:  {report.adjoint_maximum_absolute_error:g}\n"
            f"CPU time:             {report.cpu_elapsed_s:.3f} s\n"
            f"GPU time:             {report.gpu_elapsed_s:.3f} s\n"
            f"Speedup:              {report.speedup:.2f}x\n"
            f"Output:               {report.output_path}\n"
            f"Overall:              {'PASS' if report.passed else 'FAIL'}"
        )
        return 0 if report.passed else 1
    except (OSError, ValueError) as exc:
        print(f"Device comparison error:\n  {exc}", file=sys.stderr)
    return 2


def _validate_kspace_reference(
    shifted_gt: Path,
    multicoil_reference: Path,
    output: Path,
) -> int:
    try:
        report = validate_image_reference(
            shifted_gt, multicoil_reference, output
        )
        print(
            "Shifted-GT/all-coil adjoint validation\n"
            f"Support correlation:  {report.correlation:g}\n"
            f"Center offset voxels: {report.center_offset_voxels}\n"
            f"Center offset mm:     {report.center_offset_mm}\n"
            f"GT boundary energy:   {report.gt_boundary_energy_ratio:g}\n"
            f"Adj boundary energy:  {report.adjoint_boundary_energy_ratio:g}\n"
            "Intended orientation: "
            + ("best match\n" if report.intended_orientation_is_best else "NOT best\n")
            + f"Output:               {report.output_path}"
        )
        return 0 if report.intended_orientation_is_best else 1
    except (OSError, ValueError) as exc:
        print(f"Image reference validation error:\n  {exc}", file=sys.stderr)
    return 2


def _diagnose_kspace_fov(
    configuration: Path,
    *,
    coil: int,
    kspace_path: Path | None,
    support_threshold: float,
    support_margin_mm: float,
    device_id: int,
    output: Path | None,
    overwrite: bool,
) -> int:
    """Compare impulse PSFs and one saved tissue k-space over fixed FOVs."""

    try:
        config = load_config(configuration)
        if not config.coils.enabled or config.coils.sensitivity_map is None:
            raise SensitivityMapError(
                "an enabled sensitivity map is required to define the "
                "logical 500-mm reference grid"
            )
        info = inspect_sensitivity_map(config.coils.sensitivity_map)
        if not 0 <= coil < info.coil_count:
            raise SensitivityMapError(
                f"coil must be between 0 and {info.coil_count - 1}"
            )
        transforms = build_coordinate_transforms(
            patient_position=config.phantom.patient_position,
            coordinate_mode=config.sequence.coordinate_mode,
            sequence_orientation=config.sequence.orientation,
        )
        logical_shape = sensitivity_shape_in_logical_frame(
            info,
            stored_axis_order=config.coils.axis_order,
            dcs_to_logical=transforms.dcs_to_logical,
        )
        logical_voxel = np.abs(transforms.pcs_to_logical) @ np.asarray(
            config.phantom.voxel_size_mm, dtype=np.float64
        )
        sequence = read_sequence(config.sequence)
        resolution = np.asarray(sequence.resolution_mm, dtype=np.float64).reshape(-1)
        if resolution.size == 1:
            resolution = np.repeat(resolution, 3)
        if resolution.size != 3:
            raise TrajectoryPreparationError(
                "sequence resolution must contain one or three values"
            )

        contrast_path = contrast_frame_path(config, 1)
        prepared = prepare_contrast_for_encoding(
            contrast_path,
            logical_shape,
            source_to_target=transforms.pcs_to_logical,
            source_frame="XCAT PCS [Sag, Cor, Tra]",
            target_frame="Pulseq logical [x, y, z]",
            target_axis_patient_directions=(
                transforms.logical_axis_patient_directions
            ),
        )
        support = measure_centered_signal_support(
            prepared.image,
            voxel_size_mm=tuple(float(value) for value in logical_voxel),
            threshold_fraction=support_threshold,
            margin_mm=support_margin_mm,
            fov_rounding_mm=tuple(float(value) for value in resolution),
        )
        padded_fov = tuple(
            float(size * voxel)
            for size, voxel in zip(
                logical_shape, logical_voxel, strict=True
            )
        )
        del prepared

        source = (
            kspace_path
            if kspace_path is not None
            else (
                config.run.output_root
                / "kspace"
                / "debug"
                / (
                    f"sigpy_3d_frame_0001_coil_{coil:02d}_"
                    "forward_adjoint.mat"
                )
            )
        ).expanduser().resolve(strict=False)
        if not source.is_file():
            raise FileNotFoundError(f"saved tissue k-space does not exist: {source}")
        content = loadmat(source, variable_names=["kspace"])
        if "kspace" not in content:
            raise ValueError(f"saved debug file has no kspace variable: {source}")
        kspace = np.asarray(content["kspace"], dtype=np.complex64)

        native_fov_values = np.asarray(
            sequence.fov_mm, dtype=np.float64
        ).reshape(-1)
        if native_fov_values.size == 1:
            native_fov_values = np.repeat(native_fov_values, 3)
        if native_fov_values.size != 3:
            raise TrajectoryPreparationError(
                "sequence FOV must contain one or three values"
            )
        candidates = {
            "padded_500mm": padded_fov,
            "sequence_native": tuple(
                float(value) for value in native_fov_values
            ),
            "support_derived": support.derived_fov_mm,
        }
        destination = (
            output
            if output is not None
            else (
                config.run.output_root
                / "kspace"
                / "debug"
                / (
                    f"fov_psf_frame_0001_coil_{coil:02d}_os1p5.mat"
                )
            )
        )
        report = run_fov_psf_diagnostic(
            kspace=kspace,
            kspace_path=source,
            kx_per_m=sequence.kx,
            ky_per_m=sequence.ky,
            kz_per_m=sequence.kz,
            density_compensation=sequence.density_compensation,
            resolution_mm=tuple(float(value) for value in resolution),
            support=support,
            candidate_fovs_mm=candidates,
            output_path=destination,
            device_id=device_id,
            overwrite=overwrite,
        )
        print(format_fov_psf_diagnostic(report))
        return 0
    except ConfigurationLoadError as exc:
        print(f"Configuration error:\n  {exc}", file=sys.stderr)
    except ValidationError as exc:
        print(format_validation_error(exc), file=sys.stderr)
    except SequenceReadError as exc:
        print(f"Sequence error:\n  {exc}", file=sys.stderr)
    except SensitivityMapError as exc:
        print(f"Sensitivity-map error:\n  {exc}", file=sys.stderr)
    except OrientationTransformError as exc:
        print(f"Orientation error:\n  {exc}", file=sys.stderr)
    except (EncodingInputError, TrajectoryPreparationError) as exc:
        print(f"Encoding-input error:\n  {exc}", file=sys.stderr)
    except (NufftBackendError, FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"K-space diagnostic error:\n  {exc}", file=sys.stderr)
    return 2


def _export_contrast_nrrd(
    configuration: Path,
    *,
    output: Path | None,
    overwrite: bool,
) -> int:
    try:
        config = load_config(configuration)
        frame_plan = plan_xcat_frames(config, debug_one_frame=False)
        paths = tuple(
            contrast_frame_path(config, frame.index)
            for frame in frame_plan.frames
        )
        destination = (
            output
            if output is not None
            else (
                config.run.output_root
                / "exports"
                / (
                    f"phantom_{config.run.id}_"
                    f"{config.sequence.contrast.model}_4d.nrrd"
                )
            )
        )

        def show_progress(completed: int, total: int) -> None:
            if completed == 1 or completed % 10 == 0 or completed == total:
                print(
                    f"NRRD frame {completed}/{total}",
                    flush=True,
                )

        report = export_contrast_series_nrrd(
            paths,
            destination,
            voxel_size_mm=config.phantom.voxel_size_mm,
            time_step_s=config.timeline.xcat_time_step_s,
            overwrite=overwrite,
            progress=show_progress,
        )
        print("\n" + format_nrrd_export(report))
        return 0
    except ConfigurationLoadError as exc:
        print(f"Configuration error:\n  {exc}", file=sys.stderr)
    except ValidationError as exc:
        print(format_validation_error(exc), file=sys.stderr)
    except (XcatParameterError, XcatFramePlanError) as exc:
        print(f"NRRD frame-plan error:\n  {exc}", file=sys.stderr)
    except NrrdExportError as exc:
        print(f"NRRD export error:\n  {exc}", file=sys.stderr)
    return 2


def _export_labels_nrrd(
    configuration: Path,
    *,
    output: Path | None,
    overwrite: bool,
) -> int:
    try:
        config = load_config(configuration)
        if not config.outputs.save_tissue_labels_nrrd:
            raise NrrdExportError(
                "outputs.save_tissue_labels_nrrd is false"
            )
        frame_plan = plan_xcat_frames(config, debug_one_frame=False)
        export_time_step_s = (
            config.outputs.tissue_labels_nrrd_time_step_s
        )
        stride = round(
            export_time_step_s / config.timeline.xcat_time_step_s
        )
        selected_frames = frame_plan.frames[::stride]
        paths = tuple(
            frame.label_path
            for frame in selected_frames
            if frame.label_path is not None
        )
        if len(paths) != len(selected_frames):
            raise NrrdExportError(
                "every selected frame must have a label path"
            )
        time_step_ms = round(export_time_step_s * 1e3)
        destination = (
            output
            if output is not None
            else (
                config.run.output_root
                / "exports"
                / (
                    f"phantom_{config.run.id}_tissue_labels_"
                    f"{time_step_ms}ms_4d.nrrd"
                )
            )
        )

        def show_progress(completed: int, total: int) -> None:
            if completed == 1 or completed % 10 == 0 or completed == total:
                print(f"Label NRRD frame {completed}/{total}", flush=True)

        report = export_label_series_nrrd(
            paths,
            destination,
            voxel_size_mm=config.phantom.voxel_size_mm,
            time_step_s=export_time_step_s,
            overwrite=overwrite,
            progress=show_progress,
        )
        print("\n" + format_nrrd_export(report))
        return 0
    except ConfigurationLoadError as exc:
        print(f"Configuration error:\n  {exc}", file=sys.stderr)
    except ValidationError as exc:
        print(format_validation_error(exc), file=sys.stderr)
    except (XcatParameterError, XcatFramePlanError) as exc:
        print(f"NRRD frame-plan error:\n  {exc}", file=sys.stderr)
    except NrrdExportError as exc:
        print(f"NRRD export error:\n  {exc}", file=sys.stderr)
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "validate":
        return _validate(args.configuration)
    if args.command == "inspect-reuse":
        return _inspect_reuse(args.configuration)
    if args.command == "inspect-cache":
        return _inspect_artifact_cache(args.configuration)
    if args.command == "adopt-legacy-cache":
        return _adopt_legacy_cache(
            args.configuration,
            labels_only=args.labels_only,
            overwrite=args.overwrite,
        )
    if args.command == "inspect-sequence":
        return _inspect_sequence(args.configuration, args.matlab_reference)
    if args.command == "compare-bssfp":
        return _compare_bssfp(
            args.configuration,
            args.labels,
            args.matlab_image,
            chunk_slices=args.chunk_slices,
            atol=args.atol,
            rtol=args.rtol,
        )
    if args.command == "prepare-xcat":
        return _prepare_xcat(
            args.configuration,
            output=args.output,
            debug_one_frame=args.debug_one_frame,
        )
    if args.command == "plan-xcat":
        return _plan_xcat(
            args.configuration,
            debug_one_frame=args.debug_one_frame,
        )
    if args.command == "run-xcat":
        return _run_xcat(
            args.configuration,
            dry_run=args.dry_run,
            debug_one_frame=args.debug_one_frame,
        )
    if args.command == "generate-dynamic-cycle":
        return _generate_dynamic_cycle(
            args.configuration,
            chunk_slices=args.chunk_slices,
            regenerate_from_frame=args.regenerate_from_frame,
        )
    if args.command == "compare-xcat-labels":
        return _compare_xcat_labels(
            args.configuration,
            args.matlab_reference,
            binary=args.binary,
            chunk_slices=args.chunk_slices,
        )
    if args.command == "convert-xcat-labels":
        return _convert_xcat_labels(
            args.configuration,
            debug_one_frame=args.debug_one_frame,
            chunk_slices=args.chunk_slices,
            overwrite=args.overwrite,
        )
    if args.command == "generate-contrast":
        return _generate_spatially_varying_fa_contrast(
            args.configuration,
            debug_one_frame=args.debug_one_frame,
            chunk_slices=args.chunk_slices,
            overwrite=args.overwrite,
        )
    if args.command == "export-contrast-nrrd":
        return _export_contrast_nrrd(
            args.configuration,
            output=args.output,
            overwrite=args.overwrite,
        )
    if args.command == "export-labels-nrrd":
        return _export_labels_nrrd(
            args.configuration,
            output=args.output,
            overwrite=args.overwrite,
        )
    if args.command == "generate-balloon-debug":
        return _generate_balloon_debug(
            args.configuration,
            overwrite=args.overwrite,
        )
    if args.command == "generate-balloon-path-debug":
        return _generate_balloon_path_debug(
            args.configuration,
            center_spacing_mm=args.center_spacing_mm,
            overwrite=args.overwrite,
        )
    if args.command == "prepare-kspace-inputs":
        return _prepare_kspace_inputs(
            args.configuration,
            rebuild_rss_cache=args.rebuild_rss_cache,
        )
    if args.command == "generate-kspace-debug":
        return _generate_kspace_debug(
            args.configuration,
            coil=args.coil,
            device_id=args.device_id,
        )
    if args.command == "generate-kspace-all-coils-debug":
        return _generate_kspace_all_coils_debug(args.configuration)
    if args.command == "generate-balloon-kspace-debug":
        return _generate_kspace_all_coils_debug(
            args.configuration,
            gd_balloon=True,
        )
    if args.command == "validate-balloon-kspace-linearity":
        return _validate_balloon_kspace_linearity(args.configuration)
    if args.command == "generate-fullysampled-reference":
        return _generate_fullysampled_reference(
            args.configuration,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            overwrite=args.overwrite,
        )
    if args.command == "plan-acquisition":
        return _plan_acquisition(args.configuration)
    if args.command == "generate-tissue-kspace-library":
        return _generate_tissue_kspace_library(
            args.configuration,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
    if args.command == "generate-tissue-adjoint-reference":
        return _generate_tissue_adjoint_reference(
            args.configuration,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            overwrite=args.overwrite,
            allow_missing=args.allow_missing,
        )
    if args.command == "generate-dynamic-acquisition":
        return _generate_dynamic_acquisition(
            args.configuration,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            view_order_cycles=args.view_order_cycles,
            save_adjoint_debug=args.save_adjoint_debug,
        )
    if args.command == "generate-dynamic-fullysampled-reference":
        return _generate_dynamic_reference(
            args.configuration, overwrite=args.overwrite
        )
    if args.command == "generate-curved-line-profile":
        return _generate_curved_line_profile(
            args.configuration,
            input_path=args.input,
            overwrite=args.overwrite,
        )
    if args.command == "generate-three-position-reference-debug":
        return _generate_three_position_reference_debug(
            args.configuration,
            overwrite=args.overwrite,
        )
    if args.command == "compare-nufft-devices":
        return _compare_nufft_devices(
            args.cpu_reference, args.gpu_reference, args.output
        )
    if args.command == "validate-kspace-reference":
        return _validate_kspace_reference(
            args.shifted_gt, args.multicoil_reference, args.output
        )
    if args.command == "diagnose-kspace-fov":
        return _diagnose_kspace_fov(
            args.configuration,
            coil=args.coil,
            kspace_path=args.kspace,
            support_threshold=args.support_threshold,
            support_margin_mm=args.support_margin_mm,
            device_id=args.device_id,
            output=args.output,
            overwrite=args.overwrite,
        )

    parser.print_help()
    return 0
