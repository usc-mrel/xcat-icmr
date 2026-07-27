"""Command-line entry point for XCAT-iCMR."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from pydantic import ValidationError
from scipy.io import whosmat

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
    format_reduced_nufft_validation,
    format_logical_input_preview,
    format_sigpy_reference_validation,
    format_prepared_contrast,
    prepare_contrast_for_encoding,
    prepare_encoding_grids,
    run_reduced_nufft_validation,
    save_logical_input_preview,
    validate_sigpy_reference,
)
from xcat_icmr.exporting import (
    NrrdExportError,
    export_contrast_series_nrrd,
    format_nrrd_export,
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
)
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
        if not config.outputs.save_contrast_images:
            raise ContrastGenerationError(
                "outputs.save_contrast_images is false; enable it to write "
                "contrast images"
            )
        sequence = read_sequence(config.sequence)
        library = get_tissue_library(
            config.sequence.contrast.tissue_library
        )
        frame_plan = plan_xcat_frames(
            config, debug_one_frame=debug_one_frame
        )
        expected_shape = xcat_label_shape(config)
        contrast_directory = config.run.output_root / "contrast"
        reports = []
        for frame in frame_plan.frames:
            if frame.label_path is None:
                raise ContrastGenerationError(
                    "contrast generation currently requires "
                    "outputs.save_tissue_labels to be true"
                )
            image_path = (
                contrast_directory
                / (
                    f"phantom_{config.run.id}_act_{frame.index}_"
                    f"{config.sequence.contrast.model}.mat"
                )
            )
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

        image_path = (
            config.run.output_root
            / "contrast"
            / (
                f"phantom_{config.run.id}_act_1_"
                f"{config.sequence.contrast.model}.mat"
            )
        )
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
        patient_direction = (
            transforms.logical_axis_patient_directions[logical_axis]
        )
        expected_rf_direction = {
            "Sag": "LR",
            "Cor": "AP",
            "Tra": "SI",
        }[patient_direction[1:]]
        if expected_rf_direction != config.sequence.rf_direction:
            raise SliceProfileError(
                "Pulseq RF gradient and sequence.rf_direction disagree: "
                f"logical {excitation.gradient_channel} maps to "
                f"{patient_direction} ({expected_rf_direction}), but YAML "
                f"declares {config.sequence.rf_direction}"
            )
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
        contrast_directory = config.run.output_root / "contrast"
        frame_plan = plan_xcat_frames(
            config, debug_one_frame=debug_one_frame
        )
        profile_path = (
            contrast_directory
            / f"phantom_{config.run.id}_rf_slice_profile.mat"
        )
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
            stem = f"phantom_{config.run.id}_act_{frame.index}"
            report = generate_rf_profile_bssfp_contrast(
                label_path=frame.label_path,
                profile=profile,
                transforms=transforms,
                pcs_voxel_size_mm=config.phantom.voxel_size_mm,
                library=library,
                te_ms=sequence.te_ms,
                tr_ms=sequence.tr_ms,
                profile_output_path=profile_path,
                image_output_path=(
                    contrast_directory
                    / f"{stem}_{config.sequence.contrast.model}.mat"
                ),
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
    expected_shape: tuple[int, int, int],
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
    return entries.get(variable_name) == (expected_shape, "single")


def _generate_dynamic_cycle(
    configuration: Path,
    *,
    chunk_slices: int,
) -> int:
    """Generate, consume, and clean one complete XCAT motion cycle."""

    try:
        config = load_config(configuration)
        if not config.outputs.save_tissue_labels:
            raise XcatLabelConversionError(
                "streaming generation requires save_tissue_labels: true"
            )
        if not config.outputs.save_contrast_images:
            raise RfProfileContrastError(
                "streaming generation requires save_contrast_images: true"
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
        preflight = preflight_xcat_invocation(
            config,
            parameters,
            frames,
            allow_partial_outputs=True,
        )
        if not preflight.passed:
            print(format_xcat_preflight(preflight, dry_run=False))
            return 2

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
        patient_direction = (
            transforms.logical_axis_patient_directions[logical_axis]
        )
        expected_rf_direction = {
            "Sag": "LR",
            "Cor": "AP",
            "Tra": "SI",
        }[patient_direction[1:]]
        if expected_rf_direction != config.sequence.rf_direction:
            raise SliceProfileError(
                "Pulseq RF gradient maps to "
                f"{patient_direction} ({expected_rf_direction}), but YAML "
                f"declares {config.sequence.rf_direction}"
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
        expected_shape = xcat_label_shape(config)
        contrast_directory = config.run.output_root / "contrast"
        profile_path = (
            contrast_directory
            / f"phantom_{config.run.id}_rf_slice_profile.mat"
        )
        library = get_tissue_library(
            config.sequence.contrast.tissue_library
        )
        # The small shared profile is conservatively rewritten on the first
        # generated frame; label and contrast frames remain independently
        # resumable.
        profile_written = False
        total = len(frames.frames)
        reused_labels = 0
        reused_contrasts = 0

        def consume(frame) -> None:
            nonlocal profile_written, reused_labels, reused_contrasts
            if frame.label_path is None:
                raise XcatLabelConversionError(
                    f"frame {frame.index} has no label destination"
                )
            label_existed = _mat_variable_matches(
                frame.label_path, "P", expected_shape
            )
            if frame.label_path.exists() and not label_existed:
                raise XcatLabelConversionError(
                    f"existing label failed validation: {frame.label_path}"
                )
            if label_existed:
                reused_labels += 1
            else:
                convert_xcat_labels_to_mat(
                    open_xcat_binary(config, frame.binary_path),
                    frame.label_path,
                    chunk_slices=chunk_slices,
                    overwrite=False,
                )
            if not _mat_variable_matches(frame.label_path, "P", expected_shape):
                raise XcatLabelConversionError(
                    f"label verification failed: {frame.label_path}"
                )

            # The raw is redundant only after the label has passed reopening.
            frame.binary_path.unlink()

            stem = f"phantom_{config.run.id}_act_{frame.index}"
            contrast_path = (
                contrast_directory
                / f"{stem}_{config.sequence.contrast.model}.mat"
            )
            contrast_valid = _mat_variable_matches(
                contrast_path, "image", expected_shape
            )
            if label_existed and contrast_valid:
                reused_contrasts += 1
            else:
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
            print(
                f"Frame {frame.index}/{total}: label verified, raw removed, "
                f"contrast {'reused' if label_existed and contrast_valid else 'generated'}",
                flush=True,
            )

        result = execute_streaming_xcat_invocation(
            config,
            frames,
            preflight,
            consume,
        )
        print(
            "\nDynamic cycle generation\n"
            f"Frames:             {result.consumed_frame_count}/{total}\n"
            f"Labels reused:      {reused_labels}\n"
            f"Contrasts reused:   {reused_contrasts}\n"
            "Raw files retained: 0\n"
            f"stdout:             {result.stdout_log}\n"
            f"stderr:             {result.stderr_log}\n"
            "Verification:       PASS"
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


def _generate_kspace_debug(
    configuration: Path,
    *,
    coil: int,
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

        image_path = (
            config.run.output_root
            / "contrast"
            / (
                f"phantom_{config.run.id}_act_1_"
                f"{config.sequence.contrast.model}.mat"
            )
        )
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
        encoding_grids = prepare_encoding_grids(
            ground_truth_shape=prepared.image.shape,
            ground_truth_voxel_size_mm=config.phantom.voxel_size_mm,
            sequence_resolution_mm=sequence.resolution_mm,
        )
        output_path = (
            config.run.output_root
            / "kspace"
            / "debug"
            / (
                f"sigpy_3d_frame_0001_coil_{coil:02d}_"
                "forward_adjoint.mat"
            )
        )
        report = run_reduced_nufft_validation(
            prepared.image,
            normalized_coil,
            kx_per_m=sequence.kx,
            ky_per_m=sequence.ky,
            kz_per_m=sequence.kz,
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


def _export_contrast_nrrd(
    configuration: Path,
    *,
    output: Path | None,
    overwrite: bool,
) -> int:
    try:
        config = load_config(configuration)
        frame_plan = plan_xcat_frames(config, debug_one_frame=False)
        contrast_directory = config.run.output_root / "contrast"
        paths = tuple(
            contrast_directory
            / (
                f"phantom_{config.run.id}_act_{frame.index}_"
                f"{config.sequence.contrast.model}.mat"
            )
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "validate":
        return _validate(args.configuration)
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
    if args.command == "prepare-kspace-inputs":
        return _prepare_kspace_inputs(
            args.configuration,
            rebuild_rss_cache=args.rebuild_rss_cache,
        )
    if args.command == "generate-kspace-debug":
        return _generate_kspace_debug(
            args.configuration,
            coil=args.coil,
        )

    parser.print_help()
    return 0
