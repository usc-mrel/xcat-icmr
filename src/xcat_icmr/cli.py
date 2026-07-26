"""Command-line entry point for XCAT-iCMR."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from pydantic import ValidationError

from xcat_icmr.config import (
    ConfigurationLoadError,
    format_summary,
    load_config,
    validate_paths,
)
from xcat_icmr.config.loader import format_validation_error
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
    plan_xcat_frames,
    prepare_xcat_parameter_file,
    preflight_xcat_invocation,
    execute_xcat_invocation,
)
from xcat_icmr.sequence import (
    MatlabReferenceError,
    SequenceReadError,
    compare_to_matlab,
    format_matlab_comparison,
    format_sequence_summary,
    read_sequence,
)
from xcat_icmr.signal import (
    ContrastGenerationError,
    MatlabSignalReferenceError,
    compare_bssfp_to_matlab,
    format_bssfp_matlab_comparison,
    format_contrast_generation,
    generate_bssfp_contrast,
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
        help="MATLAB v7.3 file containing the cropped P label volume",
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
        help="convert cropped XCAT binaries to verified MATLAB label files",
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
        for frame in frame_plan.frames:
            if frame.label_path is None:
                raise XcatLabelConversionError(
                    f"frame {frame.index} has no planned label path"
                )
            reports.append(
                convert_xcat_labels_to_mat(
                    open_xcat_binary(config, frame.binary_path),
                    frame.label_path,
                    chunk_slices=chunk_slices,
                    overwrite=overwrite,
                )
            )

        print(f"XCAT label conversion ({len(reports)} frame(s))")
        for index, report in enumerate(reports):
            if index:
                print()
            print(format_xcat_label_conversion(report))
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
        expected_shape = (
            config.phantom.crop.rows[1] - config.phantom.crop.rows[0],
            config.phantom.crop.columns[1]
            - config.phantom.crop.columns[0],
            config.phantom.slice_range.end
            - config.phantom.slice_range.start
            + 1,
        )
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
        return _generate_contrast(
            args.configuration,
            debug_one_frame=args.debug_one_frame,
            chunk_slices=args.chunk_slices,
            overwrite=args.overwrite,
        )

    parser.print_help()
    return 0
