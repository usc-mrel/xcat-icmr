"""Self-describing HDF5 contract for simulated reconstruction inputs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import h5py
import numpy as np

from xcat_icmr import __version__
from xcat_icmr.cache import tissue_kspace_cache_entry
from xcat_icmr.coils import (
    inspect_sensitivity_map,
    load_normalized_coil_roi_in_logical_frame,
    prepare_rss_normalization,
    sensitivity_shape_in_logical_frame,
)
from xcat_icmr.encoding.sigpy_backend import (
    DEFAULT_NUFFT_KERNEL_WIDTH,
    DEFAULT_NUFFT_OVERSAMPLING,
)
from xcat_icmr.encoding.trajectory import prepare_physical_sigpy_trajectory
from xcat_icmr.sequence import build_coordinate_transforms, read_sequence
from xcat_icmr.signal import read_pulseq_excitation

if TYPE_CHECKING:
    from xcat_icmr.acquisition.schedule import AcquisitionSchedule
    from xcat_icmr.config.models import SimulationConfig


ACQUISITION_SCHEMA_NAME = "xcat-icmr-acquisition"
ACQUISITION_SCHEMA_VERSION = 1


class AcquisitionSchemaError(ValueError):
    """Raised when an acquisition cannot satisfy the reconstruction contract."""


@dataclass(frozen=True)
class AcquisitionInspection:
    path: Path
    schema_name: str
    schema_version: int
    storage: str
    kspace_shape: tuple[int, int, int]
    trajectory_shape: tuple[int, int, int]
    sensitivity_shape: tuple[int, int, int, int]
    reconstruction_shape: tuple[int, int, int]
    target_fov_mm: tuple[float, float, float]
    voxel_size_mm: tuple[float, float, float]
    frame_count: int
    trs_per_frame: int
    frame_duration_s: float
    effective_tr_s: float
    sequence_filename: str
    sequence_signature: str
    nufft_backend: str
    nufft_oversampling: float
    nufft_kernel_width: float
    matching_reference: str | None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _three_tuple(values, *, kind=float) -> tuple:
    array = np.asarray(values).reshape(-1)
    if array.size != 3:
        raise AcquisitionSchemaError("expected exactly three spatial values")
    return tuple(kind(value) for value in array)


def inspect_acquisition(path: str | Path) -> AcquisitionInspection:
    """Validate the inexpensive structural reconstruction contract."""

    resolved = Path(path).expanduser().resolve(strict=False)
    if not resolved.is_file():
        raise AcquisitionSchemaError(f"acquisition file does not exist: {resolved}")
    try:
        with h5py.File(resolved, "r") as handle:
            schema_name = str(handle.attrs.get("xcat_icmr_schema", ""))
            schema_version = int(handle.attrs.get("xcat_icmr_schema_version", 0))
            schema_status = str(handle.attrs.get("xcat_icmr_schema_status", ""))
            if schema_name != ACQUISITION_SCHEMA_NAME:
                raise AcquisitionSchemaError(
                    f"unsupported or missing acquisition schema: {schema_name!r}"
                )
            if schema_version != ACQUISITION_SCHEMA_VERSION:
                raise AcquisitionSchemaError(
                    f"unsupported acquisition schema version {schema_version}"
                )
            if schema_status != "complete":
                raise AcquisitionSchemaError(
                    f"acquisition reconstruction metadata is {schema_status or 'incomplete'}"
                )

            kspace = handle.get("kspace")
            coordinates = handle.get("trajectory/coordinates")
            dcf = handle.get("trajectory/density_compensation")
            acquired = handle.get("trajectory_tr_index_zero_based")
            sensitivities = handle.get("encoding/sensitivity_maps")
            sensitivity_complete = handle.get("encoding/sensitivity_complete")
            starts = handle.get("frame_start_tr_zero_based")
            stops = handle.get("frame_stop_tr_exclusive")
            required = (kspace, coordinates, dcf, acquired, sensitivities, starts, stops)
            if not all(isinstance(value, h5py.Dataset) for value in required):
                raise AcquisitionSchemaError(
                    "acquisition is missing one or more reconstruction datasets"
                )
            assert isinstance(kspace, h5py.Dataset)
            assert isinstance(coordinates, h5py.Dataset)
            assert isinstance(dcf, h5py.Dataset)
            assert isinstance(acquired, h5py.Dataset)
            assert isinstance(sensitivities, h5py.Dataset)
            assert isinstance(starts, h5py.Dataset)
            assert isinstance(stops, h5py.Dataset)
            if kspace.ndim != 3 or kspace.dtype != np.dtype(np.complex64):
                raise AcquisitionSchemaError(
                    "kspace must be complex64 [sample, acquired_TR, coil]"
                )
            if coordinates.ndim != 3 or coordinates.shape[2] != 3:
                raise AcquisitionSchemaError(
                    "trajectory coordinates must be [sample, arm, xyz]"
                )
            if coordinates.dtype != np.dtype(np.float32):
                raise AcquisitionSchemaError("trajectory coordinates must be float32")
            if dcf.shape != coordinates.shape[:2] or dcf.dtype != np.dtype(np.float32):
                raise AcquisitionSchemaError(
                    "density compensation must be float32 [sample, arm]"
                )
            if acquired.shape != (kspace.shape[1],):
                raise AcquisitionSchemaError(
                    "acquired trajectory index count does not match k-space TRs"
                )
            if sensitivities.ndim != 4 or sensitivities.dtype != np.dtype(np.complex64):
                raise AcquisitionSchemaError(
                    "sensitivity maps must be complex64 [x, y, z, coil]"
                )
            if sensitivities.shape[3] != kspace.shape[2]:
                raise AcquisitionSchemaError(
                    "sensitivity-map and k-space coil counts differ"
                )
            if not isinstance(sensitivity_complete, h5py.Dataset) or not np.all(
                sensitivity_complete[:]
            ):
                raise AcquisitionSchemaError("sensitivity-map preparation is incomplete")
            reconstruction_shape = _three_tuple(
                handle.attrs["reconstruction_shape"], kind=int
            )
            if sensitivities.shape[:3] != reconstruction_shape:
                raise AcquisitionSchemaError(
                    "sensitivity-map shape differs from reconstruction_shape"
                )
            if starts.ndim != 1 or stops.shape != starts.shape or starts.size == 0:
                raise AcquisitionSchemaError("frame boundaries are invalid")
            start_values = np.asarray(starts[:], dtype=np.int64)
            stop_values = np.asarray(stops[:], dtype=np.int64)
            if (
                start_values[0] != 0
                or stop_values[-1] != kspace.shape[1]
                or np.any(stop_values <= start_values)
                or np.any(start_values[1:] != stop_values[:-1])
            ):
                raise AcquisitionSchemaError(
                    "frame boundaries do not continuously cover acquired TRs"
                )
            acquired_values = np.asarray(acquired[:], dtype=np.int64)
            if np.any(acquired_values < 0) or np.any(
                acquired_values >= coordinates.shape[1]
            ):
                raise AcquisitionSchemaError("acquired trajectory indices are out of range")

            return AcquisitionInspection(
                path=resolved,
                schema_name=schema_name,
                schema_version=schema_version,
                storage=("virtual" if kspace.is_virtual else "materialized"),
                kspace_shape=tuple(int(value) for value in kspace.shape),
                trajectory_shape=tuple(int(value) for value in coordinates.shape),
                sensitivity_shape=tuple(int(value) for value in sensitivities.shape),
                reconstruction_shape=reconstruction_shape,
                target_fov_mm=_three_tuple(handle.attrs["target_fov_mm"]),
                voxel_size_mm=_three_tuple(handle.attrs["reconstruction_voxel_size_mm"]),
                frame_count=int(starts.size),
                trs_per_frame=int(handle.attrs["trs_per_frame"]),
                frame_duration_s=float(handle.attrs["frame_duration_s"]),
                effective_tr_s=float(handle.attrs["effective_tr_s"]),
                sequence_filename=str(handle.attrs["pulseq_sequence_filename"]),
                sequence_signature=str(handle.attrs["pulseq_signature"]),
                nufft_backend=str(handle.attrs["nufft_backend"]),
                nufft_oversampling=float(handle.attrs["nufft_oversampling"]),
                nufft_kernel_width=float(handle.attrs["nufft_kernel_width"]),
                matching_reference=(
                    str(handle.attrs["matching_fullysampled_reference"])
                    if "matching_fullysampled_reference" in handle.attrs
                    else None
                ),
            )
    except OSError as exc:
        raise AcquisitionSchemaError(f"could not read acquisition {resolved}: {exc}") from exc


def _existing_contract_is_valid(path: Path) -> bool:
    try:
        inspect_acquisition(path)
        return True
    except (AcquisitionSchemaError, KeyError, TypeError, ValueError):
        return False


def embed_reconstruction_contract(
    path: str | Path,
    config: "SimulationConfig",
    *,
    schedule: "AcquisitionSchedule",
    matching_reference: str | Path | None,
    acquisition_id: str | None = None,
    experiment_directory: str | Path | None = None,
    overwrite_metadata: bool = False,
    progress: Callable[[str], None] | None = None,
) -> AcquisitionInspection:
    """Embed exact trajectory, DCF, coils, grids, and provenance in k-space."""

    # Imported here to keep acquisition inspection independent of the encoding
    # pipeline and avoid a package-import cycle through acquisition.__init__.
    from xcat_icmr.encoding.fullysampled_reference import _resample_complex
    from xcat_icmr.encoding.tissue_library import _grid

    destination = Path(path).expanduser().resolve(strict=False)
    if _existing_contract_is_valid(destination) and not overwrite_metadata:
        with h5py.File(destination, "r+") as handle:
            if acquisition_id is not None:
                handle.attrs["acquisition_id"] = acquisition_id
            if experiment_directory is not None:
                handle.attrs["experiment_directory"] = str(
                    Path(experiment_directory).expanduser().resolve(strict=False)
                )
            handle.attrs["run_output_root"] = str(config.run.output_root)
            control_points = config.intervention.gd_balloon.path.control_points_file
            if control_points is not None:
                handle.attrs["control_points_filename"] = control_points.name
            handle.attrs["velocity_cm_per_s"] = (
                config.intervention.gd_balloon.movement.velocity_cm_per_s
            )
        return inspect_acquisition(destination)

    sequence = read_sequence(config.sequence)
    transforms = build_coordinate_transforms(
        patient_position=config.phantom.patient_position,
        coordinate_mode=config.sequence.coordinate_mode,
        sequence_orientation=config.sequence.orientation,
    )
    high_shape, reconstruction_shape, logical_voxel, scaled_k = _grid(
        config, sequence, transforms
    )
    trajectory = prepare_physical_sigpy_trajectory(
        *scaled_k,
        fov_mm=config.encoding.target_fov_mm,
        matrix_shape=reconstruction_shape,
    )
    coordinates = trajectory.coordinates.reshape(
        trajectory.arm_count, trajectory.sample_count, 3
    ).transpose(1, 0, 2)
    dcf = np.asarray(sequence.density_compensation, dtype=np.float32)
    maximum_dcf = float(np.max(dcf))
    if not np.isfinite(maximum_dcf) or maximum_dcf <= 0:
        raise AcquisitionSchemaError("density compensation has invalid maximum")
    dcf = np.asarray(dcf / np.float32(maximum_dcf), dtype=np.float32)

    coil_info = inspect_sensitivity_map(config.coils.sensitivity_map)
    logical_coil_shape = sensitivity_shape_in_logical_frame(
        coil_info,
        stored_axis_order=config.coils.axis_order,
        dcs_to_logical=transforms.dcs_to_logical,
    )
    if any(
        requested > available
        for requested, available in zip(high_shape, logical_coil_shape, strict=True)
    ):
        raise AcquisitionSchemaError("target FOV exceeds sensitivity-map support")
    coil_offset = (
        np.asarray(logical_coil_shape, dtype=np.int64)
        - np.asarray(high_shape, dtype=np.int64)
    ) // 2
    coil_slices = tuple(
        slice(int(start), int(start + size))
        for start, size in zip(coil_offset, high_shape, strict=True)
    )
    normalization = prepare_rss_normalization(
        coil_info,
        tissue_kspace_cache_entry(config).directory / "sensitivity_rss.npy",
    )
    excitation = read_pulseq_excitation(sequence.sequence_path)

    try:
        with h5py.File(destination, "r+") as handle:
            kspace = handle.get("kspace")
            if not isinstance(kspace, h5py.Dataset) or kspace.ndim != 3:
                raise AcquisitionSchemaError(
                    "acquisition must contain [sample, acquired_TR, coil] kspace"
                )
            if kspace.shape[0] != sequence.sample_count:
                raise AcquisitionSchemaError("k-space and trajectory sample counts differ")
            if kspace.shape[2] != coil_info.coil_count:
                raise AcquisitionSchemaError("k-space and sensitivity coil counts differ")
            if schedule.acquisition_count < kspace.shape[1]:
                raise AcquisitionSchemaError("schedule is shorter than stored k-space")

            handle.attrs["xcat_icmr_schema"] = ACQUISITION_SCHEMA_NAME
            handle.attrs["xcat_icmr_schema_version"] = ACQUISITION_SCHEMA_VERSION
            handle.attrs["xcat_icmr_schema_status"] = "preparing"
            for group_name in ("trajectory", "encoding", "provenance"):
                if group_name in handle:
                    del handle[group_name]

            trajectory_group = handle.create_group("trajectory")
            trajectory_group.create_dataset(
                "coordinates",
                data=np.asarray(coordinates, dtype=np.float32),
                chunks=(min(sequence.sample_count, 128), 1, 3),
                compression="lzf",
            )
            trajectory_group["coordinates"].attrs["axis_order"] = "sample,trajectory_arm,kx_ky_kz"
            trajectory_group["coordinates"].attrs["coordinate_frame"] = "Pulseq logical"
            trajectory_group["coordinates"].attrs["units"] = "SigPy grid coordinates"
            trajectory_group["coordinates"].attrs["flattening_for_sigpy"] = "arm-major; samples contiguous"
            trajectory_group.create_dataset(
                "density_compensation",
                data=dcf,
                chunks=(min(sequence.sample_count, 128), 1),
                compression="lzf",
            )
            trajectory_group["density_compensation"].attrs["axis_order"] = "sample,trajectory_arm"
            trajectory_group["density_compensation"].attrs["normalization"] = "divided by full-trajectory maximum"
            trajectory_group["density_compensation"].attrs["source_maximum"] = maximum_dcf
            trajectory_group["acquired_arm_index_zero_based"] = handle[
                "trajectory_tr_index_zero_based"
            ]

            encoding_group = handle.create_group("encoding")
            sensitivity = encoding_group.create_dataset(
                "sensitivity_maps",
                shape=reconstruction_shape + (coil_info.coil_count,),
                dtype=np.complex64,
                chunks=tuple(min(size, 32) for size in reconstruction_shape) + (1,),
                compression="lzf",
            )
            sensitivity.attrs["axis_order"] = "logical_x,logical_y,logical_z,coil"
            sensitivity.attrs["coordinate_frame"] = "Pulseq logical"
            sensitivity.attrs["normalization"] = "voxelwise RSS before linear resampling; no post-resampling division"
            sensitivity.attrs["spatial_shift_mm"] = 0.0
            complete = encoding_group.create_dataset(
                "sensitivity_complete",
                shape=(coil_info.coil_count,),
                dtype=np.uint8,
            )
            for coil_index in range(coil_info.coil_count):
                logical = load_normalized_coil_roi_in_logical_frame(
                    coil_info,
                    coil_index,
                    normalization,
                    coil_slices,
                    stored_axis_order=config.coils.axis_order,
                    dcs_to_logical=transforms.dcs_to_logical,
                )
                low = _resample_complex(logical, reconstruction_shape)
                if not np.all(np.isfinite(low)):
                    raise AcquisitionSchemaError(
                        f"effective sensitivity map {coil_index} is non-finite"
                    )
                sensitivity[..., coil_index] = low
                complete[coil_index] = 1
                handle.flush()
                if progress:
                    progress(
                        f"Reconstruction metadata: sensitivity coil "
                        f"{coil_index + 1}/{coil_info.coil_count}"
                    )

            provenance = handle.create_group("provenance")
            provenance.attrs["simulation_config_json"] = config.model_dump_json()
            provenance.attrs["view_order_file"] = str(config.acquisition.view_order.file)
            provenance.attrs["view_order_sha256"] = _sha256(
                config.acquisition.view_order.file
            )
            provenance.attrs["sensitivity_source_file"] = str(coil_info.path)
            provenance.attrs["sensitivity_source_dataset"] = coil_info.dataset_name
            source_stat = coil_info.path.stat()
            provenance.attrs["sensitivity_source_size_bytes"] = source_stat.st_size
            provenance.attrs["sensitivity_source_mtime_ns"] = source_stat.st_mtime_ns

            frame_starts = handle.get("frame_start_tr_zero_based")
            frame_stops = handle.get("frame_stop_tr_exclusive")
            if not isinstance(frame_starts, h5py.Dataset) or not isinstance(
                frame_stops, h5py.Dataset
            ):
                raise AcquisitionSchemaError("grouped frame boundaries are missing")
            frames = handle.create_group("frames")
            frames["start_tr_zero_based"] = frame_starts
            frames["stop_tr_exclusive"] = frame_stops
            frames["tr_time_s"] = handle["time_s"]
            frames["frame_index_zero_based"] = handle["frame_index_zero_based"]

            reconstruction_voxel = np.asarray(
                config.encoding.target_fov_mm, dtype=np.float64
            ) / np.asarray(reconstruction_shape, dtype=np.float64)
            handle.attrs["package_version"] = __version__
            handle.attrs["pulseq_sequence_filename"] = sequence.sequence_path.name
            handle.attrs["pulseq_sequence_path"] = str(sequence.sequence_path)
            handle.attrs["pulseq_signature"] = sequence.signature
            handle.attrs["pulseq_te_ms"] = sequence.te_ms
            handle.attrs["pulseq_tr_ms"] = sequence.tr_ms
            handle.attrs["pulseq_flip_angle_deg"] = sequence.flip_angle_deg
            handle.attrs["effective_tr_s"] = schedule.effective_tr_s
            handle.attrs["trs_per_frame"] = int(frame_stops[0] - frame_starts[0])
            handle.attrs["frame_duration_s"] = float(
                kspace.attrs.get("frame_duration_s", schedule.frame_duration_s)
            )
            handle.attrs["target_fov_mm"] = config.encoding.target_fov_mm
            handle.attrs["high_resolution_shape"] = high_shape
            handle.attrs["high_resolution_voxel_size_mm"] = logical_voxel
            handle.attrs["reconstruction_shape"] = reconstruction_shape
            handle.attrs["reconstruction_voxel_size_mm"] = reconstruction_voxel
            handle.attrs["image_axis_order"] = "logical_x,logical_y,logical_z"
            handle.attrs["logical_axis_patient_directions"] = (
                transforms.logical_axis_patient_directions
            )
            handle.attrs["pcs_to_logical"] = transforms.pcs_to_logical
            handle.attrs["logical_to_dcs"] = transforms.logical_to_dcs
            handle.attrs["patient_position"] = config.phantom.patient_position
            handle.attrs["sequence_coordinate_mode"] = config.sequence.coordinate_mode
            handle.attrs["sequence_orientation"] = config.sequence.orientation
            handle.attrs["rf_logical_axis_zero_based"] = excitation.logical_axis
            handle.attrs["rf_center_shift_mm"] = config.sequence.rf_profile.center_shift_mm
            handle.attrs["nufft_backend"] = "sigpy"
            handle.attrs["nufft_oversampling"] = DEFAULT_NUFFT_OVERSAMPLING
            handle.attrs["nufft_kernel_width"] = DEFAULT_NUFFT_KERNEL_WIDTH
            handle.attrs["kspace_composition"] = "tissue + additive_Gd"
            handle.attrs["coil_combination_contract"] = "sum(conj(sensitivity) * coil_image); no sensitivity denominator"
            if matching_reference is not None:
                handle.attrs["matching_fullysampled_reference"] = str(
                    Path(matching_reference).expanduser().resolve(strict=False)
                )
            if acquisition_id is not None:
                handle.attrs["acquisition_id"] = acquisition_id
            if experiment_directory is not None:
                handle.attrs["experiment_directory"] = str(
                    Path(experiment_directory).expanduser().resolve(strict=False)
                )
            handle.attrs["run_output_root"] = str(config.run.output_root)
            control_points = config.intervention.gd_balloon.path.control_points_file
            if control_points is not None:
                handle.attrs["control_points_filename"] = control_points.name
            handle.attrs["velocity_cm_per_s"] = (
                config.intervention.gd_balloon.movement.velocity_cm_per_s
            )
            handle.attrs["xcat_icmr_schema_status"] = "complete"
            handle.flush()
    except OSError as exc:
        raise AcquisitionSchemaError(
            f"could not embed reconstruction metadata in {destination}: {exc}"
        ) from exc

    return inspect_acquisition(destination)


def format_acquisition_inspection(report: AcquisitionInspection) -> str:
    """Format a concise reconstruction-input preflight."""

    return "\n".join(
        (
            "Self-describing simulated acquisition",
            f"Schema:              {report.schema_name} v{report.schema_version}",
            f"K-space:             {report.kspace_shape} complex64 ({report.storage})",
            f"Trajectory:          {report.trajectory_shape} float32",
            f"Sensitivity maps:    {report.sensitivity_shape} complex64",
            f"Reconstruction grid: {report.reconstruction_shape}",
            f"FOV:                 {report.target_fov_mm} mm",
            f"Voxel size:          {report.voxel_size_mm} mm",
            f"Frames:              {report.frame_count}",
            f"TRs per frame:       {report.trs_per_frame}",
            f"Frame duration:      {report.frame_duration_s * 1e3:g} ms",
            f"Effective TR:        {report.effective_tr_s * 1e3:g} ms",
            f"Sequence:            {report.sequence_filename}",
            f"Signature:           {report.sequence_signature}",
            f"NUFFT:               {report.nufft_backend}, OS {report.nufft_oversampling:g}, width {report.nufft_kernel_width:g}",
            f"Matching reference:  {report.matching_reference or 'not recorded'}",
            f"Input:               {report.path}",
        )
    )
