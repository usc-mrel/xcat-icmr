"""TR-streamed additive tissue and moving-Gd multicoil acquisition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import TYPE_CHECKING, Callable

import h5py
import numpy as np

from xcat_icmr.acquisition.schedule import AcquisitionSchedule, build_acquisition_schedule
from xcat_icmr.acquisition.storage import StorageEstimate, estimate_dynamic_acquisition_storage, require_free_space
from xcat_icmr.cache import (
    artifact_cache_status,
    dynamic_acquisition_cache_entry,
    tissue_kspace_cache_entry,
    write_artifact_manifest,
)
from xcat_icmr.coils import (
    inspect_sensitivity_map,
    load_normalized_coil_roi_in_logical_frame,
    prepare_rss_normalization,
    sensitivity_shape_in_logical_frame,
)
from xcat_icmr.encoding.tissue_library import _grid, tissue_library_frame_path
from xcat_icmr.intervention import (
    calculate_gd_bssfp_signal,
    interpolate_cubic_arc_length,
    load_balloon_path,
    rasterize_sparse_balloon,
)
from xcat_icmr.intervention.roi_encoding import PersistentSparseRoiEncoder
from xcat_icmr.phantom import plan_xcat_frames
from xcat_icmr.sequence import build_coordinate_transforms, read_sequence
from xcat_icmr.sequence.orientation import map_spatial_indices, reoriented_spatial_shape
from xcat_icmr.signal import calculate_rf_profile_bssfp_contrast, generate_slice_profile, read_pulseq_excitation
from xcat_icmr.tissue import get_tissue_library

if TYPE_CHECKING:
    from xcat_icmr.config.models import SimulationConfig


class DynamicAcquisitionError(ValueError):
    """Raised when the combined acquisition cannot be generated safely."""


@dataclass(frozen=True)
class DynamicAcquisitionPlan:
    output_path: Path
    schedule: AcquisitionSchedule
    shape: tuple[int, int, int]
    storage: StorageEstimate
    completed_trs: int
    view_order_cycles: int | None


@dataclass(frozen=True)
class DynamicAcquisitionResult:
    plan: DynamicAcquisitionPlan
    generated_trs: int
    reused_trs: int
    elapsed_s: float
    adjoint_debug_path: Path | None = None


def _carrier(library, name: str):
    for group in library.groups:
        if group.name.lower() == name.lower():
            return group.properties
    raise DynamicAcquisitionError(f"carrier tissue {name!r} is absent from the tissue library")


def _map_pcs_sparse_to_high(
    indices: np.ndarray,
    values: np.ndarray,
    *,
    pcs_shape: tuple[int, int, int],
    pcs_to_logical: np.ndarray,
    high_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    mapped, valid = map_spatial_indices(
        indices, source_shape=pcs_shape, source_to_target=pcs_to_logical
    )
    oriented_shape = np.asarray(reoriented_spatial_shape(pcs_shape, pcs_to_logical))
    mapped = mapped.astype(np.int64) + ((np.asarray(high_shape) - oriented_shape) // 2)[None, :]
    valid &= np.all(mapped >= 0, axis=1) & np.all(mapped < np.asarray(high_shape)[None, :], axis=1)
    if not np.any(valid):
        raise DynamicAcquisitionError("the balloon lies outside the target encoding FOV")
    return mapped[valid].astype(np.int32), np.asarray(values, dtype=np.float32).reshape(-1)[valid]


def _ensure_output(path: Path, shape: tuple[int, int, int], schedule: AcquisitionSchedule, *, overwrite: bool) -> tuple[h5py.File, h5py.Dataset, h5py.Dataset]:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "a"
    handle = h5py.File(path, mode)
    data = handle.get("kspace")
    complete = handle.get("tr_complete")
    valid = isinstance(data, h5py.Dataset) and data.shape == shape and data.dtype == np.dtype(np.complex64) and isinstance(complete, h5py.Dataset) and complete.shape == (shape[1],)
    if not valid:
        if data is not None or complete is not None:
            handle.close()
            raise DynamicAcquisitionError("existing acquisition has an incompatible schema; use --overwrite")
        data = handle.create_dataset("kspace", shape=shape, dtype=np.complex64, chunks=(shape[0], 1, shape[2]))
        complete = handle.create_dataset("tr_complete", shape=(shape[1],), dtype=np.uint8)
        data.attrs["axis_order"] = "sample,simulation_TR,coil"
        data.attrs["effective_tr_s"] = schedule.effective_tr_s
        data.attrs["frame_duration_s"] = schedule.frame_duration_s
        data.attrs["trajectory_tr_count"] = schedule.trajectory_tr_count
        data.attrs["view_order_cycle_length"] = schedule.view_order_cycle_length
        data.attrs["complete_view_order_cycles"] = schedule.complete_view_order_cycles
        data.attrs["partial_view_order_cycle_tr_count"] = (
            schedule.partial_view_order_cycle_tr_count
        )
        data.attrs["composition"] = "tissue + additive_Gd"
    schedule_values = {
        "time_s": schedule.time_s,
        "frame_index_zero_based": schedule.frame_index_zero_based,
        "cardiac_phase_index_zero_based": schedule.cardiac_phase_index_zero_based,
        "trajectory_tr_index_zero_based": schedule.trajectory_tr_index_zero_based,
    }
    for name, values in schedule_values.items():
        existing = handle.get(name)
        array = np.asarray(values)
        if existing is None:
            handle.create_dataset(name, data=array)
        elif existing.shape != array.shape or not np.array_equal(existing[:], array):
            handle.close()
            raise DynamicAcquisitionError(
                f"existing acquisition schedule dataset {name!r} differs; use --overwrite"
            )
    return handle, data, complete


def plan_dynamic_acquisition(
    config: "SimulationConfig",
    *,
    check_free_space: bool = False,
    view_order_cycles: int | None = None,
) -> DynamicAcquisitionPlan:
    if artifact_cache_status(tissue_kspace_cache_entry(config)).state != "HIT":
        raise DynamicAcquisitionError("the complete tissue k-space library is required")
    sequence = read_sequence(config.sequence)
    phases = len(plan_xcat_frames(config, debug_one_frame=False).frames)
    schedule = build_acquisition_schedule(
        config,
        actual_tr_s=sequence.tr_ms * 1e-3,
        trajectory_tr_count=sequence.arm_count,
        cardiac_phase_count=phases,
        view_order_cycles=view_order_cycles,
    )
    coils = inspect_sensitivity_map(config.coils.sensitivity_map).coil_count
    shape = (sequence.sample_count, schedule.acquisition_count, coils)
    storage = estimate_dynamic_acquisition_storage(*shape)
    output_directory = dynamic_acquisition_cache_entry(config).directory
    if view_order_cycles is not None:
        output_directory = (
            output_directory
            / "debug"
            / f"view_order_cycles_{view_order_cycles:04d}"
        )
    output = output_directory / "combined_multicoil_kspace.h5"
    completed = 0
    if output.is_file():
        try:
            with h5py.File(output, "r") as handle:
                if "tr_complete" in handle:
                    completed = int(np.count_nonzero(handle["tr_complete"][:]))
        except OSError:
            completed = 0
    if check_free_space and completed < schedule.acquisition_count:
        require_free_space(output.parent, storage.bytes * (schedule.acquisition_count - completed) // schedule.acquisition_count)
    return DynamicAcquisitionPlan(
        output, schedule, shape, storage, completed, view_order_cycles
    )


def generate_dynamic_acquisition(
    config: "SimulationConfig",
    *,
    overwrite: bool = False,
    dry_run: bool = False,
    view_order_cycles: int | None = None,
    save_adjoint_debug: bool = False,
    progress: Callable[[str], None] | None = None,
) -> DynamicAcquisitionResult:
    """Gather cached tissue arms and add the sparse moving Gd signal per TR."""

    started = time.perf_counter()
    if not config.intervention.gd_balloon.enabled:
        raise DynamicAcquisitionError("the Gd balloon must be enabled")
    if config.intervention.gd_balloon.composition.mode != "additive":
        raise DynamicAcquisitionError(
            "dynamic acquisition currently requires additive Gd composition"
        )
    plan = plan_dynamic_acquisition(
        config,
        check_free_space=not dry_run,
        view_order_cycles=view_order_cycles,
    )
    if dry_run:
        predicted_adjoint = (
            plan.output_path.with_name("combined_adjoint_4d.h5")
            if save_adjoint_debug
            else None
        )
        return DynamicAcquisitionResult(
            plan,
            0,
            plan.completed_trs,
            time.perf_counter() - started,
            predicted_adjoint,
        )
    if save_adjoint_debug and view_order_cycles is None:
        raise DynamicAcquisitionError(
            "--save-adjoint-debug requires --view-order-cycles so debug "
            "images cannot be confused with the full experiment"
        )
    if (
        not overwrite
        and plan.completed_trs == plan.schedule.acquisition_count
    ):
        adjoint_debug_path = None
        if save_adjoint_debug:
            from xcat_icmr.acquisition.dynamic_adjoint import (
                generate_dynamic_adjoint_debug,
            )

            adjoint_debug_path = generate_dynamic_adjoint_debug(
                config,
                plan=plan,
                overwrite=False,
                progress=progress,
            )
        return DynamicAcquisitionResult(
            plan,
            0,
            plan.completed_trs,
            time.perf_counter() - started,
            adjoint_debug_path,
        )
    sequence = read_sequence(config.sequence)
    transforms = build_coordinate_transforms(
        patient_position=config.phantom.patient_position,
        coordinate_mode=config.sequence.coordinate_mode,
        sequence_orientation=config.sequence.orientation,
    )
    high_shape, reconstruction_shape, logical_voxel, scaled_k = _grid(config, sequence, transforms)
    coil_info = inspect_sensitivity_map(config.coils.sensitivity_map)
    logical_coil_shape = sensitivity_shape_in_logical_frame(
        coil_info, stored_axis_order=config.coils.axis_order, dcs_to_logical=transforms.dcs_to_logical
    )
    normalization = prepare_rss_normalization(coil_info, tissue_kspace_cache_entry(config).directory / "sensitivity_rss.npy")
    offset = (np.asarray(logical_coil_shape) - np.asarray(high_shape)) // 2

    def coil_roi_loader(coil_index: int, slices: tuple[slice, slice, slice]) -> np.ndarray:
        full_slices = tuple(slice(int(item.start) + int(offset[axis]), int(item.stop) + int(offset[axis])) for axis, item in enumerate(slices))
        return load_normalized_coil_roi_in_logical_frame(
            coil_info, coil_index, normalization, full_slices,
            stored_axis_order=config.coils.axis_order, dcs_to_logical=transforms.dcs_to_logical,
        )

    frames = plan_xcat_frames(config, debug_one_frame=False).frames
    first_label = frames[0].label_path
    if first_label is None:
        raise DynamicAcquisitionError("the first XCAT label path is unavailable")
    excitation = read_pulseq_excitation(sequence.sequence_path)
    profile = generate_slice_profile(
        excitation,
        matrix_size=logical_coil_shape[excitation.logical_axis],
        voxel_size_mm=logical_voxel[excitation.logical_axis],
        center_shift_mm=config.sequence.rf_profile.center_shift_mm,
    )
    library = get_tissue_library(config.sequence.contrast.tissue_library)
    scratch_contrast, pcs_axis, _, applied_flip = calculate_rf_profile_bssfp_contrast(
        label_path=first_label, profile=profile, transforms=transforms,
        pcs_voxel_size_mm=config.phantom.voxel_size_mm, library=library,
        te_ms=sequence.te_ms, tr_ms=sequence.tr_ms,
    )
    pcs_shape = tuple(int(v) for v in scratch_contrast.shape)
    del scratch_contrast
    path_config = config.intervention.gd_balloon.path
    loaded_path = load_balloon_path(path_config.control_points_file, coordinate_system=path_config.coordinate_system)
    curve = interpolate_cubic_arc_length(loaded_path.control_points_lps_mm)
    positions = curve.positions_at_times_s(
        plan.schedule.time_s,
        velocity_cm_per_s=config.intervention.gd_balloon.movement.velocity_cm_per_s,
        start_time_s=config.intervention.gd_balloon.movement.start_time_s,
        traversal=config.intervention.gd_balloon.movement.traversal,
    )
    carrier = _carrier(library, config.intervention.gd_balloon.contrast_agent.carrier_tissue)
    gd_profile = calculate_gd_bssfp_signal(
        carrier=carrier,
        concentration_mM=(
            config.intervention.gd_balloon.contrast_agent.concentration.value_mM
        ),
        flip_angle_deg=applied_flip,
        te_ms=sequence.te_ms,
        tr_ms=sequence.tr_ms,
        relaxivity_library=(
            config.intervention.gd_balloon.contrast_agent.relaxivity_library
        ),
    )
    if progress:
        progress(
            f"Gd bSSFP profile: T1={gd_profile.t1_ms:.3f} ms, "
            f"T2={gd_profile.t2_ms:.3f} ms; calculated once"
        )
    encoder = PersistentSparseRoiEncoder(
        global_shape=high_shape,
        voxel_size_mm=logical_voxel,
        coil_count=coil_info.coil_count,
        coil_roi_loader=coil_roi_loader,
        kx_per_m=scaled_k[0],
        ky_per_m=scaled_k[1],
        kz_per_m=scaled_k[2],
        acquisition_matrix_shape=reconstruction_shape,
        rf_center_shift_mm=config.sequence.rf_profile.center_shift_mm,
        rf_axis_voxel_size_mm=logical_voxel[excitation.logical_axis],
        rf_logical_axis=excitation.logical_axis,
        device_id=config.compute.device_id,
        progress=progress,
    )
    if progress:
        progress(
            "Dynamic Gd encoding: full trajectory and normalized coil maps "
            f"resident on {encoder.device_name}"
        )
    handle, output, complete = _ensure_output(plan.output_path, plan.shape, plan.schedule, overwrite=overwrite)
    output.attrs["gd_t1_ms"] = gd_profile.t1_ms
    output.attrs["gd_t2_ms"] = gd_profile.t2_ms
    output.attrs["gd_concentration_mM"] = gd_profile.concentration_mM
    output.attrs["gd_signal_model"] = "steady-state on-resonance bSSFP"
    output.attrs["gd_fov_centering"] = "rf-profile"
    output.attrs["gd_rf_center_shift_mm"] = (
        config.sequence.rf_profile.center_shift_mm
    )
    output.attrs["gd_rf_logical_axis_zero_based"] = excitation.logical_axis
    center_dataset = handle.get("balloon_center_lps_mm")
    if center_dataset is None:
        handle.create_dataset(
            "balloon_center_lps_mm",
            data=np.asarray(positions, dtype=np.float32),
        )
    elif (
        center_dataset.shape != positions.shape
        or not np.allclose(center_dataset[:], positions, rtol=0.0, atol=1e-5)
    ):
        handle.close()
        raise DynamicAcquisitionError(
            "existing balloon positions differ; use --overwrite"
        )
    generated = reused = 0
    try:
        pending = np.flatnonzero(np.asarray(complete[:]) == 0)
        if pending.size:
            if progress:
                progress(
                    "Gathering cached tissue k-space once per cardiac phase"
                )
            pending_phases = plan.schedule.cardiac_phase_index_zero_based[pending]
            for phase_zero in np.unique(pending_phases):
                simulation_trs = pending[pending_phases == phase_zero]
                trajectory_trs = (
                    plan.schedule.trajectory_tr_index_zero_based[simulation_trs]
                )
                unique_trs, inverse = np.unique(
                    trajectory_trs, return_inverse=True
                )
                with h5py.File(
                    tissue_library_frame_path(config, int(phase_zero) + 1),
                    "r",
                ) as tissue_handle:
                    block = np.asarray(
                        tissue_handle["kspace"][:, unique_trs, :],
                        dtype=np.complex64,
                    )
                output[:, simulation_trs, :] = block[:, inverse, :]
            handle.flush()
        reused = int(plan.schedule.acquisition_count - pending.size)
        for generated_index, global_tr in enumerate(pending, start=1):
            trajectory_tr = int(plan.schedule.trajectory_tr_index_zero_based[global_tr])
            tissue_arm = np.asarray(
                output[:, global_tr, :], dtype=np.complex64
            )
            support = rasterize_sparse_balloon(
                positions[global_tr], volume_shape=pcs_shape,
                voxel_size_mm=config.phantom.voxel_size_mm,
                diameter_mm=config.intervention.gd_balloon.geometry.diameter_mm,
                shape=config.intervention.gd_balloon.geometry.shape,
            )
            profile_start = int(support.bounding_box_start_ijk[pcs_axis])
            profile_stop = profile_start + support.occupancy.shape[pcs_axis]
            local_profile = gd_profile.values[profile_start:profile_stop]
            broadcast_shape = [1, 1, 1]
            broadcast_shape[pcs_axis] = local_profile.size
            local_signal = np.broadcast_to(
                local_profile.reshape(broadcast_shape),
                support.occupancy.shape,
            )
            occupied = support.occupancy > 0
            indices, values = _map_pcs_sparse_to_high(
                support.occupied_indices_ijk(),
                support.occupancy[occupied] * local_signal[occupied],
                pcs_shape=pcs_shape, pcs_to_logical=transforms.pcs_to_logical, high_shape=high_shape,
            )
            gd_encoded = encoder.encode(
                indices,
                values,
                trajectory_tr=trajectory_tr,
            ).kspace[:, 0, :]
            output[:, global_tr, :] = tissue_arm + gd_encoded
            complete[global_tr] = 1
            generated += 1
            if generated_index % plan.schedule.trs_per_frame == 0:
                handle.flush()
            if progress and (
                generated == 1
                or generated % plan.schedule.trs_per_frame == 0
            ):
                progress(f"Dynamic acquisition: {global_tr + 1}/{plan.schedule.acquisition_count} TRs")
    finally:
        handle.close()
    del encoder
    adjoint_debug_path = None
    if save_adjoint_debug:
        from xcat_icmr.acquisition.dynamic_adjoint import (
            generate_dynamic_adjoint_debug,
        )

        adjoint_debug_path = generate_dynamic_adjoint_debug(
            config,
            plan=plan,
            overwrite=overwrite,
            progress=progress,
        )
    if (
        plan.view_order_cycles is None
        and generated + reused == plan.schedule.acquisition_count
    ):
        write_artifact_manifest(
            dynamic_acquisition_cache_entry(config), status="complete",
            frame_count=plan.schedule.frame_count,
            completed_frame_indices=list(range(1, plan.schedule.frame_count + 1)),
            outputs=[plan.output_path],
        )
    return DynamicAcquisitionResult(
        plan,
        generated,
        reused,
        time.perf_counter() - started,
        adjoint_debug_path,
    )


def format_dynamic_acquisition(result: DynamicAcquisitionResult) -> str:
    return "\n".join(
        (
            "Combined dynamic multicoil acquisition",
            f"Shape:              {result.plan.shape} complex64",
            f"TRs per frame:      {result.plan.schedule.trs_per_frame}",
            f"Complete frames:    {result.plan.schedule.frame_count}",
            (
                "View-order cycles: full experiment"
                if result.plan.view_order_cycles is None
                else f"View-order cycles: {result.plan.view_order_cycles}"
            ),
            f"Estimated storage:  {result.plan.storage.gib:.2f} GiB",
            f"Generated/reused:   {result.generated_trs}/{result.reused_trs}",
            f"Elapsed:            {result.elapsed_s:.3f} s",
            f"Output:             {result.plan.output_path}",
            f"Adjoint debug:      {result.adjoint_debug_path or 'not requested'}",
        )
    )
