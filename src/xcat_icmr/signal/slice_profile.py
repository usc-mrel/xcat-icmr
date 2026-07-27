"""Pulseq excitation extraction and one-dimensional Bloch slice profiles."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile

import numpy as np


class SliceProfileError(ValueError):
    """Raised when an RF slice profile cannot be derived safely."""


@dataclass(frozen=True)
class PulseqExcitation:
    """One RF event and its simultaneously active selection gradient."""

    sequence_path: Path
    block_index: int
    gradient_channel: str
    logical_axis: int
    block_duration_s: float
    raster_time_s: float
    rf_delay_s: float
    rf_duration_s: float
    rf_frequency_offset_hz: float
    rf_phase_offset_rad: float
    rf_waveform_hz: np.ndarray
    gradient_waveform_hz_per_m: np.ndarray
    nominal_flip_angle_deg: float


@dataclass(frozen=True)
class SliceProfile:
    """Bloch-simulated excitation profile on a centered logical axis."""

    excitation: PulseqExcitation
    positions_mm: np.ndarray
    complex_mxy: np.ndarray
    normalized_magnitude: np.ndarray
    phase_rad: np.ndarray
    effective_flip_angle_deg: np.ndarray
    fwhm_mm: float
    center_shift_mm: float = 0.0


def _pypulseq_sequence_class():
    cache = Path(tempfile.gettempdir()) / "xcat-icmr-matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    try:
        from pypulseq.Sequence.sequence import Sequence
    except (ImportError, RuntimeError) as exc:
        raise SliceProfileError(f"could not import PyPulseq: {exc}") from exc
    return Sequence


def _gradient_samples(
    gradient: object,
    *,
    sample_times_s: np.ndarray,
) -> np.ndarray:
    waveform = getattr(gradient, "waveform", None)
    times = getattr(gradient, "tt", None)
    if waveform is None or times is None:
        raise SliceProfileError(
            "the selected Pulseq gradient must expose waveform and tt arrays"
        )
    values = np.asarray(waveform, dtype=np.float64).reshape(-1)
    time_points = (
        np.asarray(times, dtype=np.float64).reshape(-1)
        + float(getattr(gradient, "delay", 0.0))
    )
    if values.size != time_points.size or values.size < 2:
        raise SliceProfileError(
            "gradient waveform and time arrays must have matching lengths"
        )
    if not np.all(np.diff(time_points) > 0):
        raise SliceProfileError(
            "gradient waveform time points must be strictly increasing"
        )
    return np.interp(
        sample_times_s,
        time_points,
        values,
        left=0.0,
        right=0.0,
    )


def read_pulseq_excitation(path: str | Path) -> PulseqExcitation:
    """Read the first RF block with exactly one active gradient channel."""

    sequence_path = Path(path).expanduser().resolve(strict=False)
    if not sequence_path.is_file():
        raise SliceProfileError(
            f"Pulseq sequence does not exist: {sequence_path}"
        )
    Sequence = _pypulseq_sequence_class()
    sequence = Sequence()
    try:
        sequence.read(str(sequence_path))
    except Exception as exc:
        raise SliceProfileError(
            f"could not read Pulseq sequence {sequence_path}: {exc}"
        ) from exc

    selected: tuple[int, object, str, object] | None = None
    for block_index in sequence.block_events:
        block = sequence.get_block(int(block_index))
        rf = getattr(block, "rf", None)
        if rf is None:
            continue
        gradients = tuple(
            (channel, getattr(block, f"g{channel}", None))
            for channel in ("x", "y", "z")
        )
        gradients = tuple(item for item in gradients if item[1] is not None)
        if len(gradients) != 1:
            raise SliceProfileError(
                f"RF block {block_index} must contain exactly one selection "
                f"gradient; found {[item[0] for item in gradients]}"
            )
        selected = (int(block_index), rf, gradients[0][0], gradients[0][1])
        block_duration_s = float(block.block_duration)
        break
    if selected is None:
        raise SliceProfileError(
            f"no RF block with a selection gradient was found: {sequence_path}"
        )

    block_index, rf, channel, gradient = selected
    rf_signal = np.asarray(rf.signal, dtype=np.complex128).reshape(-1)
    rf_times = np.asarray(rf.t, dtype=np.float64).reshape(-1)
    if rf_signal.size < 2 or rf_signal.size != rf_times.size:
        raise SliceProfileError("RF signal and time arrays are inconsistent")
    raster_time_s = float(np.median(np.diff(rf_times)))
    if not np.isfinite(raster_time_s) or raster_time_s <= 0:
        raise SliceProfileError("RF raster time must be positive and finite")

    sample_count = int(np.ceil(block_duration_s / raster_time_s))
    rf_waveform = np.zeros(sample_count, dtype=np.complex128)
    start = int(round(float(rf.delay) / raster_time_s))
    stop = start + rf_signal.size
    if start < 0 or stop > sample_count:
        raise SliceProfileError(
            "RF event does not fit inside its Pulseq block"
        )
    rf_waveform[start:stop] = rf_signal * np.exp(
        1j * float(rf.phase_offset)
    )
    sample_times = (
        np.arange(sample_count, dtype=np.float64) + 0.5
    ) * raster_time_s
    gradient_waveform = _gradient_samples(
        gradient, sample_times_s=sample_times
    )
    nominal_flip_angle_deg = float(
        np.degrees(
            2.0
            * np.pi
            * np.abs(np.sum(rf_signal) * raster_time_s)
        )
    )
    return PulseqExcitation(
        sequence_path=sequence_path,
        block_index=block_index,
        gradient_channel=channel,
        logical_axis={"x": 0, "y": 1, "z": 2}[channel],
        block_duration_s=block_duration_s,
        raster_time_s=raster_time_s,
        rf_delay_s=float(rf.delay),
        rf_duration_s=float(rf.shape_dur),
        rf_frequency_offset_hz=float(rf.freq_offset),
        rf_phase_offset_rad=float(rf.phase_offset),
        rf_waveform_hz=rf_waveform,
        gradient_waveform_hz_per_m=gradient_waveform,
        nominal_flip_angle_deg=nominal_flip_angle_deg,
    )


def simulate_bloch_profile(
    rf_waveform_hz: np.ndarray,
    gradient_waveform_hz_per_m: np.ndarray,
    *,
    raster_time_s: float,
    positions_m: np.ndarray,
    rf_frequency_offset_hz: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate equilibrium magnetization through one RF/gradient block."""

    rf = np.asarray(rf_waveform_hz, dtype=np.complex128).reshape(-1)
    gradient = np.asarray(
        gradient_waveform_hz_per_m, dtype=np.float64
    ).reshape(-1)
    positions = np.asarray(positions_m, dtype=np.float64).reshape(-1)
    if rf.shape != gradient.shape or rf.size == 0:
        raise SliceProfileError(
            "RF and gradient waveforms must be non-empty and equally sized"
        )
    if (
        not np.isfinite(raster_time_s)
        or raster_time_s <= 0
        or not np.all(np.isfinite(rf))
        or not np.all(np.isfinite(gradient))
        or not np.all(np.isfinite(positions))
    ):
        raise SliceProfileError("Bloch inputs must be finite with positive dt")

    magnetization = np.zeros((positions.size, 3), dtype=np.float64)
    magnetization[:, 2] = 1.0
    for rf_sample, gradient_sample in zip(rf, gradient, strict=True):
        effective_hz = np.column_stack(
            (
                np.full(positions.size, rf_sample.real),
                np.full(positions.size, rf_sample.imag),
                gradient_sample * positions - rf_frequency_offset_hz,
            )
        )
        norm_hz = np.linalg.norm(effective_hz, axis=1)
        active = norm_hz > 0
        if not np.any(active):
            continue
        axes = np.zeros_like(effective_hz)
        axes[active] = (
            effective_hz[active] / norm_hz[active, np.newaxis]
        )
        angle = -2.0 * np.pi * norm_hz * raster_time_s
        cosine = np.cos(angle)[:, np.newaxis]
        sine = np.sin(angle)[:, np.newaxis]
        projection = np.sum(axes * magnetization, axis=1)[:, np.newaxis]
        magnetization = (
            magnetization * cosine
            + np.cross(axes, magnetization) * sine
            + axes * projection * (1.0 - cosine)
        )
    return (
        magnetization[:, 0] + 1j * magnetization[:, 1],
        magnetization[:, 2],
    )


def generate_slice_profile(
    excitation: PulseqExcitation,
    *,
    matrix_size: int,
    voxel_size_mm: float,
    center_shift_mm: float = 0.0,
) -> SliceProfile:
    """Generate a centered-grid profile with RF-only spatial displacement."""

    if matrix_size <= 0:
        raise SliceProfileError("matrix_size must be positive")
    if not np.isfinite(voxel_size_mm) or voxel_size_mm <= 0:
        raise SliceProfileError("voxel_size_mm must be positive and finite")
    if not np.isfinite(center_shift_mm):
        raise SliceProfileError("center_shift_mm must be finite")
    positions_mm = (
        np.arange(matrix_size, dtype=np.float64) - matrix_size // 2
    ) * voxel_size_mm
    # A scanner displaces the selected slab by modulating only the RF phase
    # while leaving the selection gradient and spatial coordinates fixed.
    # PyPulseq gradients use Hz/m. For a requested center r0, applying
    # phi(t) = -2*pi*r0*integral(G(t)dt) makes G(t)*(r-r0) the effective
    # spatial frequency in the RF rotating frame. The midpoint integral
    # aligns the phase with our midpoint-sampled RF/gradient arrays.
    gradient = excitation.gradient_waveform_hz_per_m
    gradient_integral = (
        np.cumsum(gradient, dtype=np.float64) - 0.5 * gradient
    ) * excitation.raster_time_s
    rf_phase_shift_rad = (
        -2.0 * np.pi * center_shift_mm * 1e-3 * gradient_integral
    )
    shifted_rf = excitation.rf_waveform_hz * np.exp(
        1j * rf_phase_shift_rad
    )
    complex_mxy, mz = simulate_bloch_profile(
        shifted_rf,
        gradient,
        raster_time_s=excitation.raster_time_s,
        positions_m=positions_mm * 1e-3,
        rf_frequency_offset_hz=excitation.rf_frequency_offset_hz,
    )
    magnitude = np.abs(complex_mxy)
    maximum = float(np.max(magnitude))
    if maximum <= 0 or not np.isfinite(maximum):
        raise SliceProfileError("Bloch simulation produced no excitation")
    normalized = magnitude / maximum
    effective_flip = np.degrees(
        np.arccos(np.clip(mz, -1.0, 1.0))
    )
    half_max = normalized >= 0.5
    fwhm_mm = (
        float(positions_mm[half_max][-1] - positions_mm[half_max][0])
        if np.any(half_max)
        else 0.0
    )
    return SliceProfile(
        excitation=excitation,
        positions_mm=positions_mm,
        complex_mxy=np.asarray(complex_mxy, dtype=np.complex64),
        normalized_magnitude=np.asarray(normalized, dtype=np.float32),
        phase_rad=np.asarray(np.angle(complex_mxy), dtype=np.float32),
        effective_flip_angle_deg=np.asarray(
            effective_flip, dtype=np.float32
        ),
        fwhm_mm=fwhm_mm,
        center_shift_mm=float(center_shift_mm),
    )
