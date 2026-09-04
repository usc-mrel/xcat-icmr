from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

from xcat_icmr.sequence import build_coordinate_transforms
from xcat_icmr.signal import (
    PulseqExcitation,
    SliceProfile,
    generate_rf_profile_bssfp_contrast,
    generate_slice_profile,
    simulate_bloch_profile,
)
from xcat_icmr.signal.bssfp import bssfp_signal_from_tissue_properties
from xcat_icmr.tissue import (
    LEGACY_MATLAB_055T,
    XcatLabel,
    map_labels_to_tissue_properties,
)


def make_excitation(
    rf: np.ndarray,
    gradient: np.ndarray,
    *,
    dt: float,
) -> PulseqExcitation:
    return PulseqExcitation(
        sequence_path=Path("test.seq"),
        block_index=1,
        gradient_channel="y",
        logical_axis=1,
        block_duration_s=rf.size * dt,
        raster_time_s=dt,
        rf_delay_s=0.0,
        rf_duration_s=rf.size * dt,
        rf_frequency_offset_hz=0.0,
        rf_phase_offset_rad=0.0,
        rf_waveform_hz=np.asarray(rf, dtype=np.complex128),
        gradient_waveform_hz_per_m=np.asarray(
            gradient, dtype=np.float64
        ),
        nominal_flip_angle_deg=65.0,
    )


def test_hard_pulse_bloch_profile_matches_flip_angle() -> None:
    dt = 1e-6
    sample_count = 1000
    amplitude_hz = np.deg2rad(65.0) / (
        2.0 * np.pi * sample_count * dt
    )
    rf = np.full(sample_count, amplitude_hz, dtype=np.complex128)

    mxy, mz = simulate_bloch_profile(
        rf,
        np.zeros(sample_count),
        raster_time_s=dt,
        positions_m=np.asarray((-0.1, 0.0, 0.1)),
    )

    np.testing.assert_allclose(
        np.abs(mxy), np.sin(np.deg2rad(65.0)), atol=1e-12
    )
    np.testing.assert_allclose(
        mz, np.cos(np.deg2rad(65.0)), atol=1e-12
    )


def test_zero_sampled_profile_uses_requested_matrix_size() -> None:
    dt = 1e-6
    rf = np.full(10, 1000.0, dtype=np.complex128)
    profile = generate_slice_profile(
        make_excitation(rf, np.zeros(10), dt=dt),
        matrix_size=8,
        voxel_size_mm=2.0,
    )

    np.testing.assert_array_equal(
        profile.positions_mm,
        np.asarray((-8, -6, -4, -2, 0, 2, 4, 6)),
    )
    assert profile.normalized_magnitude.shape == (8,)


def test_center_shift_modulates_only_rf_and_moves_profile_positive() -> None:
    dt = 1e-6
    sample_count = 1000
    amplitude_hz = np.deg2rad(65.0) / (
        2.0 * np.pi * sample_count * dt
    )
    excitation = make_excitation(
        np.full(sample_count, amplitude_hz, dtype=np.complex128),
        np.full(sample_count, 20_000.0),
        dt=dt,
    )

    centered = generate_slice_profile(
        excitation,
        matrix_size=101,
        voxel_size_mm=1.0,
    )
    shifted = generate_slice_profile(
        excitation,
        matrix_size=101,
        voxel_size_mm=1.0,
        center_shift_mm=10.0,
    )

    # The immutable source event and gradient are unchanged. A +10 mm RF
    # displacement makes the shifted response at r equal the base at r-10.
    np.testing.assert_array_equal(
        shifted.excitation.rf_waveform_hz,
        centered.excitation.rf_waveform_hz,
    )
    np.testing.assert_array_equal(
        shifted.excitation.gradient_waveform_hz_per_m,
        centered.excitation.gradient_waveform_hz_per_m,
    )
    np.testing.assert_allclose(
        shifted.effective_flip_angle_deg[10:],
        centered.effective_flip_angle_deg[:-10],
        atol=2e-4,
    )
    assert shifted.center_shift_mm == 10.0


def test_generates_spatially_varying_fa_bssfp(
    tmp_path: Path,
) -> None:
    labels = np.full((4, 2, 2), XcatLabel.MUSCLE, dtype=np.float32)
    properties = map_labels_to_tissue_properties(
        labels, LEGACY_MATLAB_055T
    )
    label_path = tmp_path / "labels.mat"
    savemat(label_path, {"P": labels})

    positions = np.arange(8, dtype=np.float32) - 4
    magnitude = np.linspace(0.1, 0.8, 8, dtype=np.float32)
    flips = np.linspace(5, 65, 8, dtype=np.float32)
    excitation = make_excitation(
        np.ones(2), np.zeros(2), dt=1e-6
    )
    profile = SliceProfile(
        excitation=excitation,
        positions_mm=positions,
        complex_mxy=magnitude.astype(np.complex64),
        normalized_magnitude=magnitude,
        phase_rad=np.zeros(8, dtype=np.float32),
        effective_flip_angle_deg=flips,
        fwhm_mm=4.0,
    )
    transforms = build_coordinate_transforms(
        patient_position="HFS",
        coordinate_mode="XYZ-in-TRA",
        sequence_orientation="COR",
    )
    effective_path = tmp_path / "effective.mat"

    report = generate_rf_profile_bssfp_contrast(
        label_path=label_path,
        profile=profile,
        transforms=transforms,
        pcs_voxel_size_mm=(1.0, 1.0, 1.0),
        library=LEGACY_MATLAB_055T,
        te_ms=0.7395,
        tr_ms=4.97,
        profile_output_path=tmp_path / "profile.mat",
        image_output_path=effective_path,
        chunk_slices=1,
    )

    effective_image = loadmat(effective_path)["image"]
    assert report.logical_axis == 1
    assert report.pcs_axis == 0
    assert report.patient_direction == "+Sag"
    assert report.center_shift_mm == 0.0
    assert effective_image.shape == labels.shape
    assert report.image_path == effective_path.resolve()
    expected_effective = bssfp_signal_from_tissue_properties(
        properties,
        flip_angle_deg=flips[2:6, np.newaxis, np.newaxis],
        te_ms=0.7395,
        tr_ms=4.97,
    )
    np.testing.assert_allclose(effective_image, expected_effective)
