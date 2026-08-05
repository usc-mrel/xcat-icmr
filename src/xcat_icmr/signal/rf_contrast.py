"""Comparison bSSFP volumes using a Pulseq-derived excitation profile."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile

import numpy as np
from scipy.io import loadmat, savemat

from xcat_icmr.sequence.orientation import CoordinateTransforms
from xcat_icmr.signal.bssfp import bssfp_signal_from_tissue_properties
from xcat_icmr.signal.slice_profile import SliceProfile
from xcat_icmr.tissue import TissueLibrary, map_labels_to_tissue_properties


class RfProfileContrastError(ValueError):
    """Raised when RF-profile comparison contrasts cannot be generated."""


@dataclass(frozen=True)
class RfProfileContrastGeneration:
    """Verified Pulseq profile and spatially varying-FA bSSFP output."""

    profile_path: Path
    image_path: Path
    image_shape: tuple[int, int, int]
    full_profile_length: int
    applied_profile_length: int
    logical_axis: int
    pcs_axis: int
    patient_direction: str
    center_shift_mm: float
    signal_range: tuple[float, float]
    fwhm_mm: float


def _load_required_array(
    path: str | Path,
    variable_name: str,
) -> tuple[Path, np.ndarray]:
    resolved = Path(path).expanduser().resolve(strict=False)
    if not resolved.is_file():
        raise RfProfileContrastError(f"input file does not exist: {resolved}")
    try:
        content = loadmat(
            resolved,
            variable_names=[variable_name],
            squeeze_me=False,
        )
    except (OSError, ValueError, NotImplementedError) as exc:
        raise RfProfileContrastError(
            f"could not read MATLAB file {resolved}: {exc}"
        ) from exc
    if variable_name not in content:
        raise RfProfileContrastError(
            f"MATLAB file {resolved} is missing {variable_name!r}"
        )
    return resolved, np.asarray(content[variable_name])


def _write_verified_mat(
    path: str | Path,
    variables: dict[str, object],
    *,
    expected_image_shape: tuple[int, int, int] | None = None,
    overwrite: bool,
) -> Path:
    destination = Path(path).expanduser().resolve(strict=False)
    if destination.exists() and not overwrite:
        raise RfProfileContrastError(
            f"output already exists: {destination}; pass --overwrite"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        savemat(
            temporary_path,
            variables,
            appendmat=False,
            do_compression=False,
        )
        names = ["image"] if expected_image_shape is not None else list(variables)
        reopened = loadmat(
            temporary_path,
            variable_names=names,
            squeeze_me=False,
        )
        if expected_image_shape is not None:
            image = reopened.get("image")
            if image is None or image.shape != expected_image_shape:
                raise RfProfileContrastError(
                    "saved RF-profile contrast shape changed during writing"
                )
            if image.dtype != np.dtype(np.float32):
                raise RfProfileContrastError(
                    "saved RF-profile contrast dtype is not float32"
                )
        temporary_path.chmod(0o644)
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination


def _profile_on_pcs_axis(
    profile: SliceProfile,
    transforms: CoordinateTransforms,
    *,
    pcs_shape: tuple[int, int, int],
    pcs_voxel_size_mm: tuple[float, float, float],
) -> tuple[int, np.ndarray, np.ndarray]:
    logical_axis = profile.excitation.logical_axis
    row = transforms.pcs_to_logical[logical_axis]
    pcs_axis = int(np.argmax(np.abs(row)))
    sign = float(row[pcs_axis])
    pcs_positions_mm = (
        np.arange(pcs_shape[pcs_axis], dtype=np.float64)
        - pcs_shape[pcs_axis] // 2
    ) * pcs_voxel_size_mm[pcs_axis]
    logical_positions_mm = sign * pcs_positions_mm
    lower = float(profile.positions_mm[0])
    upper = float(profile.positions_mm[-1])
    if np.any(logical_positions_mm < lower) or np.any(
        logical_positions_mm > upper
    ):
        raise RfProfileContrastError(
            "the full logical slice profile does not cover the PCS image grid"
        )
    magnitude = np.interp(
        logical_positions_mm,
        profile.positions_mm,
        profile.normalized_magnitude,
    ).astype(np.float32)
    effective_flip = np.interp(
        logical_positions_mm,
        profile.positions_mm,
        profile.effective_flip_angle_deg,
    ).astype(np.float32)
    return pcs_axis, magnitude, effective_flip


def generate_rf_profile_bssfp_contrast(
    *,
    label_path: str | Path,
    profile: SliceProfile,
    transforms: CoordinateTransforms,
    pcs_voxel_size_mm: tuple[float, float, float],
    library: TissueLibrary,
    te_ms: float,
    tr_ms: float,
    profile_output_path: str | Path,
    image_output_path: str | Path,
    chunk_slices: int = 8,
    overwrite: bool = False,
    write_profile: bool = True,
) -> RfProfileContrastGeneration:
    """Generate the sole bSSFP image using the local Pulseq flip angle."""

    if chunk_slices <= 0:
        raise RfProfileContrastError("chunk_slices must be positive")
    _, labels = _load_required_array(label_path, "P")
    if labels.ndim != 3:
        raise RfProfileContrastError(
            f"label image must be three-dimensional; got {labels.shape}"
        )
    pcs_shape = tuple(int(value) for value in labels.shape)
    pcs_axis, magnitude_profile, effective_flip_profile = (
        _profile_on_pcs_axis(
            profile,
            transforms,
            pcs_shape=pcs_shape,
            pcs_voxel_size_mm=pcs_voxel_size_mm,
        )
    )
    broadcast_shape = [1, 1, 1]
    broadcast_shape[pcs_axis] = pcs_shape[pcs_axis]
    flip_grid = effective_flip_profile.reshape(broadcast_shape)

    effective_flip = np.empty(pcs_shape, dtype=np.float32)
    try:
        for start in range(0, pcs_shape[2], chunk_slices):
            stop = min(start + chunk_slices, pcs_shape[2])
            properties = map_labels_to_tissue_properties(
                labels[:, :, start:stop],
                library,
                dtype=np.float32,
            )
            local_flip = (
                flip_grid[:, :, start:stop]
                if pcs_axis == 2
                else flip_grid
            )
            effective_flip[:, :, start:stop] = (
                bssfp_signal_from_tissue_properties(
                    properties,
                    flip_angle_deg=local_flip,
                    te_ms=te_ms,
                    tr_ms=tr_ms,
                    off_resonance_enabled=False,
                    dtype=np.float32,
                )
            )
    except ValueError as exc:
        raise RfProfileContrastError(str(exc)) from exc

    if not np.all(np.isfinite(effective_flip)):
        raise RfProfileContrastError(
            "RF-profile contrast contains non-finite values"
        )

    profile_variables = {
            "positions_mm": profile.positions_mm.astype(np.float32),
            "complex_mxy": profile.complex_mxy,
            "normalized_magnitude": profile.normalized_magnitude,
            "phase_rad": profile.phase_rad,
            "effective_flip_angle_deg": (
                profile.effective_flip_angle_deg
            ),
            "fwhm_mm": np.asarray([[profile.fwhm_mm]], dtype=np.float32),
            "nominal_flip_angle_deg": np.asarray(
                [[profile.excitation.nominal_flip_angle_deg]],
                dtype=np.float32,
            ),
            "rf_center_shift_mm": np.asarray(
                [[profile.center_shift_mm]], dtype=np.float32
            ),
            "gradient_channel": profile.excitation.gradient_channel,
            "logical_axis_zero_based": np.asarray(
                [[profile.excitation.logical_axis]], dtype=np.int32
            ),
            "pcs_axis_zero_based": np.asarray([[pcs_axis]], dtype=np.int32),
            "logical_axis_patient_direction": (
                transforms.logical_axis_patient_directions[
                    profile.excitation.logical_axis
                ]
            ),
            "pcs_to_dcs": transforms.pcs_to_dcs,
            "logical_to_dcs": transforms.logical_to_dcs,
            "pcs_to_logical": transforms.pcs_to_logical,
            "applied_normalized_magnitude": magnitude_profile,
            "applied_effective_flip_angle_deg": effective_flip_profile,
            "pcs_image_shape": np.asarray(pcs_shape, dtype=np.int32),
        }
    resolved_profile_path = Path(profile_output_path).expanduser().resolve(
        strict=False
    )
    profile_path = (
        _write_verified_mat(
            resolved_profile_path,
            profile_variables,
            overwrite=overwrite,
        )
        if write_profile
        else resolved_profile_path
    )
    image_path = _write_verified_mat(
        image_output_path,
        {"image": effective_flip},
        expected_image_shape=pcs_shape,
        overwrite=overwrite,
    )
    return RfProfileContrastGeneration(
        profile_path=profile_path,
        image_path=image_path,
        image_shape=pcs_shape,
        full_profile_length=profile.positions_mm.size,
        applied_profile_length=magnitude_profile.size,
        logical_axis=profile.excitation.logical_axis,
        pcs_axis=pcs_axis,
        patient_direction=(
            transforms.logical_axis_patient_directions[
                profile.excitation.logical_axis
            ]
        ),
        center_shift_mm=profile.center_shift_mm,
        signal_range=(
            float(np.min(effective_flip)),
            float(np.max(effective_flip)),
        ),
        fwhm_mm=profile.fwhm_mm,
    )


def format_rf_profile_contrast_generation(
    report: RfProfileContrastGeneration,
) -> str:
    """Format the sole verified spatially varying-FA bSSFP output."""

    rf_direction = {
        "Sag": "LR",
        "Cor": "AP",
        "Tra": "SI",
    }[report.patient_direction[1:]]
    return "\n".join(
        (
            "Pulseq RF-profile bSSFP contrast",
            f"Image frame:             XCAT PCS [Sag, Cor, Tra]",
            f"Image shape:             {report.image_shape}",
            f"RF logical axis:         {report.logical_axis} (zero-based)",
            f"Applied PCS axis:        {report.pcs_axis} (zero-based)",
            (
                "Patient direction:       "
                f"{rf_direction} ({report.patient_direction})"
            ),
            f"RF center shift:         {report.center_shift_mm:g} mm",
            f"Full logical profile:    {report.full_profile_length} samples",
            f"Applied cropped profile: {report.applied_profile_length} samples",
            f"Profile FWHM:            {report.fwhm_mm:g} mm",
            f"Profile metadata:        {report.profile_path}",
            f"Contrast image:          {report.image_path}",
            (
                "  signal range:          "
                f"{report.signal_range[0]:g} to "
                f"{report.signal_range[1]:g}"
            ),
            "Verification:             PASS",
        )
    )
