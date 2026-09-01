from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from scipy.io import savemat

from xcat_icmr.coils import (
    inspect_sensitivity_map,
    load_normalized_coil,
    load_normalized_coil_in_logical_frame,
    load_normalized_coil_roi_in_logical_frame,
    prepare_rss_normalization,
    sensitivity_shape_in_logical_frame,
)
from xcat_icmr.encoding import (
    center_padding,
    prepare_contrast_for_encoding,
)
from xcat_icmr.sequence import build_coordinate_transforms


def write_legacy_complex_map(path: Path, logical: np.ndarray) -> None:
    stored = logical.transpose(3, 0, 1, 2)
    compound = np.empty(
        stored.shape,
        dtype=np.dtype([("real", "<f4"), ("imag", "<f4")]),
    )
    compound["real"] = stored.real
    compound["imag"] = stored.imag
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset(
            "sens",
            data=compound,
            chunks=(1, 2, 2, 2),
        )
        dataset.attrs["MATLAB_class"] = np.bytes_(b"single")


def test_uses_every_file_coil_and_normalizes_rss_safely(
    tmp_path: Path,
) -> None:
    logical = np.empty((3, 4, 5, 2), dtype=np.complex64)
    logical[..., 0] = 3 + 0j
    logical[..., 1] = 0 + 4j
    logical[0, 0, 0, :] = 0
    path = tmp_path / "sens.mat"
    cache = tmp_path / "rss.npy"
    write_legacy_complex_map(path, logical)

    info = inspect_sensitivity_map(path)
    report = prepare_rss_normalization(info, cache)
    first = load_normalized_coil(info, 0, report)
    second = load_normalized_coil(info, 1, report)
    normalized_rss = np.sqrt(np.abs(first) ** 2 + np.abs(second) ** 2)

    assert info.coil_count == 2
    assert info.logical_spatial_shape == (3, 4, 5)
    assert info.stored_shape == (2, 3, 4, 5)
    assert report.nonfinite_value_count == 0
    assert report.supported_voxel_count == logical[..., 0].size - 1
    assert report.background_voxel_count == 1
    assert normalized_rss[0, 0, 0] == 0
    np.testing.assert_allclose(normalized_rss[normalized_rss > 0], 1.0)
    np.testing.assert_allclose(first[1, 2, 3], 0.6 + 0j)
    np.testing.assert_allclose(second[1, 2, 3], 0 + 0.8j)


def test_reuses_matching_rss_cache(tmp_path: Path) -> None:
    logical = np.ones((2, 3, 4, 3), dtype=np.complex64)
    path = tmp_path / "sens.mat"
    cache = tmp_path / "rss.npy"
    write_legacy_complex_map(path, logical)
    info = inspect_sensitivity_map(path)

    first = prepare_rss_normalization(info, cache)
    second = prepare_rss_normalization(info, cache)

    assert not first.reused_cache
    assert second.reused_cache


def test_center_padding_matches_existing_script_convention() -> None:
    assert center_padding(
        (470, 250, 193), (500, 500, 500)
    ) == ((15, 15), (125, 125), (153, 154))


def test_prepares_float32_contrast_on_coil_grid(tmp_path: Path) -> None:
    source = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    path = tmp_path / "image.mat"
    savemat(path, {"image": source})

    prepared = prepare_contrast_for_encoding(path, (5, 8, 9))

    assert prepared.padding == ((1, 1), (2, 2), (2, 2))
    assert prepared.image.shape == (5, 8, 9)
    assert prepared.image.dtype == np.float32
    np.testing.assert_array_equal(
        prepared.image[1:4, 2:6, 2:7], source
    )


def test_reorients_declared_dcs_coil_to_cor_logical_axes(
    tmp_path: Path,
) -> None:
    logical = np.ones((3, 4, 5, 1), dtype=np.complex64)
    phases = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    logical[..., 0] = np.exp(1j * phases / 20)
    path = tmp_path / "sens.mat"
    cache = tmp_path / "rss.npy"
    write_legacy_complex_map(path, logical)
    info = inspect_sensitivity_map(path)
    normalization = prepare_rss_normalization(info, cache)
    transforms = build_coordinate_transforms(
        patient_position="HFS",
        coordinate_mode="XYZ-in-TRA",
        sequence_orientation="COR",
    )

    stored_coil = load_normalized_coil(info, 0, normalization)
    logical_coil = load_normalized_coil_in_logical_frame(
        info,
        0,
        normalization,
        stored_axis_order=("X", "Y", "Z"),
        dcs_to_logical=transforms.dcs_to_logical,
    )

    assert sensitivity_shape_in_logical_frame(
        info,
        stored_axis_order=("X", "Y", "Z"),
        dcs_to_logical=transforms.dcs_to_logical,
    ) == (5, 3, 4)
    np.testing.assert_allclose(
        logical_coil,
        np.transpose(stored_coil, (2, 0, 1)),
    )

    roi = load_normalized_coil_roi_in_logical_frame(
        info,
        0,
        normalization,
        (slice(1, 4), slice(1, 3), slice(1, 4)),
        stored_axis_order=("X", "Y", "Z"),
        dcs_to_logical=transforms.dcs_to_logical,
    )
    np.testing.assert_allclose(roi, logical_coil[1:4, 1:3, 1:4])
