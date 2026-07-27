from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import savemat

from xcat_icmr.exporting import export_contrast_series_nrrd


def test_streams_frames_with_first_spatial_axis_fastest(
    tmp_path: Path,
) -> None:
    first = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    second = first + 100
    paths = []
    for index, image in enumerate((first, second), start=1):
        path = tmp_path / f"frame_{index}.mat"
        savemat(path, {"image": image})
        paths.append(path)

    destination = tmp_path / "series.nrrd"
    report = export_contrast_series_nrrd(
        paths,
        destination,
        voxel_size_mm=(1.0, 2.0, 3.0),
        time_step_s=0.005,
    )

    content = destination.read_bytes()
    header, payload = content.split(b"\n\n", maxsplit=1)
    assert b"sizes: 2 3 4 2" in header
    assert b"space: left-posterior-superior" in header
    assert b"xcat_icmr_time_step_s:=0.005" in header
    values = np.frombuffer(payload, dtype="<f4")
    np.testing.assert_array_equal(values[:24], first.ravel(order="F"))
    np.testing.assert_array_equal(values[24:], second.ravel(order="F"))
    assert report.spatial_shape == (2, 3, 4)
    assert report.frame_count == 2
    assert report.data_size_bytes == 2 * 3 * 4 * 2 * 4
