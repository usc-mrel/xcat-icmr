from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy.io import savemat

from xcat_icmr.config import load_config
from xcat_icmr.encoding.tissue_cache import (
    TissueKspaceCacheError,
    _average_temporal_images,
    _build_temporal_groups,
    _ensure_reference_datasets,
    _valid_kspace,
)
from xcat_icmr.phantom import plan_xcat_frames


FIXTURE = Path(__file__).parents[1] / "fixtures" / "valid_simulation.yaml"


def test_valid_kspace_requires_one_complex64_variable(tmp_path: Path) -> None:
    shape = (3, 4, 2)
    path = tmp_path / "frame.mat"
    savemat(path, {"kspace": np.zeros(shape, dtype=np.complex64)})

    assert _valid_kspace(path, shape)


def test_valid_kspace_rejects_wrong_shape_or_extra_variables(
    tmp_path: Path,
) -> None:
    path = tmp_path / "frame.mat"
    savemat(
        path,
        {
            "kspace": np.zeros((3, 4, 2), dtype=np.complex64),
            "time_s": np.asarray([[0.0]]),
        },
    )

    assert not _valid_kspace(path, (3, 4, 2))
    assert not _valid_kspace(tmp_path / "missing.mat", (3, 4, 2))


def test_creates_one_resumable_complex_4d_reference(tmp_path: Path) -> None:
    path = tmp_path / "reference.h5"
    with h5py.File(path, "a") as handle:
        image, complete = _ensure_reference_datasets(
            handle, (4, 5, 6, 7), overwrite=False
        )
        assert image.shape == (4, 5, 6, 7)
        assert image.dtype == np.dtype(np.complex64)
        assert complete.shape == (7,)
        assert complete.dtype == np.dtype(np.uint8)
        complete[3] = 1

    with h5py.File(path, "a") as handle:
        _, complete = _ensure_reference_datasets(
            handle, (4, 5, 6, 7), overwrite=False
        )
        assert complete[3] == 1


def test_groups_xcat_frames_on_the_configured_kspace_time_grid() -> None:
    config = load_config(FIXTURE)
    frames = plan_xcat_frames(config, debug_one_frame=False)
    groups = _build_temporal_groups(
        frames.frames,
        frames_per_group=config.timeline.xcat_frames_per_reference_frame,
        xcat_time_step_s=config.timeline.xcat_time_step_s,
    )

    assert len(groups) == 20
    assert tuple(frame.index for frame in groups[0].xcat_frames) == tuple(
        range(1, 11)
    )
    assert groups[0].window_start_s == pytest.approx(0.0)
    assert groups[0].window_end_s == pytest.approx(0.05)
    assert groups[0].representative_time_s == pytest.approx(0.025)


def test_rejects_nonuniform_final_temporal_group() -> None:
    config = load_config(FIXTURE)
    frames = plan_xcat_frames(config, debug_one_frame=False)

    with pytest.raises(TissueKspaceCacheError, match="cannot be divided"):
        _build_temporal_groups(
            frames.frames,
            frames_per_group=11,
            xcat_time_step_s=0.005,
        )


def test_averages_temporal_images_without_changing_signal_scale() -> None:
    result = _average_temporal_images(
        iter(
            (
                np.full((2, 3, 4), 2.0, dtype=np.float32),
                np.full((2, 3, 4), 4.0, dtype=np.float32),
            )
        )
    )

    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, np.full((2, 3, 4), 3.0))
