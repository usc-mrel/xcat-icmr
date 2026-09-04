from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from xcat_icmr.acquisition.schedule import (
    AcquisitionScheduleError,
    build_acquisition_schedule,
    load_view_order,
)
from xcat_icmr.acquisition.storage import (
    estimate_dynamic_acquisition_storage,
    estimate_tissue_library_storage,
)
from xcat_icmr.config import load_config


def _config():
    return load_config(Path(__file__).parents[1] / "fixtures/valid_simulation.yaml")


def test_schedule_snaps_close_tr_and_drops_incomplete_tail(tmp_path: Path) -> None:
    config = _config()
    order_path = tmp_path / "order.csv"
    order_path.write_text(
        "trajectory_tr_index_zero_based\n2\n0\n1\n", encoding="utf-8"
    )
    config = config.model_copy(
        update={
            "timeline": config.timeline.model_copy(update={"duration_s": 0.119}),
            "acquisition": config.acquisition.model_copy(
                update={
                    "frame_duration_s": 0.055,
                    "view_order": config.acquisition.view_order.model_copy(
                        update={"file": order_path}
                    ),
                }
            ),
        }
    )
    result = build_acquisition_schedule(
        config,
        actual_tr_s=0.00497,
        trajectory_tr_count=3,
        cardiac_phase_count=200,
    )
    assert result.effective_tr_s == pytest.approx(0.005)
    assert result.trs_per_frame == 11
    assert result.frame_count == 2
    assert result.acquisition_count == 22
    assert result.dropped_duration_s == pytest.approx(0.009)
    assert result.trajectory_tr_index_zero_based[:6].tolist() == [2, 0, 1, 2, 0, 1]
    assert result.view_order_cycle_length == 3
    assert result.complete_view_order_cycles == 7
    assert result.partial_view_order_cycle_tr_count == 1
    assert result.cardiac_phase_index_zero_based[:3].tolist() == [0, 1, 2]


def test_view_order_allows_repeated_and_omitted_indices(tmp_path: Path) -> None:
    path = tmp_path / "repeated.csv"
    path.write_text("tr\n2\n2\n0\n", encoding="utf-8")
    result = load_view_order(path, variable="tr", trajectory_tr_count=4)
    assert result.tolist() == [2, 2, 0]


@pytest.mark.parametrize("invalid_index", [-1, 4])
def test_view_order_rejects_out_of_range_indices(
    tmp_path: Path, invalid_index: int
) -> None:
    path = tmp_path / "bad.csv"
    path.write_text(f"tr\n0\n{invalid_index}\n", encoding="utf-8")
    with pytest.raises(AcquisitionScheduleError, match="between 0 and 3"):
        load_view_order(path, variable="tr", trajectory_tr_count=4)


def test_view_order_supported_file_formats(tmp_path: Path) -> None:
    expected = np.asarray([2, 0, 2], dtype=np.int64)
    csv_path = tmp_path / "order.csv"
    txt_path = tmp_path / "order.txt"
    npy_path = tmp_path / "order.npy"
    mat_path = tmp_path / "order.mat"
    csv_path.write_text("tr\n2\n0\n2\n", encoding="utf-8")
    np.savetxt(txt_path, expected, fmt="%d")
    np.save(npy_path, expected.reshape(1, -1))
    savemat(mat_path, {"tr": expected.reshape(-1, 1)})
    for path in (csv_path, txt_path, npy_path, mat_path):
        result = load_view_order(path, variable="tr", trajectory_tr_count=3)
        assert result.tolist() == expected.tolist()


def test_one_view_order_cycle_sets_exact_debug_duration(tmp_path: Path) -> None:
    config = _config()
    order_path = tmp_path / "cycle.csv"
    order_path.write_text(
        "trajectory_tr_index_zero_based\n2\n0\n1\n",
        encoding="utf-8",
    )
    config = config.model_copy(
        update={
            "acquisition": config.acquisition.model_copy(
                update={
                    "frame_duration_s": 0.015,
                    "view_order": config.acquisition.view_order.model_copy(
                        update={"file": order_path}
                    ),
                }
            )
        }
    )
    result = build_acquisition_schedule(
        config,
        actual_tr_s=0.00497,
        trajectory_tr_count=3,
        cardiac_phase_count=200,
        view_order_cycles=1,
    )
    assert result.acquisition_count == 3
    assert result.frame_count == 1
    assert result.retained_duration_s == pytest.approx(0.015)
    assert result.trajectory_tr_index_zero_based.tolist() == [2, 0, 1]


def test_storage_estimates_are_complex64() -> None:
    tissue = estimate_tissue_library_storage(1500, 1232, 16, 200)
    dynamic = estimate_dynamic_acquisition_storage(1500, 11_407, 16)
    assert tissue.dtype == "complex64"
    assert tissue.bytes == 1500 * 1232 * 16 * 200 * 8
    assert dynamic.bytes == 1500 * 11_407 * 16 * 8
