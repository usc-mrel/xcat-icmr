from pathlib import Path

from xcat_icmr.cache import (
    dynamic_acquisition_cache_entry,
    undersampled_acquisition_cache_entry,
)
from xcat_icmr.config import load_config
from xcat_icmr.undersampling import resolve_temporal_grouping
from xcat_icmr.undersampling import resolve_bracketing_multiples


def _config():
    return load_config(Path(__file__).parents[1] / "fixtures/valid_simulation.yaml")


def test_integer_grouping_reports_actual_time_and_dropped_tail() -> None:
    grouping = resolve_temporal_grouping(
        source_frame_duration_s=0.055,
        source_trs_per_frame=11,
        available_tr_count=1232,
        source_frames_per_output_frame=9,
    )
    assert grouping.available_source_frames == 112
    assert grouping.output_frame_count == 12
    assert grouping.retained_source_frames == 108
    assert grouping.dropped_source_frames == 4
    assert grouping.trs_per_output_frame == 99
    assert grouping.retained_tr_count == 1188
    assert grouping.output_frame_duration_s == 0.495


def test_requested_duration_resolves_floor_and_ceil_multiples() -> None:
    assert resolve_bracketing_multiples(
        target_frame_duration_s=0.300,
        source_frame_duration_s=0.055,
    ) == (5, 6)
    assert resolve_bracketing_multiples(
        target_frame_duration_s=0.275,
        source_frame_duration_s=0.055,
    ) == (5,)


def test_grouping_changes_only_derived_cache_identity() -> None:
    config = _config()
    source_id = dynamic_acquisition_cache_entry(config).cache_id
    grouped_id = undersampled_acquisition_cache_entry(
        config, source_frames_per_output_frame=5, view_order_cycles=1
    ).cache_id
    assert dynamic_acquisition_cache_entry(config).cache_id == source_id
    assert (
        undersampled_acquisition_cache_entry(
            config, source_frames_per_output_frame=6, view_order_cycles=1
        ).cache_id
        != grouped_id
    )
