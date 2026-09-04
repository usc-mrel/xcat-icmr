from __future__ import annotations

from pathlib import Path

from xcat_icmr.cache import (
    artifact_cache_status,
    contrast_cache_entry,
    label_cache_entry,
    stage_digest,
    stage_reuse_status,
    tissue_kspace_cache_entry,
    write_artifact_manifest,
    write_stage_manifest,
)
from xcat_icmr.config import load_config


def test_stage_digests_have_expected_dependency_boundaries(tmp_path: Path) -> None:
    config = load_config(Path(__file__).parents[1] / "fixtures/valid_simulation.yaml")
    baseline = {
        stage: stage_digest(config, stage)
        for stage in ("labels", "contrast", "fullysampled_kspace")
    }

    changed = config.model_copy(
        update={
            "undersampling": config.undersampling.model_copy(
                update={"frame_duration_s": 0.2}
            )
        }
    )
    assert stage_digest(changed, "labels") == baseline["labels"]
    assert stage_digest(changed, "contrast") == baseline["contrast"]
    assert (
        stage_digest(changed, "fullysampled_kspace")
        == baseline["fullysampled_kspace"]
    )

    shifted = config.model_copy(
        update={
            "sequence": config.sequence.model_copy(
                update={
                    "rf_profile": config.sequence.rf_profile.model_copy(
                        update={"center_shift_mm": 10.0}
                    )
                }
            )
        }
    )
    assert stage_digest(shifted, "labels") == baseline["labels"]
    assert stage_digest(shifted, "contrast") != baseline["contrast"]
    assert (
        stage_digest(shifted, "fullysampled_kspace")
        != baseline["fullysampled_kspace"]
    )


def test_manifest_requires_unchanged_outputs(tmp_path: Path) -> None:
    config = load_config(Path(__file__).parents[1] / "fixtures/valid_simulation.yaml")
    config = config.model_copy(
        update={"run": config.run.model_copy(update={"output_root": tmp_path})}
    )
    output = tmp_path / "labels" / "frame.mat"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"valid")

    write_stage_manifest(config, "labels", [output])
    assert stage_reuse_status(config, "labels").reusable

    output.write_bytes(b"changed")
    status = stage_reuse_status(config, "labels")
    assert not status.reusable
    assert status.reason.startswith("output changed")


def test_artifact_ids_ignore_run_name_and_undersampling() -> None:
    config = load_config(Path(__file__).parents[1] / "fixtures/valid_simulation.yaml")
    baseline = (
        label_cache_entry(config).cache_id,
        contrast_cache_entry(config).cache_id,
        tissue_kspace_cache_entry(config).cache_id,
    )
    changed = config.model_copy(
        update={
            "run": config.run.model_copy(
                update={"id": "another-run", "output_root": Path("outputs/other")}
            ),
            "undersampling": config.undersampling.model_copy(
                update={"frame_duration_s": 0.2}
            ),
        }
    )
    assert (
        label_cache_entry(changed).cache_id,
        contrast_cache_entry(changed).cache_id,
        tissue_kspace_cache_entry(changed).cache_id,
    ) == baseline


def test_artifact_dependency_chain_changes_only_downstream() -> None:
    config = load_config(Path(__file__).parents[1] / "fixtures/valid_simulation.yaml")
    baseline = (
        label_cache_entry(config).cache_id,
        contrast_cache_entry(config).cache_id,
        tissue_kspace_cache_entry(config).cache_id,
    )
    shifted = config.model_copy(
        update={
            "sequence": config.sequence.model_copy(
                update={
                    "rf_profile": config.sequence.rf_profile.model_copy(
                        update={"center_shift_mm": 15.0}
                    )
                }
            )
        }
    )
    shifted_ids = (
        label_cache_entry(shifted).cache_id,
        contrast_cache_entry(shifted).cache_id,
        tissue_kspace_cache_entry(shifted).cache_id,
    )
    assert shifted_ids[0] == baseline[0]
    assert shifted_ids[1] != baseline[1]
    assert shifted_ids[2] != baseline[2]

    temporally_aggregated = config.model_copy(
        update={
            "timeline": config.timeline.model_copy(
                update={"reference_time_step_s": 0.01}
            )
        }
    )
    temporal_ids = (
        label_cache_entry(temporally_aggregated).cache_id,
        contrast_cache_entry(temporally_aggregated).cache_id,
        tissue_kspace_cache_entry(temporally_aggregated).cache_id,
    )
    assert temporal_ids[0] == baseline[0]
    assert temporal_ids[1] == baseline[1]
    # The full tissue library is indexed at the XCAT cardiac-phase raster.
    # Changing only the later image-reference grouping must not invalidate it.
    assert temporal_ids[2] == baseline[2]


def test_artifact_manifest_reports_miss_partial_and_hit(tmp_path: Path) -> None:
    config = load_config(Path(__file__).parents[1] / "fixtures/valid_simulation.yaml")
    config = config.model_copy(
        update={
            "run": config.run.model_copy(
                update={"output_root": tmp_path / "run"}
            )
        }
    )
    entry = label_cache_entry(config)
    assert artifact_cache_status(entry).state == "MISS"

    output = entry.directory / "frames" / "label_frame_0001.mat"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"label")
    write_artifact_manifest(
        entry,
        status="partial",
        frame_count=2,
        completed_frame_indices=[1],
        outputs=[output],
    )
    assert artifact_cache_status(entry).state == "PARTIAL"

    write_artifact_manifest(
        entry,
        status="complete",
        frame_count=1,
        completed_frame_indices=[1],
        outputs=[output],
    )
    assert artifact_cache_status(entry).state == "HIT"
