from __future__ import annotations

from pathlib import Path

from xcat_icmr.cache import (
    stage_digest,
    stage_reuse_status,
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
