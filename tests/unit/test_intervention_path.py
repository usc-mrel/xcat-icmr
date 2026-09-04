from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from xcat_icmr.config.models import SimulationConfig
from xcat_icmr.intervention import (
    AUTO_DURATION_REFERENCE_SPEED_CM_PER_S,
    BalloonPathError,
    cubic_path_length_mm,
    interpolate_cubic_arc_length,
    load_balloon_path,
    resolve_simulation_duration,
)


FIXTURE = Path(__file__).parents[1] / "fixtures" / "valid_simulation.yaml"


def _config_with_path(tmp_path: Path, positions: list[list[float]]) -> SimulationConfig:
    path = tmp_path / "path.json"
    path.write_text(
        json.dumps(
            {
                "markups": [
                    {
                        "type": "Curve",
                        "coordinateSystem": "LPS",
                        "coordinateUnits": "mm",
                        "controlPoints": [
                            {"position": position} for position in positions
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    data = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    data["timeline"]["duration_s"] = "auto"
    balloon = data["intervention"]["gd_balloon"]
    balloon["enabled"] = True
    balloon["path"]["control_points_file"] = str(path)
    return SimulationConfig.model_validate(data)


def test_auto_duration_uses_fixed_half_cm_per_second(tmp_path: Path) -> None:
    config = _config_with_path(tmp_path, [[0, 0, 0], [10, 0, 0]])
    config.intervention.gd_balloon.movement.velocity_cm_per_s = 2.0

    result = resolve_simulation_duration(config)

    assert AUTO_DURATION_REFERENCE_SPEED_CM_PER_S == 0.5
    assert result.automatic
    assert result.path_length_mm == pytest.approx(10.0)
    assert result.duration_s == pytest.approx(2.0)
    assert result.reference_speed_cm_per_s == pytest.approx(0.5)


def test_numeric_duration_does_not_read_balloon_path() -> None:
    data = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    config = SimulationConfig.model_validate(data)

    result = resolve_simulation_duration(config)

    assert not result.automatic
    assert result.duration_s == pytest.approx(1.0)
    assert result.path_length_mm is None


def test_cubic_path_length_for_straight_points() -> None:
    positions = np.asarray([[0, 0, 0], [4, 0, 0], [10, 0, 0]])
    assert cubic_path_length_mm(positions) == pytest.approx(10.0)


def test_auto_duration_rejects_zero_length_path(tmp_path: Path) -> None:
    config = _config_with_path(tmp_path, [[1, 2, 3], [1, 2, 3]])
    with pytest.raises(BalloonPathError, match="positive finite length"):
        resolve_simulation_duration(config)


def test_loads_ras_markup_as_lps(tmp_path: Path) -> None:
    path = tmp_path / "ras.json"
    path.write_text(
        json.dumps(
            {
                "markups": [
                    {
                        "type": "Curve",
                        "coordinateSystem": "RAS",
                        "coordinateUnits": "mm",
                        "controlPoints": [
                            {"position": [1, 2, 3]},
                            {"position": [4, 5, 6]},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    loaded = load_balloon_path(path)
    np.testing.assert_array_equal(
        loaded.control_points_lps_mm,
        [[-1, -2, 3], [-4, -5, 6]],
    )


def test_positions_follow_constant_physical_speed_and_hold_last() -> None:
    path = interpolate_cubic_arc_length(
        np.asarray([[0, 0, 0], [10, 0, 0], [20, 0, 0]])
    )
    positions = path.positions_at_times_s(
        np.asarray([0.0, 1.0, 2.0, 5.0]),
        velocity_cm_per_s=0.5,
        start_time_s=1.0,
    )
    np.testing.assert_allclose(
        positions,
        [[0, 0, 0], [0, 0, 0], [5, 0, 0], [20, 0, 0]],
        atol=1e-6,
    )


def test_round_trip_returns_to_start_without_looping() -> None:
    path = interpolate_cubic_arc_length(
        np.asarray([[0, 0, 0], [5, 0, 0], [10, 0, 0]])
    )
    positions = path.positions_at_times_s(
        np.asarray([0.0, 0.5, 1.0, 1.5, 2.0, 3.0]),
        velocity_cm_per_s=1.0,
        start_time_s=0.0,
        traversal="round-trip",
    )
    np.testing.assert_allclose(
        positions,
        [[0, 0, 0], [5, 0, 0], [10, 0, 0], [5, 0, 0], [0, 0, 0], [0, 0, 0]],
        atol=1e-6,
    )
