"""Independent regressions for geometric input boundaries and billiards paths."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from billiards import normalize_pockets, plan_shots
from billiards_table import TableCalibration


def test_profile_rejects_dimensions_that_planner_cannot_use():
    """Pocket gaps can fit even when the table is too narrow for the planner."""
    with pytest.raises(ValueError, match="small|diameter|radius|fit"):
        TableCalibration(
            width_cm=80, height_cm=20, ball_radius_cm=6, pocket_width_cm=13,
            inset_cm=1, camera_matrix=np.array([[700., 0., 487.], [0., 715., 351.], [0., 0., 1.]]),
            image_size=(1024, 768), origin=np.array([-0.4, -0.1, 1.65]),
            rotation=np.eye(3), normal_sign=-1, signature={})


@pytest.mark.parametrize("invalid_id", [[], {}, ["TM"]])
def test_unhashable_pocket_ids_raise_validation_error(invalid_id):
    with pytest.raises(ValueError, match="Pocket IDs"):
        normalize_pockets({"width": 254., "height": 127., "ball_radius": 2.8575,
                           "pocket_width": 11.5, "pockets": [{"id": invalid_id}]})


def _state(balls):
    return {"table": {"width": 254., "height": 127., "ball_radius": 2.8575,
                      "pocket_width": 11.5}, "group": "solids", "balls": balls}


def test_post_impact_collision_accounts_for_obstacle_position_uncertainty():
    state = _state([{"id": 0, "x": 100., "y": 100.},
                    {"id": 1, "x": 127., "y": 50.},
                    {"id": 9, "x": 145., "y": 62., "position_uncertainty_cm": 1.}])
    shot = next(shot for shot in plan_shots(state)["shots"] if shot["id"] == "1-TM")
    # The nominal continuation is y=55.715: 6.285 cm from ball 9's center.
    # This clears the nominal diameter but intersects the supplied 1 cm envelope.
    assert shot["scratch_risk"]["event"] == "ball_collision"
    assert shot["scratch_risk"]["level"] == "unknown"
    assert shot["scratch_risk"]["stop"] is None


def test_immediate_post_impact_rail_contact_does_not_escape_table():
    state = _state([{"id": 0, "x": 50., "y": 25.},
                    {"id": 1, "x": 84.65229719023142, "y": 6.176740253492252}])
    shot = next(shot for shot in plan_shots(state)["shots"] if shot["id"] == "1-BR")
    assert shot["ghost"]["y"] == pytest.approx(2.8575)
    for point in shot["scratch_risk"]["cue_path"]:
        assert 0 <= point["x"] <= 254
        assert 0 <= point["y"] <= 127


def test_corner_scratch_stops_at_first_physical_edge():
    state = _state([{"id": 0, "x": 76.07487493331391, "y": 55.22121973545772},
                    {"id": 1, "x": 210.73881460481795, "y": 76.61665989278865}])
    shot = next(shot for shot in plan_shots(state)["shots"] if shot["id"] == "1-TR")
    assert shot["scratch_risk"]["event"] == "scratch"
    assert shot["scratch_risk"]["pocket_id"] == "TL"
    last = shot["scratch_risk"]["cue_path"][-1]
    assert last["x"] == pytest.approx(0)
    assert 0 <= last["y"] <= 127


def test_measured_mouth_widths_override_unused_default_opening_geometry():
    state = {"table": {"width": 80., "height": 40., "ball_radius": 2.,
                       "pocket_width": 30., "pockets": [
                           {"id": pocket_id, "width": 8.}
                           for pocket_id in ("TL", "TM", "TR", "BL", "BM", "BR")]},
             "group": "solids", "balls": [{"id": 0, "x": 40., "y": 30.},
                                             {"id": 1, "x": 40., "y": 15.}]}
    assert len(normalize_pockets(state["table"])) == 6
    result = plan_shots(state)
    assert any(shot["id"] == "1-TM" for shot in result["shots"])


@pytest.mark.parametrize("default_width", [3., 50.])
def test_profile_and_planner_both_use_fully_overridden_mouths(default_width):
    calibration = TableCalibration(
        width_cm=80, height_cm=40, ball_radius_cm=2, pocket_width_cm=default_width,
        inset_cm=1, camera_matrix=np.array([[700., 0., 487.], [0., 715., 351.], [0., 0., 1.]]),
        image_size=(1024, 768), origin=np.array([-0.4, -0.2, 1.65]),
        rotation=np.eye(3), normal_sign=-1, signature={},
        pockets=[{"id": key, "width": 8.} for key in ("TL", "TM", "TR", "BL", "BM", "BR")])
    result = plan_shots({"table": calibration.table, "group": "solids",
                         "balls": [{"id": 0, "x": 40., "y": 30.},
                                   {"id": 1, "x": 40., "y": 15.}]})
    assert result["status"] == "ok"
