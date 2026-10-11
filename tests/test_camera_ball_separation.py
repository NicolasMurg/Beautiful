"""Ball spacing is invariant to camera position, but needs image bearings."""

import copy

import numpy as np
import pytest

from billiards_table import camera_ball_separation
from billiards_map import build_table_map


K = np.array([[550., 0., 320.], [0., 540., 240.], [0., 0., 1.]])
RADIUS_CM = 2.5


def observation(center, kind="camera_z", reference="center"):
    measured = np.asarray(center, float)
    if reference == "surface":
        measured = measured - RADIUS_CM / 100 * measured / np.linalg.norm(measured)
    pixel = K @ measured
    return {"pixel": {"x": float(pixel[0] / pixel[2]), "y": float(pixel[1] / pixel[2])},
            "distance_m": float(measured[2] if kind == "camera_z" else np.linalg.norm(measured)),
            "distance_kind": kind, "measurement_reference": reference,
            "camera_id": "left-rectified", "frame_id": 41}


@pytest.mark.parametrize("kind", ["camera_z", "range"])
@pytest.mark.parametrize("reference", ["surface", "center"])
@pytest.mark.parametrize("angle", [0., 1.05])
def test_ball_spacing_does_not_require_table_origin_or_camera_mounting_distance(kind, reference, angle):
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])
    for camera_offset in (np.array([0.2, -0.3, 1.4]), np.array([-0.6, 0.1, 2.8])):
        a = rotation @ np.array([0.3, 0.2, 0.025]) + camera_offset
        b = rotation @ np.array([0.6, 0.6, 0.025]) + camera_offset
        result = camera_ball_separation(observation(a, kind, reference), observation(b, kind, reference),
                                        K, ball_radius_cm=RADIUS_CM)
        assert result["center_distance_cm"] == pytest.approx(50.)
        assert result["surface_gap_cm"] == pytest.approx(45.)


def test_equal_camera_distances_can_describe_widely_separated_balls():
    first = observation([0., 0., 2.], kind="range")
    second = observation([1.2, 0., 1.6], kind="range")
    assert first["distance_m"] == second["distance_m"] == 2.
    result = camera_ball_separation(first, second, K, ball_radius_cm=RADIUS_CM)
    assert result["center_distance_cm"] == pytest.approx(np.sqrt(1.6) * 100)
    assert result["center_distance_cm"] != abs(first["distance_m"] - second["distance_m"]) * 100


def test_distance_only_measurements_are_rejected():
    first, second = observation([0., 0., 2.]), observation([0.5, 0., 2.])
    first.pop("pixel")
    with pytest.raises(ValueError, match="image position"):
        camera_ball_separation(first, second, K, ball_radius_cm=RADIUS_CM)


@pytest.mark.parametrize("field,value", [("camera_id", "right"), ("frame_id", 42),
                                         ("calibration_signature", "different-calibration")])
def test_cannot_mix_camera_origins_or_frames_without_an_extrinsic_transform(field, value):
    first, second = observation([0., 0., 2.]), observation([0.5, 0., 2.])
    second[field] = value
    with pytest.raises(ValueError, match="same"):
        camera_ball_separation(first, second, K, ball_radius_cm=RADIUS_CM)


def test_map_exports_current_pair_distances_without_a_pocket_history():
    table = {"width": 120., "height": 60., "ball_radius": 2.5, "pocket_width": 10.}
    candidates = [{"key": "a", "position": {"x": 30., "y": 10., "position_uncertainty_cm": 0.1}},
                  {"key": "b", "position": {"x": 60., "y": 50., "position_uncertainty_cm": 0.2}}]
    before = copy.deepcopy(candidates)
    result = build_table_map(table, candidates, labels={"a": 0, "b": 9}, confirmed=True)
    assert result["status"] == "confirmed"
    assert result["ball_separations"] == [{"keys": ["a", "b"], "ball_ids": [0, 9],
                                           "center_distance_cm": 50., "surface_gap_cm": 45.,
                                           "position_margin_cm": pytest.approx(0.3)}]
    assert "pocketed_ball_ids" not in result
    assert candidates == before
