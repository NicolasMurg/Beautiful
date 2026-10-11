"""Independent checks for the camera-measurement to playable-map boundary."""

import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from billiards_map import build_table_map, map_camera_measurements, planner_state_from_map
from billiards_table import TableCalibration


def calibrated_camera():
    angle = np.deg2rad(31.)
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])
    return TableCalibration(
        width_cm=120., height_cm=60., ball_radius_cm=2.5,
        pocket_width_cm=10., inset_cm=6.,
        camera_matrix=np.array([[710., 0., 480.], [0., 690., 360.], [0., 0., 1.]]),
        image_size=(960, 720), origin=np.array([-.6, -.3, 1.7]),
        rotation=rotation, normal_sign=-1., signature={"camera": "left-rectified", "revision": 3},
        pockets=[{"id": "TM", "x": 54., "width": 9.}, {"id": "BR", "blocked": True}],
    )


def measurement(calibration, key, x, y, *, kind="camera_z", reference="surface"):
    center = calibration.origin + calibration.rotation @ np.array([
        x / 100., y / 100., calibration.normal_sign * calibration.ball_radius_cm / 100.])
    observed = center.copy()
    if reference == "surface":
        observed -= calibration.ball_radius_cm / 100. * center / np.linalg.norm(center)
    projected = calibration.camera_matrix @ center
    return {
        "key": key,
        "pixel": {"x": float(projected[0] / projected[2]), "y": float(projected[1] / projected[2])},
        "distance_m": float(observed[2] if kind == "camera_z" else np.linalg.norm(observed)),
        "distance_kind": kind, "measurement_reference": reference,
        "distance_uncertainty_m": .002, "confidence": .92,
    }, center


def sample_map():
    calibration = calibrated_camera()
    observations = [measurement(calibration, "cue", 22., 20.)[0],
                    measurement(calibration, "object", 77., 31.)[0]]
    return map_camera_measurements(calibration, observations,
                                   calibration_signature=calibration.signature,
                                   labels={"cue": 0, "object": 3}, confirmed=True), calibration


@pytest.mark.parametrize("kind", ["camera_z", "range"])
@pytest.mark.parametrize("reference", ["surface", "center"])
def test_metric_record_to_map_preserves_center_without_second_radius_correction(kind, reference):
    calibration = calibrated_camera()
    points = [(22., 20.), (77., 31.), (54., 1.)]
    records, centers = zip(*(measurement(calibration, str(i), x, y, kind=kind, reference=reference)
                            for i, (x, y) in enumerate(points)))
    mapped = map_camera_measurements(calibration, list(records),
                                    calibration_signature=calibration.signature,
                                    labels={"0": 0, "1": 3, "2": 9}, confirmed=True)
    assert mapped["status"] == "confirmed"
    for ball, expected_xy, expected_xyz in zip(mapped["balls"], points, centers):
        assert (ball["x"], ball["y"]) == pytest.approx(expected_xy, abs=1e-10)
        assert ball["measurement"]["camera_xyz_m"] == pytest.approx(expected_xyz, abs=1e-12)
        assert ball["measurement"]["distance_kind"] == kind
        assert ball["measurement"]["measurement_reference"] == reference
    state = planner_state_from_map(mapped, "open")
    assert np.array([(ball["x"], ball["y"]) for ball in state["balls"]]) == pytest.approx(np.array(points))
    assert all(ball["position_uncertainty_cm"] > 0 for ball in state["balls"])
    assert json.loads(json.dumps(mapped, allow_nan=False)) == mapped
    pockets = {p["id"]: p for p in state["table"]["pockets"]}
    assert pockets["TM"]["x"] == 54.
    assert pockets["BR"]["blocked"] is True


@pytest.mark.parametrize("field,value", [
    ("distance_m", None), ("distance_m", True), ("distance_m", "1.7"),
    ("distance_m", float("nan")), ("distance_m", float("inf")),
    ("distance_m", 0.), ("distance_m", -1.), ("distance_m", [1.7]),
    ("distance_uncertainty_m", -.01), ("distance_uncertainty_m", True),
    ("pixel_uncertainty_px", 0.), ("distance_kind", "relative"),
    ("measurement_reference", "unknown"), ("confidence", float("nan")),
    ("pixel", {"x": None, "y": 2}),
])
def test_bad_scalar_record_stays_unmapped_and_never_implies_pocketed_ball(field, value):
    calibration = calibrated_camera()
    cue = measurement(calibration, "cue", 22., 20.)[0]
    obj = measurement(calibration, "object", 77., 31.)[0]
    obj[field] = value
    mapped = map_camera_measurements(calibration, [cue, obj],
                                    calibration_signature=calibration.signature,
                                    labels={"cue": 0, "object": 3}, confirmed=True)
    assert mapped["status"] == "invalid"
    assert mapped["complete"] is False
    assert [item["key"] for item in mapped["unmapped"]] == ["object"]
    assert "pocketed_ball_ids" not in mapped
    with pytest.raises(ValueError):
        planner_state_from_map(mapped, "solids")


@pytest.mark.parametrize("missing", ["pixel", "distance_m", "distance_kind", "measurement_reference"])
def test_incomplete_measurement_is_retained_for_review(missing):
    calibration = calibrated_camera()
    observation = measurement(calibration, "candidate", 22., 20.)[0]
    del observation[missing]
    mapped = map_camera_measurements(calibration, [observation],
                                    calibration_signature=calibration.signature)
    assert mapped["unmapped"][0]["key"] == "candidate"
    assert not mapped["balls"]
    assert not mapped["complete"]


def test_wrong_camera_calibration_cannot_reuse_other_camera_measurement():
    calibration = calibrated_camera()
    observation = measurement(calibration, "candidate", 22., 20.)[0]
    with pytest.raises(ValueError, match="signature"):
        map_camera_measurements(calibration, [observation],
                                calibration_signature={"camera": "right-rectified", "revision": 3})


def test_unknown_identity_keeps_obstacle_and_blocks_planner_even_when_confirmed_requested():
    calibration = calibrated_camera()
    measurements = [measurement(calibration, "cue", 22., 20.)[0],
                    measurement(calibration, "unknown", 77., 31.)[0]]
    mapped = map_camera_measurements(calibration, measurements,
                                    calibration_signature=calibration.signature,
                                    labels={"cue": 0}, confirmed=True)
    assert mapped["status"] == "review"
    assert len(mapped["balls"]) == 2
    assert mapped["balls"][1]["id"] is None
    assert "pocketed_ball_ids" not in mapped
    with pytest.raises(ValueError):
        planner_state_from_map(mapped, "solids")


def test_unresolved_map_cannot_bypass_review_by_changing_only_status_metadata():
    mapped, _ = sample_map()
    mapped["unmapped"] = [{"key": "missed-object", "reason": "Invalid camera depth"}]
    mapped["issues"] = ["Candidate missed-object has no table coordinates."]
    with pytest.raises(ValueError):
        planner_state_from_map(mapped, "solids")


@pytest.mark.parametrize("frame_ids", [(12, 13), (12, None)])
def test_observation_adapter_rejects_mixed_frame_records(frame_ids):
    calibration = calibrated_camera()
    observations = [measurement(calibration, "cue", 22., 20.)[0],
                    measurement(calibration, "object", 77., 31.)[0]]
    for observation, frame_id in zip(observations, frame_ids):
        if frame_id is not None:
            observation["frame_id"] = frame_id
    with pytest.raises(ValueError, match="frame"):
        map_camera_measurements(calibration, observations,
                                calibration_signature=calibration.signature,
                                labels={"cue": 0, "object": 3}, confirmed=True)


@pytest.mark.parametrize("change", ["duplicate_id", "missing_id", "invalid_coordinate", "overlap"])
def test_confirmed_map_metadata_does_not_allow_changed_invalid_layout(change):
    mapped, _ = sample_map()
    if change == "duplicate_id":
        mapped["balls"][1]["id"] = mapped["balls"][0]["id"]
    elif change == "missing_id":
        mapped["balls"][1]["id"] = None
    elif change == "invalid_coordinate":
        mapped["balls"][1]["x"] = float("nan")
    else:
        mapped["balls"][1]["x"] = mapped["balls"][0]["x"]
        mapped["balls"][1]["y"] = mapped["balls"][0]["y"]
    with pytest.raises(ValueError):
        planner_state_from_map(mapped, "solids")


def test_planner_extraction_isolated_from_map_and_does_not_pass_measurement_units_as_position():
    mapped, _ = sample_map()
    original = copy.deepcopy(mapped)
    state = planner_state_from_map(mapped, "solids")
    assert set(state["balls"][0]) == {"id", "x", "y", "position_uncertainty_cm", "confidence"}
    assert state["balls"][0]["x"] == pytest.approx(22.)
    state["balls"][0]["x"] = 100.
    state["table"]["pockets"][0]["blocked"] = True
    assert mapped == original


def test_ball_without_table_coordinates_does_not_treat_depth_as_table_position():
    calibration = calibrated_camera()
    mapped = build_table_map(calibration.table,
                             [{"key": "raw-distance", "depth_m": 1.7}],
                             {"raw-distance": 0}, confirmed=True)
    assert mapped["status"] == "invalid"
    assert not mapped["balls"]
    assert "distance alone" in mapped["unmapped"][0]["reason"]
