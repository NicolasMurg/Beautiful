"""Four-ball calibration defaults, automatic references and persistent alignment."""

import copy
import itertools
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import calibrate_table as alignment
import map_table_state as mapping
from billiards_table import TableCalibration
from test_ball_alignment import (GAP, HEIGHT, K, POCKET, RADIUS, SIGNATURE, SIZE,
                                 WIDTH, observation, scene)


DIMENSIONS = ["--table-length-cm", str(WIDTH), "--table-width-cm", str(HEIGHT),
              "--ball-diameter-cm", str(RADIUS * 2), "--pocket-width-cm", str(POCKET)]


def pixel_dict(record):
    record = copy.deepcopy(record)
    record["pixel"] = dict(zip(("x", "y"), record["pixel"]))
    return record


def test_default_is_three_inch_ball_surface_clearance_without_marker_inset():
    args = alignment.build_parser().parse_args(DIMENSIONS)
    assert args.reference_type == "balls"
    assert alignment.fit_options(args)["cushion_gap_cm"] == pytest.approx(7.62)
    assert "inset_cm" not in alignment.fit_options(args)
    bad = alignment.build_parser().parse_args([*DIMENSIONS, "--marker-inset-cm", "10"])
    with pytest.raises(ValueError, match="requires --reference-type markers"):
        alignment.fit_options(bad)


def test_all_input_orders_identify_same_corners_without_ball_identity():
    balls = [pixel_dict(ball) for ball in scene()[0]]
    for permutation in itertools.permutations(balls):
        ordered = alignment.order_ball_candidates(list(permutation))
        assert [ball["name"] for ball in ordered] == ["TL", "TR", "BR", "BL"]
        assert [ball["pixel"] for ball in ordered] == [ball["pixel"] for ball in balls]


@pytest.mark.parametrize("count", [0, 1, 3, 5])
def test_never_silently_selects_four_from_wrong_candidate_count(count):
    balls = [pixel_dict(ball) for ball in scene()[0]]
    candidates = (balls + balls[:1])[:count]
    with pytest.raises(ValueError, match="exactly four"):
        alignment.order_ball_candidates(candidates)


@pytest.mark.parametrize("kind,reference", [("camera_z", "surface"), ("range", "center")])
def test_saved_cli_alignment_maps_new_layout_after_all_reference_balls_removed(tmp_path, kind, reference):
    balls, rotation, origin, _ = scene(angle=55, kind=kind, reference=reference)
    balls = [pixel_dict(ball) for ball in balls]
    source = tmp_path / "reference-balls.json"
    common = {"camera_matrix": K.tolist(), "image_size": list(SIZE),
              "calibration_signature": SIGNATURE, "pixel_space": "rectified"}
    source.write_text(json.dumps(dict(common, balls=[balls[2], balls[0], balls[3], balls[1]])), encoding="utf-8")
    profile = tmp_path / "table.json"
    assert alignment.main([*DIMENSIONS, "--measurements", str(source), "--output", str(profile)]) == 0
    calibrated = TableCalibration.load(profile, SIGNATURE)
    assert calibrated.reference_type == "balls"
    assert calibrated.inset_cm == pytest.approx(GAP + RADIUS)
    assert calibrated.origin == pytest.approx(origin, abs=1e-12)
    saved = profile.read_bytes()
    play_balls = []
    for key, x, y in (("cue", 28., 31.), ("target", 84., 35.)):
        center = origin + rotation @ (np.array([x, y, -RADIUS]) / 100)
        play_balls.append(pixel_dict(observation(center, kind, reference, key=key)))
    state = mapping.build_physical_state(
        profile, dict(common, measurements=play_balls), labels={"cue": 0, "target": 9}, confirmed=True)
    assert state["status"] == "confirmed" and len(state["balls"]) == 2
    assert (state["balls"][0]["x"], state["balls"][0]["y"]) == pytest.approx((28., 31.))
    assert (state["balls"][1]["x"], state["balls"][1]["y"]) == pytest.approx((84., 35.))
    assert profile.read_bytes() == saved  # Play observations never refit the alignment.


@pytest.mark.parametrize("count", [3, 4, 5])
def test_live_detects_only_four_balls_on_frozen_frame_and_saves_without_clicks(monkeypatch, tmp_path, count):
    import live_billiards
    import stereo_camera
    import stereo_depth
    from test_live_billiards import physical_scene

    inset = GAP + RADIUS
    corners = [(0, inset, inset, (240, 240, 240)),
               (1, WIDTH - inset, inset, (40, 220, 240)),
               (3, WIDTH - inset, HEIGHT - inset, (20, 20, 230)),
               (2, inset, HEIGHT - inset, (220, 40, 20)),
               (8, WIDTH / 2, HEIGHT / 2, (15, 15, 15))]
    truth, frame, depth, _ = physical_scene(ball_layout=corners[:count])
    events = SimpleNamespace(reads=0, closed=False, destroyed=False, texts=[], keys=0)
    monkeypatch.setattr(alignment.time, "monotonic", lambda: 10.)
    monkeypatch.setattr(live_billiards, "calibration_signature", lambda *args: truth.signature)
    monkeypatch.setattr(stereo_depth, "StereoDepthEstimator", lambda *args: SimpleNamespace(
        image_size=truth.image_size, rectified_camera_matrix=truth.camera_matrix,
        estimate=lambda left, right: (left, right, depth)))

    class Camera:
        def __init__(self, *args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            events.closed = True

        def read(self):
            events.reads += 1
            assert events.reads == 1, "All four references must come from the frozen pair"
            return frame, frame, {"received_monotonic_s": 10., "pair_accepted": True}

    def key(delay):
        events.keys += 1
        if events.keys == 1:
            return ord(" ")
        if events.keys == 2:
            return 13  # Try saving even when the count is wrong; that must fail.
        return ord("q")

    monkeypatch.setattr(stereo_camera, "StereoCamera", Camera)
    monkeypatch.setattr(cv2, "namedWindow", lambda *args: None)
    monkeypatch.setattr(cv2, "setMouseCallback", lambda *args: None)
    monkeypatch.setattr(cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(cv2, "putText", lambda *args, **kwargs: events.texts.append(str(args[1])))
    monkeypatch.setattr(cv2, "waitKeyEx", key)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: setattr(events, "destroyed", True))
    output = tmp_path / "table.json"
    result = alignment.main([*DIMENSIONS, "--left-camera", "0", "--right-camera", "1",
                             "--baseline-inches", "5", "--stereo-calibration", str(tmp_path / "rig.npz"),
                             "--output", str(output)])
    assert events.closed and events.destroyed
    if count == 4:
        assert result == 0
        fitted = TableCalibration.load(output, truth.signature)
        assert fitted.reference_type == "balls"
        assert fitted.origin == pytest.approx(truth.origin, abs=.005)
        assert fitted.rotation == pytest.approx(truth.rotation, abs=.005)
        assert set(("TL", "TR", "BR", "BL")).issubset(events.texts)
    else:
        assert result == 1 and not output.exists()
        assert "exactly four" in " ".join(events.texts).lower()
