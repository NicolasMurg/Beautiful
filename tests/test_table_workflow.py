"""Standalone alignment -> camera measurements -> map -> planner integration."""

import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import calibrate_table as alignment
import capture_ball_measurements as capture
import map_table_state as mapping
from billiards import plan_shots
from billiards_table import TableCalibration
from table_workflow import read_json


ROOT = Path(__file__).resolve().parents[1]
DIMENSIONS = ["--reference-type", "markers", "--table-length-cm", "120", "--table-width-cm", "60",
              "--ball-diameter-cm", "5", "--pocket-width-cm", "10", "--marker-inset-cm", "6"]
K = np.array([[700., 0., 400.], [0., 710., 300.], [0., 0., 1.]])
SIZE = [800, 600]
TRUTH = [(0, 20., 40.), (3, 72., 28.), (9, 96., 40.)]


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def physical_measurements(angle=60., kind="camera_z", reference="surface"):
    theta = np.deg2rad(angle)
    c, s = np.cos(theta), np.sin(theta)
    rotation = np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])
    origin = np.array([-.6, -.15, 1.8])

    def measured(point):
        pixel = K @ point
        return {"pixel": {"x": float(pixel[0] / pixel[2]), "y": float(pixel[1] / pixel[2])},
                "distance_m": float(point[2] if kind == "camera_z" else np.linalg.norm(point)),
                "distance_kind": kind, "frame_id": 1, "camera_id": "rectified_left"}

    common = {"calibration_signature": {"test_camera": "rig-a", "image_size": SIZE},
              "image_size": SIZE, "pixel_space": "rectified"}
    markers = []
    for name, (x, y) in zip(("TL", "TR", "BR", "BL"), ((6, 6), (114, 6), (114, 54), (6, 54))):
        markers.append(dict(measured(origin + rotation @ np.array([x, y, 0.]) / 100), name=name))
    records = []
    for i, (number, x, y) in enumerate(TRUTH):
        center = origin + rotation @ np.array([x, y, -2.5]) / 100
        surface = center - .025 * center / np.linalg.norm(center)
        records.append(dict(measured(surface if reference == "surface" else center), key=str(i + 1),
                            measurement_reference=reference, distance_uncertainty_m=.001))
    return (dict(common, camera_matrix=K.tolist(), markers=markers),
            dict(common, measurements=records, observed_at=1234567890.), rotation, origin)


def profile_and_observations(tmp_path):
    markers, balls, _, _ = physical_measurements()
    source, profile = tmp_path / "markers.json", tmp_path / "table.json"
    write(source, markers)
    assert alignment.main(["--measurements", str(source), "--output", str(profile), *DIMENSIONS]) == 0
    return profile, balls


@pytest.mark.parametrize("angle", [0., 60.])
@pytest.mark.parametrize("kind", ["camera_z", "range"])
@pytest.mark.parametrize("reference", ["surface", "center"])
def test_standalone_alignment_then_map_recovers_physical_positions(tmp_path, angle, kind, reference):
    markers, balls, expected_rotation, expected_origin = physical_measurements(angle, kind, reference)
    source, profile = tmp_path / "markers.json", tmp_path / "table.json"
    write(source, markers)
    assert alignment.main(["--measurements", str(source), "--output", str(profile), *DIMENSIONS]) == 0
    calibrated = TableCalibration.load(profile, balls["calibration_signature"])
    assert calibrated.origin == pytest.approx(expected_origin, abs=1e-10)
    assert calibrated.rotation == pytest.approx(expected_rotation, abs=1e-10)
    observations = write(tmp_path / "balls.json", balls)
    labels = write(tmp_path / "labels.json", {str(i + 1): ball[0] for i, ball in enumerate(TRUTH)})
    output, planner, image = tmp_path / "state.json", tmp_path / "planner.json", tmp_path / "map.png"
    assert mapping.main(["--table-profile", str(profile), "--measurements", str(observations),
                         "--labels", str(labels), "--confirmed", "--output", str(output),
                         "--planner-state", str(planner), "--image", str(image)]) == 0
    world = read_json(output)
    assert world["status"] == "confirmed" and world["units"] == "cm"
    assert world["observation_kind"] == "snapshot"
    assert len(world["table"]["pockets"]) == 6 and len(world["rails"]) > 4
    assert len(world["ball_separations"]) == 3
    assert "pocketed_ball_ids" not in world
    for ball, (number, x, y) in zip(world["balls"], TRUTH):
        assert ball["id"] == number
        assert (ball["x"], ball["y"]) == pytest.approx((x, y), abs=1e-9)
        assert ball["measurement"]["measurement_reference"] == reference
    expected = np.hypot(72. - 20., 28. - 40.)
    assert world["ball_separations"][0]["center_distance_cm"] == pytest.approx(expected)
    assert world["ball_separations"][0]["surface_gap_cm"] == pytest.approx(expected - 5.)
    assert image.read_bytes().startswith(b"\x89PNG")
    state = read_json(planner)
    before = copy.deepcopy(state)
    assert plan_shots(state, "best shot for stripes")["target_ball_ids"] == [9]
    assert plan_shots(state, "best shot for solids")["target_ball_ids"] == [3]
    assert state == before


@pytest.mark.parametrize("failure", ["no_pixel", "swapped", "scale", "mixed_frame", "mixed_camera"])
def test_marker_alignment_rejects_inconsistent_or_incomplete_measurements(tmp_path, failure):
    markers, _, _, _ = physical_measurements()
    if failure == "no_pixel":
        markers["markers"][0].pop("pixel")
    elif failure == "swapped":
        markers["markers"][0], markers["markers"][1] = markers["markers"][1], markers["markers"][0]
    elif failure == "scale":
        for marker in markers["markers"]:
            marker["distance_m"] *= 1.2
    elif failure == "mixed_frame":
        markers["markers"][1]["frame_id"] = 2
    else:
        markers["markers"][1]["camera_id"] = "right"
    path, output = write(tmp_path / "markers.json", markers), tmp_path / "table.json"
    assert alignment.main(["--measurements", str(path), "--output", str(output), *DIMENSIONS]) == 2
    assert not output.exists()


@pytest.mark.parametrize("failure", ["signature", "resolution", "raw_pixels", "distance_only", "camera_id"])
def test_mapper_rejects_wrong_frame_or_missing_direction(tmp_path, failure):
    profile, balls = profile_and_observations(tmp_path)
    if failure == "signature":
        balls["calibration_signature"] = {"test_camera": "other"}
    elif failure == "resolution":
        balls["image_size"] = [400, 300]
    elif failure == "raw_pixels":
        balls["pixel_space"] = "raw"
    elif failure == "distance_only":
        balls["measurements"][0].pop("pixel")
    else:
        balls["measurements"][1]["camera_id"] = "right"
    observations = write(tmp_path / "balls.json", balls)
    output = tmp_path / "state.json"
    assert mapping.main(["--table-profile", str(profile), "--measurements", str(observations),
                         "--output", str(output)]) == 2
    if failure == "distance_only":
        world = read_json(output)
        assert world["status"] == "invalid"
        assert len(world["unmapped"]) == 1 and len(world["balls"]) == 2
    else:
        assert not output.exists()


def test_unidentified_balls_map_without_inventing_identity_or_allowing_planning(tmp_path):
    profile, balls = profile_and_observations(tmp_path)
    world = mapping.build_physical_state(profile, balls)
    assert len(world["balls"]) == 3 and world["status"] == "review"
    assert all(ball["id"] is None for ball in world["balls"])
    path = write(tmp_path / "measurements.json", balls)
    output, planner = tmp_path / "state.json", tmp_path / "planner.json"
    assert mapping.main(["--table-profile", str(profile), "--measurements", str(path),
                         "--output", str(output), "--planner-state", str(planner)]) == 2
    assert not output.exists() and not planner.exists()
    assert mapping.main(["--table-profile", str(profile), "--measurements", str(path), "--confirmed",
                         "--output", str(output), "--planner-state", str(planner)]) == 2
    assert read_json(output)["status"] == "review" and not planner.exists()


def test_failed_depth_preserves_candidate_and_diagnostic(tmp_path):
    profile, balls = profile_and_observations(tmp_path)
    balls["measurements"][1] = {"key": "2", "frame_id": 1, "camera_id": "rectified_left",
                                "error": "Insufficient stereo depth on this ball"}
    world = mapping.build_physical_state(profile, balls, labels={"1": 0, "2": 3, "3": 9}, confirmed=True)
    assert world["status"] == "invalid"
    assert world["unmapped"] == [{"key": "2", "reason": "Insufficient stereo depth on this ball"}]


def test_output_cannot_replace_inputs_and_nonfinite_json_is_rejected(tmp_path):
    profile, balls = profile_and_observations(tmp_path)
    source = write(tmp_path / "balls.json", balls)
    before = profile.read_bytes()
    assert mapping.main(["--table-profile", str(profile), "--measurements", str(source),
                         "--output", str(profile)]) == 2
    assert profile.read_bytes() == before
    for content in ('{"distance_m": NaN}', '{"image_size":[],"image_size":[800,600]}'):
        source.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError):
            read_json(source)


def test_cli_help_and_headless_import_do_not_require_opencv(tmp_path):
    # A subprocess catches path/import and __main__ regressions, not just function calls.
    code = "import sys; sys.modules['cv2'] = None; import calibrate_table, map_table_state"
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    for script in ("calibrate_table.py", "capture_ball_measurements.py", "map_table_state.py"):
        result = subprocess.run([sys.executable, str(ROOT / script), "--help"],
                                cwd=tmp_path, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert "--output" in result.stdout


def test_capture_exports_surface_measurements_and_hints_without_confirming_labels(tmp_path):
    # Real stereo localizer/detector/session, with ray-rendered spheres.
    from test_live_billiards import reviewed_session
    session, _ = reviewed_session()
    batch = capture.snapshot_measurements(session, frame_id="snapshot-a", observed_at=42.)
    assert batch["reviewed"] is False
    assert all("id" not in record for record in batch["measurements"])
    assert all(record["measurement_reference"] == "surface" for record in batch["measurements"])
    profile = tmp_path / "table.json"
    session.calibration.save(profile)
    world = mapping.build_physical_state(profile, batch, labels=session.labels, confirmed=True)
    for original, mapped in zip(session.candidates, world["balls"]):
        assert (mapped["x"], mapped["y"]) == pytest.approx(
            (original["position"]["x"], original["position"]["y"]), abs=1e-9)
    assert capture.review_image(session).shape[:2] == (550, 640)
    session.candidates[1].pop("position")
    session.candidates[1]["error"] = "No metric depth"
    failed_batch = capture.snapshot_measurements(session, frame_id="snapshot-b", observed_at=43.)
    assert failed_batch["measurements"][1]["error"] == "No metric depth"
    assert "distance_m" not in failed_batch["measurements"][1]


@pytest.mark.parametrize("accepted", [True, False])
def test_capture_waits_for_still_frames_and_closes_camera(monkeypatch, tmp_path, accepted):
    from test_live_billiards import physical_scene
    calibration, frame, depth, _ = physical_scene()
    profile = tmp_path / "table.json"
    calibration.save(profile)
    events = SimpleNamespace(reads=0, closed=False)
    clock = [10.]
    monkeypatch.setattr(capture.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(capture, "calibration_signature", lambda *args: calibration.signature)
    monkeypatch.setattr(capture, "StereoDepthEstimator", lambda *args: SimpleNamespace(
        image_size=(640, 480), estimate=lambda left, right: (left, right, depth)))

    class Camera:
        def __init__(self, *args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            events.closed = True

        def read(self):
            events.reads += 1
            clock[0] += .6
            return frame, frame, {"received_monotonic_s": clock[0], "pair_accepted": accepted}

    monkeypatch.setattr(capture, "StereoCamera", Camera)
    output, image = tmp_path / "measurements.json", tmp_path / "review.png"
    args = ["--left-camera", "0", "--right-camera", "1", "--baseline-inches", "5",
            "--stereo-calibration", str(tmp_path / "rig.npz"), "--table-profile", str(profile),
            "--output", str(output), "--image", str(image), "--timeout-seconds", "3"]
    assert capture.main(args) == (0 if accepted else 2)
    assert events.closed and events.reads >= 3
    if accepted:
        assert len(read_json(output)["measurements"]) == 3
        assert image.read_bytes().startswith(b"\x89PNG")
    else:
        assert not output.exists() and not image.exists()


@pytest.mark.parametrize("fresh", [True, False])
def test_live_alignment_freezes_one_frame_for_all_clicks(monkeypatch, tmp_path, fresh):
    import cv2
    import live_billiards
    import stereo_camera
    import stereo_depth
    from test_live_billiards import physical_scene
    _, frame, depth, _ = physical_scene(ball_layout=[])
    events = SimpleNamespace(reads=0, closed=False, destroyed=False, mouse=None, keys=0)
    monkeypatch.setattr(alignment.time, "monotonic", lambda: 10.)
    monkeypatch.setattr(live_billiards, "calibration_signature", lambda *args: {"test": "frozen-rig"})
    monkeypatch.setattr(stereo_depth, "StereoDepthEstimator", lambda *args: SimpleNamespace(
        image_size=(640, 480), rectified_camera_matrix=np.array([[400., 0., 320.], [0., 400., 240.], [0., 0., 1.]]),
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
            if fresh:
                assert events.reads == 1, "Marker selection must not use subsequent frames"
            return frame, frame, {"received_monotonic_s": 10. if fresh else 2., "pair_accepted": True}

    def wait_key(delay):
        events.keys += 1
        if events.keys == 1:
            return ord(" ")
        if not fresh:
            return ord("q")
        for x, y in ((140, 160), (500, 160), (500, 320), (140, 320)):
            events.mouse(cv2.EVENT_LBUTTONDOWN, x, y, 0, None)
        return 13

    monkeypatch.setattr(stereo_camera, "StereoCamera", Camera)
    monkeypatch.setattr(cv2, "namedWindow", lambda *args: None)
    monkeypatch.setattr(cv2, "setMouseCallback", lambda window, callback: setattr(events, "mouse", callback))
    monkeypatch.setattr(cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(cv2, "waitKeyEx", wait_key)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: setattr(events, "destroyed", True))
    output = tmp_path / "table.json"
    assert alignment.main(["--left-camera", "0", "--right-camera", "1", "--stereo-calibration",
                           str(tmp_path / "rig.npz"), "--baseline-inches", "5", "--output", str(output),
                           *DIMENSIONS]) == (0 if fresh else 1)
    assert events.closed and events.destroyed
    if fresh:
        assert TableCalibration.load(output, {"test": "frozen-rig"}).fit_error_cm < .001
    else:
        assert not output.exists() and events.reads == 2
