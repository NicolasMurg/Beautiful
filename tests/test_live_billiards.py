"""Headless live integration using synthetic physical stereo observations.

Camera I/O is replaced; the table transform, sphere localization, detector,
review session, shot planner and overlay renderer run their real code.
"""

from pathlib import Path
import queue
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import live_billiards as live
from billiards_session import PoolSession
from billiards_table import TableCalibration
from billiards_vision import BallDetector


IMAGE_SIZE = (640, 480)
CAMERA_MATRIX = np.array([[400., 0, 320.], [0, 400., 240.], [0, 0, 1.]])
RADIUS_CM = 2.8575


def physical_scene(signature=None, ball_layout=None):
    """Known table plus ray-rendered sphere depths, never planner example state."""
    depth = np.full(IMAGE_SIZE[::-1], 1.2)
    calibration = TableCalibration.fit(
        [[140, 160], [500, 160], [500, 320], [140, 320]], depth,
        CAMERA_MATRIX, IMAGE_SIZE, 120, 60, RADIUS_CM, 11.5, 6,
        signature or {"test": "synthetic physical calibration"})
    frame = np.full((*IMAGE_SIZE[::-1], 3), (40, 105, 30), np.uint8)
    yy, xx = np.indices(depth.shape)
    rays = np.stack(((xx - 320) / 400, (yy - 240) / 400, np.ones_like(xx)), axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    balls = ball_layout if ball_layout is not None else [
        (0, 25, 30, (240, 240, 240)), (3, 65, 25, (20, 20, 230)),
        (8, 100, 40, (15, 15, 15))]
    for number, x, y, color in balls:
        center = calibration.origin + np.array([x, y, -RADIUS_CM]) / 100
        projection = rays @ center
        discriminant = projection ** 2 - (np.dot(center, center) - (RADIUS_CM / 100) ** 2)
        visible = discriminant >= 0
        distance = projection[visible] - np.sqrt(discriminant[visible])
        depth[visible] = distance * rays[visible, 2]
        frame[visible] = color
    return calibration, frame, depth, balls


def reviewed_session(ball_layout=None):
    calibration, frame, depth, balls = physical_scene(ball_layout=ball_layout)
    session = PoolSession(calibration, BallDetector(5, 18), "solids")
    for now in (10.0, 10.6, 11.2):
        session.observe(frame, now, now=now)
    candidates = session.scan(frame, depth, now=11.2)
    assert len(candidates) == len(balls)
    for candidate in candidates:
        assert "error" not in candidate
        position = candidate["position"]
        truth = min(balls, key=lambda ball: np.hypot(position["x"] - ball[1], position["y"] - ball[2]))
        assert (position["x"], position["y"]) == pytest.approx(truth[1:3], abs=0.3)
        # This is the operator-confirmation boundary, using known test labels.
        session.assign(candidate["key"], truth[0])
    session.confirm()
    result = session.ask("What is my best shot?", now=11.2)
    assert result["shots"]
    return session, frame


def test_real_detector_geometry_session_and_planner_render_physical_shot(monkeypatch):
    session, frame = reviewed_session()
    arrows, polygons, markers, texts = [], [], [], []
    for name, collected in (("arrowedLine", arrows), ("polylines", polygons),
                            ("drawMarker", markers), ("putText", texts)):
        original = getattr(cv2, name)

        def record(*args, _original=original, _collected=collected, **kwargs):
            _collected.append(args)
            return _original(*args, **kwargs)

        monkeypatch.setattr(cv2, name, record)
    canvas = live.render(frame, session, selected=session.candidates[0]["key"])
    assert canvas.dtype == np.uint8
    assert canvas.shape == (760, IMAGE_SIZE[0] + 530, 3)
    assert np.array_equal(canvas[0, 0], frame[0, 0])
    assert np.count_nonzero(np.any(canvas[:480, :640] != frame, axis=2)) > 100
    assert len(arrows) == 2
    assert any(len(call[1][0]) == 4 for call in polygons)  # Physical cushion boundary.
    assert any(len(call[1][0]) == 40 for call in polygons)  # Projected ghost sphere section.
    assert any(call[3] == cv2.MARKER_DIAMOND for call in markers)
    displayed = " ".join(str(call[1]) for call in texts)
    assert "CONFIRMED" in displayed
    assert "Default target filter: SOLIDS" in displayed
    assert "This request: SOLIDS" in displayed
    assert "pocketed" not in displayed.lower()
    assert "Stereo Z" in displayed and "table (" in displayed
    assert session.shot["ball_name"] in displayed
    assert "Cut angle:" in displayed
    # Shot endpoints must be projected at ball-center height, not onto the felt.
    expected = [tuple(round(v) for v in session.calibration.project(
        p["x"], p["y"], height_cm=RADIUS_CM)) for p in session.shot["cue_path"]]
    assert arrows[0][1:3] == tuple(expected)


@pytest.mark.parametrize("question,target_label", [("best shot for stripes", "STRIPES"),
                                                    ("shoot ball 9", "BALL 9")])
def test_request_filter_display_does_not_assign_player_or_change_scene(monkeypatch, question, target_label):
    session, frame = reviewed_session([(0, 25, 30, (240, 240, 240)),
                                       (3, 65, 25, (20, 20, 230)),
                                       (9, 100, 40, (40, 220, 240))])
    before = session.state()
    result = session.ask(question, now=11.2)
    assert result["eligible_ball_ids"] == [9]
    texts = []
    monkeypatch.setattr(cv2, "putText", lambda *args, **kwargs: texts.append(args[1]))
    live.render(frame, session)
    displayed = " ".join(texts)
    assert "Default target filter: SOLIDS" in displayed
    assert f"This request: {target_label}" in displayed
    assert session.state() == before


def test_parser_help_and_defaults_require_physical_stereo_arguments(capsys):
    parser = live.build_parser()
    args = parser.parse_args(["--left-camera", "0", "--right-camera", "2",
                              "--stereo-calibration", "rig.npz"])
    assert args.left_camera == "0" and args.right_camera == "2"
    assert args.baseline_inches == 5.0
    assert args.ball_diameter_cm == 5.715
    assert args.group == "open"
    assert args.model is None and not args.voice and not args.speak
    assert ".local" in args.table_profile.parts
    with pytest.raises(SystemExit) as result:
        parser.parse_args(["--help"])
    assert result.value.code == 0
    help_text = capsys.readouterr().out
    assert "--stereo-calibration" in help_text and "--table-length-cm" in help_text
    assert "--voice" in help_text and "--speak" in help_text


def test_profile_signature_binds_file_baseline_resolution_and_rectified_matrix(tmp_path):
    path = tmp_path / "stereo.npz"
    np.savez(path, fixture=np.array([1, 2, 3]))
    estimator = SimpleNamespace(image_size=IMAGE_SIZE, rectified_camera_matrix=CAMERA_MATRIX.copy())
    signature = live.calibration_signature(path, estimator, 0.127)
    calibration, _, _, _ = physical_scene(signature)
    profile = tmp_path / "table.json"
    calibration.save(profile)
    loaded = TableCalibration.load(profile, live.calibration_signature(path, estimator, 0.127))
    assert loaded.polygon == pytest.approx(calibration.polygon)
    with pytest.raises(ValueError, match="signature"):
        TableCalibration.load(profile, live.calibration_signature(path, estimator, 0.2))
    estimator.rectified_camera_matrix[1, 2] += 1
    with pytest.raises(ValueError, match="signature"):
        TableCalibration.load(profile, live.calibration_signature(path, estimator, 0.127))
    estimator.rectified_camera_matrix = CAMERA_MATRIX.copy()
    estimator.image_size = (1280, 960)
    with pytest.raises(ValueError, match="signature"):
        TableCalibration.load(profile, live.calibration_signature(path, estimator, 0.127))
    estimator.image_size = IMAGE_SIZE
    np.savez(path, fixture=np.array([4, 5, 6]))
    with pytest.raises(ValueError, match="signature"):
        TableCalibration.load(profile, live.calibration_signature(path, estimator, 0.127))


def fake_live_environment(monkeypatch, tmp_path, fail_capture=False, voice=False):
    calibration, frame, depth, _ = physical_scene()
    events = SimpleNamespace(entered=False, exited=False, destroyed=False, shown=[],
                             speaker_stopped=0, reads=0, worker_closed=False)
    stereo_file = tmp_path / "stereo.npz"
    np.savez(stereo_file, fixture=np.array([1]))
    arguments = ["live_billiards.py", "--left-camera", "0", "--right-camera", "1",
                 "--stereo-calibration", str(stereo_file), "--table-profile", str(tmp_path / "table.json"),
                 "--table-length-cm", "120", "--table-width-cm", "60",
                 "--pocket-width-cm", "11.5", "--marker-inset-cm", "6"]
    if voice:
        arguments.append("--voice")
    monkeypatch.setattr(sys, "argv", arguments)

    class Estimator:
        def __init__(self, *args, **kwargs):
            self.image_size = IMAGE_SIZE
            self.rectified_camera_matrix = CAMERA_MATRIX.copy()

        def estimate(self, left, right):
            return left, right, depth.copy()

    class Camera:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            events.entered = True
            return self

        def __exit__(self, *args):
            events.exited = True

        def read(self):
            events.reads += 1
            if fail_capture or events.reads > 3:
                raise RuntimeError("Synthetic capture failure")
            return frame.copy(), frame.copy(), {"pair_accepted": True,
                                                 "received_monotonic_s": time.monotonic()}

    class Speaker:
        def __init__(self, *args, **kwargs):
            pass

        def stop(self):
            events.speaker_stopped += 1

        def say(self, text):
            pass

    monkeypatch.setattr(live, "StereoDepthEstimator", Estimator)
    monkeypatch.setattr(live, "StereoCamera", Camera)
    monkeypatch.setattr(live, "LocalSpeaker", Speaker)
    monkeypatch.setattr(live, "terminal_commands", lambda: queue.Queue())
    monkeypatch.setattr(cv2, "namedWindow", lambda *args: None)
    monkeypatch.setattr(cv2, "setMouseCallback", lambda *args: None)
    monkeypatch.setattr(cv2, "imshow", lambda window, canvas: events.shown.append(canvas.copy()))
    monkeypatch.setattr(cv2, "waitKeyEx", lambda delay: ord("q"))
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: setattr(events, "destroyed", True))
    return events


def test_main_capture_failure_closes_camera_speech_and_gui(monkeypatch, tmp_path, capsys):
    import speech
    events = fake_live_environment(monkeypatch, tmp_path, fail_capture=True, voice=True)
    worker = SimpleNamespace(close=lambda: setattr(events, "worker_closed", True))
    monkeypatch.setattr(speech, "worker_from_args", lambda args: worker)
    with pytest.raises(SystemExit) as result:
        live.main()
    assert result.value.code == 1
    assert events.entered and events.exited and events.destroyed and events.worker_closed
    assert events.speaker_stopped >= 1
    assert "Synthetic capture failure" in capsys.readouterr().err


def test_main_missing_whisper_keeps_camera_and_typed_workflow_available(monkeypatch, tmp_path, capsys):
    import speech
    events = fake_live_environment(monkeypatch, tmp_path, voice=True)

    def missing_worker(args):
        raise FileNotFoundError("synthetic missing whisper-cli")

    monkeypatch.setattr(speech, "worker_from_args", missing_worker)
    assert live.main() is None
    assert events.entered and events.exited and events.destroyed
    assert events.reads == 1 and len(events.shown) == 1
    assert events.shown[0].shape == (760, 1170, 3)
    output = capsys.readouterr().out
    assert "Voice unavailable" in output and "Typed questions remain available" in output


def test_missing_optional_speaker_is_nonfatal(monkeypatch, capsys):
    monkeypatch.setattr(live.shutil, "which", lambda name: None)
    speaker = live.LocalSpeaker(enabled=True)
    speaker.say("Geometric shot instructions")
    speaker.stop()
    assert speaker.process is None
    assert "Spoken replies unavailable" in capsys.readouterr().out


def test_bank_overlay_renders_every_leg_and_reports_risk(monkeypatch):
    session, frame = reviewed_session([(0, 40, 45, (240, 240, 240)),
                                       (3, 65, 30, (20, 20, 230))])
    candidates = session.plan["shots"]
    banks = [i for i, shot in enumerate(candidates) if shot["shot_type"] == "bank"]
    assert banks, "Physical test layout should admit a bank candidate"
    session.shot_index = banks[0]
    arrows, texts = [], []
    monkeypatch.setattr(cv2, "arrowedLine", lambda *args, **kwargs: arrows.append(args))
    monkeypatch.setattr(cv2, "putText", lambda *args, **kwargs: texts.append(args[1]))
    live.render(frame, session)
    assert len(arrows) == 3  # Cue approach plus both object-ball bank legs.
    displayed = " ".join(texts)
    assert "BANK" in displayed and "Scratch estimate:" in displayed
    assert session.shot["bank"]["rail"] in displayed


def test_export_requires_review_and_preserves_measured_state(tmp_path):
    import json
    session, _ = reviewed_session()
    path = tmp_path / "capture" / "layout.json"
    live.export_reviewed_state(session, path)
    exported = json.loads(path.read_text(encoding="utf-8"))
    assert exported["balls"] == session.state()["balls"]
    assert exported["observation"]["reviewed"] is True
    assert exported["observation"]["source"] == "physical_calibrated_stereo"
    assert exported["virtual_map"]["status"] == "confirmed"
    assert "pocketed_ball_ids" not in exported["virtual_map"]
    assert exported["virtual_map"]["table"] == exported["table"]
    assert [(b["id"], b["x"], b["y"]) for b in exported["virtual_map"]["balls"]] == [
        (b["id"], b["x"], b["y"]) for b in exported["balls"]]
    original = path.read_bytes()
    session.invalidate("Motion")
    with pytest.raises(ValueError, match="Confirm"):
        live.export_reviewed_state(session, path)
    assert path.read_bytes() == original
    assert not list(path.parent.glob("*.tmp"))


def test_configuration_files_are_validated_before_use(tmp_path):
    import json
    path = tmp_path / "pockets.json"
    path.write_text(json.dumps([{"id": "TM", "x": 60, "width": 10}]), encoding="utf-8")
    assert live.read_json_option(path, list, "pockets")[0]["id"] == "TM"
    with pytest.raises(ValueError, match="JSON dict"):
        live.read_json_option(path, dict, "class map")
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError, match="Cannot read"):
        live.read_json_option(path, dict, "class map")


def test_physical_scan_map_contains_true_table_coordinates_and_measurement_provenance():
    session, _ = reviewed_session()
    mapped = session.virtual_map()
    assert mapped["status"] == "confirmed" and mapped["units"] == "cm"
    assert len(mapped["table"]["pockets"]) == 6 and len(mapped["rails"]) == 6
    expected = {0: (25, 30), 3: (65, 25), 8: (100, 40)}
    for ball in mapped["balls"]:
        assert (ball["x"], ball["y"]) == pytest.approx(expected[ball["id"]], abs=0.3)
        assert ball["measurement"]["distance_kind"] == "camera_z"
        assert ball["measurement"]["measurement_reference"] == "surface"
        assert ball["measurement"]["pixel_space"] == "rectified"
        assert len(ball["measurement"]["camera_xyz_m"]) == 3
    assert mapped["observed_at"] == 11.2
    assert "pocketed_ball_ids" not in mapped


def test_shot_planner_consumes_the_same_measured_map_shown_to_the_player(monkeypatch):
    import billiards_session
    session, _ = reviewed_session()
    mapped = session.virtual_map()
    original = billiards_session.plan_shots
    received = []

    def capture(state, question=""):
        received.append(state)
        return original(state, question)

    monkeypatch.setattr(billiards_session, "plan_shots", capture)
    session.ask(now=11.2)
    assert received[0]["table"] == mapped["table"]
    for ball, observed in zip(received[0]["balls"], mapped["balls"]):
        assert ball == {key: observed[key] for key in
                        ("id", "x", "y", "confidence", "position_uncertainty_cm")}


def test_virtual_view_is_overhead_without_camera_projection_and_invalidates_after_motion(monkeypatch):
    session, frame = reviewed_session()

    def no_camera_projection(*args, **kwargs):
        raise AssertionError("Overhead view must use metric map coordinates")

    monkeypatch.setattr(session.calibration, "project", no_camera_projection)
    texts = []
    original_text = cv2.putText

    def capture_text(*args, **kwargs):
        texts.append(args[1])
        return original_text(*args, **kwargs)

    monkeypatch.setattr(cv2, "putText", capture_text)
    canvas = live.render(frame, session, map_view=True)
    assert canvas.shape == (760, 1170, 3)
    assert "VIRTUAL TABLE | CONFIRMED" in texts
    assert not np.array_equal(canvas[:480, :640], frame)
    session.invalidate("Table moved")
    texts.clear()
    live.render(frame, session, map_view=True)
    assert any("UNAVAILABLE" in value for value in texts)
    assert session.virtual_map()["balls"] == []
    assert "pocketed_ball_ids" not in session.virtual_map()
    assert session.shot is None
