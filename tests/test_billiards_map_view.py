"""The overhead map must preserve measured geometry and reject invented poses."""

import copy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import billiards_map_view as view
from billiards import normalize_pockets


def snapshot(status="confirmed"):
    table = {"width": 120.0, "height": 60.0, "ball_radius": 2.8575,
             "pocket_width": 11.5, "pockets": [{"id": "TM", "x": 65, "blocked": True}]}
    table["pockets"] = normalize_pockets(table)
    return {"table": table, "status": status, "issues": [], "balls": [
        {"key": 7, "id": 0, "x": 30.0, "y": 20.0,
         "position_uncertainty_cm": 0.4},
        {"key": 8, "id": 13, "x": 50.0, "y": 40.0,
         "position_uncertainty_cm": 1.3},
        {"key": 9, "id": None, "x": 90.0, "y": 20.0},
    ]}


def shot():
    return {"cue_path": [{"x": 30, "y": 20}, {"x": 44, "y": 40}],
            "object_path": [{"x": 50, "y": 40}, {"x": 100, "y": 2.8575},
                            {"x": 120, "y": 60}],
            "ghost": {"x": 44, "y": 40},
            "bank": {"point": {"x": 100, "y": 2.8575}},
            "scratch_risk": {"cue_path": [{"x": 44, "y": 40}, {"x": 44, "y": 55}]}}


def drawing_calls(monkeypatch):
    calls = {name: [] for name in ("circle", "line", "arrowedLine", "putText", "drawMarker")}
    for name, records in calls.items():
        original = getattr(view.cv2, name)

        def record(*args, _records=records, _original=original, **kwargs):
            _records.append((args[1:], kwargs))
            return _original(*args, **kwargs)

        monkeypatch.setattr(view.cv2, name, record)
    return calls


def texts(calls):
    return [args[0] for args, _ in calls["putText"]]


def filled_balls(calls):
    return [args for args, _ in calls["circle"] if args[3] == -1]


@pytest.mark.parametrize("size", [(640, 480), (320, 240), (1280, 720), (720, 1280)])
def test_renders_exact_shape_and_preserves_input(size):
    table_map, recommendation = snapshot(), shot()
    original_map, original_shot = copy.deepcopy(table_map), copy.deepcopy(recommendation)
    image = view.draw_table_map(*size, table_map, recommendation, selected=8)
    assert image.shape == (size[1], size[0], 3)
    assert image.dtype == np.uint8
    assert np.unique(image.reshape(-1, 3), axis=0).shape[0] > 20
    assert table_map == original_map
    assert recommendation == original_shot


def test_ball_positions_and_radii_share_one_metric_scale(monkeypatch):
    calls = drawing_calls(monkeypatch)
    view.draw_table_map(640, 480, snapshot())
    balls = filled_balls(calls)
    assert len(balls) == 3
    p, q, _ = [entry[0] for entry in balls]
    scale_x, scale_y = (q[0] - p[0]) / 20, (q[1] - p[1]) / 20
    assert scale_x == pytest.approx(scale_y, abs=0.05)
    assert balls[0][1] == pytest.approx(2.8575 * scale_x, abs=1)
    assert all(0 < center[0] < 640 and 0 < center[1] < 480 for center, *_ in balls)
    assert any("120 x 60 cm" in text for text in texts(calls))
    assert any("Grid" in text and "cm" in text for text in texts(calls))
    assert "+x" in texts(calls) and "+y" in texts(calls)


def test_scaled_physical_table_has_same_virtual_geometry(monkeypatch):
    calls = drawing_calls(monkeypatch)
    first = snapshot()
    view.draw_table_map(640, 480, first)
    original = filled_balls(calls)
    calls["circle"].clear()
    second = copy.deepcopy(first)
    for name in ("width", "height", "ball_radius", "pocket_width"):
        second["table"][name] *= 2
    for pocket in second["table"]["pockets"]:
        for name in ("x", "y", "width"):
            pocket[name] *= 2
    for ball in second["balls"]:
        for name in ("x", "y", "position_uncertainty_cm"):
            if name in ball:
                ball[name] *= 2
    view.draw_table_map(640, 480, second)
    assert filled_balls(calls) == original


def test_labels_blocked_pockets_selection_and_uncertainty(monkeypatch):
    calls = drawing_calls(monkeypatch)
    view.draw_table_map(640, 480, snapshot(), selected=8)
    assert set(["0", "13", "?", "TM X", "TL", "TR", "BL", "BM", "BR"]) <= set(texts(calls))
    balls = filled_balls(calls)
    selected_center, radius = balls[1][:2]
    assert any(args[:4] == (selected_center, radius + 4, (255, 255, 255), 2)
               for args, _ in calls["circle"])
    rings = [args for args, _ in calls["circle"] if args[2] == view.WARNING]
    assert len(rings) == 2
    assert rings[0][1] > radius and rings[1][1] > rings[0][1]


def test_confirmed_bank_draws_all_paths_ghost_and_conditional_continuation(monkeypatch):
    calls = drawing_calls(monkeypatch)
    view.draw_table_map(640, 480, snapshot(), shot())
    cue_arrows = [args for args, _ in calls["arrowedLine"] if args[2] == view.CUE_PATH]
    object_arrows = [args for args, _ in calls["arrowedLine"] if args[2] == view.OBJECT_PATH]
    assert len(cue_arrows) == 1
    assert len(object_arrows) == 2
    assert object_arrows[0][1] == object_arrows[1][0]
    assert any(args[2] == view.CUE_PATH for args, _ in calls["circle"])
    assert any(args[2] == view.CONTINUATION for args, _ in calls["line"])
    assert calls["drawMarker"]
    assert any("conditional stun" in text for text in texts(calls))


@pytest.mark.parametrize("status", ["review", "invalid", "unknown"])
def test_unconfirmed_or_invalid_map_never_displays_guidance(monkeypatch, status):
    calls = drawing_calls(monkeypatch)
    state = snapshot(status)
    state["issues"] = ["Ball identities need review"]
    view.draw_table_map(640, 480, state, shot())
    assert len(filled_balls(calls)) == 3
    assert all(args[2] == view.TEXT for args, _ in calls["arrowedLine"])
    assert not calls["drawMarker"]
    assert any("NO GUIDANCE" in text for text in texts(calls))
    assert "Ball identities need review" in texts(calls)


@pytest.mark.parametrize("position", [{"x": None, "y": 20}, {"x": float("nan"), "y": 20},
                                      {"x": 30, "y": 80}, {"x": -4, "y": 20},
                                      {"x": True, "y": 20}])
def test_missing_or_bad_coordinates_are_not_placed_on_table(monkeypatch, position):
    calls = drawing_calls(monkeypatch)
    state = snapshot()
    state["balls"][0].update(position)
    view.draw_table_map(640, 480, state, shot())
    assert len(filled_balls(calls)) == 2
    assert any("1 invalid positions omitted" in text for text in texts(calls))
    assert all(args[2] == view.TEXT for args, _ in calls["arrowedLine"])


@pytest.mark.parametrize("state", [None, {}, {"table": {}}, {"table": {"width": float("nan")}}])
def test_missing_calibration_is_unavailable_without_sample_balls(monkeypatch, state):
    calls = drawing_calls(monkeypatch)
    view.draw_table_map(640, 480, state, shot())
    assert not calls["circle"]
    assert not calls["arrowedLine"]
    assert any("UNAVAILABLE" in text for text in texts(calls))


def test_empty_map_does_not_draw_shot_or_default_balls(monkeypatch):
    calls = drawing_calls(monkeypatch)
    state = snapshot()
    state["balls"] = []
    view.draw_table_map(640, 480, state, shot())
    assert not calls["circle"]
    assert any("No observed ball positions" in text for text in texts(calls))
    assert all(args[2] == view.TEXT for args, _ in calls["arrowedLine"])


@pytest.mark.parametrize("size", [(0, 480), (640, -1), (640.5, 480), (True, 480)])
def test_rejects_invalid_output_dimensions(size):
    with pytest.raises(ValueError, match="dimensions"):
        view.draw_table_map(*size, snapshot())


@pytest.mark.parametrize("size", [(1, 1), (100, 60), (239, 199)])
def test_tiny_panels_return_exact_size_without_rendering_a_false_table(monkeypatch, size):
    calls = drawing_calls(monkeypatch)
    image = view.draw_table_map(*size, snapshot(), shot())
    assert image.shape == (size[1], size[0], 3)
    assert not calls["circle"]
    assert not calls["arrowedLine"]
