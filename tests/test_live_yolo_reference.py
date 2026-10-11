"""Exercise the viewer's click/confirm/reset loop with deterministic fakes."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import depth
import live_yolo


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class FakeBoxes:
    def __init__(self):
        self.xyxy = FakeTensor([[80, 40, 145, 248], [250, 40, 330, 240]])
        self.cls = FakeTensor([0, 0])
        self.id = FakeTensor([7, 12])
        self.data = SimpleNamespace(device="cpu")

    def __len__(self):
        return 2


def test_reference_viewer_requires_click_then_confirm_and_reset_clears_correction(
        tmp_path, monkeypatch, capsys):
    weights = tmp_path / "test.pt"
    weights.touch()
    profile = tmp_path / "reference.json"
    profile.write_text(json.dumps({"name": "fixture", "height_m": 0.2,
                                   "diameter_m": 0.06}), encoding="utf-8")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    calls = {"depth": 0, "track": 0, "keys": 0, "released": False}
    callback = {}

    class FakeDepth:
        is_metric = True
        last_latency_ms = 10

        def __init__(self, *args):
            pass

        def estimate(self, image):
            calls["depth"] += 1
            return np.ones(image.shape[:2], dtype=np.float32)

    class FakeModel:
        predictor = SimpleNamespace(device="cpu")

        def __init__(self, path):
            pass

        def track(self, image, **kwargs):
            calls["track"] += 1
            return [SimpleNamespace(boxes=FakeBoxes(), names={0: "bottle"},
                                    plot=lambda: image.copy())]

    def release():
        calls["released"] = True

    camera = SimpleNamespace(isOpened=lambda: True, set=lambda *args: None,
                             read=lambda: (True, frame.copy()), release=release)

    def wait_key(delay):
        calls["keys"] += 1
        if calls["keys"] == 1:
            callback["mouse"](1, 100, 100, None, None)
            return ord("c")
        return {2: 0, 3: ord("r"), 4: ord("q")}[calls["keys"]]

    cv2 = SimpleNamespace(
        CAP_V4L2=0, CAP_PROP_FOURCC=1, CAP_PROP_FRAME_WIDTH=2,
        CAP_PROP_FRAME_HEIGHT=3, CAP_PROP_BUFFERSIZE=4, WINDOW_NORMAL=5,
        EVENT_LBUTTONDOWN=1, FONT_HERSHEY_SIMPLEX=6, WND_PROP_VISIBLE=7,
        VideoCapture=lambda *args: camera, VideoWriter_fourcc=lambda *args: 0,
        namedWindow=lambda *args: None, resizeWindow=lambda *args: None,
        setMouseCallback=lambda window, function: callback.update(mouse=function),
        putText=lambda *args: None, rectangle=lambda *args: None,
        imshow=lambda *args: None, waitKey=wait_key,
        getWindowProperty=lambda *args: 1, destroyAllWindows=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeModel))
    monkeypatch.setattr(depth, "DepthEstimator", FakeDepth)
    monkeypatch.setattr(depth, "load_fx", lambda *args: None)
    monkeypatch.setattr(depth, "load_fy", lambda *args: None)
    ticks = iter(range(100))
    monkeypatch.setattr(live_yolo, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    monkeypatch.setattr(sys, "argv", [
        "live_yolo.py", "--model", str(weights), "--device", "cpu",
        "--depth-model", "fixture-metric-model", "--reference-profile", str(profile),
        "--reference-distance-cm", "60", "--depth-every", "3", "--frames", "4",
    ])
    live_yolo.main()
    output = [json.loads(line) for line in capsys.readouterr().out.splitlines()
              if line.startswith("{")]
    assert [r["frame_id"] for r in output] == [1, 2, 3, 4]
    assert output[0]["reference"] is None
    assert output[1]["reference"]["track_id"] == 7
    assert output[1]["reference"]["reference_depth_m"] == 0.6
    assert output[1]["objects"][1]["raw_depth_m"] == 1.0
    assert output[1]["objects"][1]["depth_m"] == 0.6
    assert output[3]["reference"] is None
    assert output[3]["objects"][1]["depth_m"] == 1.0
    assert calls == {"depth": 4, "track": 4, "keys": 4, "released": True}
