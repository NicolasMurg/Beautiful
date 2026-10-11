"""USB capture ordering, failure cleanup, and honest host timing metadata."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stereo_camera


def fake_devices(monkeypatch, second_open=True, grab_ok=True):
    events, cameras = [], []

    class Camera:
        def __init__(self, source, backend):
            self.index = len(cameras)
            self.released = False
            cameras.append(self)
        def isOpened(self): return self.index == 0 or second_open
        def set(self, *args): pass
        def grab(self):
            events.append(('grab', self.index))
            return grab_ok
        def retrieve(self):
            events.append(('retrieve', self.index))
            return True, np.zeros((10, 10, 3), np.uint8)
        def release(self): self.released = True

    monkeypatch.setitem(sys.modules, 'cv2', SimpleNamespace(
        VideoCapture=Camera, CAP_V4L2=0, CAP_ANY=1, CAP_PROP_FOURCC=2,
        CAP_PROP_FRAME_WIDTH=3, CAP_PROP_FRAME_HEIGHT=4, CAP_PROP_FPS=5,
        CAP_PROP_BUFFERSIZE=6, VideoWriter_fourcc=lambda *args: 0))
    return events, cameras


def test_grab_both_before_retrieve_and_do_not_claim_exposure_sync(monkeypatch):
    events, cameras = fake_devices(monkeypatch)
    timestamps = iter((1., 1.01, 1.02))
    monkeypatch.setattr(stereo_camera, 'time', SimpleNamespace(monotonic=lambda: next(timestamps)))
    with stereo_camera.StereoCamera('0', '2') as camera:
        _, _, metadata = camera.read()
    assert events == [('grab', 0), ('grab', 1), ('retrieve', 0), ('retrieve', 1)]
    assert metadata['pair_accepted'] is True
    assert metadata['exposure_synchronized'] is False
    assert all(camera.released for camera in cameras)


def test_large_host_gap_rejects_pair(monkeypatch):
    fake_devices(monkeypatch)
    timestamps = iter((1., 1.1, 1.11))
    monkeypatch.setattr(stereo_camera, 'time', SimpleNamespace(monotonic=lambda: next(timestamps)))
    with stereo_camera.StereoCamera('0', '2') as camera:
        _, _, metadata = camera.read()
    assert metadata['pair_accepted'] is False


def test_open_failure_releases_both_handles(monkeypatch):
    _, cameras = fake_devices(monkeypatch, second_open=False)
    with pytest.raises(RuntimeError, match='Cannot open'):
        stereo_camera.StereoCamera('0', '2')
    assert all(camera.released for camera in cameras)


def test_grab_failure_releases_cameras_without_reusing_old_frames(monkeypatch):
    _, cameras = fake_devices(monkeypatch, grab_ok=False)
    with pytest.raises(RuntimeError, match='stopped delivering'):
        with stereo_camera.StereoCamera('0', '2') as camera:
            camera.read()
    assert all(camera.released for camera in cameras)


def test_same_camera_cannot_be_used_twice(monkeypatch):
    _, cameras = fake_devices(monkeypatch)
    with pytest.raises(ValueError, match='distinct'):
        stereo_camera.StereoCamera('0', '0')
    assert not cameras
