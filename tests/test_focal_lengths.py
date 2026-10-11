"""Focal-length resize/validation checks using an in-memory OpenCV file adapter."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from depth import load_fx, load_fy


def fake_calibration(tmp_path, monkeypatch, matrix, width=640, height=480):
    path = tmp_path / "calibration.yaml"
    path.touch()
    nodes = {
        "camera_matrix": SimpleNamespace(mat=lambda: matrix),
        "image_width": SimpleNamespace(real=lambda: width),
        "image_height": SimpleNamespace(real=lambda: height),
    }
    storage = SimpleNamespace(getNode=lambda name: nodes[name], release=lambda: None)
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(
        FILE_STORAGE_READ=0, FileStorage=lambda *args: storage))
    return path


def test_focal_lengths_scale_each_axis_using_its_actual_frame_dimension(tmp_path, monkeypatch):
    path = fake_calibration(tmp_path, monkeypatch, np.diag([600.0, 700.0, 1]))
    assert load_fx(path, 320) == 300.0
    assert load_fy(path, 960) == 1400.0


@pytest.mark.parametrize("matrix, width, height", [
    (np.diag([float("nan"), float("inf"), 1]), 640, 480),
    (np.diag([600.0, 700.0, 1]), 0, 0),
    (np.eye(2), 640, 480),
])
def test_invalid_intrinsics_or_unknown_original_size_disable_geometry(
        tmp_path, monkeypatch, matrix, width, height):
    path = fake_calibration(tmp_path, monkeypatch, matrix, width, height)
    assert load_fx(path, 640) is None
    assert load_fy(path, 480) is None
