import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from depth import box_depth, box_width_m, load_fx  # noqa: E402


def test_box_depth_uses_inner_region_not_background():
    depth_map = np.full((100, 100), 5.0, np.float32)  # background
    depth_map[30:70, 30:70] = 0.5                     # object fills the box center
    assert box_depth(depth_map, (20, 20, 80, 80)) == 0.5


def test_box_depth_ignores_nan_and_nonpositive():
    depth_map = np.full((100, 100), 0.8, np.float32)
    depth_map[40:50, 40:60] = np.nan
    depth_map[50:60, 40:60] = 0.0
    assert math.isclose(box_depth(depth_map, (20, 20, 80, 80)), 0.8, rel_tol=1e-6)


def test_box_depth_none_when_too_few_valid_or_outside():
    depth_map = np.full((100, 100), np.nan, np.float32)
    assert box_depth(depth_map, (20, 20, 80, 80)) is None
    assert box_depth(np.ones((100, 100)), (200, 200, 300, 300)) is None


def test_width_unknown_without_fx_or_depth():
    assert box_width_m((0, 0, 100, 50), 0.5, None) is None
    assert box_width_m((0, 0, 100, 50), None, 600.0) is None


def test_width_formula():
    # 120 px at 0.5 m with fx=600 px -> 0.10 m
    assert math.isclose(box_width_m((100, 0, 220, 50), 0.5, 600.0), 0.10)


def test_load_fx_missing_file_and_rescale(tmp_path):
    assert load_fx(tmp_path / 'missing.yaml') is None
    import cv2
    path = tmp_path / 'calib.yaml'
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    fs.write('image_width', 640)
    fs.write('camera_matrix', np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]]))
    fs.release()
    assert math.isclose(load_fx(path, 640), 600.0)
    assert math.isclose(load_fx(path, 320), 300.0)
