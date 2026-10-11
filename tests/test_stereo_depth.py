"""Stereo scale, occlusion rejection, calibration, and visible-size checks."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stereo_depth import (DEFAULT_BASELINE_M, StereoGeometry, StereoDepthEstimator,
                          disparity_to_depth, left_right_validity, object_dimensions)
from live_yolo_stereo import stereo_object_records


def geometry():
    return StereoGeometry(500, 500, 320, 320)


def calibration_file(tmp_path, baseline=DEFAULT_BASELINE_M):
    path = tmp_path / 'stereo.npz'
    k = np.array([[250., 0, 160], [0, 250., 120], [0, 0, 1]])
    np.savez(path, K1=k, K2=k, D1=np.zeros(5), D2=np.zeros(5), R=np.eye(3),
             T=np.array([-baseline, 0, 0]), image_size=np.array([320, 240]))
    return path


def test_five_inches_establishes_metric_scale():
    assert DEFAULT_BASELINE_M == pytest.approx(0.127)
    depth = disparity_to_depth(np.array([[50., 100.]]), geometry())
    assert depth[0].tolist() == pytest.approx([1.27, 0.635])


def test_principal_point_offset_and_unknown_disparity():
    g = StereoGeometry(500, 500, 320, 315)
    values = disparity_to_depth(np.array([[55., 5., 0., -1., np.nan, np.inf]]), g)
    assert values[0, 0] == pytest.approx(1.27)
    assert np.isnan(values[0, 1:]).all()
    assert np.isnan(disparity_to_depth(np.ones((2, 2)) * 50, geometry(),
                                     np.zeros((2, 2), dtype=bool))).all()


def test_left_right_check_rejects_mismatch_out_of_view_and_invalid_roi():
    left = np.full((10, 20), 3.)
    right = np.full((10, 20), -3.)
    valid = left_right_validity(left, right, (0, 0, 20, 10), (0, 0, 20, 10))
    assert not valid[:, :3].any()
    assert valid[:, 3:].all()
    right[:, 5] = -10
    valid = left_right_validity(left, right, (0, 0, 20, 10), (0, 0, 20, 10))
    assert not valid[:, 8].any()
    restricted = left_right_validity(left, right, (6, 2, 10, 6), (0, 0, 20, 10))
    assert not restricted[:2].any()
    assert not restricted[:, :6].any()


def test_projected_dimensions_scale_with_pixels_and_depth():
    result = object_dimensions(np.full((300, 400), 0.5), [50, 40, 110, 240], geometry())
    assert result['depth_m'] == 0.5
    assert result['projected_bbox_width_m'] == pytest.approx(0.06)
    assert result['projected_bbox_height_m'] == pytest.approx(0.2)


def test_missing_mixed_and_partial_regions_are_not_reported_as_full_size():
    box = [50, 40, 115, 248]
    unknown = object_dimensions(np.full((300, 400), np.nan), box, geometry())
    assert unknown['depth_m'] is None
    mixed = np.full((300, 400), 0.5)
    mixed[144:] = 2
    result = object_dimensions(mixed, box, geometry())
    assert result['measurement_status'] == 'mixed_depth_region'
    assert result['depth_m'] is None
    result = object_dimensions(np.ones((300, 400)), [0, 40, 65, 248], geometry())
    assert result['depth_m'] == 1
    assert result['projected_bbox_width_m'] is None


def test_wrong_baseline_and_reversed_cameras_fail_instead_of_inventing_scale(tmp_path):
    pytest.importorskip('cv2')
    with pytest.raises(ValueError, match='disagrees'):
        StereoDepthEstimator(calibration_file(tmp_path, baseline=0.254), num_disparities=64)
    with pytest.raises(ValueError, match='left camera'):
        StereoDepthEstimator(calibration_file(tmp_path, baseline=-0.127), num_disparities=64)


def test_actual_opencv_matching_recovers_known_synthetic_depth(tmp_path):
    pytest.importorskip('cv2')
    estimator = StereoDepthEstimator(calibration_file(tmp_path), num_disparities=64)
    rng = np.random.default_rng(42)
    left = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
    right = np.zeros_like(left)
    right[:, :-25] = left[:, 25:]
    _, _, depth = estimator.estimate(left, right)
    central = depth[60:180, 90:220]
    assert np.isfinite(central).mean() > 0.7
    assert np.nanmedian(central) == pytest.approx(1.27, rel=0.01)
    with pytest.raises(ValueError, match='resolution'):
        estimator.estimate(left[:, :200], right[:, :200])


def test_same_class_objects_have_independent_depth_and_size():
    class Tensor:
        def __init__(self, values): self.values = values
        def cpu(self): return self
        def tolist(self): return self.values
    class Boxes:
        xyxy = Tensor([[20, 20, 80, 120], [120, 20, 180, 120]])
        cls = Tensor([0, 0])
        id = Tensor([7, 12])
        def __len__(self): return 2
    result = SimpleNamespace(boxes=Boxes(), names={0: 'bottle'})
    depth = np.ones((150, 200))
    depth[:, 100:] = 2
    records = stereo_object_records(result, depth, geometry(), 9)
    assert [r['track_id'] for r in records] == [7, 12]
    assert [r['depth_m'] for r in records] == [1, 2]
    assert [r['projected_bbox_width_m'] for r in records] == pytest.approx([0.12, 0.24])


def test_calibration_recovers_independently_generated_camera_geometry():
    cv2 = pytest.importorskip('cv2')
    from calibrate_stereo import solve_calibration
    k = np.array([[500., 0, 320], [0, 500., 240], [0, 0, 1]])
    board = np.zeros((54, 3), np.float32)
    board[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * 0.018
    objects, left_points, right_points = [], [], []
    for i in range(20):
        rvec = np.array([0.08 * (i % 5 - 2), 0.07 * (i % 4 - 1.5), 0.02 * (i - 10)])
        tvec = np.array([-0.02 + (i % 4) * 0.025, -0.09 + (i % 5) * 0.018,
                         0.55 + (i % 6) * 0.07])
        left, _ = cv2.projectPoints(board, rvec, tvec, k, np.zeros(5))
        right, _ = cv2.projectPoints(board, rvec, tvec + [-0.127, 0, 0], k, np.zeros(5))
        objects.append(board.copy())
        left_points.append(left.astype(np.float32))
        right_points.append(right.astype(np.float32))
    calibration = solve_calibration(objects, left_points, right_points, (640, 480))
    assert calibration['calibrated_baseline_m'] == pytest.approx(0.127, rel=0.005)
    assert calibration['rms_stereo_px'] < 0.01
