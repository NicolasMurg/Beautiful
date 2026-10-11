"""Calibrated stereo geometry in meters. OpenCV is imported only when needed."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_BASELINE_M = 5 * 0.0254


@dataclass(frozen=True)
class StereoGeometry:
    fx: float
    fy: float
    cx_left: float
    cx_right: float
    baseline_m: float = DEFAULT_BASELINE_M

    def __post_init__(self):
        values = (self.fx, self.fy, self.cx_left, self.cx_right, self.baseline_m)
        if not all(np.isfinite(v) for v in values) or min(self.fx, self.fy, self.baseline_m) <= 0:
            raise ValueError("Stereo geometry must have finite positive focal lengths/baseline")


def disparity_to_depth(disparity_px, geometry, valid_mask=None):
    """Z = fx * B / (d - (cx_left - cx_right)); unknown pixels are NaN."""
    disparity = np.asarray(disparity_px, dtype=np.float64)
    effective = disparity - (geometry.cx_left - geometry.cx_right)
    valid = np.isfinite(effective) & (effective > 0)
    if valid_mask is not None:
        if np.shape(valid_mask) != disparity.shape:
            raise ValueError("Validity mask and disparity shapes must match")
        valid &= valid_mask
    depth = np.full(disparity.shape, np.nan)
    np.divide(geometry.fx * geometry.baseline_m, effective, out=depth, where=valid)
    depth[~np.isfinite(depth)] = np.nan
    return depth.astype(np.float32)


def left_right_validity(left_disparity, right_disparity, left_roi, right_roi, tolerance=1.0):
    """Reject unmatched/occluded pixels using reverse matching and rectified ROIs."""
    left, right = np.asarray(left_disparity), np.asarray(right_disparity)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("Stereo disparities must be equally sized 2D arrays")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Consistency tolerance must be finite and nonnegative")
    h, w = left.shape
    yy, xx = np.indices((h, w))
    safe = np.where(np.isfinite(left), left, 0)
    rx = np.rint(xx - safe).astype(np.int64)
    lx, ly, lw, lh = left_roi
    x, y, rw, rh = right_roi
    valid = (np.isfinite(left) & (left > 0) & (rx >= 0) & (rx < w)
             & (xx >= lx) & (xx < lx + lw) & (yy >= ly) & (yy < ly + lh)
             & (rx >= x) & (rx < x + rw) & (yy >= y) & (yy < y + rh))
    reverse = right[yy, np.clip(rx, 0, w - 1)]
    return valid & np.isfinite(reverse) & (np.abs(left + reverse) <= tolerance)


def object_dimensions(depth_map, xyxy, geometry, inner=0.4, min_valid=30,
                      min_valid_fraction=0.25, max_relative_iqr=0.15, valid_roi=None):
    """Projected bounding-box width/height at median visible depth, not full 3D size.

    The central box region is an approximation to object pixels; segmentation
    is not available in this detection-only baseline. Mixed depth is rejected.
    """
    empty = {"depth_m": None, "projected_bbox_width_m": None,
             "projected_bbox_height_m": None, "valid_depth_fraction": 0.0,
             "depth_iqr_m": None, "measurement_status": "insufficient_stereo_depth"}
    if not 0 < inner <= 1 or min_valid < 1:
        raise ValueError("Sampling requires 0 < inner <= 1 and positive min_valid")
    if len(xyxy) != 4 or not np.isfinite(xyxy).all():
        return empty
    h, w = depth_map.shape
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return empty
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    xa, xb = max(0, round(cx - bw * inner / 2)), min(w, round(cx + bw * inner / 2))
    ya, yb = max(0, round(cy - bh * inner / 2)), min(h, round(cy + bh * inner / 2))
    if xb <= xa or yb <= ya:
        return empty
    patch = depth_map[ya:yb, xa:xb]
    values = patch[np.isfinite(patch) & (patch > 0)]
    fraction = values.size / patch.size
    empty["valid_depth_fraction"] = float(fraction)
    if values.size < min_valid or fraction < min_valid_fraction:
        return empty
    z = float(np.median(values))
    q1, q3 = np.percentile(values, (25, 75))
    spread = float(q3 - q1)
    empty["depth_iqr_m"] = spread
    if spread / z > max_relative_iqr:
        empty["measurement_status"] = "mixed_depth_region"
        return empty
    empty["depth_m"] = z
    roi = valid_roi or (0, 0, w, h)
    rx, ry, rw, rh = roi
    if (x1 <= max(0, rx) or y1 <= max(0, ry)
            or x2 >= min(w, rx + rw) or y2 >= min(h, ry + rh)):
        empty["measurement_status"] = "partial_box_size_unknown"
        return empty
    empty.update(projected_bbox_width_m=bw * z / geometry.fx,
                 projected_bbox_height_m=bh * z / geometry.fy,
                 measurement_status="projected_bbox_estimate")
    return empty


class StereoDepthEstimator:
    def __init__(self, calibration_path, baseline_m=DEFAULT_BASELINE_M,
                 num_disparities=192, block_size=5, baseline_tolerance=0.05):
        import cv2
        if not np.isfinite(baseline_m) or baseline_m <= 0:
            raise ValueError("Baseline must be finite positive meters")
        if num_disparities <= 0 or num_disparities % 16:
            raise ValueError("num_disparities must be a positive multiple of 16")
        if block_size < 3 or block_size % 2 == 0:
            raise ValueError("block_size must be odd and at least 3")
        with np.load(Path(calibration_path), allow_pickle=False) as data:
            required = ("K1", "D1", "K2", "D2", "R", "T", "image_size")
            if any(key not in data for key in required):
                raise ValueError("Incomplete stereo calibration; run calibrate_stereo.py")
            k1, d1, k2, d2, rotation, translation, image_size = [data[k] for k in required]
        for matrix in (k1, d1, k2, d2, rotation, translation):
            if not np.isfinite(matrix).all():
                raise ValueError("Calibration contains nonfinite values")
        if k1.shape != (3, 3) or k2.shape != (3, 3) or rotation.shape != (3, 3):
            raise ValueError("Invalid calibration matrix shapes")
        if translation.size != 3 or image_size.shape != (2,):
            raise ValueError("Invalid translation/image size")
        if np.any(image_size <= 0) or np.any(image_size != np.rint(image_size)):
            raise ValueError("Calibration resolution must contain positive integers")
        self.image_size = tuple(int(v) for v in image_size)
        if num_disparities >= self.image_size[0]:
            raise ValueError("Disparity search must be narrower than the image")
        estimated_baseline = float(np.linalg.norm(translation))
        if estimated_baseline <= 0 or abs(estimated_baseline / baseline_m - 1) > baseline_tolerance:
            raise ValueError(f"Calibrated baseline {estimated_baseline:.4f} m disagrees with "
                             f"measured {baseline_m:.4f} m; check mounting/units and recalibrate")
        # Keep calibrated camera orientation, using the measured separation for metric scale.
        translation = translation.reshape(3, 1) * (baseline_m / estimated_baseline)
        r1, r2, p1, p2, _, self.left_roi, self.right_roi = cv2.stereoRectify(
            k1, d1, k2, d2, self.image_size, rotation, translation,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
        if abs(p2[1, 3]) > 1e-8 or p2[0, 3] >= 0:
            raise ValueError("Use horizontal stereo with the physical left camera as left input")
        self.geometry = StereoGeometry(float(p1[0, 0]), float(p1[1, 1]),
                                       float(p1[0, 2]), float(p2[0, 2]), baseline_m)
        # Full rectified intrinsics are needed for metric XYZ/table coordinates,
        # including the vertical principal point omitted by StereoGeometry.
        self.rectified_camera_matrix = p1[:, :3].copy()
        self._cv2 = cv2
        self._left_maps = cv2.initUndistortRectifyMap(k1, d1, r1, p1, self.image_size, cv2.CV_32FC1)
        self._right_maps = cv2.initUndistortRectifyMap(k2, d2, r2, p2, self.image_size, cv2.CV_32FC1)
        common = dict(numDisparities=num_disparities, blockSize=block_size,
                      P1=8 * block_size ** 2, P2=32 * block_size ** 2,
                      uniquenessRatio=10, speckleWindowSize=50, speckleRange=2,
                      disp12MaxDiff=1, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        self._left_matcher = cv2.StereoSGBM_create(minDisparity=0, **common)
        self._right_min = 1 - num_disparities
        self._right_matcher = cv2.StereoSGBM_create(minDisparity=self._right_min, **common)

    def estimate(self, left, right):
        cv2 = self._cv2
        if left.shape[:2][::-1] != self.image_size or right.shape[:2][::-1] != self.image_size:
            raise ValueError("Capture resolution differs from calibration; use matching mode")
        left_rect = cv2.remap(left, *self._left_maps, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right, *self._right_maps, cv2.INTER_LINEAR)
        lg = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
        rg = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)
        dl = self._left_matcher.compute(lg, rg).astype(np.float32) / 16.0
        dr = self._right_matcher.compute(rg, lg).astype(np.float32) / 16.0
        dr[dr <= self._right_min - 1] = np.nan
        valid = left_right_validity(dl, dr, self.left_roi, self.right_roi)
        depth = disparity_to_depth(dl, self.geometry, valid)
        return left_rect, right_rect, depth
