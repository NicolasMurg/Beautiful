"""Calibrate a fixed USB stereo rig using a measured checkerboard.

SPACE saves a stationary paired view; C solves after at least 15 views; Q quits.
Square size establishes metric calibration scale; --baseline-inches validates it.
"""

import argparse
from pathlib import Path

import numpy as np

from stereo_camera import StereoCamera
from stereo_depth import DEFAULT_BASELINE_M


def solve_calibration(object_points, left_points, right_points, image_size,
                      expected_baseline_m=DEFAULT_BASELINE_M, max_rms_px=2.0):
    import cv2
    if len(object_points) < 15 or not (len(object_points) == len(left_points) == len(right_points)):
        raise ValueError('At least 15 paired calibration views are required')
    rms1, k1, d1, _, _ = cv2.calibrateCamera(object_points, left_points, image_size, None, None)
    rms2, k2, d2, _, _ = cv2.calibrateCamera(object_points, right_points, image_size, None, None)
    rms, k1, d1, k2, d2, rotation, translation, _, _ = cv2.stereoCalibrate(
        object_points, left_points, right_points, k1, d1, k2, d2, image_size,
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
    if not np.isfinite([rms1, rms2, rms]).all() or max(rms1, rms2, rms) > max_rms_px:
        raise ValueError(f'Calibration error too high: left={rms1:.3f}, right={rms2:.3f}, '
                         f'stereo={rms:.3f} px. Capture more varied/sharp paired views.')
    baseline = float(np.linalg.norm(translation))
    if abs(baseline / expected_baseline_m - 1) > 0.05:
        raise ValueError(f'Calibration estimates {baseline * 100:.2f} cm separation; '
                         f'expected {expected_baseline_m * 100:.2f} cm. Check square '
                         'measurement, camera order and mounting; do not reuse old calibration.')
    r1, r2, p1, p2, _, _, _ = cv2.stereoRectify(
        k1, d1, k2, d2, image_size, rotation, translation,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    if abs(p2[1, 3]) > 1e-8 or p2[0, 3] >= 0:
        raise ValueError('Use a horizontal rig; check physical left/right camera order')
    errors = []
    for lp, rp in zip(left_points, right_points):
        left_rect = cv2.undistortPoints(lp, k1, d1, R=r1, P=p1)
        right_rect = cv2.undistortPoints(rp, k2, d2, R=r2, P=p2)
        errors.extend(np.abs(left_rect[:, 0, 1] - right_rect[:, 0, 1]).tolist())
    epipolar_p95 = float(np.percentile(errors, 95))
    if epipolar_p95 > max_rms_px:
        raise ValueError(f'Rectified vertical mismatch p95={epipolar_p95:.3f} px; recalibrate')
    return dict(K1=k1, D1=d1, K2=k2, D2=d2, R=rotation, T=translation,
                image_size=np.array(image_size), expected_baseline_m=expected_baseline_m,
                calibrated_baseline_m=baseline, rms_left_px=rms1, rms_right_px=rms2,
                rms_stereo_px=rms, epipolar_p95_px=epipolar_p95)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--left-camera', required=True)
    parser.add_argument('--right-camera', required=True)
    parser.add_argument('--cols', type=int, default=9, help='Inner corners per row')
    parser.add_argument('--rows', type=int, default=6, help='Inner corners per column')
    parser.add_argument('--square-mm', type=float, required=True, help='Actual measured printed square side')
    parser.add_argument('--baseline-inches', type=float, default=5.0)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--output', type=Path, default=Path('stereo_calibration.npz'))
    args = parser.parse_args()
    if min(args.cols, args.rows) < 3 or min(args.width, args.height) <= 0:
        parser.error('Use at least 3 inner corners on each axis and positive resolution')
    if not np.isfinite([args.square_mm, args.baseline_inches]).all() or min(
            args.square_mm, args.baseline_inches) <= 0:
        parser.error('Square size and baseline must be finite positive values')
    import cv2
    pattern = (args.cols, args.rows)
    board = np.zeros((args.cols * args.rows, 3), np.float32)
    board[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * args.square_mm / 1000
    objects, left_points, right_points = [], [], []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    image_size = None
    print('Keep cameras fixed. Hold checkerboard still before SPACE. '
          'Collect >=15 varied views across the shared field of view.', flush=True)
    with StereoCamera(args.left_camera, args.right_camera, args.width, args.height) as camera:
        try:
            while True:
                left, right, metadata = camera.read()
                if left.shape != right.shape or left.shape[:2][::-1] != (args.width, args.height):
                    raise RuntimeError('Both cameras must deliver the requested calibration resolution')
                image_size = (args.width, args.height)
                views, corners = [], []
                for image in (left, right):
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    found, points = cv2.findChessboardCorners(gray, pattern)
                    view = image.copy()
                    if found:
                        points = cv2.cornerSubPix(gray, points, (11, 11), (-1, -1), criteria)
                        cv2.drawChessboardCorners(view, pattern, points, True)
                    corners.append(points if found else None)
                    cv2.putText(view, f'Views: {len(objects)} | SPACE: save | C: solve | Q: quit',
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                    views.append(view)
                cv2.imshow('Stereo calibration: left | right', np.hstack(views))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    return
                if key == ord(' '):
                    if not metadata['pair_accepted'] or any(p is None for p in corners):
                        print('Pair not saved: show all corners in both views and hold still', flush=True)
                        continue
                    objects.append(board.copy())
                    left_points.append(corners[0])
                    right_points.append(corners[1])
                    print(f'Saved paired view {len(objects)}', flush=True)
                if key in (ord('c'), ord('C')):
                    if len(objects) < 15:
                        print('Need at least 15 paired views', flush=True)
                        continue
                    try:
                        calibration = solve_calibration(objects, left_points, right_points,
                                                        image_size, args.baseline_inches * 0.0254)
                    except ValueError as exc:
                        print(f'Calibration rejected: {exc}', flush=True)
                        continue
                    np.savez_compressed(args.output, **calibration)
                    print(f'Saved {args.output}; stereo RMS {calibration["rms_stereo_px"]:.3f} px; '
                          f'baseline {calibration["calibrated_baseline_m"] * 100:.2f} cm. '
                          'Validate distances on separate measured objects.', flush=True)
                    return
        finally:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
