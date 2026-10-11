#!/usr/bin/env python3
"""Calibrate webcam intrinsics with a printed checkerboard; writes camera_calibration.yaml.

Hold the board at varied angles/positions across the frame. SPACE saves a view when
corners are found, C calibrates (needs >= 10 views), Q quits. Recalibrate if the
resolution or lens/focus changes.
"""
import argparse
from pathlib import Path
import sys

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', default='0', help='Camera index or /dev/video path')
    parser.add_argument('--cols', type=int, default=9, help='Inner corners per row')
    parser.add_argument('--rows', type=int, default=6, help='Inner corners per column')
    parser.add_argument('--square-mm', type=float, required=True, help='Measured square size in mm')
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parent / 'camera_calibration.yaml')
    args = parser.parse_args()

    pattern = (args.cols, args.rows)
    board = np.zeros((args.cols * args.rows, 3), np.float32)
    board[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * args.square_mm / 1000.0
    source = int(args.camera) if args.camera.isdecimal() else args.camera
    backend = cv2.CAP_V4L2 if sys.platform.startswith('linux') else cv2.CAP_ANY
    camera = cv2.VideoCapture(source, backend)
    if not camera.isOpened():
        raise SystemExit(f'Camera {args.camera} could not open')
    # Same capture mode as live_yolo.py so the intrinsics match.
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    object_points, image_points, size = [], [], None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise SystemExit('Camera stopped delivering frames')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            size = gray.shape[::-1]
            found, corners = cv2.findChessboardCorners(gray, pattern)
            view = frame.copy()
            if found:
                cv2.drawChessboardCorners(view, pattern, corners, found)
            cv2.putText(view, f'views: {len(image_points)} | SPACE save, C calibrate, Q quit',
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow('Calibrate', view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') and found:
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                object_points.append(board)
                image_points.append(corners)
                print(f'Saved view {len(image_points)}', flush=True)
            elif key in (ord('c'), ord('C')):
                if len(image_points) < 10:
                    print('Need at least 10 views', flush=True)
                    continue
                break
            elif key in (ord('q'), ord('Q'), 27):
                return
    finally:
        camera.release()
        cv2.destroyAllWindows()

    rms, camera_matrix, dist, _, _ = cv2.calibrateCamera(object_points, image_points, size, None, None)
    fs = cv2.FileStorage(str(args.output), cv2.FILE_STORAGE_WRITE)
    fs.write('image_width', size[0])
    fs.write('image_height', size[1])
    fs.write('camera_matrix', camera_matrix)
    fs.write('dist_coeffs', dist)
    fs.write('rms_reprojection_error_px', rms)
    fs.release()
    print(f'RMS reprojection error {rms:.3f} px; fx {camera_matrix[0, 0]:.1f}, '
          f'fy {camera_matrix[1, 1]:.1f}; wrote {args.output}', flush=True)


if __name__ == '__main__':
    main()
