"""YOLO on rectified stereo frames with distance and projected box-size estimates."""

import argparse
import json
from pathlib import Path
import time

from stereo_camera import StereoCamera
from stereo_depth import DEFAULT_BASELINE_M, StereoDepthEstimator, object_dimensions


def stereo_object_records(result, depth_map, geometry, frame_id, valid_roi=None):
    records = []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return records
    coordinates = boxes.xyxy.cpu().tolist()
    classes = boxes.cls.cpu().tolist()
    ids = boxes.id.cpu().tolist() if boxes.id is not None else [None] * len(coordinates)
    for box, class_id, track_id in zip(coordinates, classes, ids):
        dimensions = object_dimensions(depth_map, box, geometry, valid_roi=valid_roi)
        records.append(dict(frame_id=frame_id,
                            track_id=int(track_id) if track_id is not None else None,
                            object=result.names[int(class_id)], bbox_xyxy=box,
                            depth_source='calibrated_stereo',
                            size_source='projected_yolo_bbox_at_median_depth', **dimensions))
    return records


def run(left_camera, right_camera, calibration_path, model_path, device='0',
        baseline_m=DEFAULT_BASELINE_M, num_disparities=192, frames=0, no_display=False):
    import cv2
    import numpy as np
    from ultralytics import YOLO
    estimator = StereoDepthEstimator(calibration_path, baseline_m, num_disparities)
    yolo = YOLO(str(model_path))
    width, height = estimator.image_size
    frame_id, last_report = 0, 0.0
    print(f'Stereo baseline {baseline_m * 100:.2f} cm; rectified focal length '
          f'{estimator.geometry.fx:.1f} px. USB exposure synchronization is unverified; '
          'use static/slow scenes and validate physical accuracy.', flush=True)
    with StereoCamera(left_camera, right_camera, width, height) as camera:
        try:
            while frames == 0 or frame_id < frames:
                left, right, capture = camera.read()
                frame_id += 1
                if not capture['pair_accepted']:
                    print(json.dumps(dict(frame_id=frame_id, status='capture_gap_rejected',
                                          objects=[], capture=capture)), flush=True)
                    if not no_display and left.shape == right.shape:
                        cv2.putText(left, 'Capture timing rejected | Q: quit', (10, 22),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
                        cv2.imshow('YOLO stereo: left | right', np.hstack((left, right)))
                        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
                            break
                    continue
                left_rect, right_rect, depth_map = estimator.estimate(left, right)
                result = yolo.track(left_rect, persist=True, tracker='bytetrack.yaml',
                                    device=device, imgsz=320, conf=0.25, verbose=False)[0]
                records = stereo_object_records(result, depth_map, estimator.geometry,
                                                frame_id, estimator.left_roi)
                now = time.monotonic()
                if frame_id == 1 or now - last_report >= 1 or frame_id == frames:
                    print(json.dumps(dict(frame_id=frame_id, objects=records, capture=capture,
                                          baseline_m=baseline_m,
                                          processing_age_ms=(now - capture['received_monotonic_s']) * 1000),
                                     allow_nan=False), flush=True)
                    last_report = now
                if not no_display:
                    annotated = result.plot()
                    for record in records:
                        z = record['depth_m']
                        bw, bh = record['projected_bbox_width_m'], record['projected_bbox_height_m']
                        label = 'Stereo Z: unknown' if z is None else f'Z~{z:.3f}m'
                        if bw is not None and bh is not None:
                            label += f' bbox W~{bw * 100:.1f} H~{bh * 100:.1f}cm'
                        else:
                            label += ' size unknown'
                        x, y = int(record['bbox_xyxy'][0]), int(record['bbox_xyxy'][3]) - 5
                        cv2.putText(annotated, label, (max(0, x), max(15, y)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                    cv2.putText(annotated, f'Stereo {baseline_m * 100:.1f}cm | projected size estimates | Q: quit',
                                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                    cv2.imshow('YOLO stereo: left | right', np.hstack((annotated, right_rect)))
                    if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
                        break
        except KeyboardInterrupt:
            print('Stopped by user.', flush=True)
        finally:
            if not no_display:
                cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--left-camera', required=True)
    parser.add_argument('--right-camera', required=True)
    parser.add_argument('--stereo-calibration', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=Path(__file__).parent / 'yolo26n.pt')
    parser.add_argument('--device', default='0')
    parser.add_argument('--baseline-inches', type=float, default=5.0)
    parser.add_argument('--stereo-disparities', type=int, default=192)
    parser.add_argument('--frames', type=int, default=0)
    parser.add_argument('--no-display', action='store_true')
    args = parser.parse_args()
    if args.frames < 0:
        parser.error('--frames must be nonnegative')
    if not args.model.is_file() or not args.stereo_calibration.is_file():
        parser.error('Existing YOLO weights and a stereo calibration file are required')
    run(args.left_camera, args.right_camera, args.stereo_calibration, args.model,
        args.device, args.baseline_inches * 0.0254, args.stereo_disparities,
        args.frames, args.no_display)


if __name__ == '__main__':
    main()
