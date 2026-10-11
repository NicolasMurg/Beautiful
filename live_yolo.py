#!/usr/bin/env python3
"""Run YOLO on frames read directly from a local V4L2 webcam."""
import argparse
from pathlib import Path
import time

import cv2
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path,
                        default=Path(__file__).resolve().parent / 'yolo26n.pt')
    parser.add_argument('--camera', default='0', help='Camera index or /dev/video path')
    parser.add_argument('--device', default='0', help='CUDA index, or cpu explicitly')
    parser.add_argument('--frames', type=int, default=0, help='Stop after N frames; 0 runs until Q')
    parser.add_argument('--no-display', action='store_true', help='Validate capture/inference without a window')
    parser.add_argument('--depth-model', default=None,
                        help='Optional Hugging Face depth model id or local path, e.g. '
                             'depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf')
    parser.add_argument('--depth-every', type=int, default=3, help='Run depth every N frames')
    parser.add_argument('--calibration', type=Path,
                        default=Path(__file__).resolve().parent / 'camera_calibration.yaml',
                        help='calibrate_camera.py output; enables metric width when present')
    args = parser.parse_args()
    if args.frames < 0:
        parser.error('--frames must be zero or positive')
    if args.depth_every < 1:
        parser.error('--depth-every must be at least 1')
    if not args.model.is_file():
        parser.error(f'Model file does not exist: {args.model}')

    print(f'Model: {args.model}; requested inference device: {args.device}', flush=True)
    model = YOLO(str(args.model))
    depth = None
    if args.depth_model:
        import depth as depth_tools
        depth = depth_tools.DepthEstimator(args.depth_model, args.device)
        depth_map, depth_frame, depth_latencies, fx = None, 0, [], None
    source = int(args.camera) if args.camera.isdecimal() else args.camera
    camera = cv2.VideoCapture(source, cv2.CAP_V4L2)
    window = 'Live YOLO'
    count = 0
    try:
        if not camera.isOpened():
            raise RuntimeError(f'Camera {args.camera} could not open')
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not args.no_display:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window, 800, 600)
        print('Reading camera frames directly. Press Q in the window or Ctrl+C to stop.', flush=True)
        report_started = time.monotonic()
        while args.frames == 0 or count < args.frames:
            ok, frame = camera.read()
            if not ok or frame is None:
                raise RuntimeError(f'Camera stopped delivering frames after {count} successful predictions')
            result = model.predict(frame, device=args.device, imgsz=320,
                                   conf=0.25, verbose=False)[0]
            count += 1
            if count == 1:
                print(f'First prediction succeeded: frame {frame.shape[1]}x{frame.shape[0]}, '
                      f'{len(result.boxes)} detections; inference device {model.predictor.device}; '
                      f'result device {result.boxes.data.device}',
                      flush=True)
                report_started = time.monotonic()
            if depth is not None:
                if depth_map is None or (count - 1) % args.depth_every == 0:
                    depth_map, depth_frame = depth.estimate(frame), count
                    depth_latencies.append(depth.last_latency_ms)
                    if len(depth_latencies) == 1:
                        fx = depth_tools.load_fx(args.calibration, frame.shape[1])
                        print(f'First depth map succeeded: {depth.last_latency_ms:.0f} ms; '
                              f'range {depth_map.min():.2f}..{depth_map.max():.2f} '
                              f'{"m" if depth.is_metric else "(relative)"}; '
                              f'fx {"%.1f px" % fx if fx else "unknown (width disabled)"}',
                              flush=True)
                box_labels = []
                for xyxy in result.boxes.xyxy.tolist():
                    z = depth_tools.box_depth(depth_map, xyxy)
                    if not depth.is_metric:
                        text = 'rel ?' if z is None else f'rel {z:.2f}'
                    else:
                        width = depth_tools.box_width_m(xyxy, z, fx)
                        text = ('Z ?' if z is None else f'Z~{z:.2f}m') + \
                               (' W ?' if width is None else f' W~{width * 100:.0f}cm')
                    box_labels.append((xyxy, text))
            if not args.no_display:
                annotated = result.plot()
                if depth is not None:
                    for xyxy, text in box_labels:
                        cv2.putText(annotated, text, (int(xyxy[0]) + 2, int(xyxy[3]) - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                    cv2.putText(annotated, f'Depth: {args.depth_model.split("/")[-1]} '
                                f'{depth.last_latency_ms:.0f} ms, {count - depth_frame} frames old',
                                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(annotated, f'Device: {model.predictor.device} | Q: quit', (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow(window, annotated)
                if (cv2.waitKey(1) & 0xFF) in (ord('q'), ord('Q'), 27):
                    break
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
            if count > 1 and (count - 1) % 30 == 0:
                elapsed = time.monotonic() - report_started
                print(f'{count} frames processed; last 30 frames: {30 / elapsed:.1f} FPS; '
                      f'{len(result.boxes)} current detections', flush=True)
                if depth is not None:
                    recent = sorted(depth_latencies[-30:])
                    print(f'  depth median {recent[len(recent) // 2]:.0f} ms; '
                          + ', '.join(text for _, text in box_labels), flush=True)
                report_started = time.monotonic()
    except KeyboardInterrupt:
        print('\nStopped by user.', flush=True)
    finally:
        camera.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        print(f'Camera released; {count} frames processed.', flush=True)


if __name__ == '__main__':
    main()
