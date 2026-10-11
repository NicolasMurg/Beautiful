#!/usr/bin/env python3
"""Run YOLO on frames read directly from a local V4L2 webcam."""
import argparse
import json
import math
from pathlib import Path
import time


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
    parser.add_argument('--reference-profile', type=Path,
                        help='Measured bottle JSON; click that bottle, then C to calibrate')
    parser.add_argument('--reference-distance-cm', type=float,
                        help='Measured camera-axis distance to the reference surface; '
                             'keep the bottle stationary after calibration')
    parser.add_argument('--stereo-calibration', type=Path,
                        help='Stereo calibration NPZ; enables geometric stereo depth/size')
    parser.add_argument('--right-camera', help='Right capture device; --camera is the left')
    parser.add_argument('--baseline-inches', type=float, default=5.0,
                        help='Measured lens-center spacing for stereo; default 5 inches')
    parser.add_argument('--stereo-disparities', type=int, default=192,
                        help='Stereo disparity search, positive multiple of 16')
    args = parser.parse_args()
    if args.frames < 0:
        parser.error('--frames must be zero or positive')
    if args.depth_every < 1:
        parser.error('--depth-every must be at least 1')
    if not args.model.is_file():
        parser.error(f'Model file does not exist: {args.model}')
    if args.stereo_calibration:
        if not args.right_camera or not args.stereo_calibration.is_file():
            parser.error('Stereo mode needs --right-camera and an existing --stereo-calibration')
        if args.depth_model or args.reference_profile or args.reference_distance_cm is not None:
            parser.error('Use stereo mode without monocular depth/reference arguments')
        from live_yolo_stereo import run
        run(args.camera, args.right_camera, args.stereo_calibration, args.model, args.device,
            args.baseline_inches * 0.0254, args.stereo_disparities, args.frames, args.no_display)
        return
    if args.right_camera:
        parser.error('--right-camera requires --stereo-calibration')
    reference = None
    if args.reference_distance_cm is not None:
        if (not math.isfinite(args.reference_distance_cm)
                or args.reference_distance_cm <= 0 or not args.reference_profile):
            parser.error('--reference-distance-cm needs a positive finite distance and '
                         '--reference-profile')
    if args.reference_profile:
        from reference_depth import (BottleReference, ReferenceDepthCalibrator,
                                     adjust_records, bottle_at_pixel)
        if not args.depth_model or 'metric' not in args.depth_model.lower():
            parser.error('Bottle scale correction requires a metric --depth-model')
        if args.no_display:
            parser.error('Reference mode needs the display to select and confirm the bottle')
        if args.reference_distance_cm is None and not args.calibration.is_file():
            parser.error('Reference mode needs --calibration or --reference-distance-cm; '
                         'bottle dimensions alone cannot establish distance')
        try:
            profile = BottleReference.load(args.reference_profile)
            reference = ReferenceDepthCalibrator(
                profile, None if args.reference_distance_cm is None
                else args.reference_distance_cm / 100.0)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            parser.error(f'Invalid reference profile: {exc}')
        # Always align the selected bottle's box and model depth to the same frame.
        args.depth_every = 1

    import cv2
    from ultralytics import YOLO

    print(f'Model: {args.model}; requested inference device: {args.device}', flush=True)
    model = YOLO(str(args.model))
    depth = None
    if args.depth_model:
        import depth as depth_tools
        depth = depth_tools.DepthEstimator(args.depth_model, args.device)
        depth_map, depth_frame, depth_latencies, fx = None, 0, [], None
    if reference is not None:
        from live_yolo_object_depth import object_depth_records
        if not depth.is_metric:
            raise RuntimeError('Reference scale correction requires metric model output')
        reference_records = []
        reference_selection = {'track_id': None}
        reference_report_at = 0.0
        reference_frame_size = None
        fy = None

        def select_reference(event, x, y, flags, parameter):
            if event == cv2.EVENT_LBUTTONDOWN:
                selected_id = bottle_at_pixel(reference_records, x, y)
                reference_selection['track_id'] = selected_id
                reference.reset('Click one tracked bottle' if selected_id is None
                                else f'Selected ID {selected_id}; upright/full view, then C')
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
            if reference is not None:
                cv2.setMouseCallback(window, select_reference)
                print(f'Reference: {profile.name}, height {profile.height_m * 100:.2f} cm, '
                      f'diameter {profile.diameter_m * 100:.2f} cm. '
                      'Click this specific bottle, C confirms upright/full view, R resets. '
                      'Depth runs every frame. Corrected depths remain estimates.', flush=True)
        print('Reading camera frames directly. Press Q in the window or Ctrl+C to stop.', flush=True)
        report_started = time.monotonic()
        while args.frames == 0 or count < args.frames:
            ok, frame = camera.read()
            if not ok or frame is None:
                raise RuntimeError(f'Camera stopped delivering frames after {count} successful predictions')
            if reference is not None:
                result = model.track(frame, persist=True, tracker='bytetrack.yaml',
                                     device=args.device, imgsz=320, conf=0.25, verbose=False)[0]
            else:
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
                if reference is not None:
                    frame_size = (frame.shape[1], frame.shape[0])
                    if frame_size != reference_frame_size:
                        fx = depth_tools.load_fx(args.calibration, frame_size[0])
                        fy = depth_tools.load_fy(args.calibration, frame_size[1])
                        reference_frame_size = frame_size
                    reference_records = object_depth_records(result, depth_map, count)
                    adjustment = reference.update(reference_records, count, frame_size)
                    corrected_records = adjust_records(reference_records, adjustment)
                    report_now = time.monotonic()
                    if count == 1 or report_now - reference_report_at >= 1:
                        print(json.dumps({'frame_id': count, 'reference': adjustment,
                                          'reference_status': reference.status,
                                          'objects': corrected_records}, allow_nan=False),
                              flush=True)
                        reference_report_at = report_now
                box_labels = []
                samples = ([(r['bbox_xyxy'], r['depth_m']) for r in corrected_records]
                           if reference is not None else
                           [(xyxy, depth_tools.box_depth(depth_map, xyxy))
                            for xyxy in result.boxes.xyxy.tolist()])
                for xyxy, z in samples:
                    if not depth.is_metric:
                        text = 'rel ?' if z is None else f'rel {z:.2f}'
                    else:
                        width = depth_tools.box_width_m(xyxy, z, fx)
                        text = ('Z ?' if z is None else f'Z~{z:.2f}m') + \
                               (' W ?' if width is None else f' W~{width * 100:.0f}cm')
                        if reference is not None and reference.adjustment is not None:
                            text += ' (ref est.)'
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
                if reference is not None:
                    cv2.putText(annotated, reference.status, (10, 75),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                    cv2.putText(annotated, 'Click reference | C: confirm | R: reset',
                                (10, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                    for record in reference_records:
                        if record['track_id'] == reference_selection['track_id']:
                            x1, y1, x2, y2 = (int(v) for v in record['bbox_xyxy'])
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, f'Device: {model.predictor.device} | Q: quit', (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow(window, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    break
                if reference is not None and key in (ord('c'), ord('C')):
                    reference.calibrate(reference_records, reference_selection['track_id'],
                                        count, reference_frame_size, (fx, fy))
                    print(reference.status, flush=True)
                elif reference is not None and key in (ord('r'), ord('R')):
                    reference.reset()
                    reference_selection['track_id'] = None
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
