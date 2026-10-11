#!/usr/bin/env python3
"""Align using four corner balls, each three inches clear of both adjacent cushions.

Live: SPACE freezes a stereo frame and detects the only four visible balls.
Check TL/TR/BR/BL labels; ENTER saves. R recaptures; Q/Esc cancels.
Use --measurements for four pixel/distance readings from another metric provider,
or --reference-type markers for the older flat-marker workflow.
See TABLE_WORKFLOW.md for placement, input formats and the next steps.
"""

from jetson_paths import PROJECT_ROOT, RUNTIME_ROOT

import argparse
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(PROJECT_ROOT))
from billiards_table import TableCalibration, sample_depth, unproject
from table_workflow import LOCAL, distinct_outputs, read_json, validate_pixel_frame


WINDOW = "Four-point table alignment"
CUSHION_GAP_CM = 7.62


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--measurements", type=Path, help="Headless reference-ball observations JSON")
    parser.add_argument("--reference-type", choices=("balls", "markers"), default="balls",
                        help="Default: four balls, each surface 3 inches from both cushion lines")
    parser.add_argument("--left-camera")
    parser.add_argument("--right-camera")
    parser.add_argument("--stereo-calibration", type=Path)
    parser.add_argument("--baseline-inches", type=float, help="Measured camera separation; required for live mode")
    parser.add_argument("--stereo-disparities", type=int, default=192)
    parser.add_argument("--stereo-backend", choices=("sgbm", "fast-foundation"), default="sgbm",
                        help="Neural stereo runs on the accepted frozen pair; live preview uses SGBM")
    parser.add_argument("--table-length-cm", type=float, required=True, help="Inner long dimension (table x axis)")
    parser.add_argument("--table-width-cm", type=float, required=True, help="Inner short dimension (table y axis)")
    parser.add_argument("--table-long-axis", choices=("left-to-right", "top-to-bottom"),
                        default="left-to-right",
                        help="Long dimension in frozen view: TL→TR or TL→BL")
    parser.add_argument("--ball-diameter-cm", type=float, required=True)
    parser.add_argument("--pocket-width-cm", type=float, required=True, help="Smallest measured mouth opening")
    parser.add_argument("--marker-inset-cm", type=float, help="Only for --reference-type markers: center inset from cushions")
    parser.add_argument("--model", type=Path, help="Optional existing YOLO ball weights for automatic ball detection")
    parser.add_argument("--manual-balls", action="store_true",
                        help="Freeze and click each ball center then edge in TL/TR/BR/BL order")
    parser.add_argument("--device", default="0", help="YOLO device, e.g. 0 or cpu")
    parser.add_argument("--yolo-confidence", type=float, default=0.20)
    parser.add_argument("--min-ball-radius-px", type=int, default=3)
    parser.add_argument("--max-ball-radius-px", type=int, default=40)
    parser.add_argument("--max-error-cm", type=float, default=1.0, help="Maximum rigid-fit and dimension error")
    parser.add_argument("--pockets-config", type=Path, help="Optional JSON list of measured pocket overrides")
    parser.add_argument("--output", type=Path, default=LOCAL / "table.json")
    return parser


def fit_options(args):
    options = dict(width_cm=args.table_length_cm, height_cm=args.table_width_cm,
                   ball_radius_cm=args.ball_diameter_cm / 2, pocket_width_cm=args.pocket_width_cm,
                   max_error_cm=args.max_error_cm)
    if args.table_long_axis == "top-to-bottom":
        options['width_cm'], options['height_cm'] = options['height_cm'], options['width_cm']
    if args.reference_type == "balls":
        if args.marker_inset_cm is not None:
            raise ValueError("--marker-inset-cm requires --reference-type markers. Ball alignment uses a 3-inch surface gap.")
        options["cushion_gap_cm"] = CUSHION_GAP_CM
        inset = CUSHION_GAP_CM + options["ball_radius_cm"]
    else:
        if args.marker_inset_cm is None:
            raise ValueError("Flat-marker alignment requires --marker-inset-cm.")
        options["inset_cm"] = inset = args.marker_inset_cm
    if not all(np.isfinite(v) and v > 0 for v in options.values()):
        raise ValueError("All dimensions and the calibration tolerance must be finite and positive.")
    if 2 * inset >= min(args.table_length_cm, args.table_width_cm):
        raise ValueError("Reference center inset must be less than half either table dimension.")
    if args.pockets_config:
        options["pockets"] = read_json(args.pockets_config)
        if not isinstance(options["pockets"], list):
            raise ValueError("Pocket configuration must be a JSON list.")
    return options


def order_ball_candidates(candidates):
    """Order exactly four corners clockwise from the top edge in the image."""
    if not isinstance(candidates, list) or len(candidates) != 4:
        count = len(candidates) if isinstance(candidates, list) else 0
        raise ValueError(f"Need exactly four visible balls; detected {count}. Remove other balls or fix missed/false detections, then R to recapture.")
    try:
        points = np.array([[ball["pixel"]["x"], ball["pixel"]["y"]] for ball in candidates], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Each reference ball needs its measured image center.") from exc
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("Reference ball image centers must be finite x/y points.")
    delta = points - points.mean(axis=0)
    indices = np.argsort(np.arctan2(delta[:, 1], delta[:, 0]))
    ordered = points[indices]
    # The top edge is the edge with the highest midpoint in the image.
    start = int(np.argmin((ordered[:, 1] + np.roll(ordered[:, 1], -1)) / 2))
    indices = np.roll(indices, -start)
    ordered = points[indices]
    edges = np.roll(ordered, -1, axis=0) - ordered
    crosses = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
    if (np.any(crosses <= 1.) or np.any(np.linalg.norm(edges, axis=1) < 8.)
            or ordered[0, 0] >= ordered[1, 0] or ordered[3, 0] >= ordered[2, 0]):
        raise ValueError("The four balls must form four distinct visible table corners, with the long axis left to right.")
    return [dict(candidates[int(index)], name=name)
            for name, index in zip(("TL", "TR", "BR", "BL"), indices)]


def manual_ball(center, edge, previous, size, minimum=3, maximum=40):
    center, edge = np.asarray(center, dtype=float), np.asarray(edge, dtype=float)
    if center.shape != (2,) or edge.shape != (2,) or not np.isfinite([center, edge]).all():
        raise ValueError("Click a finite center and visible ball edge.")
    radius = float(np.linalg.norm(edge - center))
    if not minimum <= radius <= maximum:
        raise ValueError(f"Ball radius must be {minimum:g}..{maximum:g}px. Click its visible edge again.")
    if np.any(center - radius < 0) or np.any(center + radius >= np.asarray(size)):
        raise ValueError("Reference ball must be fully inside the image.")
    if any(np.linalg.norm(center - [b['pixel']['x'], b['pixel']['y']]) <= radius + b['radius_px']
           for b in previous):
        raise ValueError("Choose a different corner ball; this selection overlaps an earlier ball.")
    return dict(pixel=dict(x=float(center[0]), y=float(center[1])), radius_px=radius,
                source="operator_center_and_edge", identity_confirmed=False)


def align_measurements(value, reference_type="balls", **options):
    size = validate_pixel_frame(value)
    if "camera_matrix" not in value:
        raise ValueError("Reference measurements require the matching calibrated camera_matrix.")
    if reference_type == "balls":
        balls = order_ball_candidates(value.get("balls"))
        return TableCalibration.fit_ball_measurements(
            balls, value["camera_matrix"], size, signature=value["calibration_signature"], **options)
    return TableCalibration.fit_measurements(
        value.get("markers"), value["camera_matrix"], size,
        signature=value["calibration_signature"], **options)


def neural_pair_depth(estimator, matcher, left, right):
    """Use full-resolution rectified stereo with the original reverse/ROI checks."""
    import cv2
    from stereo_depth import left_right_validity, disparity_to_depth
    if left.shape[:2][::-1] != estimator.image_size or right.shape[:2][::-1] != estimator.image_size:
        raise ValueError("Capture resolution differs from calibration")
    a = cv2.remap(left, *estimator._left_maps, cv2.INTER_LINEAR)
    b = cv2.remap(right, *estimator._right_maps, cv2.INTER_LINEAR)
    dl, dr = matcher.both(a, b)
    valid = left_right_validity(dl, dr, estimator.left_roi, estimator.right_roi)
    return a, b, disparity_to_depth(dl, estimator.geometry, valid)


def align_live(args, options):
    import cv2
    from live_billiards import calibration_signature
    from stereo_camera import StereoCamera
    from stereo_depth import StereoDepthEstimator

    if any(v is None for v in (args.left_camera, args.right_camera, args.stereo_calibration,
                               args.baseline_inches)):
        raise ValueError("Live alignment requires --left-camera, --right-camera, --stereo-calibration and --baseline-inches.")
    if not np.isfinite(args.baseline_inches) or args.baseline_inches <= 0:
        raise ValueError("Measured baseline must be finite and positive.")
    baseline = args.baseline_inches * 0.0254
    estimator = StereoDepthEstimator(args.stereo_calibration, baseline, args.stereo_disparities)
    signature = calibration_signature(args.stereo_calibration, estimator, baseline)
    matcher = None
    if args.stereo_backend == "fast-foundation":
        import os
        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
        import torch
        from fast_stereo_backend import FastStereoMatcher
        torch.set_num_threads(2)
        print("Loading neural stereo. SPACE will process one stationary frozen pair; please wait.", flush=True)
        matcher = FastStereoMatcher(args.device, max_disp=96, iters=4)
    width, height = estimator.image_size
    ball_mode = args.reference_type == "balls"
    manual = ball_mode and args.manual_balls
    detector = None
    if ball_mode and not manual:
        from billiards_vision import BallDetector
        detector = BallDetector(args.min_ball_radius_px, args.max_ball_radius_px,
                                yolo_model=args.model, yolo_device=args.device,
                                yolo_conf=args.yolo_confidence)
    frozen = None
    points = []
    references = []
    pending_center = None
    notice = ("Place ONLY 4 balls, each surface 3 inches from BOTH adjacent cushion lines."
              if ball_mode else "Keep markers flat and rig fixed. SPACE freezes one accepted image.")

    def on_mouse(event, x, y, flags, userdata):
        nonlocal pending_center, notice
        if (manual and event == cv2.EVENT_LBUTTONDOWN and frozen is not None
                and 0 <= x < width and 0 <= y < height and len(references) < 4):
            name = ("TL", "TR", "BR", "BL")[len(references)]
            if pending_center is None:
                pending_center = (x, y)
                notice = f"{name}: click the outer edge of this ball. Backspace undoes."
            else:
                try:
                    ball = manual_ball(pending_center, (x, y), references, (width, height),
                                       max(3, args.min_ball_radius_px), args.max_ball_radius_px)
                    references.append(dict(ball, name=name))
                    points.append(pending_center)
                    pending_center = None
                    notice = (f"Click {('TL', 'TR', 'BR', 'BL')[len(references)]} ball CENTER, then its EDGE."
                              if len(references) < 4 else "Check all four circles/labels. ENTER validates and saves.")
                except ValueError as exc:
                    notice = str(exc)
        if (not ball_mode and event == cv2.EVENT_LBUTTONDOWN and frozen is not None
                and 0 <= x < width and 0 <= y < height and len(points) < 4):
            points.append((x, y))

    if ball_mode:
        print(f"Use ONLY four equal-size balls, one near each corner. Leave a 3-inch (7.62 cm) gap\n"
              f"from each ball SURFACE to BOTH adjacent inner cushion lines. Center inset: "
              f"{CUSHION_GAP_CM + options['ball_radius_cm']:.4f} cm.\n" +
              ("SPACE: freeze | click CENTER then EDGE for TL, TR, BR, BL | Backspace: undo | ENTER: save"
               if manual else "SPACE: detect on frozen frame | check TL/TR/BR/BL | ENTER: save | R: recapture | Q: cancel"), flush=True)
    else:
        print("Place marker centers at the measured inset from both adjacent cushions.\n"
              "SPACE: freeze | click TL, TR, BR, BL | ENTER: save | R: recapture | Q: cancel", flush=True)
    try:
        with StereoCamera(args.left_camera, args.right_camera, width, height) as camera:
            cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(WINDOW, on_mouse)
            while True:
                if frozen is None:
                    left, right, capture = camera.read()
                    frame, _, depth = estimator.estimate(left, right)
                    age = time.monotonic() - capture["received_monotonic_s"]
                    fresh = capture["pair_accepted"] and np.isfinite(age) and 0 <= age <= 3.0
                else:
                    frame, depth = frozen
                view = np.zeros((height + 90, width, 3), dtype=np.uint8)
                view[:height] = frame
                for name, point in zip(("TL", "TR", "BR", "BL"), points):
                    center = tuple(int(round(v)) for v in point)
                    cv2.drawMarker(view, center, (0, 255, 255), cv2.MARKER_CROSS, 15, 2)
                    cv2.putText(view, name, (center[0] + 7, center[1] - 7),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                if manual:
                    for ball in references:
                        center = tuple(int(round(ball["pixel"][axis])) for axis in ("x", "y"))
                        cv2.circle(view, center, int(round(ball["radius_px"])), (0, 255, 255), 1)
                    if pending_center is not None:
                        cv2.drawMarker(view, pending_center, (255, 255, 0), cv2.MARKER_CROSS, 9, 1)
                status = (f"FROZEN: {len(points)}/4 {'balls' if ball_mode else 'markers'} | ENTER saves | R recaptures"
                          if frozen is not None else "LIVE | SPACE freezes | Q cancels")
                for i, line in enumerate((status, notice[:95], notice[95:190])):
                    cv2.putText(view, line, (8, height + 22 + 24 * i),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 235, 235), 1)
                cv2.imshow(WINDOW, view)
                key = cv2.waitKeyEx(20)
                if key < 0:
                    continue
                key &= 0xFF
                if key in (27, ord("q"), ord("Q")):
                    return None
                if key in (ord("r"), ord("R")):
                    frozen = None
                    points.clear()
                    references.clear()
                    pending_center = None
                    notice = "Live again. SPACE freezes a new frame."
                elif key in (8, 127):
                    if manual:
                        if pending_center is not None:
                            pending_center = None
                        elif references:
                            references.pop()
                            points.pop()
                        notice = "Click the next ball CENTER, then its EDGE in TL/TR/BR/BL order."
                    elif points and not ball_mode:
                        points.pop()
                elif key == ord(" ") and frozen is None:
                    if fresh:
                        if matcher is not None:
                            print("Processing accepted frozen stereo pair with neural stereo...", flush=True)
                            frame, _, depth = neural_pair_depth(estimator, matcher, left, right)
                        frozen = (frame.copy(), depth.copy())
                        diagnostic = RUNTIME_ROOT / 'table-depth-diagnostic-new'
                        diagnostic.mkdir(exist_ok=True)
                        np.savez_compressed(diagnostic / 'selected_pair.npz',
                                            left=left, right=right, left_rect=frame, depth=depth,
                                            backend=args.stereo_backend)
                        if manual:
                            notice = "Click TL ball CENTER, then its EDGE; repeat for TR, BR, BL."
                        elif ball_mode:
                            try:
                                polygon = np.array([[0, 0], [width - 1, 0],
                                                    [width - 1, height - 1], [0, height - 1]])
                                references = order_ball_candidates(detector.detect(frozen[0], polygon))
                                points = [(ball["pixel"]["x"], ball["pixel"]["y"]) for ball in references]
                                notice = "Check that TL/TR/BR/BL match the four corner balls. ENTER validates and saves."
                            except ValueError as exc:
                                references, points = [], []
                                notice = str(exc)
                                print(notice, file=sys.stderr)
                        else:
                            notice = "Click marker centers TL, TR, BR, BL on this frozen image."
                    else:
                        notice = "Capture timing rejected or stale. Wait for an accepted frame."
                elif key in (10, 13) and frozen is not None:
                    try:
                        if ball_mode:
                            if len(references) != 4:
                                raise ValueError("Select all four reference balls before saving." if manual else
                                                 "Exactly four reference balls must be detected before saving. R to recapture.")
                            if manual:
                                ordered = order_ball_candidates(references)
                                if [b['pixel'] for b in ordered] != [b['pixel'] for b in references]:
                                    raise ValueError("Selected order does not match TL/TR/BR/BL. Use R and select again.")
                            import json
                            (RUNTIME_ROOT / 'table-depth-diagnostic-new' /
                             'selected_balls.json').write_text(json.dumps(references, indent=2))
                            for name, point, ball in zip(("TL", "TR", "BR", "BL"), points, references):
                                sampling_radius = max(1.5, min(4.0, ball['radius_px'] * 0.22))
                                try:
                                    sample_depth(depth, point, radius_px=sampling_radius)
                                except ValueError as exc:
                                    raise ValueError(f"{name} ball: {exc}") from exc
                            centers = []
                            for point, ball in zip(points, references):
                                sampled = sample_depth(depth, point,
                                    radius_px=max(1.5, min(4.0, ball['radius_px'] * 0.22)))
                                center = unproject(point, sampled['depth_m'], estimator.rectified_camera_matrix)
                                center += options['ball_radius_cm'] / 100 * center / np.linalg.norm(center)
                                centers.append(center)
                            inset = options['cushion_gap_cm'] + options['ball_radius_cm']
                            for label, i, j, span in (('top', 0, 1, options['width_cm']),
                                                     ('bottom', 3, 2, options['width_cm']),
                                                     ('left', 0, 3, options['height_cm']),
                                                     ('right', 1, 2, options['height_cm'])):
                                print(f"Reference {label}: stereo center spacing "
                                      f"{100*np.linalg.norm(centers[j]-centers[i]):.2f} cm; "
                                      f"expected {span-2*inset:.2f} cm", flush=True)
                            return TableCalibration.fit_balls(
                                points, depth, estimator.rectified_camera_matrix, estimator.image_size,
                                ball_radii_px=[ball["radius_px"] for ball in references],
                                signature=signature, **options)
                        return TableCalibration.fit(
                            points, depth, estimator.rectified_camera_matrix, estimator.image_size,
                            signature=signature, **options)
                    except (ValueError, TypeError) as exc:
                        notice = str(exc)
                        print(f"Alignment rejected: {exc}", file=sys.stderr)
    finally:
        cv2.destroyAllWindows()


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.manual_balls and (args.reference_type != 'balls' or args.measurements):
            raise ValueError("--manual-balls requires live ball alignment.")
        distinct_outputs([args.measurements, args.stereo_calibration, args.pockets_config, args.model], [args.output])
        options = fit_options(args)
        if args.measurements:
            if any(v is not None for v in (args.left_camera, args.right_camera,
                                           args.stereo_calibration, args.baseline_inches)):
                raise ValueError("Choose --measurements or live camera arguments, not both.")
            calibration = align_measurements(read_json(args.measurements), reference_type=args.reference_type, **options)
        else:
            calibration = align_live(args, options)
        if calibration is None:
            print("Alignment cancelled; no profile saved.")
            return 1
        calibration.save(args.output)
        print(f"Saved table alignment: {args.output}\n"
              f"Fit residual {calibration.fit_error_cm:.3f} cm; "
              f"dimension error {calibration.max_dimension_error_cm:.3f} cm.\n"
              "Verify independent table points. Keep cameras and table fixed; reference balls/markers may now be moved.")
        return 0
    except (OSError, UnicodeError, ValueError, TypeError, RuntimeError, OverflowError) as exc:
        print(f"{parser.prog}: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
