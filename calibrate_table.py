#!/usr/bin/env python3
"""Align using four corner balls, each three inches clear of both adjacent cushions.

Live: SPACE freezes a stereo frame and detects the only four visible balls.
Check TL/TR/BR/BL labels; ENTER saves. R recaptures; Q/Esc cancels.
Use --measurements for four pixel/distance readings from another metric provider,
or --reference-type markers for the older flat-marker workflow.
See TABLE_WORKFLOW.md for placement, input formats and the next steps.
"""

import argparse
from pathlib import Path
import sys
import time

import numpy as np

from billiards_table import TableCalibration
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
    parser.add_argument("--table-length-cm", type=float, required=True, help="Inner long dimension (table x axis)")
    parser.add_argument("--table-width-cm", type=float, required=True, help="Inner short dimension (table y axis)")
    parser.add_argument("--ball-diameter-cm", type=float, required=True)
    parser.add_argument("--pocket-width-cm", type=float, required=True, help="Smallest measured mouth opening")
    parser.add_argument("--marker-inset-cm", type=float, help="Only for --reference-type markers: center inset from cushions")
    parser.add_argument("--model", type=Path, help="Optional existing YOLO ball weights for automatic ball detection")
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
    width, height = estimator.image_size
    ball_mode = args.reference_type == "balls"
    detector = None
    if ball_mode:
        from billiards_vision import BallDetector
        detector = BallDetector(args.min_ball_radius_px, args.max_ball_radius_px,
                                yolo_model=args.model, yolo_device=args.device,
                                yolo_conf=args.yolo_confidence)
    frozen = None
    points = []
    references = []
    notice = ("Place ONLY 4 balls, each surface 3 inches from BOTH adjacent cushion lines."
              if ball_mode else "Keep markers flat and rig fixed. SPACE freezes one accepted image.")

    def on_mouse(event, x, y, flags, userdata):
        if (not ball_mode and event == cv2.EVENT_LBUTTONDOWN and frozen is not None
                and 0 <= x < width and 0 <= y < height and len(points) < 4):
            points.append((x, y))

    if ball_mode:
        print(f"Use ONLY four equal-size balls, one near each corner. Leave a 3-inch (7.62 cm) gap\n"
              f"from each ball SURFACE to BOTH adjacent inner cushion lines. Center inset: "
              f"{CUSHION_GAP_CM + options['ball_radius_cm']:.4f} cm.\n"
              "SPACE: detect on frozen frame | check TL/TR/BR/BL | ENTER: save | R: recapture | Q: cancel", flush=True)
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
                    notice = "Live again. SPACE freezes a new frame."
                elif key in (8, 127) and points and not ball_mode:
                    points.pop()
                elif key == ord(" ") and frozen is None:
                    if fresh:
                        frozen = (frame.copy(), depth.copy())
                        if ball_mode:
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
                                raise ValueError("Exactly four reference balls must be detected before saving. R to recapture.")
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
