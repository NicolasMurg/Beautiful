#!/usr/bin/env python3
"""Capture one stationary stereo ball measurement batch for map_table_state.py.

Saves metric surface depths, image positions, identity hints and an annotated
review image. This does not confirm identities or completeness. The files are
snapshots; capture again after any ball or camera moves. See TABLE_WORKFLOW.md.
"""

import argparse
import copy
from pathlib import Path
import sys
import time
import uuid

import numpy as np

from billiards_session import PoolSession
from billiards_table import TableCalibration
from billiards_vision import BallDetector
from live_billiards import calibration_signature
from stereo_camera import StereoCamera
from stereo_depth import StereoDepthEstimator
from table_workflow import (LOCAL, atomic_write, distinct_outputs, json_bytes,
                            png_bytes, read_json)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-camera", required=True)
    parser.add_argument("--right-camera", required=True)
    parser.add_argument("--stereo-calibration", type=Path, required=True)
    parser.add_argument("--baseline-inches", type=float, required=True, help="Actual measured camera separation")
    parser.add_argument("--stereo-disparities", type=int, default=192)
    parser.add_argument("--table-profile", type=Path, default=LOCAL / "table.json")
    parser.add_argument("--model", type=Path, help="Optional existing YOLO ball weights; OpenCV proposals also run")
    parser.add_argument("--class-map", type=Path, help="Optional JSON mapping model classes to ball numbers")
    parser.add_argument("--device", default="0", help="YOLO device, e.g. 0 or cpu")
    parser.add_argument("--yolo-confidence", type=float, default=0.20)
    parser.add_argument("--min-ball-radius-px", type=int, default=3)
    parser.add_argument("--max-ball-radius-px", type=int, default=40)
    parser.add_argument("--distance-uncertainty-m", type=float, default=0.,
                        help="Additional measured depth error allowance beyond sampled depth spread")
    parser.add_argument("--timeout-seconds", type=float, default=30., help="Wait limit for a fresh stationary scan")
    parser.add_argument("--output", type=Path, default=LOCAL / "measurements.json")
    parser.add_argument("--image", type=Path, default=LOCAL / "measurement-review.png")
    return parser


def snapshot_measurements(session, *, frame_id, observed_at, distance_uncertainty_m=0.):
    """Export raw observations so the mapper performs sphere correction once."""
    if session.snapshot is None:
        raise ValueError("Scan a stationary physical table before exporting measurements.")
    if not np.isfinite(distance_uncertainty_m) or distance_uncertainty_m < 0:
        raise ValueError("Distance uncertainty must be finite and nonnegative.")
    measurements = []
    for candidate in session.candidates:
        if candidate["key"] in session.ignored:
            continue
        record = {"key": candidate["key"], "frame_id": frame_id, "camera_id": "rectified_left",
                  "pixel": copy.deepcopy(candidate["pixel"]), "radius_px": candidate["radius_px"],
                  "confidence": candidate.get("confidence", 1.),
                  "distance_kind": "camera_z", "measurement_reference": "surface"}
        for name in ("suggested_id", "suggestion_source", "model_suggested_id", "identity_conflict"):
            if name in candidate:
                record[name] = candidate[name]
        if "error" in candidate:
            record["error"] = candidate["error"]
        else:
            position = candidate["position"]
            record.update(distance_m=position["depth_m"],
                          distance_uncertainty_m=position["depth_iqr_m"] + distance_uncertainty_m,
                          pixel_uncertainty_px=1.0)
        measurements.append(record)
    return {"version": 1, "calibration_signature": copy.deepcopy(session.calibration.signature),
            "image_size": list(session.calibration.image_size), "pixel_space": "rectified",
            "observed_at": observed_at, "source": "stationary_stereo_capture",
            "reviewed": False, "measurements": measurements}


def review_image(session):
    import cv2
    frame = session.snapshot
    height, width = frame.shape[:2]
    image = np.zeros((height + 70, width, 3), dtype=np.uint8)
    image[:height] = frame
    cv2.polylines(image, [np.rint(session.calibration.polygon).astype(np.int32)], True, (90, 230, 150), 1)
    for candidate in session.candidates:
        pixel = candidate["pixel"]
        x, y = int(round(pixel["x"])), int(round(pixel["y"]))
        color = (80, 80, 255) if "error" in candidate else (0, 230, 255)
        cv2.circle(image, (x, y), max(3, round(candidate["radius_px"])), color, 1)
        cv2.putText(image, f"key {candidate['key']}", (max(0, x - 20), max(15, y - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    for i, line in enumerate(("REVIEW SNAPSHOT: keys are NOT ball numbers.",
                              "Check every real ball; label keys in JSON. Red = unusable measurement.")):
        cv2.putText(image, line, (8, height + 24 + i * 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (230, 235, 235), 1)
    return image


def capture_once(args):
    baseline = args.baseline_inches * 0.0254
    estimator = StereoDepthEstimator(args.stereo_calibration, baseline, args.stereo_disparities)
    signature = calibration_signature(args.stereo_calibration, estimator, baseline)
    calibration = TableCalibration.load(args.table_profile, signature)
    yolo = None
    if args.model:
        from ultralytics import YOLO
        yolo = YOLO(str(args.model))
    class_map = read_json(args.class_map) if args.class_map else None
    detector = BallDetector(args.min_ball_radius_px, args.max_ball_radius_px,
                            yolo_model=yolo, yolo_device=args.device,
                            yolo_conf=args.yolo_confidence, class_to_ball_id=class_map)
    session = PoolSession(calibration, detector)
    start = time.monotonic()
    print("Waiting for at least one second of a still table. Keep hands and cues out of view.", flush=True)
    with StereoCamera(args.left_camera, args.right_camera, *estimator.image_size) as camera:
        while time.monotonic() - start < args.timeout_seconds:
            left, right, metadata = camera.read()
            frame, _, depth = estimator.estimate(left, right)
            now = time.monotonic()
            session.observe(frame, metadata["received_monotonic_s"],
                            accepted=metadata["pair_accepted"], now=now)
            if session.settled(now):
                observed_at = time.time() - (now - metadata["received_monotonic_s"])
                session.scan(frame, depth, now=now)
                return snapshot_measurements(
                    session, frame_id=uuid.uuid4().hex, observed_at=observed_at,
                    distance_uncertainty_m=args.distance_uncertainty_m), review_image(session)
    raise ValueError("No fresh stationary scan before timeout. Check capture timing, motion and camera view.")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        distinct_outputs([args.table_profile, args.stereo_calibration, args.model, args.class_map],
                         [args.output, args.image])
        if args.image.suffix.lower() != ".png":
            raise ValueError("--image must have a .png extension.")
        if not all(np.isfinite(v) and v > 0 for v in (args.baseline_inches, args.timeout_seconds)):
            raise ValueError("Baseline and timeout must be finite and positive.")
        if not np.isfinite(args.distance_uncertainty_m) or args.distance_uncertainty_m < 0:
            raise ValueError("Distance uncertainty must be finite and nonnegative.")
        if args.class_map and not args.model:
            raise ValueError("--class-map requires --model.")
        if args.model and not args.model.is_file():
            raise ValueError("--model must be an existing local weights file.")
        batch, image = capture_once(args)
        content, image_content = json_bytes(batch), png_bytes(image)
        atomic_write(args.output, content)
        atomic_write(args.image, image_content)
        failed = sum("error" in item for item in batch["measurements"])
        print(f"Saved {len(batch['measurements'])} candidates ({failed} unusable): {args.output}\n"
              f"Review image: {args.image}\n"
              "Check for missed balls and review identities before requesting shots.")
        return 0
    except (OSError, UnicodeError, ValueError, TypeError, RuntimeError, OverflowError) as exc:
        print(f"{parser.prog}: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
