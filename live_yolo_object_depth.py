#!/usr/bin/env python3
"""Track webcam objects and overlay raw metric depth estimates on the YOLO view.

Each estimate is the median of valid predictions in the center of that object's
box, using detection and depth from the same frame. These are monocular depth
estimates, not measured distances. No normalization or display clipping is used.
"""

import argparse
import json
from pathlib import Path
import sys
import time

from depth import box_depth


CHECKPOINT = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
WINDOW = "YOLO object depth"


def object_depth_records(result, depth_map, frame_id, sample_fraction=0.4):
    """Associate current boxes/IDs with raw depth; missing estimates remain None."""
    records = []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return records

    coordinates = boxes.xyxy.cpu().tolist()
    classes = boxes.cls.cpu().tolist()
    track_ids = (
        boxes.id.cpu().tolist() if boxes.id is not None
        else [None] * len(coordinates)
    )
    for xyxy, class_id, track_id in zip(coordinates, classes, track_ids):
        records.append({
            "frame_id": frame_id,
            "track_id": int(track_id) if track_id is not None else None,
            "object": result.names[int(class_id)],
            "depth_m": box_depth(depth_map, xyxy, inner=sample_fraction),
            "bbox_xyxy": xyxy,
        })
    return records


def depth_label(record):
    """Format an object's estimate without clamping it or claiming measurement."""
    value = record["depth_m"]
    estimate = "unknown" if value is None else f"{value:.3f} m (est.)"
    identity = (
        "" if record["track_id"] is None else f" [ID {record['track_id']}]"
    )
    return f"Depth of {record['object']}{identity}: {estimate}"


def draw_depth_labels(image, records, cv2):
    """Draw readable depth text inside each object's box on the live image."""
    height, width = image.shape[:2]
    for record in records:
        text = depth_label(record)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, scale, thickness
        )
        if text_width + 8 > width:
            scale *= max(1, width - 8) / (text_width + 8)
            (text_width, text_height), baseline = cv2.getTextSize(
                text, font, scale, thickness
            )
        x1, y1, _, y2 = record["bbox_xyxy"]
        x = max(0, min(int(x1) + 3, width - text_width - 8))
        y = max(text_height + 6, min(int(y2) - 6, height - baseline - 4))
        if y < y1:
            y = max(text_height + 6, min(int(y1) + text_height + 6,
                                       height - baseline - 4))
        color = (0, 255, 255) if record["depth_m"] is not None else (0, 165, 255)
        cv2.rectangle(image, (x, y - text_height - 4),
                      (min(width - 1, x + text_width + 8), y + baseline + 3),
                      (0, 0, 0), -1)
        cv2.putText(image, text, (x + 4, y), font, scale, color, thickness,
                    cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path,
                        default=Path(__file__).resolve().parent / "yolo26n.pt")
    parser.add_argument("--camera", default="0", help="Camera index or /dev/video path")
    parser.add_argument("--device", default="0", help="CUDA index, or cpu explicitly")
    parser.add_argument("--sample-fraction", type=float, default=0.4,
                        help="Central fraction of each box's width/height to sample")
    parser.add_argument("--frames", type=int, default=0,
                        help="Stop after N frames; 0 runs until Q or Ctrl+C")
    parser.add_argument("--no-display", action="store_true",
                        help="Print numerical estimates without opening a window")
    args = parser.parse_args()
    if not 0 < args.sample_fraction <= 1:
        parser.error("--sample-fraction must be greater than 0 and at most 1")
    if args.frames < 0:
        parser.error("--frames must be zero or positive")
    if args.device != "cpu" and not args.device.isdecimal():
        parser.error("--device must be a CUDA index such as 0, or cpu")

    # Keep heavyweight libraries out of argument parsing and helper imports.
    import cv2
    import torch
    from ultralytics import YOLO
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu for an explicit CPU run")
    device = torch.device("cpu" if args.device == "cpu" else f"cuda:{args.device}")

    print(f"Loading YOLO: {args.model}", flush=True)
    yolo = YOLO(str(args.model))
    print(f"Loading metric indoor depth model: {CHECKPOINT}", flush=True)
    processor = AutoImageProcessor.from_pretrained(CHECKPOINT, use_fast=False)
    depth_model = AutoModelForDepthEstimation.from_pretrained(
        CHECKPOINT, use_safetensors=True
    ).to(device).eval()

    source = int(args.camera) if args.camera.isdecimal() else args.camera
    backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
    camera = cv2.VideoCapture(source, backend)
    frame_id = 0
    last_report = 0.0
    try:
        if not camera.isOpened():
            raise RuntimeError(f"Camera {args.camera} could not open")
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not args.no_display:
            cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        print("Depth is estimated camera-space Z in meters; validate against measured distances.",
              flush=True)
        print("Numerical object records print once per second. Press Q/Esc or Ctrl+C to stop.",
              flush=True)

        while args.frames == 0 or frame_id < args.frames:
            ok, frame = camera.read()
            if not ok or frame is None:
                raise RuntimeError("Camera stopped delivering frames")
            frame_id += 1
            received_at = time.monotonic()
            height, width = frame.shape[:2]

            # Persist track IDs across consecutive frames from this camera.
            # Pass the original frame to both models before drawing any overlay.
            result = yolo.track(
                frame, persist=True, tracker="bytetrack.yaml", device=args.device,
                imgsz=320, conf=0.25, verbose=False
            )[0]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inputs = processor(images=rgb, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.inference_mode():
                output = depth_model(**inputs)
                depth_map = torch.nn.functional.interpolate(
                    output.predicted_depth.unsqueeze(1), size=(height, width),
                    mode="bilinear", align_corners=False
                )[0, 0].float().cpu().numpy()

            # Only the raw resized prediction enters the median calculation.
            records = object_depth_records(result, depth_map, frame_id,
                                           args.sample_fraction)
            now = time.monotonic()
            if frame_id == 1 or now - last_report >= 1.0 or frame_id == args.frames:
                print(json.dumps({"frame_id": frame_id,
                                  "frame_received_monotonic_s": received_at,
                                  "objects": records}, allow_nan=False), flush=True)
                last_report = now

            if not args.no_display:
                annotated = result.plot()
                draw_depth_labels(annotated, records, cv2)
                cv2.putText(annotated, "Estimated object depth | Q/Esc: quit",
                            (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow(WINDOW, annotated)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                    break
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
    except KeyboardInterrupt:
        print("\nStopped by user.", flush=True)
    finally:
        camera.release()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
