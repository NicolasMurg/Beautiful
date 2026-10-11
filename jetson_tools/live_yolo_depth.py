from pathlib import Path
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

assert torch.cuda.is_available(), "CUDA is unavailable"

MAX_DISPLAY_DEPTH = 3.0
CHECKPOINT = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
WINDOW = "YOLO + Indoor Depth"

print("Loading YOLO...")
yolo = YOLO(str(Path(__file__).resolve().parent / "yolo26n.pt"))

print("Loading depth model...")
processor = AutoImageProcessor.from_pretrained(
    CHECKPOINT, use_fast=False
)
depth_model = AutoModelForDepthEstimation.from_pretrained(
    CHECKPOINT, use_safetensors=True
).to("cuda").eval()

camera = cv2.VideoCapture(0, cv2.CAP_V4L2)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

clicked_pixel = None
last_report = 0.0


def select_pixel(event, x, y, flags, parameter):
    global clicked_pixel
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_pixel = (x, y)


try:
    if not camera.isOpened():
        raise RuntimeError("Camera could not open")

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, select_pixel)

    print("Click the right-hand image for estimated depth.")
    print("Press Q in the window to stop.")

    while True:
        ok, frame = camera.read()
        if not ok:
            print("Camera stopped delivering frames")
            break

        height, width = frame.shape[:2]

        detection = yolo.predict(
            frame,
            device=0,
            imgsz=320,
            conf=0.25,
            verbose=False,
        )[0]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Use the checkpoint's default preprocessing.
        # This can be slower than the previous 280-pixel setting.
        inputs = processor(images=rgb, return_tensors="pt")
        inputs = {
            key: value.to("cuda")
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            output = depth_model(**inputs)
            depth = torch.nn.functional.interpolate(
                output.predicted_depth.unsqueeze(1),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )[0, 0].float().cpu().numpy()

        valid_mask = np.isfinite(depth) & (depth > 0)
        valid_values = depth[valid_mask]

        # Report raw estimates, before display clipping.
        now = time.monotonic()
        if now - last_report >= 1.0:
            if valid_values.size:
                values = np.percentile(
                    valid_values, [5, 50, 95]
                )
                print(
                    "Depth estimates [m], p5/median/p95:",
                    np.round(values, 3),
                )
            else:
                print("No valid depth estimates")
            last_report = now

        # Fixed 0-3 m mapping. Invalid pixels are black.
        safe_depth = np.where(valid_mask, depth, 0.0)
        scaled = np.rint(
            np.clip(safe_depth / MAX_DISPLAY_DEPTH, 0, 1)
            * 255
        ).astype(np.uint8)

        heatmap = cv2.applyColorMap(
            scaled, cv2.COLORMAP_TURBO
        )
        heatmap[~valid_mask] = (0, 0, 0)

        # Inspect a pixel in the right-hand depth image.
        if clicked_pixel is not None:
            click_x, click_y = clicked_pixel
            x = click_x - width
            y = click_y

            if 0 <= x < width and 0 <= y < height:
                cv2.drawMarker(
                    heatmap, (x, y), (255, 255, 255),
                    cv2.MARKER_CROSS, 16, 2,
                )

                if valid_mask[y, x]:
                    value = float(depth[y, x])
                    label = f"Pixel estimate: {value:.2f} m"
                    if value >= MAX_DISPLAY_DEPTH:
                        label += " (color clipped)"
                else:
                    label = "Pixel estimate: invalid"

                cv2.putText(
                    heatmap, label, (10, height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2,
                )

        boxes = detection.plot()

        cv2.putText(
            boxes, "YOLO detections", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 255, 255), 2,
        )
        cv2.putText(
            heatmap, "Estimated depth | black = invalid",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 255, 255), 1,
        )

        # Add a fixed, labeled color legend.
        gradient = np.linspace(
            0, 255, width
        ).astype(np.uint8)
        gradient = np.tile(gradient, (18, 1))
        colorbar = cv2.applyColorMap(
            gradient, cv2.COLORMAP_TURBO
        )

        legend = np.zeros((55, width, 3), dtype=np.uint8)
        legend[:18] = colorbar

        for fraction in (0, 0.25, 0.5, 0.75, 1):
            meters = fraction * MAX_DISPLAY_DEPTH
            label = f"{meters:.1f}m"
            if fraction == 1:
                label += "+"

            position = int(fraction * (width - 1))
            position = min(position, width - 65)
            cv2.putText(
                legend, label, (position, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1,
            )

        left = np.vstack((
            boxes, np.zeros_like(legend)
        ))
        right = np.vstack((heatmap, legend))

        cv2.imshow(WINDOW, np.hstack((left, right)))

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
