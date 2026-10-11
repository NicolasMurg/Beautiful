import cv2
import numpy as np
import torch
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

assert torch.cuda.is_available(), "CUDA is unavailable"

checkpoint = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"

print("Loading YOLO...")
yolo = YOLO("yolo26n.pt")

print("Loading depth model; first run downloads its files...")
processor = AutoImageProcessor.from_pretrained(
    checkpoint, use_fast=False
)
depth_model = AutoModelForDepthEstimation.from_pretrained(
    checkpoint, use_safetensors=True
).to("cuda").eval()

camera = cv2.VideoCapture(0, cv2.CAP_V4L2)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

try:
    if not camera.isOpened():
        raise RuntimeError("Camera could not open")

    print("Starting live view. Press Q in the window to stop.")

    while True:
        ok, frame = camera.read()
        if not ok:
            print("Camera stopped delivering frames")
            break

        # Both models analyze the same captured frame.
        detection = yolo.predict(
            frame, device=0, imgsz=320,
            conf=0.25, verbose=False
        )[0]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = processor(
            images=rgb,
            return_tensors="pt",
            size={"height": 280, "width": 280},
        )
        inputs = {key: value.to("cuda")
                  for key, value in inputs.items()}

        with torch.inference_mode():
            output = depth_model(**inputs)
            depth = torch.nn.functional.interpolate(
                output.predicted_depth.unsqueeze(1),
                size=frame.shape[:2],
                mode="bilinear",
                align_corners=False,
            )[0, 0].cpu().numpy()

        # Fixed visualization scale: 0 to 5 meters.
        scaled = np.nan_to_num(
            depth, nan=5.0, posinf=5.0, neginf=0.0
        )
        scaled = (np.clip(scaled / 1.5, 0, 1) * 255).astype(
            np.uint8
        )
        heatmap = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)

        boxes = detection.plot()
        cv2.putText(
            boxes, "YOLO detections", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
        cv2.putText(
            heatmap, "Estimated depth: blue near / red far",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 255, 255), 1
        )

        combined = np.hstack((boxes, heatmap))
        cv2.imshow("YOLO + Indoor Depth", combined)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    camera.release()
    cv2.destroyAllWindows()
