#!/usr/bin/env python3
"""Optional Hugging Face monocular depth for live_yolo.py (--depth-model).

Metric checkpoints (e.g. depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf)
output meters; relative checkpoints output unitless values and are never labeled
as meters. Monocular metric depth is an estimate: validate it against tape-measured
distances before trusting it. Width needs a calibrated fx (calibrate_camera.py).
"""
from pathlib import Path
import time

import numpy as np


class DepthEstimator:
    def __init__(self, model_id, device='0'):
        # Heavy imports stay here so live_yolo.py never needs them without --depth-model.
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self._torch = torch
        if device == 'cpu' or not torch.cuda.is_available():
            if device != 'cpu':
                print(f'Depth: CUDA unavailable, requested device {device}; using CPU', flush=True)
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(f'cuda:{device}' if device.isdecimal() else device)
        self.model_id = model_id
        self.is_metric = 'metric' in model_id.lower()
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id).to(self.device).eval()
        self.last_latency_ms = None
        units = 'meters' if self.is_metric else 'relative (not meters)'
        print(f'Depth model: {model_id}; device {self.device}; output {units}', flush=True)

    def estimate(self, frame_bgr):
        """Return an HxW float32 depth map aligned with the input frame."""
        torch = self._torch
        started = time.monotonic()
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        inputs = self.processor(images=rgb, return_tensors='pt').to(self.device)
        with torch.inference_mode():
            predicted = self.model(**inputs).predicted_depth
        depth = torch.nn.functional.interpolate(
            predicted.unsqueeze(1), size=frame_bgr.shape[:2], mode='bilinear', align_corners=False)
        depth_map = depth[0, 0].float().cpu().numpy()
        self.last_latency_ms = (time.monotonic() - started) * 1000
        return depth_map


def box_depth(depth_map, xyxy, inner=0.5, min_valid=20):
    """Median depth over the central `inner` fraction of a box; None if too few valid pixels."""
    h, w = depth_map.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw, hh = (x2 - x1) * inner / 2, (y2 - y1) * inner / 2
    xa, xb = max(0, int(round(cx - hw))), min(w, int(round(cx + hw)))
    ya, yb = max(0, int(round(cy - hh))), min(h, int(round(cy + hh)))
    if xb <= xa or yb <= ya:
        return None
    patch = depth_map[ya:yb, xa:xb]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if valid.size < min_valid:
        return None
    return float(np.median(valid))


def box_width_m(xyxy, z_m, fx):
    """Projected width W = w_px * Z / fx; None when depth or fx is unknown."""
    if z_m is None or fx is None or fx <= 0:
        return None
    return float(xyxy[2] - xyxy[0]) * z_m / fx


def load_fx(path, frame_width=None):
    """Read fx from calibrate_camera.py output; rescale if the frame width differs."""
    path = Path(path)
    if not path.is_file():
        return None
    import cv2
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    try:
        camera_matrix = fs.getNode('camera_matrix').mat()
        calib_width = fs.getNode('image_width').real()
    finally:
        fs.release()
    if camera_matrix is None:
        return None
    fx = float(camera_matrix[0, 0])
    if frame_width and calib_width:
        fx *= frame_width / calib_width
    return fx
