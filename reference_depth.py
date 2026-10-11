"""Scale metric predictions using one explicitly selected, measured bottle.

A reference corrects scale only; it does not repair a model's relative-depth
errors. Size-based distance assumes a fully visible upright bottle, approximately
parallel to the image plane. No hardware commands are issued by this module.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path


def positive(value):
    return value is not None and math.isfinite(value) and value > 0


@dataclass(frozen=True)
class BottleReference:
    name: str
    height_m: float
    diameter_m: float

    def __post_init__(self):
        if not positive(self.height_m) or not positive(self.diameter_m):
            raise ValueError("Reference height and diameter must be finite positive meters")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["name"], float(data["height_m"]), float(data["diameter_m"]))


class ReferenceDepthCalibrator:
    """Keep reference identity explicit and discard invalid calibration.

    With intrinsics, recompute scale using the reference's height each frame.
    With a tape distance, the reference must stay at its calibration position.
    Track loss requires deliberate reselection and recalibration.
    """

    def __init__(self, profile, measured_distance_m=None):
        if measured_distance_m is not None and not positive(measured_distance_m):
            raise ValueError("Measured reference distance must be finite positive meters")
        self.profile = profile
        self.measured_distance_m = measured_distance_m
        self.reset()

    def reset(self, reason="Click the known bottle; C confirms calibration"):
        self.track_id = None
        self.adjustment = None
        self.status = reason
        self._last_frame = None
        self._frame_size = None
        self._anchor_box = None
        self._focal_lengths = None

    def calibrate(self, records, track_id, frame_id, frame_size, focal_lengths=None):
        """Operator confirms the selected bottle is fully visible and upright."""
        self.reset()
        if track_id is None:
            self.status = "Select a tracked bottle first"
            return None
        self.track_id = track_id
        self._frame_size = tuple(frame_size)
        self._focal_lengths = focal_lengths
        return self.update(records, frame_id, frame_size)

    def update(self, records, frame_id, frame_size):
        if self.track_id is None:
            return None
        if tuple(frame_size) != self._frame_size:
            self.reset("Image size changed; recalibrate")
            return None
        if self._last_frame is not None and frame_id <= self._last_frame:
            self.reset("Out-of-order reference frame; recalibrate")
            return None
        references = [r for r in records if r["track_id"] == self.track_id]
        if len(references) != 1:
            self.reset("Reference lost or ambiguous; select it again")
            return None
        reference = references[0]
        if reference["object"] != "bottle" or reference["frame_id"] != frame_id:
            self.reset("Reference identity/frame changed; select it again")
            return None
        raw_depth = reference["depth_m"]
        if not positive(raw_depth):
            self.reset("Reference depth invalid; recalibrate")
            return None
        box = reference["bbox_xyxy"]
        if len(box) != 4 or not all(math.isfinite(v) for v in box):
            self.reset("Reference box invalid; recalibrate")
            return None
        x1, y1, x2, y2 = box
        width, height = frame_size
        box_width, box_height = x2 - x1, y2 - y1
        if (x1 <= 1 or y1 <= 1 or x2 >= width - 1 or y2 >= height - 1
                or box_width <= 0 or box_height < 40):
            self.reset("Reference must be fully visible and at least 40 pixels tall")
            return None

        if self.measured_distance_m is not None:
            if self._anchor_box is not None:
                ax1, ay1, ax2, ay2 = self._anchor_box
                # A moved bottle no longer has the supplied tape distance.
                if any(abs(a - b) / span > 0.12 for a, b, span in zip(
                        box, self._anchor_box,
                        (ax2 - ax1, ay2 - ay1, ax2 - ax1, ay2 - ay1))):
                    self.reset("Reference moved; restore measured position or restart with new distance")
                    return None
            reference_depth = self.measured_distance_m
            source = "operator_measured_distance"
        else:
            if (self._focal_lengths is None or len(self._focal_lengths) != 2
                    or not all(positive(v) for v in self._focal_lengths)):
                self.reset("Need camera calibration or a measured reference distance")
                return None
            fx, fy = self._focal_lengths
            observed_ratio = (box_width / fx) / (box_height / fy)
            expected_ratio = self.profile.diameter_m / self.profile.height_m
            if abs(observed_ratio / expected_ratio - 1) > 0.25:
                self.reset("Bottle shape/pose inconsistent; show it upright and unobscured")
                return None
            reference_depth = fy * self.profile.height_m / box_height
            source = "known_height_pinhole_estimate"

        scale = reference_depth / raw_depth
        if not positive(scale):
            self.reset("Reference scale invalid; recalibrate")
            return None
        self._anchor_box = self._anchor_box or list(box)
        self._last_frame = frame_id
        self.adjustment = {
            "frame_id": frame_id, "track_id": self.track_id,
            "scale_factor": scale, "reference_depth_m": reference_depth,
            "raw_reference_depth_m": raw_depth, "reference_source": source,
            "profile": self.profile.name, "height_m": self.profile.height_m,
            "diameter_m": self.profile.diameter_m,
        }
        self.status = f"Reference ID {self.track_id} | scale {scale:.3f} | estimates"
        return self.adjustment


def adjust_records(records, adjustment):
    """Preserve raw predictions; adjustments are valid only for their source frame."""
    adjusted = []
    for record in records:
        item = dict(record)
        raw_depth = record["depth_m"]
        item["raw_depth_m"] = raw_depth
        item["depth_source"] = "metric_model"
        if adjustment is not None and record["frame_id"] == adjustment["frame_id"]:
            scale = adjustment["scale_factor"]
            value = raw_depth * scale if positive(raw_depth) and positive(scale) else None
            item["depth_m"] = value if positive(value) else None
            item["depth_source"] = "reference_adjusted_estimate"
            item["reference_track_id"] = adjustment["track_id"]
        adjusted.append(item)
    return adjusted


def bottle_at_pixel(records, x, y):
    """Only an unambiguous click on a currently tracked bottle selects a reference."""
    candidates = [r for r in records if r["object"] == "bottle"
                  and r["track_id"] is not None
                  and r["bbox_xyxy"][0] <= x <= r["bbox_xyxy"][2]
                  and r["bbox_xyxy"][1] <= y <= r["bbox_xyxy"][3]]
    return candidates[0]["track_id"] if len(candidates) == 1 else None
