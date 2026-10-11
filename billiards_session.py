"""Review a real stereo table scan before issuing shot advice.

Ball numbers are observations confirmed by the operator, never COCO class IDs.
A scan is invalidated on scene motion, capture failure, or stale frames.
"""

import math
import time

import numpy as np

from billiards import BALLS, GROUPS, plan_shots


class SceneGuard:
    """Compare the cloth region, ignoring modest uniform exposure changes."""

    def __init__(self, polygon, image_size, threshold=25, changed_pixels=20):
        import cv2
        width, height = image_size
        self.mask = np.zeros((height, width), np.uint8)
        cv2.fillConvexPoly(self.mask, np.rint(polygon).astype(np.int32), 255)
        self.mask = cv2.erode(self.mask, np.ones((3, 3), np.uint8)) > 0
        if np.count_nonzero(self.mask) < 100:
            raise ValueError("Too little of the table is visible in the camera.")
        self.threshold = threshold
        self.changed_pixels = changed_pixels

    @staticmethod
    def gray(frame):
        import cv2
        return cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (3, 3), 0)

    def changed(self, before, after):
        if before.shape != after.shape or before.shape[:2] != self.mask.shape:
            return True
        delta = self.gray(after).astype(np.float32) - self.gray(before).astype(np.float32)
        drift = float(np.median(delta[self.mask]))
        count = np.count_nonzero(np.abs(delta[self.mask] - drift) > self.threshold)
        return count >= self.changed_pixels


class PoolSession:
    """One explicitly reviewed stationary layout, tied to fresh camera frames."""

    def __init__(self, calibration, detector, group="solids", max_frame_age=3.0):
        if group not in GROUPS:
            raise ValueError("Choose solids, stripes, or open.")
        self.calibration, self.detector, self.group = calibration, detector, group
        self.guard = SceneGuard(calibration.polygon, calibration.image_size)
        self.max_frame_age = max_frame_age
        self.revision = 0
        self.snapshot = None
        self.depth_snapshot = None
        self.candidates = []
        self.labels = {}
        self.ignored = set()
        self.confirmed = False
        self.plan = None
        self.shot_index = 0
        self.last_received = None
        self.previous = None
        self.stable_since = None
        self.stable_frames = 0
        self.message = "Wait for a still table, then SPACE to scan."

    def invalidate(self, message):
        self.snapshot = None
        self.depth_snapshot = None
        self.candidates = []
        self.labels.clear()
        self.ignored.clear()
        self.confirmed = False
        self.plan = None
        self.revision += 1
        self.message = message

    def observe(self, frame, received_at, accepted=True, now=None, depth_map=None):
        now = time.monotonic() if now is None else now
        if (not accepted or not math.isfinite(received_at)
                or not 0 <= now - received_at <= self.max_frame_age
                or self.last_received is not None and received_at <= self.last_received):
            self.last_received = None
            self.previous = None
            self.stable_since = None
            self.stable_frames = 0
            self.invalidate("Capture rejected or too old. Wait, then SPACE to rescan.")
            return False
        if self.last_received is not None and received_at - self.last_received > self.max_frame_age:
            self.invalidate("Capture gap: rescan the table.")
            self.previous = None
        self.last_received = received_at
        if self.snapshot is not None and self.guard.changed(self.snapshot, frame):
            self.invalidate("Table changed. Wait for all balls/hands to stop, then SPACE.")
        if self.snapshot is not None and depth_map is not None:
            for candidate in self.candidates:
                if candidate["key"] in self.ignored or "position" not in candidate:
                    continue
                try:
                    current = self.calibration.localize_ball(
                        candidate["pixel"], candidate["radius_px"], depth_map)
                    previous = candidate["position"]
                    if math.hypot(current["x"] - previous["x"], current["y"] - previous["y"]) > 1.0:
                        raise ValueError("Measured ball position changed by over 1 cm.")
                except ValueError as exc:
                    self.invalidate(f"Stereo ball measurement changed: {exc} SPACE to rescan.")
                    self.previous = None
                    break
        if self.previous is None or self.guard.changed(self.previous, frame):
            self.stable_since = now
            self.stable_frames = 1
        else:
            self.stable_frames += 1
        self.previous = frame.copy()
        return True

    def settled(self, now=None):
        now = time.monotonic() if now is None else now
        return (self.last_received is not None and 0 <= now - self.last_received <= self.max_frame_age
                and self.stable_since is not None and now - self.stable_since >= 1.0
                and self.stable_frames >= 3)

    def scan(self, frame, depth_map, now=None):
        if not self.settled(now):
            raise ValueError("Wait for at least one second of a still, fresh table before scanning.")
        if self.previous is None or self.guard.changed(self.previous, frame):
            self.invalidate("Table changed before the scan. Wait for a still table and try again.")
            self.stable_since = None
            self.stable_frames = 0
            raise ValueError(self.message)
        detections = self.detector.detect(frame, self.calibration.polygon)
        self.invalidate("Scan ready. Review every ball number; SHIFT-click to add a missed ball.")
        self.snapshot, self.depth_snapshot = frame.copy(), depth_map.copy()
        for detection in detections:
            self._append(detection)
        if not self.candidates:
            self.message = "No automatic candidates. SHIFT-click real ball centers to add them."
        return self.candidates

    def _append(self, detection):
        candidate = dict(detection, key=str(len(self.candidates) + 1))
        try:
            candidate["position"] = self.calibration.localize_ball(
                candidate["pixel"], candidate["radius_px"], self.depth_snapshot)
        except ValueError as exc:
            candidate["error"] = str(exc)
        self.candidates.append(candidate)
        self.confirmed, self.plan = False, None
        self.revision += 1
        return candidate

    def add_at_pixel(self, x, y):
        """Mark the center of an actual missed ball in the reviewed camera frame."""
        import cv2
        if self.snapshot is None:
            raise ValueError("SPACE to scan before marking a missed ball.")
        if cv2.pointPolygonTest(np.asarray(self.calibration.polygon, np.float32), (float(x), float(y)), False) < 0:
            raise ValueError("Select a real ball inside the calibrated table.")
        h, w = self.depth_snapshot.shape
        patch = self.depth_snapshot[max(0, int(y)-3):min(h, int(y)+4),
                                    max(0, int(x)-3):min(w, int(x)+4)]
        valid = patch[np.isfinite(patch) & (patch > 0)]
        if len(valid) < 5:
            raise ValueError("This ball has insufficient stereo depth; adjust the cameras/lighting.")
        radius = float(self.calibration.camera_matrix[0, 0] *
                       self.calibration.ball_radius_cm / 100 / np.median(valid))
        for candidate in self.candidates:
            if candidate["key"] not in self.ignored and math.hypot(
                    candidate["pixel"]["x"] - x, candidate["pixel"]["y"] - y) < radius:
                raise ValueError("There is already a candidate here. Select it or reject it first.")
        return self._append(dict(pixel={"x": float(x), "y": float(y)}, radius_px=radius,
                                 suggested_id=None, source="operator_marked_camera_pixel"))

    def candidate(self, key):
        for candidate in self.candidates:
            if candidate["key"] == str(key) and candidate["key"] not in self.ignored:
                return candidate
        raise ValueError("Select a visible ball candidate from the current scan.")

    def assign(self, key, number):
        candidate = self.candidate(key)
        if "error" in candidate:
            raise ValueError(candidate["error"])
        if isinstance(number, bool) or not isinstance(number, int) or number not in BALLS:
            raise ValueError("Use 0 for the cue ball, or a number from 1 to 15.")
        if any(n == number and k != str(key) for k, n in self.labels.items()):
            raise ValueError(f"Ball {number} already has a label. Clear its old label first.")
        self.labels[str(key)] = number
        self.confirmed, self.plan = False, None
        self.revision += 1
        self.message = f"Candidate {key}: {BALLS[number]['name']}. C confirms the complete layout."

    def clear_label(self, key):
        self.candidate(key)
        self.labels.pop(str(key), None)
        self.confirmed, self.plan = False, None
        self.revision += 1

    def reject(self, key):
        self.candidate(key)
        self.ignored.add(str(key))
        self.labels.pop(str(key), None)
        self.confirmed, self.plan = False, None
        self.revision += 1
        self.message = "Candidate ignored. Confirm that no real ball is missing before C."

    def state(self):
        active = [c for c in self.candidates if c["key"] not in self.ignored]
        if self.snapshot is None or not active:
            raise ValueError("SPACE to scan the physical table first.")
        for candidate in active:
            if "error" in candidate:
                raise ValueError(f"Candidate {candidate['key']}: {candidate['error']}")
            if candidate["key"] not in self.labels:
                raise ValueError(f"Label candidate {candidate['key']} with its actual ball number.")
        return dict(table=self.calibration.table, group=self.group,
                    balls=[dict(id=self.labels[c["key"]], x=c["position"]["x"],
                                y=c["position"]["y"]) for c in active])

    def confirm(self):
        # Missing balls are not inferred pocketed by the detector. This action is
        # an explicit operator declaration that all on-table balls are represented.
        plan_shots(self.state())  # Validate complete observed geometry and cue identity.
        self.confirmed = True
        self.revision += 1
        self.message = "Layout confirmed: all on-table balls labeled; absent numbers are pocketed. A: ask."

    def set_group(self, group):
        if group not in GROUPS:
            raise ValueError("Choose solids, stripes, or open.")
        self.group = group
        self.plan = None
        self.revision += 1
        self.message = f"Playing {group}. A: ask for the best direct shot."

    def ask(self, question="What is my best shot?", now=None):
        now = time.monotonic() if now is None else now
        if self.last_received is None or not 0 <= now - self.last_received <= self.max_frame_age:
            self.invalidate("Camera frames are stale. Rescan before asking for a shot.")
            raise ValueError(self.message)
        if not self.confirmed:
            raise ValueError("Review the real balls and press C to confirm the complete layout first.")
        result = plan_shots(self.state(), question)
        self.group = result["group"]
        self.plan = result
        self.shot_index = 0
        self.revision += 1
        self.message = result["message"]
        return result

    @property
    def shot(self):
        if self.plan and self.plan.get("shots"):
            return self.plan["shots"][self.shot_index]
        return None

    def next_shot(self):
        if self.plan and self.plan.get("shots"):
            self.shot_index = (self.shot_index + 1) % len(self.plan["shots"])
        return self.shot
