"""Review a real stereo table scan before issuing shot advice.

Ball numbers are observations confirmed by the operator, never COCO class IDs.
A scan is invalidated on scene motion, capture failure, or stale frames.
"""

import math
import copy
import re
import time

import numpy as np

from billiards import BALLS, GROUPS, plan_shots
from billiards_map import build_table_map, planner_state_from_map


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
    """One reviewed stationary layout with an optional default target filter.

    Solids/stripes requests filter this snapshot; they do not assign a player a
    group or update a game history.
    """

    def __init__(self, calibration, detector, group="open", max_frame_age=3.0,
                 planner_options=None):
        if group not in GROUPS:
            raise ValueError("Choose solids, stripes, or open.")
        if not isinstance(max_frame_age, (int, float)) or not math.isfinite(max_frame_age) or max_frame_age <= 0:
            raise ValueError("Frame age must be finite and positive.")
        self.calibration, self.detector, self.group = calibration, detector, group
        self.planner_options = dict(planner_options or {})
        self.pocket_overrides = {}
        self.guard = SceneGuard(calibration.polygon, calibration.image_size)
        self.max_frame_age = max_frame_age
        self.revision = 0
        self.snapshot = None
        self.scan_received_at = None
        self.depth_snapshot = None
        self.candidates = []
        self.parts = []
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
        self.scan_received_at = None
        self.depth_snapshot = None
        self.candidates = []
        self.parts = []
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
                    if (current.get("position_uncertainty_cm", 0) >
                            previous.get("position_uncertainty_cm", 0)
                            + max(0.1, self.calibration.ball_radius_cm * 0.1)):
                        raise ValueError("Ball position uncertainty increased; review a new scan.")
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
        if hasattr(self.detector, "detect_scene"):
            scene = self.detector.detect_scene(frame, self.calibration.polygon)
            detections, parts = scene["balls"], scene["parts"]
        else:
            detections, parts = self.detector.detect(frame, self.calibration.polygon), []
        self.invalidate("Scan ready. Review every ball number; SHIFT-click to add a missed ball.")
        self.parts = parts
        self.snapshot, self.depth_snapshot = frame.copy(), depth_map.copy()
        self.scan_received_at = self.last_received
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

    def accept_model_labels(self):
        """Copy unique model identity suggestions for review; never confirm a layout."""
        active = [c for c in self.candidates if c["key"] not in self.ignored]
        if self.snapshot is None:
            raise ValueError("SPACE to scan before reviewing model labels.")
        suggested = [c.get("model_suggested_id") for c in active]
        used = set(self.labels.values())
        accepted = 0
        for candidate in active:
            number = candidate.get("model_suggested_id")
            if (number is None or number not in BALLS or candidate.get("identity_conflict")
                    or "error" in candidate or candidate["key"] in self.labels
                    or suggested.count(number) != 1 or number in used):
                continue
            self.assign(candidate["key"], number)
            used.add(number)
            accepted += 1
        self.message = (f"Copied {accepted} unique model labels. Review every ball and press C; "
                        "unidentified or conflicting candidates still need labels.")
        return accepted

    def set_pocket_blocked(self, pocket_id, blocked):
        if pocket_id not in ("TL", "TM", "TR", "BL", "BM", "BR") or not isinstance(blocked, bool):
            raise ValueError("Use pocket TL, TM, TR, BL, BM, or BR and a boolean blocked state.")
        self.pocket_overrides[pocket_id] = blocked
        self.plan = None
        self.revision += 1

    def reject(self, key):
        self.candidate(key)
        self.ignored.add(str(key))
        self.labels.pop(str(key), None)
        self.confirmed, self.plan = False, None
        self.revision += 1
        self.message = "Candidate ignored. Confirm that no real ball is missing before C."

    def virtual_map(self):
        """The measured overhead map shown to the operator and used by planning."""
        table = copy.deepcopy(self.calibration.table)
        if self.pocket_overrides:
            pockets = {p["id"]: dict(p) for p in table.get("pockets", [])}
            for pocket_id, blocked in self.pocket_overrides.items():
                pockets.setdefault(pocket_id, {"id": pocket_id})["blocked"] = blocked
            table["pockets"] = list(pockets.values())
        result = build_table_map(table, self.candidates, self.labels, self.ignored,
                                 confirmed=self.confirmed, observed_at=self.scan_received_at)
        result["last_validated_at"] = self.last_received
        result["calibration_signature"] = copy.deepcopy(getattr(self.calibration, "signature", None))
        return result

    def state(self):
        if self.snapshot is None or not any(c["key"] not in self.ignored for c in self.candidates):
            raise ValueError("SPACE to scan the physical table first.")
        # Confirmation validates this state before declaring it reviewed. ask()
        # separately enforces review and fresh frames before issuing advice.
        return planner_state_from_map(self.virtual_map(), self.group, self.planner_options,
                                      require_confirmed=False)

    def confirm(self):
        # Review describes only the current geometry: every on-table ball must
        # be represented, including balls that block paths. Missing IDs say
        # nothing about pocket events or which group a player owns.
        plan_shots(self.state())  # Validate complete observed geometry and cue identity.
        self.confirmed = True
        self.revision += 1
        self.message = "Layout confirmed: all on-table balls labeled. A: ask for solids or stripes."

    def set_group(self, group):
        """Set the optional default target filter, without assigning a player."""
        if group not in GROUPS:
            raise ValueError("Choose solids, stripes, or open.")
        self.group = group
        self.plan = None
        self.revision += 1
        self.message = f"Target filter: {group}. A: ask for the best shot."

    def ask(self, question="What is my best shot?", now=None):
        if not isinstance(question, str) or len(question) > 500:
            raise ValueError("Use a shot request containing at most 500 characters.")
        now = time.monotonic() if now is None else now
        if self.last_received is None or not 0 <= now - self.last_received <= self.max_frame_age:
            self.invalidate("Camera frames are stale. Rescan before asking for a shot.")
            raise ValueError(self.message)
        if not self.confirmed:
            raise ValueError("Review the real balls and press C to confirm the complete layout first.")
        command = re.sub(r"[?.!]+$", "", question.strip().lower())
        if command in ("next", "next shot", "next option", "another shot", "show another shot") and self.plan:
            shot = self.next_shot()
            return dict(self.plan, response=shot["instruction"] if shot else self.plan["message"])
        if command in ("why", "why this shot", "explain", "explain this shot", "repeat", "repeat shot") and self.shot:
            text = self.shot["instruction"]
            if command not in ("repeat", "repeat shot"):
                text += " " + " ".join(self.shot["reasons"])
            return dict(self.plan, response=text)
        pocket = re.fullmatch(r"(block|unblock|open) (?:the )?(tl|tm|tr|bl|bm|br|top left|top middle|top right|bottom left|bottom middle|bottom right)(?: pocket| hole)?", command)
        if pocket:
            names = {"top left": "TL", "top middle": "TM", "top right": "TR",
                     "bottom left": "BL", "bottom middle": "BM", "bottom right": "BR"}
            pocket_id = names.get(pocket[2], pocket[2].upper())
            self.set_pocket_blocked(pocket_id, pocket[1] == "block")
            question = "What is my best shot?"
        if command in ("direct shots only", "disable bank shots", "allow bank shots", "include bank shots"):
            self.planner_options["allow_banks"] = command in ("allow bank shots", "include bank shots")
            self.plan = None
            self.revision += 1
            question = "What is my best shot?"
        result = plan_shots(self.state(), question)
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
            self.revision += 1
        return self.shot
