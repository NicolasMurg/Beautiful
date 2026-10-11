"""Ball proposals for a settled, calibrated camera view of a billiards table.

Generic COCO YOLO weights can propose ``sports ball`` locations, but do not know
pool-ball numbers. The OpenCV fallback needs no downloaded weights. Appearance
and model labels are suggestions only; the live application must ask an operator
to confirm identities and invalidate them when the physical layout changes.
"""

from pathlib import Path
import re

import numpy as np


def _label_key(value):
    """Normalize label spelling without assuming that a class index is a number."""
    return " ".join(re.sub(r"[_-]+", " ", str(value).strip().lower()).split())


def _ball_label(name):
    """Return (is_ball, numbered_identity, group) for unambiguous model names.

    Color names alone cannot distinguish solids from stripes. Class indices
    never imply ball numbers: only the model's actual name or an explicit map
    may propose an identity.
    """
    if name in {"cue ball", "white ball"}:
        return True, 0, "cue"
    words = ("zero", "one", "two", "three", "four", "five", "six", "seven",
             "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen")
    numbered = re.fullmatch(r"(?:(?:pool |billiard )?ball\s*|(?:solid|stripe|striped)\s*)?"
                           r"(\d{1,2}|" + "|".join(words) + r")(?:\s*ball)?", name)
    if numbered:
        token = numbered.group(1)
        number = int(token) if token.isdigit() else words.index(token)
        if 0 <= number <= 15:
            group = "cue" if number == 0 else "eight" if number == 8 else "solids" if number < 8 else "stripes"
            # A contradictory class name is not a reliable numbered label.
            if name.startswith("solid") and group != "solids" or name.startswith("strip") and group != "stripes":
                return True, None, None
            return True, number, group
    if name in {"solid", "solids", "solid ball"}:
        return True, None, "solids"
    if name in {"stripe", "stripes", "striped", "striped ball", "stripe ball"}:
        return True, None, "stripes"
    generic = {"sports ball", "ball", "pool ball", "billiard ball", "object ball"}
    colors = {"yellow", "blue", "red", "purple", "orange", "green", "brown", "black", "white"}
    return name in generic or name.removesuffix(" ball") in colors, None, None


def _part_label(name):
    if name in {"pocket", "hole", "corner pocket", "side pocket", "pool pocket", "billiard pocket"}:
        return "pocket"
    if name in {"rail", "cushion", "table rail", "pool rail"}:
        return "rail"
    if name in {"table", "pool table", "billiard table", "billiards table"}:
        return "table"
    if name in {"cue stick", "pool cue", "billiard cue"}:
        return "cue_stick"
    return None


def _merge_model_evidence(candidate, proposal):
    """Keep all overlapping model identities; never let a weaker box overwrite one."""
    if proposal["source"] != "yolo":
        return
    model_id = proposal.get("model_suggested_id")
    ids = set(candidate.get("model_suggested_ids", []))
    if model_id is not None:
        ids.add(int(model_id))
    groups = set(candidate.get("model_groups", []))
    if proposal.get("model_group"):
        groups.add(proposal["model_group"])
    labels = set(candidate.get("model_labels", []))
    if proposal.get("model_label"):
        labels.add(proposal["model_label"])
    candidate.update(model_suggested_ids=sorted(ids), model_groups=sorted(groups),
                     model_labels=sorted(labels),
                     model_confidence=max(candidate.get("model_confidence") or 0, proposal["confidence"]))
    conflict = len(ids) > 1 or len(groups) > 1
    candidate["identity_conflict"] = conflict
    candidate["model_suggested_id"] = next(iter(ids)) if len(ids) == 1 and not conflict else None
    candidate["group_hint"] = next(iter(groups)) if len(groups) == 1 and not conflict else None
    if conflict:
        candidate.update(suggested_id=None, model_identity_confidence=None,
                         suggestion_source="conflicting_model_labels")
    elif ids:
        source = proposal.get("model_label_source", "model_label")
        if candidate.get("suggestion_source") == "custom_model_label":
            source = "custom_model_label"
        identity_score = candidate.get("model_identity_confidence") or 0
        if model_id is not None:
            identity_score = max(identity_score, proposal["confidence"])
        candidate.update(suggested_id=next(iter(ids)), suggestion_source=source,
                         model_identity_confidence=identity_score)


def _frame_and_polygon(frame, polygon):
    import cv2

    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError("Expected a nonempty uint8 BGR image")
    if min(frame.shape[:2]) < 2:
        raise ValueError("Expected a nonempty uint8 BGR image")
    polygon = np.asarray(polygon, dtype=np.float32)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
        raise ValueError("Table polygon must contain at least three pixel points")
    if not np.isfinite(polygon).all() or abs(cv2.contourArea(polygon)) < 1:
        raise ValueError("Table polygon must be finite with nonzero area")
    if not cv2.isContourConvex(polygon):
        raise ValueError("Table polygon must be convex and ordered around its boundary")
    return frame, polygon


def _appearance(frame, x, y, radius):
    """Conservative color hints; lighting/number patches can defeat these rules."""
    import cv2

    h, w = frame.shape[:2]
    x0, x1 = max(0, int(np.floor(x - radius))), min(w, int(np.ceil(x + radius)) + 1)
    y0, y1 = max(0, int(np.floor(y - radius))), min(h, int(np.ceil(y + radius)) + 1)
    if x1 <= x0 or y1 <= y0:
        return {"suggested_id": None, "color_hint": "unknown", "stripe_hint": None}
    yy, xx = np.mgrid[y0:y1, x0:x1]
    mask = (xx - x) ** 2 + (yy - y) ** 2 <= (radius * 0.86) ** 2
    hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)[mask]
    if len(hsv) < 10:
        return {"suggested_id": None, "color_hint": "unknown", "stripe_hint": None}
    hue, saturation, value = hsv.T
    white = (saturation < 65) & (value > 160)
    black = value < 65
    colored = (saturation > 80) & (value > 65)
    white_fraction = float(np.mean(white))
    black_fraction = float(np.mean(black))
    colored_fraction = float(np.mean(colored))
    result = {"suggested_id": None, "color_hint": "unknown", "stripe_hint": None,
              "white_fraction": white_fraction}
    if white_fraction >= 0.85 and black_fraction < 0.08 and colored_fraction < 0.05:
        return dict(result, suggested_id=0, color_hint="white", stripe_hint=False)
    if black_fraction >= 0.75 and white_fraction < 0.18 and colored_fraction < 0.05:
        return dict(result, suggested_id=8, color_hint="black", stripe_hint=False)
    if colored_fraction < 0.18:
        return result
    angles = hue[colored].astype(float) * (2 * np.pi / 180)
    average = np.mean(np.exp(1j * angles))
    if abs(average) < 0.90:
        return result
    color_hue = float(np.angle(average) % (2 * np.pi) * 180 / (2 * np.pi))
    color_value = float(np.median(value[colored]))
    number, name = None, "unknown"
    if color_hue < 7 or color_hue > 174:
        number, name = 3, "red"
    elif 9 <= color_hue <= 21:
        number, name = (7, "brown") if color_value < 130 else (5, "orange")
    elif 23 <= color_hue <= 37:
        number, name = 1, "yellow"
    elif 42 <= color_hue <= 87:
        number, name = 6, "green"
    elif 98 <= color_hue <= 122:
        number, name = 2, "blue"
    elif 130 <= color_hue <= 153:
        number, name = 4, "purple"
    result["color_hint"] = name
    if number is not None and colored_fraction >= 0.6 and white_fraction <= 0.15:
        result.update(suggested_id=number, stripe_hint=False)
    elif number is not None and 0.22 <= white_fraction <= 0.75:
        result.update(suggested_id=number + 8, stripe_hint=True)
    return result


def suggest_ball_id(frame, center, radius_px):
    """Return an unconfirmed 0..15 appearance suggestion, or None.

    0 means cue ball. This does not read the printed number and cannot establish
    identity. The image must use OpenCV BGR channel order.
    """
    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Expected a uint8 BGR image")
    center = np.asarray(center, dtype=float)
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("Center must contain two finite coordinates")
    if not np.isfinite(radius_px) or radius_px <= 0:
        raise ValueError("Radius must be finite and positive")
    return _appearance(image, float(center[0]), float(center[1]), float(radius_px))["suggested_id"]


class BallDetector:
    """Fuse optional YOLO ball boxes with table-masked OpenCV circle proposals.

    ``yolo_model`` may be an existing Ultralytics object or a local weights path.
    A weights path must already exist: this module never implicitly downloads it.
    ``class_to_ball_id`` explicitly maps custom model names/class indices to
    0..15; even these labels remain unconfirmed suggestions. Names such as
    ``cue_ball``, ``ball_9`` and ``eight`` work without a map. Mapping keys may
    be integer indices, normalized names, or JSON numeric strings for indices.
    Explicit integer keys take precedence over names, then string indices.
    """

    def __init__(self, min_radius_px=3, max_radius_px=40, yolo_model=None,
                 yolo_device="0", yolo_conf=0.20, class_to_ball_id=None,
                 color_contrast=24.0):
        values = (min_radius_px, max_radius_px, yolo_conf, color_contrast)
        if not all(np.isfinite(v) for v in values):
            raise ValueError("Detector parameters must be finite")
        if min_radius_px < 2 or max_radius_px < min_radius_px:
            raise ValueError("Ball radius bounds must satisfy 2 <= min <= max")
        if not 0 < yolo_conf <= 1 or color_contrast <= 0:
            raise ValueError("YOLO confidence and color contrast must be positive")
        self.min_radius_px = float(min_radius_px)
        self.max_radius_px = float(max_radius_px)
        self.yolo_device, self.yolo_conf = yolo_device, float(yolo_conf)
        self.color_contrast = float(color_contrast)
        self.class_to_ball_id = dict(class_to_ball_id or {})
        if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or not 0 <= v <= 15
               for v in self.class_to_ball_id.values()):
            raise ValueError("Custom ball IDs must be integers from 0 through 15")
        normalized = {}
        for key, value in self.class_to_ball_id.items():
            if isinstance(key, bool) or not isinstance(key, (str, int, np.integer)):
                raise ValueError("Custom class keys must be class names or nonnegative indices")
            if isinstance(key, str):
                key = _label_key(key)
                if not key:
                    raise ValueError("Custom class names must not be empty")
            elif key < 0:
                raise ValueError("Custom class indices must not be negative")
            if key in normalized and normalized[key] != value:
                raise ValueError("Conflicting mappings for the same normalized class name")
            normalized[key] = int(value)
        self.class_to_ball_id = normalized
        if isinstance(yolo_model, (str, Path)):
            model_path = Path(yolo_model)
            if not model_path.is_file():
                raise ValueError("YOLO weights must already exist locally")
            from ultralytics import YOLO
            yolo_model = YOLO(str(model_path))
        self.yolo_model = yolo_model

    def _yolo_detections(self, frame):
        if self.yolo_model is None:
            return [], []
        results = self.yolo_model.predict(frame, device=self.yolo_device,
                                          conf=self.yolo_conf, imgsz=640, verbose=False)
        proposals, parts = [], []
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes.xyxy.cpu().tolist()
            classes = result.boxes.cls.cpu().tolist()
            scores = result.boxes.conf.cpu().tolist()
            for box, cls, score in zip(boxes, classes, scores):
                if len(box) != 4 or not np.isfinite(box).all() or not np.isfinite([cls, score]).all():
                    continue
                if not self.yolo_conf <= score <= 1 or cls < 0 or cls != int(cls):
                    continue
                class_id = int(cls)
                names = getattr(result, "names", {})
                try:
                    name = _label_key(names[class_id])
                except (KeyError, IndexError, TypeError):
                    # Explicit index mappings can work even if model metadata is absent.
                    name = ""
                custom_id = self.class_to_ball_id.get(class_id, self.class_to_ball_id.get(
                    name, self.class_to_ball_id.get(str(class_id))))
                is_ball, canonical_id, group = _ball_label(name)
                if custom_id is not None:
                    group = "cue" if custom_id == 0 else "eight" if custom_id == 8 else "solids" if custom_id < 8 else "stripes"
                role = _part_label(name)
                if custom_id is None and not is_ball and role is None:
                    continue
                x1, y1, x2, y2 = box
                bw, bh = x2 - x1, y2 - y1
                if min(bw, bh) <= 0:
                    continue
                height, width = frame.shape[:2]
                if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                    continue
                if role is not None and custom_id is None:
                    parts.append(dict(kind=role, pixel={"x": (x1 + x2) / 2, "y": (y1 + y2) / 2},
                                      box=list(map(float, box)), confidence=float(score),
                                      model_label=name, model_class_id=class_id,
                                      source="yolo", geometry_confirmed=False))
                    continue
                if min(bw, bh) / max(bw, bh) < 0.6:
                    continue
                proposals.append({"x": (x1 + x2) / 2, "y": (y1 + y2) / 2,
                                  "r": (bw + bh) / 4, "confidence": float(score),
                                  "source": "yolo", "model_suggested_id": custom_id if custom_id is not None else canonical_id,
                                  "model_label": name, "model_class_id": class_id, "model_group": group,
                                  "model_label_source": "custom_model_label" if custom_id is not None else "model_label"})
        return proposals, parts

    def _yolo_proposals(self, frame):
        proposals, _ = self._yolo_detections(frame)
        return proposals

    def detect(self, frame, table_polygon):
        """Return circle candidates inside a convex table polygon in frame pixels.

        Dictionaries contain pixel{x,y}, radius_px, box[x1,y1,x2,y2], confidence,
        source, suggested_id, and identity_confirmed=False. Confidence is a
        proposal-quality score, not a calibrated probability of a correct ball.
        model_confidence scores any YOLO evidence; model_identity_confidence
        scores the unambiguous numbered label only (None without one).
        """
        return self.detect_scene(frame, table_polygon)["balls"]

    def detect_scene(self, frame, table_polygon):
        """Return ball candidates and optional unconfirmed YOLO table-part boxes.

        Parts have kind ``pocket``, ``rail``, ``table`` or ``cue_stick``, pixel
        centers, bounding boxes and model confidence. They are review hints:
        a pocket box does not establish its measured center, width or blockage.
        Missing detections never declare a ball pocketed or a pocket open.
        """
        import cv2

        frame, polygon = _frame_and_polygon(frame, table_polygon)
        proposals, parts = self._yolo_detections(frame)
        visible_parts = []
        for part in parts:
            x1, y1, x2, y2 = part["box"]
            box_polygon = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)
            area, _ = cv2.intersectConvexConvex(polygon, box_polygon)
            if area > 0:
                visible_parts.append(part)
        return {"balls": self._detect(frame, polygon, proposals), "parts": visible_parts}

    def _detect(self, frame, polygon, proposals):
        import cv2

        height, width = frame.shape[:2]
        table_mask = np.zeros((height, width), np.uint8)
        cv2.fillConvexPoly(table_mask, np.rint(polygon).astype(np.int32), 255)
        if np.count_nonzero(table_mask) < 20:
            return []
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        cloth = np.median(lab[table_mask != 0], axis=0)
        contrast = np.linalg.norm(lab - cloth, axis=2)
        foreground = ((contrast >= self.color_contrast) & (table_mask != 0)).astype(np.uint8) * 255
        contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        proposals = list(proposals)
        for contour in contours:
            area, perimeter = cv2.contourArea(contour), cv2.arcLength(contour, True)
            if area < 8 or perimeter <= 0:
                continue
            (x, y), radius = cv2.minEnclosingCircle(contour)
            circularity = 4 * np.pi * area / perimeter ** 2
            fill = area / (np.pi * radius ** 2) if radius > 0 else 0
            if circularity >= 0.65 and fill >= 0.68:
                proposals.append({"x": x, "y": y, "r": radius,
                                  "confidence": float(min(0.90, 0.65 + circularity * 0.25)),
                                  "source": "opencv_contour"})

        # Hough can recover balls with color similar to the cloth and touching
        # foreground regions. Use original pixels, not the artificial mask edge.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 1)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1,
                                   minDist=max(4, self.min_radius_px * 1.6),
                                   param1=80, param2=12,
                                   minRadius=max(2, int(np.floor(self.min_radius_px))),
                                   maxRadius=max(2, int(np.ceil(self.max_radius_px))))
        if circles is not None:
            proposals.extend({"x": float(x), "y": float(y), "r": float(r),
                              "confidence": 0.60, "source": "opencv_hough"}
                             for x, y, r in circles[0])

        candidates = []
        for proposal in sorted(proposals, key=lambda p: (-p["confidence"], -p["r"])):
            x, y, radius = proposal["x"], proposal["y"], proposal["r"]
            if not np.isfinite([x, y, radius]).all():
                continue
            # A tiny contour epsilon keeps an exact integer radius in range.
            if radius < self.min_radius_px - 0.05 or radius > self.max_radius_px + 0.05:
                continue
            if min(x - radius, y - radius, width - 1 - x - radius, height - 1 - y - radius) < 0:
                continue
            # A hanging ball may straddle the calibrated cloth boundary at a
            # pocket mouth. Keep any center on/inside the polygon; calibrated
            # cushion geometry and operator review resolve rail false positives.
            if cv2.pointPolygonTest(polygon, (float(x), float(y)), False) < 0:
                continue
            x0, x1 = max(0, int(x - radius)), min(width, int(x + radius) + 1)
            y0, y1 = max(0, int(y - radius)), min(height, int(y + radius) + 1)
            yy, xx = np.mgrid[y0:y1, x0:x1]
            core = (xx - x) ** 2 + (yy - y) ** 2 <= (radius * 0.7) ** 2
            if not np.any(core):
                continue
            # Weak Hough rings on otherwise plain cloth are not good proposals.
            if proposal["source"] == "opencv_hough" and np.mean(contrast[y0:y1, x0:x1][core]) < self.color_contrast * 0.6:
                continue
            duplicate = next((candidate for candidate in candidates
                              if np.hypot(candidate["pixel"]["x"] - x, candidate["pixel"]["y"] - y)
                              < max(candidate["radius_px"], radius) * 0.8), None)
            if duplicate is not None:
                _merge_model_evidence(duplicate, proposal)
                continue
            appearance = _appearance(frame, x, y, radius)
            candidate = dict(pixel={"x": float(x), "y": float(y)}, radius_px=float(radius),
                             box=[x0, y0, x1, y1], confidence=proposal["confidence"],
                             source=proposal["source"], identity_confirmed=False,
                             model_suggested_id=None, model_suggested_ids=[], model_labels=[],
                             model_groups=[], model_confidence=None, model_identity_confidence=None,
                             identity_conflict=False,
                             group_hint=None, suggestion_source="appearance", **appearance)
            _merge_model_evidence(candidate, proposal)
            candidates.append(candidate)
        return sorted(candidates, key=lambda candidate: (candidate["pixel"]["y"], candidate["pixel"]["x"]))


class CandidateTracker:
    """Conservative keys for stationary review, not billiards identity tracking.

    Keys survive unique small jitter, measured from both last position and the
    original anchor. Misses, motion, radius changes, or ambiguous associations
    retire a key immediately. An unseen exchange of indistinguishable balls
    cannot be inferred; the application must invalidate a review after motion.
    """

    def __init__(self, max_distance_px=6.0, max_radius_ratio=1.4):
        if not np.isfinite([max_distance_px, max_radius_ratio]).all():
            raise ValueError("Tracker parameters must be finite")
        if max_distance_px <= 0 or max_radius_ratio < 1:
            raise ValueError("Tracker distance must be positive and radius ratio >= 1")
        self.max_distance_px = float(max_distance_px)
        self.max_radius_ratio = float(max_radius_ratio)
        self._previous = []
        self._next_key = 1

    def reset(self):
        """Discard associations without reusing keys from an earlier layout."""
        self._previous = []

    def update(self, detections):
        detections = [dict(candidate, pixel=dict(candidate["pixel"])) for candidate in detections]
        for candidate in detections:
            x, y, radius = candidate["pixel"]["x"], candidate["pixel"]["y"], candidate["radius_px"]
            if not np.isfinite([x, y, radius]).all() or radius <= 0:
                raise ValueError("Candidate coordinates and radius must be finite; radius must be positive")
        links = [[] for _ in detections]
        reverse_links = [[] for _ in self._previous]
        for current_index, candidate in enumerate(detections):
            point = np.array([candidate["pixel"]["x"], candidate["pixel"]["y"]], dtype=float)
            for previous_index, previous in enumerate(self._previous):
                old = previous["candidate"]
                old_point = np.array([old["pixel"]["x"], old["pixel"]["y"]], dtype=float)
                ratio = max(candidate["radius_px"], old["radius_px"]) / min(candidate["radius_px"], old["radius_px"])
                if (np.linalg.norm(point - old_point) <= self.max_distance_px
                        and np.linalg.norm(point - previous["anchor"]) <= self.max_distance_px
                        and ratio <= self.max_radius_ratio):
                    links[current_index].append(previous_index)
                    reverse_links[previous_index].append(current_index)
        tracked = []
        for index, candidate in enumerate(detections):
            previous_indices = links[index]
            unique = len(previous_indices) == 1 and len(reverse_links[previous_indices[0]]) == 1
            ambiguous = len(previous_indices) > 1 or any(len(reverse_links[j]) > 1 for j in previous_indices)
            if unique:
                previous = self._previous[previous_indices[0]]
                key, anchor = previous["candidate"]["key"], previous["anchor"]
            else:
                key = f"ball-{self._next_key}"
                self._next_key += 1
                anchor = np.array([candidate["pixel"]["x"], candidate["pixel"]["y"]], dtype=float)
            candidate.update(key=key, track_id=key, ambiguous=ambiguous, identity_confirmed=False)
            tracked.append({"candidate": candidate, "anchor": anchor})
        self._previous = tracked
        return [entry["candidate"] for entry in tracked]
