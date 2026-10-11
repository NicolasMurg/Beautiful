"""An explicit metric world map between camera measurements and shot planning.

The map uses centimeters in the calibrated table frame, never image pixels or
camera distance as table coordinates. Unidentified objects can be displayed but
cannot become a playable state until every remaining ball has been reviewed.
"""

import copy
import math

from billiards import ball_center_is_playable, normalize_pockets, table_cushion_segments


def _finite(value, name, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{name} must be a finite number.")
    value = float(value)
    if not math.isfinite(value) or minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise ValueError(f"{name} is outside its valid numeric range.")
    return value


def _rails(table):
    """Cushion faces, with the same measured mouth gaps used by the planner."""
    return [{"start": {"x": a[0], "y": a[1]}, "end": {"x": b[0], "y": b[1]}}
            for a, b in table_cushion_segments(table)]


def build_table_map(table, candidates, labels=None, ignored=(), *, confirmed=False,
                    observed_at=None, source="calibrated_stereo"):
    """Build a reviewable map from already localized camera candidates.

    Each candidate needs a stable review key and a position{x,y} in cm. The
    position is produced by TableCalibration, not by ball identity detection.
    Invalid measurements stay in ``unmapped`` instead of vanishing from view.
    Missing identities, overlaps, or failed measurements disable planner input.
    This function never assumes that a missing detection was pocketed.
    """
    if not isinstance(table, dict) or not isinstance(candidates, list):
        raise ValueError("Supply calibrated table geometry and a list of measured candidates.")
    if not isinstance(confirmed, bool):
        raise ValueError("Map confirmation must be boolean.")
    geometry = copy.deepcopy(table)
    geometry["pockets"] = normalize_pockets(geometry)
    labels = dict(labels or {})
    ignored = set(ignored)
    balls, unmapped, issues = [], [], []
    keys, numbered = set(), set()
    invalid = False
    for candidate in candidates:
        if not isinstance(candidate, dict) or not isinstance(candidate.get("key"), str):
            raise ValueError("Every measured candidate needs a string review key.")
        key = candidate["key"]
        if key in keys:
            raise ValueError(f"Candidate key {key} appears more than once.")
        keys.add(key)
        if key in ignored:
            continue
        try:
            if "error" in candidate:
                raise ValueError(str(candidate["error"]))
            position = candidate.get("position")
            if not isinstance(position, dict):
                raise ValueError("No calibrated table position; camera distance alone is insufficient.")
            x = _finite(position.get("x"), "Table x")
            y = _finite(position.get("y"), "Table y")
            if not ball_center_is_playable(geometry, x, y):
                raise ValueError("Position is outside the playable table or intersects a cushion.")
            error = _finite(position.get("position_uncertainty_cm", 0.), "Position uncertainty", 0., 30.)
            confidence = _finite(candidate.get("confidence", 1.), "Detection confidence", 0., 1.)
        except ValueError as exc:
            issue = f"Candidate {key}: {exc}"
            issues.append(issue)
            unmapped.append({"key": key, "reason": str(exc)})
            invalid = True
            continue
        number = labels.get(key)
        if number is None:
            issues.append(f"Label candidate {key} with its actual ball number.")
        elif isinstance(number, bool) or not isinstance(number, int) or not 0 <= number <= 15:
            issues.append(f"Candidate {key} needs an integer ball number from 0 to 15.")
            number, invalid = None, True
        elif number in numbered:
            issues.append(f"Ball {number} appears more than once; review identities.")
            invalid = True
        else:
            numbered.add(number)
        ball = {"key": key, "id": number, "x": x, "y": y,
                "position_uncertainty_cm": error, "confidence": confidence,
                "source": position.get("localization_source", source)}
        # Carry measurement provenance separately from coordinates consumed by
        # physics; never unproject or add the ball radius a second time here.
        measurement = {name: copy.deepcopy(position[name]) for name in
                       ("camera_xyz_m", "measured_camera_xyz_m", "distance_m", "distance_kind",
                        "measurement_reference", "depth_m", "center_depth_m", "depth_source",
                        "pixel_space", "camera_frame")
                       if name in position}
        if "pixel" in candidate:
            measurement["pixel"] = copy.deepcopy(candidate["pixel"])
        if measurement:
            ball["measurement"] = measurement
        balls.append(ball)
    if not balls:
        issues.append("Scan the physical table before using its map.")
    if len(balls) + len(unmapped) > 16:
        issues.append("More than 16 ball candidates; review false or duplicate detections.")
        invalid = True
    separations = []
    for index, ball in enumerate(balls):
        for other in balls[index + 1:]:
            distance = math.hypot(ball["x"] - other["x"], ball["y"] - other["y"])
            separations.append({"keys": [ball["key"], other["key"]],
                                "ball_ids": [ball["id"], other["id"]],
                                "center_distance_cm": distance,
                                "surface_gap_cm": distance - 2 * geometry["ball_radius"],
                                "position_margin_cm": ball["position_uncertainty_cm"] + other["position_uncertainty_cm"]})
            if distance < 2 * geometry["ball_radius"] - 1e-7:
                issues.append(f"Candidates {ball['key']} and {other['key']} overlap; review measurements.")
                invalid = True
    if balls and 0 not in numbered:
        issues.append("Locate and label the cue ball (0).")
    complete = bool(balls) and not issues
    return {
        "version": 1, "units": "cm",
        "coordinate_frame": {"origin": "top-left inner cushion intersection",
                             "x_axis": "toward top-right", "y_axis": "toward bottom-left"},
        "status": "invalid" if invalid else "confirmed" if confirmed and complete else "review",
        "complete": complete, "table": geometry, "rails": _rails(geometry),
        "balls": balls, "ball_separations": separations, "unmapped": unmapped, "issues": issues,
        "source": source, "observed_at": None if observed_at is None else _finite(observed_at, "Observation time"),
    }


def planner_state_from_map(table_map, group="open", planner=None, *, require_confirmed=True):
    """Extract playable coordinates only from a complete, reviewed metric map."""
    if not isinstance(table_map, dict) or table_map.get("version") != 1 or table_map.get("units") != "cm":
        raise ValueError("A calibrated centimeter table map is required.")
    if (table_map.get("complete") is not True or table_map.get("status") not in ("confirmed", "review")
            or table_map.get("unmapped") or table_map.get("issues")):
        raise ValueError(" ".join(table_map.get("issues", [])) or "The table map is incomplete.")
    if require_confirmed and table_map.get("status") != "confirmed":
        raise ValueError("Review and confirm the complete virtual table map before planning.")
    source_balls = table_map.get("balls")
    if not isinstance(source_balls, list) or any(not isinstance(ball, dict) for ball in source_balls):
        raise ValueError("The table map needs a list of measured balls.")
    candidates = [dict(ball, position=ball) for ball in source_balls]
    labels = {ball.get("key"): ball.get("id") for ball in source_balls}
    checked = build_table_map(table_map.get("table"), candidates, labels, confirmed=True)
    if not checked["complete"]:
        raise ValueError(" ".join(checked["issues"]))
    return {"table": copy.deepcopy(checked["table"]), "group": group,
            "balls": [{name: ball[name] for name in ("id", "x", "y", "position_uncertainty_cm", "confidence")}
                      for ball in checked["balls"]], "planner": dict(planner or {})}


def map_camera_measurements(calibration, measurements, *, calibration_signature,
                            labels=None, confirmed=False, observed_at=None):
    """Map scalar *metric* camera observations in the profile's pixel frame.

    Measurements need key, pixel{x,y}, distance_m, distance_kind ('camera_z' or
    'range'), and measurement_reference ('surface' or 'center'). The camera pose,
    resolution, intrinsics, and pixel rectification must match the calibration.
    This is not an adapter for relative model depth or a different camera pose.
    The whole list must describe one stationary capture; track keys are never
    interpreted as semantic ball numbers. Labels are reviewed separately.
    """
    if calibration_signature != calibration.signature:
        raise ValueError("Measurement calibration signature differs from the table profile.")
    if not isinstance(measurements, list):
        raise ValueError("Supply a list of same-frame camera measurements.")
    for field in ("frame_id", "camera_id"):
        if any(isinstance(item, dict) and field in item for item in measurements):
            values = [item.get(field) if isinstance(item, dict) else None for item in measurements]
            if (any(isinstance(value, bool) or not isinstance(value, (int, str)) for value in values)
                    or any(value != values[0] for value in values)):
                raise ValueError(f"All measurements must belong to the same {field}.")
    candidates = []
    for measurement in measurements:
        if not isinstance(measurement, dict) or not isinstance(measurement.get("key"), str):
            raise ValueError("Every camera measurement needs a string review key.")
        candidate = {"key": measurement["key"], "confidence": measurement.get("confidence", 1.)}
        if "pixel" in measurement:
            candidate["pixel"] = copy.deepcopy(measurement["pixel"])
        try:
            if "error" in measurement:
                raise ValueError(str(measurement["error"]))
            required = ("pixel", "distance_m", "distance_kind", "measurement_reference")
            if any(name not in measurement for name in required):
                raise ValueError("Need image position, metric distance, distance kind, and surface/center reference.")
            candidate["position"] = calibration.localize_ball_measurement(
                measurement["pixel"], measurement["distance_m"],
                distance_kind=measurement["distance_kind"],
                measurement_reference=measurement["measurement_reference"],
                radius_px=measurement.get("radius_px"),
                distance_uncertainty_m=measurement.get("distance_uncertainty_m", 0.),
                pixel_uncertainty_px=measurement.get("pixel_uncertainty_px", 1.))
        except (ValueError, TypeError) as exc:
            candidate["error"] = str(exc)
        candidates.append(candidate)
    return build_table_map(calibration.table, candidates, labels, confirmed=confirmed,
                           observed_at=observed_at, source="calibrated_metric_camera_measurements")
