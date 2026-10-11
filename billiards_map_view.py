"""Draw the metric world map independently of camera pixels or perspective.

The input is the table-map snapshot, not raw detections: coordinates and radii
are centimetres from the inner top-left cushion, with x right and y down.
Only confirmed snapshots may display a shot. No sample layout is substituted
when observations, calibration, or coordinates are unavailable.
"""

import math

import cv2
import numpy as np

from billiards import normalize_pockets


BACKGROUND = (28, 24, 21)
FELT = (68, 91, 38)
TEXT = (225, 231, 231)
WARNING = (65, 190, 255)
ERROR = (75, 95, 250)
CUE_PATH = (255, 220, 80)
OBJECT_PATH = (70, 245, 150)
CONTINUATION = (220, 140, 230)
_BALL_COLORS = ((235, 235, 235), (30, 215, 250), (210, 100, 35),
                (50, 55, 225), (165, 65, 145), (35, 135, 250),
                (60, 145, 50), (60, 50, 135), (25, 25, 25))


def _finite(value):
    return (not isinstance(value, (bool, np.bool_))
            and isinstance(value, (int, float, np.number))
            and math.isfinite(value))


def _label(image, value, x, y, color=TEXT, scale=0.42, centered=False,
           max_width=None):
    """Fit a one-line label to the image, including narrow preview panels."""
    height, width = image.shape[:2]
    value = str(value).replace("\n", " ")
    available = max(1, min(width - 8, max_width or width - 8))
    while len(value) > 1 and cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX,
                                           scale, 1)[0][0] > available:
        value = value[:-4] + "..." if value.endswith("...") else value[:-1] + "..."
        if value == "...":
            break
    (length, text_height), baseline = cv2.getTextSize(
        value, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    if centered:
        x -= length / 2
    x = max(3, min(round(x), width - length - 3))
    y = max(text_height + 2, min(round(y), height - baseline - 2))
    cv2.putText(image, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, 1, cv2.LINE_AA)


def _unavailable(image, reason):
    height, width = image.shape[:2]
    _label(image, "VIRTUAL TABLE - UNAVAILABLE", width / 2,
           max(18, height / 2 - 8), WARNING, centered=True)
    _label(image, reason, width / 2, height / 2 + 14, centered=True,
           scale=0.35)
    return image


def draw_table_map(width_px, height_px, table_map, shot=None, selected=None):
    """Return an exact-size uint8 BGR overhead view of measured table positions.

    ``table_map`` supplies ``table``, ``balls``, ``status`` and ``issues``.
    Tables use planner dimensions and normalized pockets; each ball supplies
    x/y, its detection ``key``, optional numbered ``id``, and optional
    ``position_uncertainty_cm``. ``selected`` is a detection key. The optional
    shot uses the planner's cue_path/object_path/ghost/scratch_risk schema.

    Missing/invalid points are omitted, never clamped to plausible positions.
    Review/invalid maps can show their valid observations, but no shot paths.
    Small positive image sizes show an unavailable notice; nonpositive or
    noninteger sizes are rejected.
    """
    if any(isinstance(v, (bool, np.bool_))
           or not isinstance(v, (int, np.integer)) or v < 1
           for v in (width_px, height_px)):
        raise ValueError("Map image dimensions must be positive integer pixels")
    image = np.full((height_px, width_px, 3), BACKGROUND, dtype=np.uint8)
    if width_px < 240 or height_px < 200:
        return _unavailable(image, "Enlarge the map view")
    if not isinstance(table_map, dict) or not isinstance(table_map.get("table"), dict):
        return _unavailable(image, "Calibrate and capture the table first")
    table = table_map["table"]
    try:
        pockets = normalize_pockets(table)
        width, height, radius = (table[name] for name in ("width", "height", "ball_radius"))
    except (KeyError, TypeError, ValueError):
        return _unavailable(image, "Metric table geometry is invalid")
    balls = table_map.get("balls")
    if not isinstance(balls, list):
        return _unavailable(image, "No valid mapped observations")

    # Use one scale on both axes; never stretch the table to fill its panel.
    top, bottom, side = 83, 103, 43
    scale = min((width_px - 2 * side) / width,
                (height_px - top - bottom) / height)
    if scale <= 0:
        return _unavailable(image, "Enlarge the map view")
    left = (width_px - width * scale) / 2
    upper = top + (height_px - top - bottom - height * scale) / 2

    def point(x, y):
        return round(left + x * scale), round(upper + y * scale)

    def coordinates(item):
        if not isinstance(item, dict):
            return None
        x, y = item.get("x"), item.get("y")
        if not (_finite(x) and _finite(y) and 0 <= x <= width and 0 <= y <= height):
            return None
        return point(x, y)

    mapped = [(ball, coordinates(ball)) for ball in balls]
    skipped = sum(center is None for _, center in mapped)
    status = table_map.get("status", "invalid")
    if status not in ("review", "confirmed", "invalid") or skipped or not mapped:
        status = "invalid"
    title = {"confirmed": "CONFIRMED", "review": "REVIEW - NO GUIDANCE",
             "invalid": "INVALID - NO GUIDANCE"}[status]
    if len(mapped) == skipped:
        title = "UNAVAILABLE - NO POSITIONS"
    status_color = OBJECT_PATH if status == "confirmed" else ERROR if status == "invalid" else WARNING
    _label(image, "VIRTUAL TABLE | " + title, 10, 20, status_color, 0.48)
    _label(image, f"{width:g} x {height:g} cm | {scale:.2f} px/cm | origin: top left (0,0)",
           10, 40, scale=0.36)
    a, b = point(0, 0), point(width, height)
    cv2.rectangle(image, (a[0] - 8, a[1] - 8), (b[0] + 8, b[1] + 8), (50, 63, 74), -1)
    cv2.rectangle(image, a, b, FELT, -1)

    # Grid spacing uses physical cm and remains readable at both demo scales.
    target_spacing = 45 / scale
    exponent = 10 ** math.floor(math.log10(target_spacing))
    grid = next(v * exponent for v in (1, 2, 5, 10) if v * exponent >= target_spacing)
    for x in np.arange(grid, width, grid):
        cv2.line(image, point(x, 0), point(x, height), (77, 102, 47), 1)
    for y in np.arange(grid, height, grid):
        cv2.line(image, point(0, y), point(width, y), (77, 102, 47), 1)
    _label(image, f"Grid {grid:g} cm", width_px - 88, 59, scale=0.34)
    # An explicit axis key avoids mistaking this view for a camera projection.
    cv2.arrowedLine(image, (14, 55), (44, 55), TEXT, 1, tipLength=0.2)
    cv2.arrowedLine(image, (14, 55), (14, 76), TEXT, 1, tipLength=0.25)
    _label(image, "+x", 48, 60, scale=0.3)
    _label(image, "+y", 19, 77, scale=0.3)

    lookup = {p["id"]: p for p in pockets}
    for pocket in pockets:
        x, y, mouth = pocket["x"], pocket["y"], pocket["width"]
        center = point(x, y)
        color = ERROR if pocket["blocked"] else (17, 17, 17)
        if pocket["id"][1] == "M":
            cv2.line(image, point(x - mouth / 2, y), point(x + mouth / 2, y),
                     color, max(3, round(radius * scale)), cv2.LINE_AA)
        else:
            dx = mouth / math.sqrt(2) * (1 if x == 0 else -1)
            dy = mouth / math.sqrt(2) * (1 if y == 0 else -1)
            polygon = np.asarray([center, point(x + dx, y), point(x, y + dy)], np.int32)
            cv2.fillConvexPoly(image, polygon, color, cv2.LINE_AA)
        _label(image, pocket["id"] + (" X" if pocket["blocked"] else ""), center[0],
               center[1] - 12 if y == 0 else center[1] + 23,
               ERROR if pocket["blocked"] else TEXT, 0.34, centered=True)
    cushions = []
    for side_name, y in (("T", 0), ("B", height)):
        l, m, r = (lookup[side_name + name] for name in ("L", "M", "R"))
        cushions += [((l["width"] / math.sqrt(2), y), (m["x"] - m["width"] / 2, y)),
                     ((m["x"] + m["width"] / 2, y), (width - r["width"] / math.sqrt(2), y))]
    for side_name, x in (("L", 0), ("R", width)):
        cushions.append(((x, lookup["T" + side_name]["width"] / math.sqrt(2)),
                         (x, height - lookup["B" + side_name]["width"] / math.sqrt(2))))
    for start, end in cushions:
        cv2.line(image, point(*start), point(*end), (105, 150, 90), 3, cv2.LINE_AA)

    def path(points, color, dashed=False):
        if not isinstance(points, list):
            return
        pixels = [coordinates(p) for p in points]
        for start, end in zip(pixels, pixels[1:]):
            if start is None or end is None:
                continue
            if dashed:
                length = math.dist(start, end)
                for distance in range(0, round(length), 11):
                    first = distance / max(1, length)
                    last = min(1, (distance + 5) / max(1, length))
                    p = tuple(round(start[i] + (end[i] - start[i]) * first) for i in (0, 1))
                    q = tuple(round(start[i] + (end[i] - start[i]) * last) for i in (0, 1))
                    cv2.line(image, p, q, color, 1, cv2.LINE_AA)
            else:
                cv2.arrowedLine(image, start, end, color, 2, cv2.LINE_AA, tipLength=0.07)

    show_shot = status == "confirmed" and isinstance(shot, dict) and bool(mapped)
    if show_shot:
        path(shot.get("cue_path"), CUE_PATH)
        path(shot.get("object_path"), OBJECT_PATH)
        continuation = shot.get("scratch_risk", {})
        if isinstance(continuation, dict):
            path(continuation.get("cue_path"), CONTINUATION, dashed=True)
        ghost = coordinates(shot.get("ghost"))
        if ghost is not None:
            cv2.circle(image, ghost, max(1, round(radius * scale)), CUE_PATH, 1, cv2.LINE_AA)
        bank = shot.get("bank")
        if isinstance(bank, dict) and coordinates(bank.get("point")) is not None:
            cv2.drawMarker(image, coordinates(bank["point"]), OBJECT_PATH,
                           cv2.MARKER_DIAMOND, 9, 1, cv2.LINE_AA)

    for ball, center in mapped:
        if center is None:
            continue
        ball_radius = max(1, round(radius * scale))
        uncertainty = ball.get("position_uncertainty_cm", 0)
        if _finite(uncertainty) and uncertainty > 0:
            extent = min((radius + uncertainty) * scale, math.hypot(width_px, height_px))
            cv2.circle(image, center, max(ball_radius + 1, round(extent)),
                       WARNING, 1, cv2.LINE_AA)
        number = ball.get("id")
        known = isinstance(number, int) and not isinstance(number, bool) and 0 <= number <= 15
        color = _BALL_COLORS[number if number <= 8 else number - 8] if known else (155, 155, 155)
        cv2.circle(image, center, ball_radius, color, -1, cv2.LINE_AA)
        cv2.circle(image, center, ball_radius, TEXT, 1, cv2.LINE_AA)
        if known and number > 8:
            cv2.ellipse(image, center, (ball_radius, max(1, ball_radius // 2)),
                        0, 0, 360, TEXT, 1, cv2.LINE_AA)
        if selected is not None and ball.get("key") == selected:
            cv2.circle(image, center, ball_radius + 4, (255, 255, 255), 2, cv2.LINE_AA)
        _label(image, str(number) if known else "?", center[0] + ball_radius + 4,
               center[1] + 4, TEXT, 0.35, max_width=30)

    footer = height_px - 66
    count = len(mapped) - skipped
    _label(image, f"{count} mapped balls | ? unknown identity | rings: position uncertainty",
           10, footer, scale=0.34)
    if show_shot:
        _label(image, "Cyan: cue + ghost | green: object | purple dashes: conditional stun",
               10, footer + 18, scale=0.34)
    else:
        _label(image, "No shot guidance" if count else "No observed ball positions - map unavailable",
               10, footer + 18, WARNING, 0.34)
    issues = table_map.get("issues", [])
    issue = str(issues[0]) if isinstance(issues, list) and issues else ""
    if skipped:
        issue = f"{skipped} invalid positions omitted. " + issue
    if issue:
        _label(image, issue, 10, footer + 38, WARNING, 0.34)
    return image
