"""Conservative, dependency-free direct-pot geometry for an observed pool table.

Coordinates are centimetres, with x rightward and y downward when looking at the
table from above. The table rectangle follows the cushion faces. Pocket openings
are idealized gaps in those faces; pocket jaws, cloth, spin, force, and cue-ball
motion after impact are not modeled. Scores rank geometry, not success probability.
"""

import math
import re


EPSILON = 1e-7
MAX_CUT_ANGLE_DEG = 80.0
GROUPS = ("solids", "stripes", "open")
_COLORS = ("white", "yellow", "blue", "red", "purple", "orange", "green", "maroon", "black")
BALLS = {
    number: {
        "name": "cue ball" if number == 0 else (
            "8 black ball" if number == 8 else
            f"{number} {_COLORS[number if number < 8 else number - 8]} "
            + ("solid" if number < 8 else "stripe")
        ),
        "color": _COLORS[number if number <= 8 else number - 8],
        "group": "cue" if number == 0 else (
            "eight" if number == 8 else "solids" if number < 8 else "stripes"
        ),
    }
    for number in range(16)
}


def default_state():
    """Return an example layout, never a replacement for camera observations."""
    positions = [(55.0, 63.5)] + [
        (115.0 + 22.0 * column, 28.0 + 17.0 * row)
        for row in range(5) for column in range(3)
    ]
    return {
        "table": {
            "width": 254.0, "height": 127.0,
            "ball_radius": 2.8575, "pocket_width": 11.5,
        },
        "balls": [
            {"id": number, "x": point[0], "y": point[1]}
            for number, point in enumerate(positions)
        ],
        "group": "open",
    }


def parse_question(question, current_group):
    """Parse a deliberately limited shot request; never silently accept extras.

    Examples: 'I am solids, what is my best shot?', 'aim for ball 10',
    'stripes', 'where should I aim?', 'shoot the 3'. An explicit group overrides
    the selected group. No claim about ball identity is inferred from this text.
    """
    if current_group not in GROUPS:
        raise ValueError("Choose a group: solids, stripes, or open.")
    if not isinstance(question, str) or len(question) > 500:
        raise ValueError("Use a shot request containing at most 500 characters.")
    normalized = question.lower().replace("\u2019", "'")
    normalized = re.sub(r"\bi'm\b", "i am", normalized)
    normalized = re.sub(r"\bwe're\b", "we are", normalized)
    normalized = re.sub(r"\bwhat's\b", "what is", normalized)
    normalized = re.sub(r"\b(?:eight[- ]?ball|eight)\b", "8 ball", normalized)
    normalized = re.sub(r"\blow balls?\b", "solids", normalized)
    normalized = re.sub(r"\bhigh balls?\b", "stripes", normalized)
    normalized = re.sub(r"\b(?:burgundy|brown)\b", "maroon", normalized)
    if re.search(r"(?:^|\s)-\s*\d", normalized):
        raise ValueError("Ball numbers must be positive integers from 1 through 15.")
    words = re.findall(r"[a-z]+|\d+|[^\w\s?,.!#'\-]", normalized)
    named_groups = set()
    for word in words:
        if word in ("solid", "solids"):
            named_groups.add("solids")
        elif word in ("stripe", "stripes"):
            named_groups.add("stripes")
        elif word == "open":
            named_groups.add("open")
    if len(named_groups) > 1:
        raise ValueError("The request contains conflicting groups; choose one group.")
    group = next(iter(named_groups), current_group)
    allowed = set(
        "i am we are my our the a an please what is should can could do does "
        "you me us give show tell suggest recommend find calculate help with "
        "best next easiest shot shots which ball balls hit shoot pot pocket "
        "aim aiming for at to where how solids solid stripes stripe open table "
        "play playing on of have in this position here current layout situation "
        "yellow blue red purple orange green maroon black white".split()
    )
    unknown = sorted({word for word in words if word not in allowed and not word.isdigit()})
    if unknown:
        raise ValueError(
            "Unsupported request: " + ", ".join(unknown)
            + ". Ask for a direct shot, optionally naming solids, stripes, or a ball number."
        )
    numbers = {int(word) for word in words if word.isdigit()}
    if len(numbers) > 1:
        raise ValueError("Choose only one target ball in each request.")
    if numbers and not 1 <= next(iter(numbers)) <= 15:
        raise ValueError("The target must be an object ball numbered 1 through 15.")
    colors = set(words).intersection(_COLORS)
    if len(colors) > 1:
        raise ValueError("Choose only one target color in each request.")
    if colors:
        color = next(iter(colors))
        if color == "white":
            raise ValueError("The white cue ball cannot be the target; name an object ball.")
        if numbers:
            if BALLS[next(iter(numbers))]["color"] != color:
                raise ValueError("The requested color and ball number do not match.")
        elif color == "black":
            numbers = {8}
        elif group == "open":
            raise ValueError("That color has a solid and a stripe; specify the group or ball number.")
        else:
            numbers = {_COLORS.index(color) + (8 if group == "stripes" else 0)}
    if question.strip() and not named_groups and not numbers and not set(words).intersection(
        {"shot", "shots", "ball", "balls", "hit", "shoot", "pot", "pocket", "aim", "aiming"}
    ):
        raise ValueError("Ask for a shot, for example: 'What is my best shot?'.")
    result = {"group": group}
    if numbers:
        result["ball_id"] = next(iter(numbers))
    return result


def _number(value, name, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number.")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}.")
    return value


def _validate_state(state):
    if not isinstance(state, dict):
        raise ValueError("Supply a table state object.")
    table = state.get("table")
    if not isinstance(table, dict):
        raise ValueError("Calibrate the table width, height, ball radius, and pocket width first.")
    width = _number(table.get("width"), "Table width", 20.0, 1000.0)
    height = _number(table.get("height"), "Table height", 20.0, 1000.0)
    radius = _number(table.get("ball_radius"), "Ball radius", 0.5, 10.0)
    pocket_width = _number(table.get("pocket_width"), "Pocket width", 1.0, 100.0)
    if 4 * radius >= min(width, height):
        raise ValueError("The table is too small for the supplied ball radius.")
    if pocket_width <= 2 * radius + EPSILON:
        raise ValueError("Pocket width must exceed the ball diameter.")
    if (pocket_width / math.sqrt(2) >= height / 2
            or pocket_width / math.sqrt(2) >= width / 2 - pocket_width / 2):
        raise ValueError("Pocket openings overlap; check table dimensions and pocket width.")
    group = state.get("group")
    if group not in GROUPS:
        raise ValueError("Choose a group: solids, stripes, or open.")
    source_balls = state.get("balls")
    if not isinstance(source_balls, list) or len(source_balls) > 16:
        raise ValueError("Supply a list of at most 16 individually identified balls.")
    balls = {}
    for item in source_balls:
        if not isinstance(item, dict):
            raise ValueError("Every ball needs an integer id and measured x/y coordinates.")
        number = item.get("id")
        if isinstance(number, bool) or not isinstance(number, int) or not 0 <= number <= 15:
            raise ValueError("Ball IDs must be integers from 0 (cue ball) through 15.")
        if number in balls:
            raise ValueError(f"Ball {number} appears more than once; confirm its identity.")
        x = _number(item.get("x"), f"Ball {number} x")
        y = _number(item.get("y"), f"Ball {number} y")
        if not (radius - EPSILON <= x <= width - radius + EPSILON
                and radius - EPSILON <= y <= height - radius + EPSILON):
            raise ValueError(f"Ball {number} is outside the playable table; recheck calibration.")
        balls[number] = (x, y)
    if 0 not in balls:
        raise ValueError("Locate and confirm the cue ball (id 0) before planning a shot.")
    entries = list(balls.items())
    for index, (number, point) in enumerate(entries):
        for other_number, other in entries[index + 1:]:
            if math.dist(point, other) < 2 * radius - EPSILON:
                raise ValueError(
                    f"Balls {number} and {other_number} overlap; confirm detections and calibration."
                )
    return width, height, radius, pocket_width, group, balls


def point_segment_distance(point, start, end):
    """Euclidean point/closed-segment distance, including zero-length segments."""
    dx, dy = end[0] - start[0], end[1] - start[1]
    denominator = dx * dx + dy * dy
    if denominator <= EPSILON * EPSILON:
        return math.dist(point, start)
    fraction = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / denominator
    fraction = min(1.0, max(0.0, fraction))
    return math.hypot(point[0] - start[0] - fraction * dx,
                      point[1] - start[1] - fraction * dy)


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segment_distance(a, b, c, d):
    ab_c, ab_d = _cross(a, b, c), _cross(a, b, d)
    cd_a, cd_b = _cross(c, d, a), _cross(c, d, b)
    if ab_c * ab_d < 0 and cd_a * cd_b < 0:
        return 0.0
    return min(point_segment_distance(a, c, d), point_segment_distance(b, c, d),
               point_segment_distance(c, a, b), point_segment_distance(d, a, b))


def _cushions(width, height, pocket_width):
    corner = pocket_width / math.sqrt(2.0)
    half = pocket_width / 2.0
    segments = []
    for y in (0.0, height):
        segments.extend([
            ((corner, y), (width / 2 - half, y)),
            ((width / 2 + half, y), (width - corner, y)),
        ])
    for x in (0.0, width):
        segments.append(((x, corner), (x, height - corner)))
    return segments


def _clearance(start, end, balls, exclude, diameter):
    """Return clearance beyond diameter; <= EPSILON includes tangent contact."""
    return min(
        (point_segment_distance(point, start, end) - diameter
         for number, point in balls.items() if number not in exclude),
        default=math.inf,
    )


def _point(point):
    return {"x": round(point[0], 6), "y": round(point[1], 6)}


def _rail_reference(cue, ghost, width, height):
    dx, dy = ghost[0] - cue[0], ghost[1] - cue[1]
    hits = []
    if dy < -EPSILON:
        hits.append((-cue[1] / dy, "top"))
    elif dy > EPSILON:
        hits.append(((height - cue[1]) / dy, "bottom"))
    if dx < -EPSILON:
        hits.append((-cue[0] / dx, "left"))
    elif dx > EPSILON:
        hits.append(((width - cue[0]) / dx, "right"))
    travel, side = min(hits)
    x = min(width, max(0.0, cue[0] + travel * dx))
    y = min(height, max(0.0, cue[1] + travel * dy))
    diamond = x / width * 8 if side in ("top", "bottom") else y / height * 4
    origin = "left" if side in ("top", "bottom") else "top"
    return {
        "side": side, "x": round(x, 6), "y": round(y, 6),
        "diamond": round(diamond, 3),
        "description": (
            f"Aiming reference only: {side} rail, diamond coordinate {diamond:.2f} "
            f"counted from the {origin}. Extend the cue-to-ghost line mentally; "
            "this is not a bank shot or a cue-ball contact instruction."
        ),
    }


def _candidate(cue, target, number, pocket, balls, table, cushions):
    width, height, radius, _ = table
    pocket_id, pocket_name, destination = pocket
    object_distance = math.dist(target, destination)
    direction = ((destination[0] - target[0]) / object_distance,
                 (destination[1] - target[1]) / object_distance)
    ghost = (target[0] - 2 * radius * direction[0],
             target[1] - 2 * radius * direction[1])
    if not (radius <= ghost[0] <= width - radius
            and radius <= ghost[1] <= height - radius):
        return None
    cue_distance = math.dist(cue, ghost)
    if cue_distance <= EPSILON:
        return None
    cosine = ((ghost[0] - cue[0]) * direction[0]
              + (ghost[1] - cue[1]) * direction[1]) / cue_distance
    if cosine <= 0:
        return None
    cut_angle = math.degrees(math.acos(min(1.0, cosine)))
    if cut_angle > MAX_CUT_ANGLE_DEG:
        return None
    # The object ball must fit past cushion edges and pocket lips along its
    # entire route. This rejects shallow approaches that clip an idealized jaw.
    if any(_segment_distance(target, destination, a, b) < radius - EPSILON
           for a, b in cushions):
        return None
    cue_clearance = _clearance(cue, ghost, balls, {0, number}, 2 * radius)
    object_clearance = _clearance(target, destination, balls, {0, number}, 2 * radius)
    clearance = min(cue_clearance, object_clearance)
    if clearance <= EPSILON:
        return None
    diagonal = math.hypot(width, height)
    tightness = max(0.0, 1.0 - clearance / (4 * radius))
    score = max(0.0, min(100.0,
        100.0 - 50.0 * (cut_angle / MAX_CUT_ANGLE_DEG) ** 1.5
        - 20.0 * cue_distance / diagonal - 20.0 * object_distance / diagonal
        - 10.0 * tightness))
    aim_angle = math.degrees(math.atan2(ghost[1] - cue[1], ghost[0] - cue[0])) % 360.0
    rail = _rail_reference(cue, ghost, width, height)
    ball_name = BALLS[number]["name"]
    reasons = [
        f"Cut angle {cut_angle:.1f} degrees; smaller cuts receive a higher score.",
        f"Cue travel {cue_distance:.1f} cm; object travel {object_distance:.1f} cm.",
        "Both modeled paths clear other observed balls and the idealized pocket opening.",
    ]
    if math.isfinite(clearance):
        reasons.append(f"Minimum extra clearance from other balls: {clearance:.1f} cm.")
    return {
        "id": f"{number}-{pocket_id}", "ball_id": number, "ball_name": ball_name,
        "pocket_id": pocket_id, "pocket_name": pocket_name,
        "ghost": _point(ghost), "cue_path": [_point(cue), _point(ghost)],
        "object_path": [_point(target), _point(destination)],
        "aim_angle_deg": round(aim_angle, 3), "cut_angle_deg": round(cut_angle, 3),
        "score": round(score, 2), "rail": rail,
        "instruction": (
            f"Aim the cue-ball center at the ghost point for the {ball_name}, "
            f"toward the {pocket_name} pocket. Aim angle {aim_angle:.1f} degrees "
            "clockwise from right in the overhead view. " + rail["description"]
        ),
        "reasons": reasons,
        "distances": {"cue": round(cue_distance, 4), "object": round(object_distance, 4)},
    }


def plan_shots(state, question=""):
    """Rank legal direct pots for confirmed ball identities and positions.

    Missing object balls are interpreted as pocketed. The caller must therefore
    explicitly confirm pocketed balls, rather than dropping occluded detections.
    This plans first contact/direct pot only; scratch risk and a resulting cue
    position are not evaluated. Invalid or ambiguous observations raise ValueError.
    """
    width, height, radius, pocket_width, group, balls = _validate_state(state)
    request = parse_question(question, group)
    group = request["group"]
    ordinary = set(range(1, 8)) if group == "solids" else set(range(9, 16))
    if group == "open":
        eligible = (set(range(1, 16)) - {8}).intersection(balls)
    else:
        eligible = ordinary.intersection(balls)
        if not eligible and 8 in balls:
            eligible = {8}
    requested = request.get("ball_id")
    if requested is not None:
        if requested not in balls:
            raise ValueError(f"Ball {requested} is not on the confirmed table layout.")
        if requested not in eligible:
            raise ValueError(f"Ball {requested} is not a legal first target for {group} in this layout.")
        eligible = {requested}
    pockets = (
        ("TL", "top left", (0.0, 0.0)),
        ("TM", "top middle", (width / 2, 0.0)),
        ("TR", "top right", (width, 0.0)),
        ("BL", "bottom left", (0.0, height)),
        ("BM", "bottom middle", (width / 2, height)),
        ("BR", "bottom right", (width, height)),
    )
    cushions = _cushions(width, height, pocket_width)
    shots = []
    for number in sorted(eligible):
        for pocket in pockets:
            candidate = _candidate(balls[0], balls[number], number, pocket, balls,
                                   (width, height, radius, pocket_width), cushions)
            if candidate is not None:
                shots.append(candidate)
    shots.sort(key=lambda shot: (-shot["score"], shot["ball_id"], shot["pocket_id"]))
    if shots:
        best = shots[0]
        message = f"Best modeled direct shot: {best['ball_name']} to {best['pocket_name']}."
    elif not eligible:
        message = "No legal target remains for the selected group; verify the group and pocketed-ball state."
    else:
        message = (
            "No clear direct pot meets the modeled clearance and cut-angle limits. "
            "Banks, combinations, spin, and defensive shots are not evaluated."
        )
    return {
        "status": "ok" if shots else "no_shot", "group": group,
        "message": message, "shots": shots, "eligible_ball_ids": sorted(eligible),
        "score_description": "Geometric ranking only; not a probability of making the shot.",
        "angle_convention": "Degrees clockwise from right in the overhead table view.",
        "assumptions": [
            "All remaining balls are confirmed; missing balls have explicitly been marked pocketed.",
            "Ideal level table, equal ball radii, and idealized pocket openings.",
            "Direct object-ball pots only, with no spin or post-impact cue-ball/scratch prediction.",
            f"Cuts exceeding {MAX_CUT_ANGLE_DEG:g} degrees are outside this planner's limit.",
            "Rail diamonds are aiming references counted from left/top, not bank instructions.",
        ],
    }
