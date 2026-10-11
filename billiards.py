"""Dependency-free candidate search for a confirmed, calibrated pool table.

Coordinates are centimetres, with x rightward and y downward when looking at the
table from above. The table rectangle follows the cushion faces. Pocket openings
are idealized gaps in those faces. Direct pots and one-cushion object-ball banks
are ranked with bounded observation error and an explicitly idealized stun-ball
continuation. This finite search is not a full pool simulator or an optimizer
over spin, combinations and arbitrary multi-rail shots. Scores are not odds.

Model references: https://drdavepoolinfo.com/faq/stun/shot/ and
https://drdavepoolinfo.com/faq/bank-kick/mirror/ . Real throw, rolling spin and
speed-dependent cushion response require calibration beyond these ideal models.
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


def parse_question(question, current_group="open"):
    """Parse a deliberately limited shot request; never silently accept extras.

    Examples: 'I am solids, what is my best shot?', 'aim for ball 10',
    'stripes', 'where should I aim?', 'shoot the 3'. An explicit group overrides
    the default target filter for this request only. A numbered ball overrides
    the default filter; an explicitly conflicting filter is rejected. No player
    assignment, match history, or claim about shot legality is inferred.
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
        "yellow blue red purple orange green maroon black white bank banks direct".split()
    )
    unknown = sorted({word for word in words if word not in allowed and not word.isdigit()})
    if unknown:
        raise ValueError(
            "Unsupported request: " + ", ".join(unknown)
            + ". Ask for a shot, optionally naming solids, stripes, or a ball number."
        )
    shot_types = set(words).intersection({"bank", "banks", "direct"})
    if "direct" in shot_types and shot_types.intersection({"bank", "banks"}):
        raise ValueError("Choose direct shots or bank shots in one request.")
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
    if numbers and named_groups and group != "open":
        number = next(iter(numbers))
        if number != 8 and BALLS[number]["group"] != group:
            raise ValueError(f"Ball {number} does not match the requested {group} target filter.")
    if question.strip() and not named_groups and not numbers and not set(words).intersection(
        {"shot", "shots", "ball", "balls", "hit", "shoot", "pot", "pocket", "aim", "aiming", "bank", "banks", "direct"}
    ):
        raise ValueError("Ask for a shot, for example: 'What is my best shot?'.")
    result = {"group": group}
    if numbers:
        result["ball_id"] = next(iter(numbers))
    if shot_types:
        result["shot_type"] = "direct" if "direct" in shot_types else "bank"
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
    # Actual mouth widths can all override this default; overlap and diameter
    # checks belong to normalize_pockets, after applying those overrides.
    cushions = _pocket_cushions(width, height, normalize_pockets(table))
    group = state.get("group", "open")
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
        if not _center_is_playable((x, y), width, height, radius, cushions):
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


def normalize_pockets(table):
    """Validate six pocket mouths, allowing partial calibrated overrides.

    Coordinates use cushion faces, so corner mouths are at table corners and
    middle mouths can move along their long rail. ``blocked`` disables a target;
    it does not remove a physical hole from the scratch model.
    """
    width = _number(table.get("width"), "Table width", 20, 1000)
    height = _number(table.get("height"), "Table height", 20, 1000)
    radius = _number(table.get("ball_radius"), "Ball radius", 0.5, 10)
    mouth = _number(table.get("pocket_width"), "Pocket width", 1, 100)
    specs = [("TL", "top left", 0., 0.), ("TM", "top middle", width / 2, 0.),
             ("TR", "top right", width, 0.), ("BL", "bottom left", 0., height),
             ("BM", "bottom middle", width / 2, height),
             ("BR", "bottom right", width, height)]
    pockets = {key: dict(id=key, name=name, x=x, y=y, width=mouth, blocked=False)
               for key, name, x, y in specs}
    overrides = table.get("pockets", [])
    if not isinstance(overrides, list) or len(overrides) > 6:
        raise ValueError("Table pockets must be a list of at most six mouth overrides.")
    seen = set()
    for item in overrides:
        if (not isinstance(item, dict) or not isinstance(item.get("id"), str)
                or item["id"] not in pockets):
            raise ValueError("Pocket IDs must be TL, TM, TR, BL, BM, or BR.")
        key = item["id"]
        if key in seen:
            raise ValueError(f"Pocket {key} appears more than once.")
        seen.add(key)
        for field in ("x", "y", "width", "blocked"):
            if field in item:
                pockets[key][field] = item[field]
    for key, _, x, y in specs:
        pocket = pockets[key]
        pocket["x"] = _number(pocket["x"], f"Pocket {key} x", 0, width)
        pocket["y"] = _number(pocket["y"], f"Pocket {key} y", 0, height)
        pocket["width"] = _number(pocket["width"], f"Pocket {key} width", 1, 100)
        if pocket["width"] <= 2 * radius + EPSILON:
            raise ValueError(f"Pocket {key} width must exceed the ball diameter.")
        if not isinstance(pocket["blocked"], bool):
            raise ValueError(f"Pocket {key} blocked must be true or false.")
        if abs(pocket["y"] - y) > EPSILON or (key[1] != "M" and abs(pocket["x"] - x) > EPSILON):
            raise ValueError(f"Pocket {key} must lie on its calibrated cushion boundary.")
    for side in ("T", "B"):
        left, middle, right = (pockets[side + key] for key in ("L", "M", "R"))
        if (left["width"] / math.sqrt(2) >= middle["x"] - middle["width"] / 2
                or middle["x"] + middle["width"] / 2 >= width - right["width"] / math.sqrt(2)):
            raise ValueError("Pocket openings overlap on a long rail.")
    for side in ("L", "R"):
        if (pockets["T" + side]["width"] + pockets["B" + side]["width"]) / math.sqrt(2) >= height:
            raise ValueError("Pocket openings overlap on a short rail.")
    return list(pockets.values())


def _pocket_cushions(width, height, pockets):
    lookup = {p["id"]: p for p in pockets}
    segments = []
    for side, y in (("T", 0.), ("B", height)):
        left, middle, right = (lookup[side + key] for key in ("L", "M", "R"))
        segments.extend([
            ((left["width"] / math.sqrt(2), y), (middle["x"] - middle["width"] / 2, y)),
            ((middle["x"] + middle["width"] / 2, y), (width - right["width"] / math.sqrt(2), y)),
        ])
    for side, x in (("L", 0.), ("R", width)):
        segments.append(((x, lookup["T" + side]["width"] / math.sqrt(2)),
                         (x, height - lookup["B" + side]["width"] / math.sqrt(2))))
    return segments


def table_cushion_segments(table):
    """Return the measured physical cushion faces, excluding pocket openings."""
    pockets = normalize_pockets(table)
    return _pocket_cushions(table["width"], table["height"], pockets)


def _center_is_playable(point, width, height, radius, cushions):
    return (0 <= point[0] <= width and 0 <= point[1] <= height
            and all(point_segment_distance(point, a, b) >= radius - EPSILON for a, b in cushions))


def ball_center_is_playable(table, x, y):
    """Check a nominal observed center against real cushion gaps, including hangers.

    Centers inside a pocket mouth may lie less than one radius from the table
    boundary. They remain observed balls wherever a current observation places
    them; proximity to a pocket does not establish that a ball was pocketed.
    Centers outside the outer rectangle are rejected: pocket shelf geometry is
    not calibrated. Invalid numeric/table input raises ValueError.
    """
    if not isinstance(table, dict):
        raise ValueError("Supply a calibrated table object.")
    pockets = normalize_pockets(table)
    width, height, radius = table["width"], table["height"], table["ball_radius"]
    if 4 * radius >= min(width, height):
        raise ValueError("The table is too small for the supplied ball radius.")
    point = (_number(x, "Ball center x"), _number(y, "Ball center y"))
    return _center_is_playable(point, width, height, radius, _pocket_cushions(width, height, pockets))


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
            "this reference does not specify the object-ball rebound or cue-ball rail contact."
        ),
    }


def _planner_options(state, radius):
    options = state.get("planner", {})
    if not isinstance(options, dict):
        raise ValueError("Planner settings must be an object.")
    allow_banks = options.get("allow_banks", True)
    if not isinstance(allow_banks, bool):
        raise ValueError("Planner allow_banks must be true or false.")
    # Parameters are measured/calibrated quantities when provided. Defaults are
    # examples for the energy heuristic, not a claim about any particular cloth.
    return {
        "allow_banks": allow_banks,
        "max_cut_angle_deg": _number(options.get("max_cut_angle_deg", MAX_CUT_ANGLE_DEG), "Maximum cut angle", 5, 85),
        "rolling_deceleration_cm_s2": _number(options.get("rolling_deceleration_cm_s2", 20), "Rolling deceleration", 0.1, 300),
        "cushion_restitution": _number(options.get("cushion_restitution", 0.8), "Cushion restitution", 0.2, 1),
        "pocket_entry_speed_cm_s": _number(options.get("pocket_entry_speed_cm_s", 8 * radius), "Pocket entry speed", 1, 150),
        "position_uncertainty_cm": _number(options.get("position_uncertainty_cm", 0), "Planner position uncertainty", 0, 30),
    }


def _observations(state, options):
    table_error = _number(state["table"].get("position_uncertainty_cm", 0), "Table position uncertainty", 0, 30)
    return {
        b["id"]: {
            "confidence": _number(b.get("confidence", 1), f"Ball {b['id']} confidence", 0, 1),
            "uncertainty": options["position_uncertainty_cm"] + max(
                table_error, _number(b.get("position_uncertainty_cm", 0), f"Ball {b['id']} position uncertainty", 0, 30)),
        } for b in state["balls"]
    }


def _observed_clearance(start, end, balls, exclude, diameter, path_error, observations):
    return min((point_segment_distance(point, start, end) - diameter - path_error
                - observations[number]["uncertainty"]
                for number, point in balls.items() if number not in exclude), default=math.inf)


def _bank_path(target, destination, side, width, height, radius):
    """Reflect across ball-center rail, then validate the actual cushion below."""
    axis = 1 if side in ("top", "bottom") else 0
    rail = {"top": radius, "bottom": height - radius,
            "left": radius, "right": width - radius}[side]
    mirror = list(destination)
    mirror[axis] = 2 * rail - destination[axis]
    delta = mirror[axis] - target[axis]
    if abs(delta) <= EPSILON:
        return None
    fraction = (rail - target[axis]) / delta
    if not EPSILON < fraction < 1 - EPSILON:
        return None
    bank = tuple(target[i] + fraction * (mirror[i] - target[i]) for i in (0, 1))
    if not radius <= bank[1 - axis] <= (width if axis == 1 else height) - radius:
        return None
    if math.dist(target, bank) <= radius or math.dist(bank, destination) <= radius:
        return None
    return [target, bank, destination]


def _ray_circle(start, direction, center, radius):
    dx, dy = start[0] - center[0], start[1] - center[1]
    b = dx * direction[0] + dy * direction[1]
    c = dx * dx + dy * dy - radius * radius
    discriminant = b * b - c
    if discriminant < 0:
        return math.inf
    hit = -b - math.sqrt(max(0., discriminant))
    return hit if hit > EPSILON else (0. if c <= 0 else math.inf)


def _cue_continuation(ghost, incoming, object_direction, impact_speed, options,
                      table, cushions, pockets, balls, number, error, observations):
    """Bounded ideal stun trajectory; stop at other balls or uncertain rail jaws.

    Equal-mass impact removes the incoming normal component. Residual tangent
    velocity and constant cloth drag give a *conditional* stop estimate. No
    assertion is made that a center-ball human stroke arrives in a stun state.
    """
    width, height, radius, _ = table
    cosine = sum(incoming[i] * object_direction[i] for i in (0, 1))
    tangent = tuple(incoming[i] - cosine * object_direction[i] for i in (0, 1))
    magnitude = math.hypot(*tangent)
    path = [ghost]
    warning = "Conditional on a stun impact; follow, draw, throw and execution error change this path."
    if magnitude <= EPSILON:
        return {"level": "low", "penalty": 0., "event": "stopped", "pocket_id": None,
                "cue_path": [_point(ghost)], "stop": _point(ghost), "warning": warning}
    direction = tuple(v / magnitude for v in tangent)
    remaining = (impact_speed * magnitude) ** 2 / (2 * options["rolling_deceleration_cm_s2"])
    position = ghost
    event, level, pocket_id = "stopped", "low", None
    # At most two cushion rebounds are interpreted; later events become unknown.
    for _ in range(3):
        hits = []
        edge_hits = []
        for axis, length in ((0, width), (1, height)):
            if abs(direction[axis]) > EPSILON:
                boundary = radius if direction[axis] < 0 else length - radius
                distance = (boundary - position[axis]) / direction[axis]
                if distance >= -EPSILON:
                    hits.append((max(0., distance), axis))
                physical_edge = 0. if direction[axis] < 0 else length
                physical_distance = (physical_edge - position[axis]) / direction[axis]
                if physical_distance >= -EPSILON:
                    edge_hits.append(max(0., physical_distance))
        rail_distance, axis = min(hits, default=(math.inf, 0))
        collision = min((_ray_circle(position, direction, point,
                                     2 * radius + error + observations[ball_id]["uncertainty"])
                         for ball_id, point in balls.items() if ball_id not in (0, number)), default=math.inf)
        # Check complete aperture before reflecting from the inset rail plane.
        edge = (0. if direction[axis] < 0 else (width if axis == 0 else height))
        # Near corners, the first physical boundary can belong to a different
        # axis from the first radius-offset cushion plane.
        edge_distance = min(edge_hits, default=math.inf)
        exit_point = tuple(position[i] + direction[i] * edge_distance for i in (0, 1))
        opening = (math.isfinite(edge_distance) and all(
            _segment_distance(position, exit_point, a, b) >= radius - EPSILON for a, b in cushions))
        if opening and edge_distance <= min(remaining, collision):
            nearest = min(pockets, key=lambda p: math.dist(exit_point, (p["x"], p["y"])))
            path.append(exit_point)
            event, level, pocket_id = "scratch", "high", nearest["id"]
            break
        if collision <= min(remaining, rail_distance):
            path.append(tuple(position[i] + direction[i] * collision for i in (0, 1)))
            event, level = "ball_collision", "unknown"
            break
        if remaining <= rail_distance:
            path.append(tuple(position[i] + direction[i] * remaining for i in (0, 1)))
            break
        hit_point = tuple(position[i] + direction[i] * rail_distance for i in (0, 1))
        path.append(hit_point)
        # A near-mouth ray can hit a curved jaw, for which flat reflection would
        # fabricate precision. Terminate and flag rather than invent a rebound.
        face = list(hit_point)
        face[axis] = edge
        on_face = any(point_segment_distance(tuple(face), a, b) <= EPSILON for a, b in cushions)
        near_mouth = any(math.dist(hit_point, (p["x"], p["y"])) < p["width"] / math.sqrt(2) + radius + error
                         for p in pockets)
        if not on_face or near_mouth:
            event, level = "pocket_jaw", "medium"
            break
        remaining = (remaining - rail_distance) * options["cushion_restitution"] ** 2
        direction = tuple(-v if i == axis else v for i, v in enumerate(direction))
        position = hit_point
    else:
        event, level = "trace_limit", "unknown"
    penalty = {"low": 0., "medium": 15., "high": 65., "unknown": 9.}[level]
    return {"level": level, "penalty": penalty, "event": event, "pocket_id": pocket_id,
            "cue_path": [_point(p) for p in path],
            "stop": _point(path[-1]) if event == "stopped" else None, "warning": warning}


def _candidate(cue, target, number, pocket, balls, table, cushions, options,
               observations, pockets, bank_side=None):
    width, height, radius, _ = table
    destination = (pocket["x"], pocket["y"])
    path = (_bank_path(target, destination, bank_side, width, height, radius)
            if bank_side else [target, destination])
    if path is None:
        return None, "bank_geometry"
    lengths = [math.dist(a, b) for a, b in zip(path, path[1:])]
    if min(lengths) <= EPSILON:
        return None, "degenerate_path"
    object_distance = sum(lengths)
    direction = tuple((path[1][i] - target[i]) / lengths[0] for i in (0, 1))
    ghost = tuple(target[i] - 2 * radius * direction[i] for i in (0, 1))
    error = max(observations[0]["uncertainty"], observations[number]["uncertainty"])
    if not (radius + error <= ghost[0] <= width - radius - error
            and radius + error <= ghost[1] <= height - radius - error):
        return None, "ghost_outside_table"
    cue_distance = math.dist(cue, ghost)
    if cue_distance <= EPSILON:
        return None, "frozen_cue_ball"
    incoming = tuple((ghost[i] - cue[i]) / cue_distance for i in (0, 1))
    cosine = sum(incoming[i] * direction[i] for i in (0, 1))
    if cosine <= 0:
        return None, "wrong_contact_side"
    cut_angle = math.degrees(math.acos(min(1., cosine)))
    if cut_angle > options["max_cut_angle_deg"]:
        return None, "cut_limit"
    # A cue ball can itself hang in a pocket mouth. Unlike two points strictly
    # inside the cushion-inset rectangle, its line to the ghost is not
    # automatically clear of the pocket jaws.
    if any(_segment_distance(cue, ghost, a, b) < radius + error - EPSILON for a, b in cushions):
        return None, "cue_path_hits_cushion"
    # Only the selected bank cushion permits intentional tangent contact.
    bank_axis = 1 if bank_side in ("top", "bottom") else 0
    bank_face = {"top": 0., "bottom": height, "left": 0., "right": width}.get(bank_side)
    for a, b in cushions:
        selected = bank_side is not None and a[bank_axis] == b[bank_axis] == bank_face
        required = radius if selected else radius + error
        if any(_segment_distance(start, end, a, b) < required - EPSILON
               for start, end in zip(path, path[1:])):
            return None, "pocket_jaw_or_cushion"
    if bank_side:
        # Reject reflection points inside a hole or near a jaw. The mirror
        # construction alone would happily bounce off an empty pocket mouth.
        contact = list(path[1])
        contact[bank_axis] = bank_face
        if not any(a[bank_axis] == b[bank_axis] == bank_face
                   and point_segment_distance(tuple(contact), a, b) < EPSILON
                   and min(math.dist(tuple(contact), a), math.dist(tuple(contact), b)) > radius + error
                   for a, b in cushions):
            return None, "bank_in_pocket_opening"
    cue_clearance = _observed_clearance(cue, ghost, balls, {0, number}, 2 * radius, error, observations)
    object_clearance = min(_observed_clearance(a, b, balls, {0, number}, 2 * radius,
                                             observations[number]["uncertainty"], observations)
                           for a, b in zip(path, path[1:]))
    clearance = min(cue_clearance, object_clearance)
    if clearance <= EPSILON:
        return None, "ball_blocks_path"
    # Solve a simple energy budget backwards from a gentle pocket-entry speed.
    # Bank loss increases launch-speed guidance; geometry still uses reflection.
    deceleration = options["rolling_deceleration_cm_s2"]
    speed_squared = options["pocket_entry_speed_cm_s"] ** 2 + 2 * deceleration * lengths[-1]
    if bank_side:
        speed_squared = speed_squared / options["cushion_restitution"] ** 2 + 2 * deceleration * lengths[0]
    impact_speed = math.sqrt(speed_squared) / cosine
    launch_speed = math.sqrt(impact_speed ** 2 + 2 * deceleration * cue_distance)
    power = "soft" if launch_speed < 100 else "medium" if launch_speed < 190 else "firm"
    continuation = _cue_continuation(ghost, incoming, direction, impact_speed, options,
                                     table, cushions, pockets, balls, number, error, observations)
    diagonal = math.hypot(width, height)
    confidence = min(observations[0]["confidence"], observations[number]["confidence"])
    mouth_lips = [(a, b) for a, b in cushions
                 if min(math.dist(a, destination), math.dist(b, destination)) <= pocket["width"] + EPSILON]
    opening_clearance = min(_segment_distance(path[-2], path[-1], a, b) - radius for a, b in mouth_lips)
    factors = {
        "cut": 42 * (cut_angle / options["max_cut_angle_deg"]) ** 1.5,
        "cue_distance": 15 * cue_distance / diagonal,
        "object_distance": 18 * object_distance / diagonal,
        "clearance": 10 * max(0., 1. - clearance / (4 * radius)),
        "pocket_margin": 6 * max(0., 1. - max(0., opening_clearance - error) / radius),
        "bank": 13. if bank_side else 0.,
        "observation": 15 * (1 - confidence) + min(12., 8 * error / radius),
        "scratch": continuation["penalty"],
        "power": min(12., max(0., launch_speed - 180) / 25),
    }
    score = max(0., min(100., 100 - sum(factors.values())))
    aim_angle = math.degrees(math.atan2(ghost[1] - cue[1], ghost[0] - cue[0])) % 360.
    rail = _rail_reference(cue, ghost, width, height)
    ball_name = BALLS[number]["name"]
    route = (f"banking off the {bank_side} cushion into" if bank_side else "toward")
    reasons = [
        f"Cut angle {cut_angle:.1f} degrees; cue travel {cue_distance:.1f} cm; object travel {object_distance:.1f} cm.",
        "Both paths clear every other observed ball and the modeled pocket jaws, including supplied position uncertainty.",
        f"Conditional stun scratch assessment: {continuation['level']} ({continuation['event'].replace('_', ' ')}).",
        f"Suggested {power} pace; speed estimate depends on calibrated cloth drag and a stun impact.",
    ]
    if bank_side:
        reasons.append("One-cushion mirror bank; real cushion angle varies with speed and spin.")
    if math.isfinite(clearance):
        reasons.append(f"Minimum additional clearance after uncertainty margins: {clearance:.2f} cm.")
    if error or confidence < 1:
        reasons.append(f"Position margin {error:.2f} cm; cue/target detection confidence {confidence:.2f}.")
    return {
        "id": f"{number}-{pocket['id']}" + (f"-bank-{bank_side}" if bank_side else ""),
        "shot_type": "bank" if bank_side else "direct", "ball_id": number, "ball_name": ball_name,
        "pocket_id": pocket["id"], "pocket_name": pocket["name"],
        "ghost": _point(ghost), "cue_path": [_point(cue), _point(ghost)],
        "object_path": [_point(p) for p in path],
        "bank": {"rail": bank_side, "point": _point(path[1])} if bank_side else None,
        "aim_angle_deg": round(aim_angle, 3), "cut_angle_deg": round(cut_angle, 3),
        "score": round(score, 2), "score_factors": {key: round(value, 3) for key, value in factors.items()},
        "rail": rail, "scratch_risk": continuation,
        "power": {"level": power, "cue_speed_cm_s": round(launch_speed, 2),
                  "model": "Heuristic cloth-drag energy budget; calibrate before relying on numerical speed."},
        "confidence": round(confidence, 4), "position_uncertainty_cm": round(error, 4),
        "clearance_cm": round(clearance, 4) if math.isfinite(clearance) else None,
        "instruction": (f"Aim the cue-ball center at the ghost point for the {ball_name}, {route} the {pocket['name']} pocket. "
                        f"Aim angle {aim_angle:.1f} degrees clockwise from right in the overhead view. "
                        f"Use {power} pace with a stun impact. "
                        + ("High modeled scratch risk: choose another shot or change the stroke. " if continuation["level"] == "high" else "")
                        + rail["description"]),
        "reasons": reasons,
        "distances": {"cue": round(cue_distance, 4), "object": round(object_distance, 4)},
    }, None


def plan_shots(state, question=""):
    """Rank direct pots and single-cushion banks for confirmed observations.

    The current layout contains only observed physical state. A group is an
    optional default target filter, overridden by the request, never a player
    assignment. Missing balls are not assumed pocketed and never unlock the
    eight ball. Explicit requests for ball 8 evaluate geometry, not game rules.
    All visible balls must be mapped because every ball can obstruct a shot;
    the planner cannot detect obstacles omitted from the supplied observations.
    Scores combine geometry, observation quality, pocket access, speed demand,
    and a conditional stun scratch estimate. Invalid input raises ValueError.
    """
    width, height, radius, pocket_width, group, balls = _validate_state(state)
    request = parse_question(question, group)
    group = request["group"]
    ordinary = set(range(1, 8)) if group == "solids" else set(range(9, 16))
    if group == "open":
        eligible = (set(range(1, 16)) - {8}).intersection(balls)
    else:
        eligible = ordinary.intersection(balls)
    requested = request.get("ball_id")
    if requested is not None:
        if requested not in balls:
            raise ValueError(f"Ball {requested} is not on the confirmed table layout.")
        eligible = {requested}
    pockets = normalize_pockets(state["table"])
    options = _planner_options(state, radius)
    if request.get("shot_type") == "bank":
        options["allow_banks"] = True
    observations = _observations(state, options)
    cushions = _pocket_cushions(width, height, pockets)
    shots = []
    rejected = {}
    considered = 0
    for number in sorted(eligible):
        for pocket in pockets:
            if pocket["blocked"]:
                rejected["pocket_marked_blocked"] = rejected.get("pocket_marked_blocked", 0) + 1
                continue
            bank_sides = [None, "top", "bottom", "left", "right"] if options["allow_banks"] else [None]
            if request.get("shot_type") == "direct":
                bank_sides = [None]
            elif request.get("shot_type") == "bank":
                bank_sides = ["top", "bottom", "left", "right"]
            for bank_side in bank_sides:
                considered += 1
                candidate, reason = _candidate(balls[0], balls[number], number, pocket, balls,
                                               (width, height, radius, pocket_width), cushions,
                                               options, observations, pockets, bank_side)
                if candidate is not None:
                    shots.append(candidate)
                else:
                    rejected[reason] = rejected.get(reason, 0) + 1
    # Deterministic tie-breaks keep the spoken recommendation stable frame to frame.
    shots.sort(key=lambda shot: (-shot["score"], shot["shot_type"] != "direct", shot["ball_id"], shot["id"]))
    if shots:
        best = shots[0]
        message = f"Best modeled {best['shot_type']} shot: {best['ball_name']} to {best['pocket_name']}."
        if best["scratch_risk"]["level"] == "high":
            message += " High scratch risk under the stun model; a different stroke or defensive play is needed."
    elif not eligible:
        message = "No observed ball matches the requested target filter; check the current layout or request another target."
    else:
        message = (
            "No clear pot meets the modeled clearance, pocket-access and cut-angle limits. "
            "Recheck occluded balls and calibration; combinations, kicks and defensive shots are outside this search."
        )
    return {
        "status": "ok" if shots else "no_shot", "group": group,
        "message": message, "shots": shots, "target_ball_ids": sorted(eligible),
        # Kept for clients that previously consumed the same target list under
        # this name. Eligibility now means filter membership, not game legality.
        "eligible_ball_ids": sorted(eligible),
        "target_filter": {"group": group, "ball_id": requested},
        "rules_evaluated": False,
        "pockets": pockets, "planner": options,
        "diagnostics": {"candidates_considered": considered, "candidates_accepted": len(shots), "rejected": rejected},
        "score_description": "Heuristic geometry and risk ranking; not a probability of making the shot or a guaranteed optimum.",
        "angle_convention": "Degrees clockwise from right in the overhead table view.",
        "assumptions": [
            "The supplied layout includes all current physical obstacles; omitted or occluded balls cannot be checked.",
            "Targets are filtered for this request only; player assignments, pocketing history and shot legality are not inferred.",
            "Ideal level table, equal ball radii, and idealized pocket openings.",
            "Direct pots and one-cushion object-ball banks; no combinations, kicks, jumps or defensive search.",
            "Cue continuation assumes an ideal stun impact and estimated cloth drag; follow, draw and throw are not modeled.",
            "Scratch screening stops at an additional ball collision, a pocket jaw or after two cushion rebounds.",
            "Measured position uncertainty is a clearance bound, not a calibrated probability distribution.",
            f"Cuts exceeding {options['max_cut_angle_deg']:g} degrees are outside this planner's limit.",
            "Rail diamonds are aiming references counted from left/top, not bank instructions.",
        ],
    }
