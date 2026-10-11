#!/usr/bin/env python3
"""Physical pool-shot adviser using the project's calibrated stereo cameras.

Run --help or read BILLIARDS.md. The virtual table is reconstructed from measured
camera observations. Ball identities and the complete stationary layout are reviewed before
any guidance. Shot scores rank modeled direct and bank shots, not make odds.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import textwrap
import tempfile
import threading
import time

import numpy as np

from billiards import BALLS, GROUPS, normalize_pockets
from billiards_session import PoolSession
from billiards_table import TableCalibration
from billiards_vision import BallDetector
from stereo_camera import StereoCamera
from stereo_depth import StereoDepthEstimator


WINDOW = "Physical billiards | calibrated stereo"
ROOT = Path(__file__).resolve().parent
COLORS = {0: (235, 235, 235), 1: (25, 210, 240), 2: (225, 110, 40),
          3: (50, 50, 225), 4: (170, 65, 150), 5: (35, 145, 245),
          6: (60, 170, 50), 7: (50, 70, 140), 8: (75, 75, 75)}


def calibration_signature(path, estimator, baseline_m):
    return {"stereo_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "baseline_m": float(baseline_m), "image_size": list(estimator.image_size),
            "camera_matrix": estimator.rectified_camera_matrix.tolist()}


def read_json_option(path, expected_type, description):
    if path is None:
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read {description}: {exc}") from exc
    if not isinstance(value, expected_type):
        raise ValueError(f"{description} must contain a JSON {expected_type.__name__}.")
    return value


def export_reviewed_state(session, path):
    """Write an explicitly reviewed observation for offline replay, atomically."""
    if not session.confirmed:
        raise ValueError("Confirm the complete observed layout before exporting it.")
    state = session.state()
    state["observation"] = {"source": "physical_calibrated_stereo", "reviewed": True,
                            "recorded_unix_s": time.time()}
    state["virtual_map"] = session.virtual_map()
    content = json.dumps(state, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temp_name = stream.name
            stream.write(content)
        os.replace(temp_name, path)
    finally:
        if temp_name is not None and os.path.exists(temp_name):
            os.unlink(temp_name)


class LocalSpeaker:
    def __init__(self, enabled=False):
        self.binary = (shutil.which("espeak-ng") or shutil.which("espeak")) if enabled else None
        self.process = None
        if enabled and self.binary is None:
            print("Spoken replies unavailable: install espeak-ng; on-screen guidance still works.", flush=True)

    def stop(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=0.5)
            self.process = None

    def say(self, text):
        self.stop()
        if self.binary and text:
            self.process = subprocess.Popen([self.binary, "-s", "155", "--", text[:1600]],
                                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                            stderr=subprocess.DEVNULL)


def terminal_commands():
    commands = queue.Queue(maxsize=16)

    def read():
        while True:
            try:
                line = sys.stdin.readline()
            except (OSError, ValueError):
                return
            if not line:
                return
            try:
                commands.put_nowait(line.strip())
            except queue.Full:
                pass

    if sys.stdin is not None:
        threading.Thread(target=read, daemon=True).start()
    return commands


def render(frame, session, selected=None, entry="", entry_mode=None,
           notice="", calibration_points=(), calibrating=False, voice_status="",
           map_view=False):
    """Draw camera observations or their measured overhead map, plus guidance."""
    import cv2
    h, w = frame.shape[:2]
    panel_width = 530
    canvas = np.full((max(h + 120, 760), w + panel_width, 3), (24, 27, 30), np.uint8)
    canvas[:h, :w] = frame
    view = canvas[:h, :w]

    def text(value, x, y, color=(225, 230, 235), scale=0.48):
        cv2.putText(canvas, str(value), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, 1, cv2.LINE_AA)

    def point(x, y, height=0):
        u, v = session.calibration.project(x, y, height_cm=height)
        return int(round(u)), int(round(v))

    if session is not None and not (map_view and not calibrating):
        table = session.calibration.table
        pockets = {p["id"]: p for p in normalize_pockets(table)}
        for pocket_id, blocked in session.pocket_overrides.items():
            pockets[pocket_id]["blocked"] = blocked
        poly = np.rint(session.calibration.polygon).astype(np.int32)
        cv2.polylines(view, [poly], True, (130, 200, 130), 1, cv2.LINE_AA)
        for pocket in pockets.values():
            p = point(pocket["x"], pocket["y"])
            color = (70, 80, 250) if pocket["blocked"] else (170, 210, 245)
            cv2.circle(view, p, 6, color, 2)
            text(pocket["id"] + (" X" if pocket["blocked"] else ""), p[0]+8, p[1]+12, color, 0.4)
        for part in session.parts:
            if part["kind"] == "table":
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in part["box"]]
            cv2.rectangle(view, (x1, y1), (x2, y2), (180, 150, 120), 1)
            text(f"YOLO {part['kind']}?", x1, y1-4, (180, 150, 120), 0.35)
        # The reference numbers refer to the table coordinate system established
        # by TL/TR/BR/BL calibration, not the user's changing viewing direction.
        for i in range(1, 8):
            for y in (0, table["height"]):
                p = point(table["width"] * i / 8, y)
                cv2.circle(view, p, 2, (220, 220, 220), -1)
                text(i, p[0] + 3, p[1] - 3, scale=0.35)
        for i in range(1, 4):
            for x in (0, table["width"]):
                p = point(x, table["height"] * i / 4)
                cv2.circle(view, p, 2, (220, 220, 220), -1)
                text(i, p[0] + 3, p[1] - 3, scale=0.35)
        for candidate in session.candidates:
            key = candidate["key"]
            if key in session.ignored:
                continue
            pixel = candidate["pixel"]
            center = (round(pixel["x"]), round(pixel["y"]))
            radius = max(4, round(candidate["radius_px"]))
            number = session.labels.get(key)
            color = (70, 80, 250) if "error" in candidate else (
                (80, 230, 120) if number is not None else (30, 210, 250))
            cv2.circle(view, center, radius + 2, color, 2)
            if key == selected:
                cv2.circle(view, center, radius + 6, (255, 255, 255), 1)
            label = f"#{key}: " + (str(number) if number is not None else "?")
            text(label, center[0] + radius + 3, center[1] - 3, color, 0.4)
        shot = session.shot
        if shot:
            r = table["ball_radius"]
            for path, color in ((shot["cue_path"], (255, 220, 80)),
                                (shot["object_path"], (70, 245, 150))):
                projected = [point(p["x"], p["y"], r) for p in path]
                for a, b in zip(projected, projected[1:]):
                    cv2.arrowedLine(view, a, b, color, 2, cv2.LINE_AA, tipLength=0.08)
            ghost = shot["ghost"]
            gp = point(ghost["x"], ghost["y"], r)
            # Project a sampled circle in the ball-center plane to preserve the
            # perspective of the physical table, rather than a fixed UI radius.
            circle = [point(ghost["x"] + r * np.cos(a), ghost["y"] + r * np.sin(a), r)
                      for a in np.linspace(0, 2*np.pi, 40)]
            cv2.polylines(view, [np.asarray(circle, np.int32)], True, (255, 220, 80), 1)
            rail = shot["rail"]
            rp = point(rail["x"], rail["y"], r)
            for t in np.arange(0, 1, 0.1):
                a = tuple(round(gp[j] + (rp[j]-gp[j])*t) for j in (0, 1))
                b = tuple(round(gp[j] + (rp[j]-gp[j])*min(1, t+.045)) for j in (0, 1))
                cv2.line(view, a, b, (170, 160, 130), 1, cv2.LINE_AA)
            cv2.drawMarker(view, rp, (255, 220, 80), cv2.MARKER_DIAMOND, 12, 2)
            continuation = shot.get("scratch_risk", {}).get("cue_path", [])
            for start, end in zip(continuation, continuation[1:]):
                a, b = point(start["x"], start["y"], r), point(end["x"], end["y"], r)
                for t in np.arange(0, 1, 0.1):
                    p = tuple(round(a[j] + (b[j]-a[j])*t) for j in (0, 1))
                    q = tuple(round(a[j] + (b[j]-a[j])*min(1, t+.05)) for j in (0, 1))
                    cv2.line(view, p, q, (220, 140, 230), 1, cv2.LINE_AA)
    for i, p in enumerate(calibration_points):
        cv2.drawMarker(view, (round(p[0]), round(p[1])), (0, 240, 255), cv2.MARKER_CROSS, 15, 2)
        text(("TL", "TR", "BR", "BL")[i], p[0]+8, p[1]-8, (0, 240, 255))

    if map_view and not calibrating:
        from billiards_map_view import draw_table_map
        canvas[:h, :w] = draw_table_map(w, h, session.virtual_map() if session else None,
                                       shot=session.shot if session else None, selected=selected)

    x, y = w + 18, 30

    def paragraph(value, color=(225, 230, 235), width=58):
        nonlocal y
        for line in textwrap.wrap(str(value), width=width) or [""]:
            text(line, x, y, color)
            y += 21
        y += 7

    text("PHYSICAL POOL COACH", x, y, (90, 230, 180), 0.62)
    y += 32
    if calibrating:
        paragraph("TABLE CALIBRATION", (30, 210, 250))
        paragraph("Place the four textured markers on the cloth at the configured inset. Click their centers: TL, TR, BR, BL.")
        paragraph("Table dimensions are between cushion faces. Markers define the top/right directions used in all guidance.")
        paragraph("Backspace undoes a point. T restarts calibration.")
    elif session:
        paragraph(f"Default target filter: {session.group.upper()} | "
                  + ("CONFIRMED" if session.confirmed else "REVIEW REQUIRED"))
        if session.plan:
            target_id = session.plan.get("target_filter", {}).get("ball_id")
            target_label = f"BALL {target_id}" if target_id is not None else session.plan["group"].upper()
            paragraph(f"This request: {target_label}")
        paragraph(session.message, (30, 210, 250))
        if selected is not None and session.snapshot is not None:
            try:
                candidate = session.candidate(selected)
                guess = candidate.get("suggested_id")
                suggestion = BALLS[guess]["name"] if guess in BALLS else "none"
                paragraph(f"Selected #{selected}. Identity suggestion: {suggestion}. Type actual number + ENTER.")
                if "position" in candidate:
                    pos = candidate["position"]
                    paragraph(f"Stereo Z {pos['depth_m']:.3f} m | table ({pos['x']:.1f}, {pos['y']:.1f}) cm"
                              + (f" | uncertainty {pos['position_uncertainty_cm']:.2f} cm"
                                 if "position_uncertainty_cm" in pos else ""))
                if "error" in candidate:
                    paragraph(candidate["error"], (70, 80, 250))
            except ValueError:
                pass
        if session.shot:
            shot = session.shot
            paragraph(f"Option {session.shot_index + 1}/{len(session.plan['shots'])}: {shot.get('shot_type', 'direct').upper()} {shot['ball_name']} -> {shot['pocket_name']}",
                      (90, 230, 180))
            paragraph(f"Cut angle: {shot['cut_angle_deg']:.1f} degrees. Aim heading: {shot['aim_angle_deg']:.1f} degrees clockwise from table-right.")
            if shot.get("bank"):
                paragraph(f"Bank off the {shot['bank']['rail']} cushion at the bend in the green path.")
            else:
                paragraph(shot["rail"]["description"])
            paragraph("Aim the cue-ball CENTER toward the outlined ghost ball. Cyan: cue path. Green: object-ball path. Purple dashes: conditional stun continuation.")
            paragraph(f"Ranking score: {shot['score']:.1f}/100 (not make odds).")
            if shot.get("scratch_risk"):
                risk = shot["scratch_risk"]
                paragraph(f"Scratch estimate: {risk.get('level', 'unknown')}. "
                          + risk.get("warning", "Ideal stun only; actual spin and speed change the path."),
                          (160, 170, 180))
            if shot.get("power"):
                paragraph(f"Suggested pace: {shot['power']['level']}; tune on your cloth.")
    if notice:
        paragraph(notice, (90, 175, 255))
    if voice_status:
        paragraph(voice_status, (170, 210, 245))
    if entry_mode:
        paragraph(("Question: " if entry_mode == "question" else "Ball number: ") + entry + "_", (255, 255, 255))

    controls = ["SPACE scan | Click ball + number + ENTER | SHIFT-click add missed ball",
                "Right-click ignore | DELETE clear label | L copy model labels | C confirm layout",
                "A shot | ? question | P voice | G default filter | N alternative | M camera/map | T calibrate | Q quit",
                "C confirms the complete current layout. Ask 'best shot for solids' or 'best shot for stripes'."]
    for i, line in enumerate(controls):
        text(line, 10, h + 23 + i*22, (195, 205, 215), 0.39)
    return canvas


def build_parser():
    from speech import add_speech_arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-camera", required=True)
    parser.add_argument("--right-camera", required=True)
    parser.add_argument("--stereo-calibration", type=Path, required=True)
    parser.add_argument("--baseline-inches", type=float, default=5.0)
    parser.add_argument("--stereo-disparities", type=int, default=192)
    parser.add_argument("--table-profile", type=Path, default=ROOT / ".local/billiards/table.json")
    parser.add_argument("--table-length-cm", type=float, help="Measured long playing dimension, between cushions")
    parser.add_argument("--table-width-cm", type=float, help="Measured short playing dimension, between cushions")
    parser.add_argument("--pocket-width-cm", type=float, help="Smallest measured pocket opening; idealized planner")
    parser.add_argument("--ball-diameter-cm", type=float, default=5.715)
    parser.add_argument("--marker-inset-cm", type=float, default=10.0)
    parser.add_argument("--group", choices=GROUPS, default="open",
                        help="Default target filter; each shot request can specify solids or stripes")
    parser.add_argument("--model", type=Path, help="Optional YOLO weights for ball proposals; OpenCV also runs")
    parser.add_argument("--class-map", type=Path, help="JSON object mapping YOLO class names/indices to ball IDs 0..15")
    parser.add_argument("--yolo-confidence", type=float, default=0.20)
    parser.add_argument("--pockets-config", type=Path, help="JSON list of measured pocket overrides (id, x, y, width, blocked)")
    parser.add_argument("--direct-only", action="store_true", help="Disable one-cushion bank candidates")
    parser.add_argument("--planner-config", type=Path, help="JSON object with measured cloth/cushion parameters and planner limits")
    parser.add_argument("--position-margin-cm", type=float,
                        help="Extra position uncertainty beyond measured localization uncertainty")
    parser.add_argument("--export-state", type=Path,
                        help="Save reviewed layouts as JSON after C, e.g. .local/billiards/latest.json")
    parser.add_argument("--map-view", action="store_true",
                        help="Start with the measured overhead map; M toggles camera/map view")
    parser.add_argument("--device", default="0", help="YOLO device, e.g. 0 or cpu")
    parser.add_argument("--min-ball-radius-px", type=int, default=3)
    parser.add_argument("--max-ball-radius-px", type=int, default=40)
    parser.add_argument("--voice", action="store_true", help="P records with existing local Whisper/automatic mic")
    parser.add_argument("--speak", action="store_true", help="Speak guidance with local espeak-ng/espeak if installed")
    add_speech_arguments(parser)
    return parser


def main():
    import cv2
    parser = build_parser()
    args = parser.parse_args()
    if not args.stereo_calibration.is_file():
        parser.error("A calibrated stereo NPZ file is required; see STEREO_DEPTH.md.")
    if args.model is not None and not args.model.is_file():
        parser.error("--model must be an existing local weights file.")
    dimensions = (args.table_length_cm, args.table_width_cm, args.pocket_width_cm)
    if any(v is not None for v in dimensions) and not all(v is not None for v in dimensions):
        parser.error("Supply --table-length-cm, --table-width-cm and --pocket-width-cm together.")
    if not args.table_profile.is_file() and not all(v is not None for v in dimensions):
        parser.error("First setup requires measured --table-length-cm, --table-width-cm and --pocket-width-cm.")
    if not all(np.isfinite(v) and v > 0 for v in (args.baseline_inches, args.ball_diameter_cm)):
        parser.error("Baseline and ball diameter must be finite and positive.")
    if not np.isfinite(args.marker_inset_cm) or args.marker_inset_cm <= 0:
        parser.error("Marker inset must be finite and positive.")
    if args.position_margin_cm is not None and (not np.isfinite(args.position_margin_cm) or args.position_margin_cm < 0):
        parser.error("Position margin must be finite and nonnegative.")
    if args.class_map is not None and args.model is None:
        parser.error("--class-map requires --model.")
    if all(v is not None for v in dimensions):
        if not all(np.isfinite(v) and v > 0 for v in dimensions):
            parser.error("Table and pocket dimensions must be finite and positive.")
        if 2 * args.marker_inset_cm >= min(dimensions[:2]):
            parser.error("Marker inset must be less than half of either table dimension.")
    speaker, worker = LocalSpeaker(args.speak), None
    try:
        estimator = StereoDepthEstimator(args.stereo_calibration, args.baseline_inches * 0.0254,
                                         args.stereo_disparities)
        signature = calibration_signature(args.stereo_calibration, estimator, args.baseline_inches * 0.0254)
        yolo = None
        class_map = read_json_option(args.class_map, dict, "class map")
        pocket_config = read_json_option(args.pockets_config, list, "pocket configuration")
        planner_options = read_json_option(args.planner_config, dict, "planner configuration") or {}
        if args.direct_only:
            planner_options["allow_banks"] = False
        if args.position_margin_cm is not None:
            planner_options["position_uncertainty_cm"] = args.position_margin_cm
        if args.model:
            from ultralytics import YOLO
            yolo = YOLO(str(args.model))
        detector = BallDetector(args.min_ball_radius_px, args.max_ball_radius_px,
                                yolo_model=yolo, yolo_device=args.device,
                                yolo_conf=args.yolo_confidence, class_to_ball_id=class_map)
        calibration, session = None, None
        notice = ""
        # Explicit dimensions start a new physical table calibration. A saved
        # profile is used only when the user omits these measurement arguments.
        if args.table_profile.is_file() and not all(v is not None for v in dimensions):
            calibration = TableCalibration.load(args.table_profile, signature=signature)
            if pocket_config is not None:
                calibration.set_pockets(pocket_config)
                calibration.save(args.table_profile)
            session = PoolSession(calibration, detector, args.group, planner_options=planner_options)
            args.table_length_cm = calibration.width_cm
            args.table_width_cm = calibration.height_cm
            args.pocket_width_cm = calibration.pocket_width_cm
            args.ball_diameter_cm = calibration.ball_radius_cm * 2
            args.marker_inset_cm = calibration.inset_cm
            pocket_config = calibration.pockets or None
        if args.voice:
            from speech import worker_from_args
            try:
                worker = worker_from_args(args)
            except (OSError, ValueError, RuntimeError) as exc:
                notice = f"Voice unavailable: {exc}. Typed questions remain available."
                print(notice, flush=True)
        calibrating = calibration is None
        calibration_points = []
        events = queue.Queue()

        def on_mouse(event, x, y, flags, _):
            if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
                events.put((event, x, y, flags))

        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, on_mouse)
        commands = terminal_commands()
        print("Physical pool coach: keep the complete table visible and the rig fixed.\n"
              "See BILLIARDS.md for marker placement. SPACE scans; label every ball; C confirms.\n"
              "Measured table geometry defines six pocket positions; pocket recognition is not required.\n"
              "Type a question here or press ? in the camera window. A asks immediately.\n"
              "Ask 'best shot for solids' or 'best shot for stripes' for this layout.\n"
              "Confirmation reviews the complete current layout; absent balls have no assigned status.", flush=True)
        selected, entry, entry_mode = None, "", None
        map_view = args.map_view
        speech_revision = None
        quit_requested = False
        width, height = estimator.image_size
        spoken_revision = None
        with StereoCamera(args.left_camera, args.right_camera, width, height) as camera:
            while not quit_requested:
                left, right, capture = camera.read()
                accepted = capture["pair_accepted"]
                left_rect, _, depth = estimator.estimate(left, right)
                now = time.monotonic()
                if session:
                    session.observe(left_rect, capture["received_monotonic_s"], accepted, now, depth_map=depth)
                    if spoken_revision != (id(session), session.revision):
                        speaker.stop()
                    if session.snapshot is None:
                        selected = None
                if not accepted:
                    notice = "Stereo capture timing rejected; calibration and scanning are blocked."
                frame_age = now - capture["received_monotonic_s"]
                frame_fresh = accepted and np.isfinite(frame_age) and 0 <= frame_age <= 3.0

                def ask(question):
                    nonlocal notice, spoken_revision
                    if session is None or calibrating:
                        raise ValueError("Calibrate the physical table first.")
                    result = session.ask(question)
                    notice = f"Asked: {question}"
                    print(json.dumps({"question": question, "source": "physical_calibrated_stereo",
                                      "result": result}, allow_nan=False), flush=True)
                    speaker.say(result.get("response") or
                                (session.shot["instruction"] if session.shot else result["message"]))
                    spoken_revision = (id(session), session.revision)

                while not events.empty():
                    event, x, y, flags = events.get_nowait()
                    if x >= width or y >= height:
                        continue
                    try:
                        if calibrating:
                            if event != cv2.EVENT_LBUTTONDOWN:
                                continue
                            if not frame_fresh:
                                raise ValueError("Wait for fresh accepted stereo frames before clicking markers.")
                            calibration_points.append((x, y))
                            if len(calibration_points) == 4:
                                calibration = TableCalibration.fit(
                                    marker_pixels=calibration_points, depth_map=depth,
                                    camera_matrix=estimator.rectified_camera_matrix,
                                    image_size=estimator.image_size,
                                    width_cm=args.table_length_cm, height_cm=args.table_width_cm,
                                    ball_radius_cm=args.ball_diameter_cm/2,
                                    pocket_width_cm=args.pocket_width_cm, inset_cm=args.marker_inset_cm,
                                    signature=signature, pockets=pocket_config)
                                calibration.save(args.table_profile)
                                session = PoolSession(calibration, detector, args.group,
                                                      planner_options=planner_options)
                                calibrating = False
                                calibration_points = []
                                notice = "Table calibrated. Remove markers if desired; wait, then SPACE to scan."
                                print(notice, flush=True)
                        elif session is not None:
                            if map_view:
                                notice = "M returns to the camera view for labeling or marking actual balls."
                                continue
                            if flags & cv2.EVENT_FLAG_SHIFTKEY and event == cv2.EVENT_LBUTTONDOWN:
                                candidate = session.add_at_pixel(x, y)
                                selected = candidate["key"]
                                entry, entry_mode = "", "number"
                            else:
                                choices = [(np.hypot(c["pixel"]["x"]-x, c["pixel"]["y"]-y), c)
                                           for c in session.candidates if c["key"] not in session.ignored]
                                if not choices:
                                    raise ValueError("SPACE to scan, then select a ball candidate.")
                                distance, candidate = min(choices, key=lambda pair: pair[0])
                                if distance > max(16, candidate["radius_px"] * 1.8):
                                    raise ValueError("Click a candidate circle, or SHIFT-click a missed real ball.")
                                selected = candidate["key"]
                                if event == cv2.EVENT_RBUTTONDOWN:
                                    session.reject(selected)
                                    selected = None
                                    entry_mode, entry = None, ""
                                else:
                                    entry, entry_mode = "", "number"
                    except (ValueError, OSError) as exc:
                        notice = str(exc)
                        if calibrating and len(calibration_points) >= 4:
                            calibration_points = []

                if worker:
                    result = worker.poll()
                    if result is not None:
                        try:
                            if result.error:
                                raise ValueError(result.error)
                            if not session or speech_revision != (id(session), session.revision):
                                raise ValueError("Layout changed during speech. Review it and ask again.")
                            if time.monotonic() - result.recorded_at > 15:
                                raise ValueError("Speech result expired. Ask again.")
                            if not result.text:
                                raise ValueError("No speech recognized. Try again or type your question.")
                            ask(result.text)
                        except (ValueError, RuntimeError) as exc:
                            notice = str(exc)

                while not commands.empty():
                    command = commands.get_nowait()
                    try:
                        if command.lower() in ("q", "quit", "exit"):
                            quit_requested = True
                        elif command:
                            ask(command)
                    except (ValueError, RuntimeError) as exc:
                        notice = str(exc)
                        print(notice, flush=True)

                cv2.imshow(WINDOW, render(left_rect, session, selected, entry, entry_mode,
                                          notice, calibration_points, calibrating,
                                          worker.status if worker else "", map_view=map_view))
                raw_key = cv2.waitKeyEx(1)
                if raw_key < 0:
                    continue
                key = raw_key & 0xFF
                delete_key = raw_key in (3014656, 65535) or key == 127
                try:
                    if entry_mode:
                        if key == 27:
                            entry_mode, entry = None, ""
                        elif delete_key and entry_mode == "number" and session and selected:
                            session.clear_label(selected)
                            entry = ""
                            notice = "Label cleared. Type the actual ball number and ENTER."
                        elif key in (10, 13):
                            mode, value = entry_mode, entry
                            entry_mode, entry = None, ""
                            if mode == "number":
                                if not session or selected is None or not value.isdecimal():
                                    raise ValueError("Click a ball, then type 0 through 15 and ENTER.")
                                session.assign(selected, int(value))
                                notice = ""
                            else:
                                ask(value)
                        elif key in (8, 127):
                            entry = entry[:-1]
                        elif 32 <= key <= 126 and len(entry) < 200:
                            if entry_mode == "question" or chr(key).isdigit():
                                entry += chr(key)
                        continue
                    if key in (ord("q"), ord("Q"), 27):
                        quit_requested = True
                    elif key in (ord("t"), ord("T")):
                        if session:
                            session.invalidate("Recalibrating table.")
                        speaker.stop()
                        calibration_points = []
                        calibrating = True
                        notice = "Click inset marker centers TL, TR, BR, BL."
                    elif calibrating:
                        if key in (8, 127) and calibration_points:
                            calibration_points.pop()
                    elif session is not None:
                        if key in (ord("m"), ord("M")):
                            map_view = not map_view
                            notice = "Measured overhead map." if map_view else "Live camera view."
                        elif key == ord(" "):
                            session.scan(left_rect, depth)
                            selected, entry_mode = None, None
                            notice = "Click each real ball and type its number; 0 = cue ball."
                        elif key in (ord("c"), ord("C")):
                            session.confirm()
                            if args.export_state is not None:
                                export_reviewed_state(session, args.export_state)
                            notice = "You confirmed every on-table ball is represented in the current layout."
                        elif key in (ord("l"), ord("L")):
                            session.accept_model_labels()
                            notice = session.message
                        elif key in (ord("a"), ord("A")):
                            ask("What is my best shot?")
                        elif key == ord("?"):
                            entry, entry_mode = "", "question"
                        elif key in (ord("g"), ord("G")):
                            session.set_group(GROUPS[(GROUPS.index(session.group)+1) % len(GROUPS)])
                            notice = ""
                        elif key in (ord("n"), ord("N")):
                            shot = session.next_shot()
                            if shot:
                                speaker.say(shot["instruction"])
                                spoken_revision = (id(session), session.revision)
                        elif key in (ord("p"), ord("P")):
                            speaker.stop()
                            if worker is None:
                                raise ValueError("Start with --voice after tools/setup_whisper.sh; typed questions also work.")
                            if not session.confirmed:
                                raise ValueError("Confirm the complete stationary layout with C before recording.")
                            if worker.request():
                                speech_revision = (id(session), session.revision)
                                notice = "Wait for Recording, then ask your question."
                            else:
                                notice = "Speech is already busy."
                        elif key in (8, 127) or raw_key in (3014656, 65535):
                            if selected:
                                session.clear_label(selected)
                except (ValueError, OSError, RuntimeError) as exc:
                    notice = str(exc)
                    print(notice, flush=True)
    except KeyboardInterrupt:
        print("Stopped.", flush=True)
    except (OSError, ValueError, RuntimeError, cv2.error) as exc:
        parser.exit(1, f"Billiards error: {exc}\n")
    finally:
        if worker:
            worker.close()
        speaker.stop()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


if __name__ == "__main__":
    main()
