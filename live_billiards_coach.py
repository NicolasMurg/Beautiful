#!/usr/bin/env python3
"""live_billiards.py plus a gentle local AI coach speaking through the JBL.

Takes every live_billiards.py argument unchanged, plus the coach options below.
The existing billiards files are not modified: this launcher substitutes a
coaching session, speaker and overlay inside its own process, then runs
live_billiards.main(). The planner still decides every shot; the model only
words it and maps casual questions onto supported requests. See COACH.md.
"""

import argparse
import sys
import textwrap
import time

import live_billiards
from billiards_session import PoolSession
from coach import (HELP_LINE, HOLE_POINTS, HOLES, CoachVoiceWorker, Voice, add_coach_arguments,
                   canonical_question, coached_line, explain_facts, explained_line, load_llm,
                   quick_intent, shot_facts)


class CoachSession(PoolSession):
    """PoolSession that also understands casual questions and follow-ups."""

    llm = None
    active = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pending_explanation = False
        CoachSession.active = self

    def _follow_up(self, intent):
        if intent not in ("why", "next") or not (self.confirmed and self.plan and self.plan.get("shots")):
            return None
        if intent == "next":
            self.next_shot()
        else:
            self.pending_explanation = True
        return self.plan

    def ask(self, question="What is my best shot?", now=None):
        # Freshness is judged when the question arrives, not after model latency.
        now = time.monotonic() if now is None else now
        try:
            return super().ask(question, now)
        except ValueError as exc:
            if not str(exc).startswith(("Unsupported request", "Ask for a shot")):
                raise
            unsupported = exc
        intent, ball, group = quick_intent(question), None, None
        if intent is None and self.llm is not None:
            parsed = self.llm.interpret(question)
            intent, ball, group = parsed["intent"], parsed["ball"], parsed["group"]
        if intent is None:
            raise unsupported
        plan = self._follow_up(intent)
        if plan is not None:
            print(f"Coach understood {question!r} as: {intent}", flush=True)
            return plan
        canonical = canonical_question(intent, ball, group) or (
            "what is my best shot" if intent in ("why", "next") else None)
        if canonical is None:
            raise ValueError(HELP_LINE)
        print(f"Coach understood {question!r} as: {canonical!r}", flush=True)
        return super().ask(canonical, now)


class CoachSpeaker:
    """Drop-in for live_billiards.LocalSpeaker that coaches the current shot."""

    worker = None

    def __init__(self, enabled=False):
        self.enabled = enabled

    def stop(self):
        self.worker.stop()

    def say(self, text):
        session = CoachSession.active
        shot = session.shot if session is not None else None
        if shot is not None and session.pending_explanation:
            session.pending_explanation = False
            facts = explain_facts(shot, len(session.plan["shots"]))
            self.worker.submit(lambda llm: explained_line(llm, facts))
        elif shot is not None and text == shot["instruction"]:
            facts = shot_facts(shot, session.calibration.table)
            self.worker.submit(lambda llm: coached_line(llm, facts))
        else:
            self.worker.submit(lambda llm: text)


def make_render(original, worker):
    def render(frame, session, *args, **kwargs):
        import cv2
        canvas = original(frame, session, *args, **kwargs)
        h, w = frame.shape[:2]
        if session is not None:
            table = session.calibration.table
            cu, cv = session.calibration.project(table["width"] / 2, table["height"] / 2, height_cm=0)
            for pocket_id, (fx, fy) in HOLE_POINTS.items():
                u, v = session.calibration.project(table["width"] * fx, table["height"] * fy, height_cm=0)
                # "H" labels sit just outside the pocket so they never hide diamond numbers.
                norm = max(1.0, ((u - cu) ** 2 + (v - cv) ** 2) ** 0.5)
                x = int(round(u + 22 * (u - cu) / norm)) - 12
                y = int(round(v + 22 * (v - cv) / norm)) + 5
                if 0 <= x < w - 24 and 12 <= y < h:
                    cv2.putText(canvas, f"H{HOLES[pocket_id]}", (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (40, 40, 40), 3, cv2.LINE_AA)
                    cv2.putText(canvas, f"H{HOLES[pocket_id]}", (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (80, 200, 255), 1, cv2.LINE_AA)
        line = worker.last_line
        if line:
            lines = textwrap.wrap("Coach: " + line, width=58)[:6]
            top = canvas.shape[0] - 24 * len(lines) - 16
            cv2.rectangle(canvas, (w + 10, top - 20), (canvas.shape[1] - 10, canvas.shape[0] - 8),
                          (45, 60, 50), -1)
            for i, value in enumerate(lines):
                cv2.putText(canvas, value, (w + 18, top + i * 24), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (150, 245, 190), 1, cv2.LINE_AA)
        return canvas
    return render


def main():
    parser = argparse.ArgumentParser(add_help=False, description="Coach options (added to live_billiards.py's)")
    add_coach_arguments(parser)
    args, rest = parser.parse_known_args()
    if "-h" in rest or "--help" in rest:
        parser.print_help()
        print()
        live_billiards.main()  # Prints the camera options and exits.
        return
    worker = CoachVoiceWorker(Voice(args.tts, args.piper_bin, args.piper_voice), load_llm(args))
    CoachSession.llm = worker.llm
    CoachSpeaker.worker = worker
    live_billiards.PoolSession = CoachSession
    live_billiards.LocalSpeaker = CoachSpeaker
    live_billiards.render = make_render(live_billiards.render, worker)
    sys.argv = [sys.argv[0]] + rest + ([] if "--speak" in rest else ["--speak"])
    live_billiards.main()


if __name__ == "__main__":
    main()
