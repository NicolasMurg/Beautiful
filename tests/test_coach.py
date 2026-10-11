"""Coach checks use fake language models and voices; nothing is downloaded."""

import threading
import time
import unittest

import numpy as np

from billiards import parse_question
from coach import (CoachVoiceWorker, canonical_question, check_reply, coached_line,
                   explain_facts, explained_line, fullness, parse_intent, quick_intent,
                   shot_facts, template_line)
from live_billiards_coach import CoachSession
from test_billiards_session import FakeCalibration, FakeDetector, camera_frame, settle

TABLE = {"width": 254.0, "height": 127.0, "ball_radius": 2.8575, "pocket_width": 11.5}


def shot(cue, ghost, pocket_id="TR", pocket_name="top right", ball=6, cut=31.0):
    return {"cue_path": [{"x": cue[0], "y": cue[1]}, {"x": ghost[0], "y": ghost[1]}],
            "ghost": {"x": ghost[0], "y": ghost[1]}, "rail": {"side": "right", "diamond": 2.1},
            "pocket_id": pocket_id, "pocket_name": pocket_name, "ball_id": ball,
            "cut_angle_deg": cut, "instruction": "planner text",
            "reasons": ["Cut angle 31.0 degrees.", "Minimum extra clearance from other balls: 7.6 cm."],
            "distances": {"cue": 80.4, "object": 60.2}}


class FakeLLM:
    def __init__(self, reply="", intent=None):
        self.reply, self.intent, self.calls = reply, intent, []

    def phrase(self, facts):
        self.calls.append("phrase")
        allowed = {"6", "3", "2"}
        return self.reply if check_reply(self.reply, allowed, allowed) else None

    def explain(self, facts):
        self.calls.append("explain")
        return None

    def interpret(self, question):
        self.calls.append(("interpret", question))
        return self.intent or {"intent": "unknown", "ball": None, "group": None}


class FactsTests(unittest.TestCase):
    def test_fullness_words(self):
        self.assertEqual(fullness(2), "straight-in shot")
        self.assertEqual(fullness(15), "three-quarter ball hit")
        self.assertEqual(fullness(30), "half-ball hit")
        self.assertEqual(fullness(50), "quarter-ball hit")
        self.assertEqual(fullness(70), "thin cut")

    def test_stance_diamond_extends_aim_line_behind_cue_horizontally(self):
        facts = shot_facts(shot((50, 63.5), (100, 63.5)), TABLE)
        self.assertEqual((facts["stance_rail"], facts["stance_diamond"]), ("left", 2.0))
        self.assertEqual((facts["aim_rail"], facts["aim_diamond"]), ("right", 2.0))
        self.assertEqual(facts["hole"], 3)
        self.assertEqual(facts["pocket"], "top right corner pocket")
        self.assertEqual(facts["hit"], "half-ball hit")

    def test_stance_diamond_vertical_and_side_pocket_numbering(self):
        facts = shot_facts(shot((127, 100), (127, 50), "TM", "top middle"), TABLE)
        self.assertEqual((facts["stance_rail"], facts["stance_diamond"]), ("bottom", 4.0))
        self.assertEqual((facts["hole"], facts["pocket"]), (2, "top middle side pocket"))

    def test_template_names_ball_hole_and_both_diamonds(self):
        line = template_line(shot_facts(shot((50, 63.5), (100, 63.5)), TABLE))
        for expected in ("diamond 2 on the left rail", "diamond 2 on the right rail",
                         "the 6 into hole 3"):
            self.assertIn(expected, line)


class ReplyCheckTests(unittest.TestCase):
    def test_reply_must_use_exactly_the_planner_numbers(self):
        allowed = {"6", "3", "2", "5.5"}
        self.assertTrue(check_reply("From diamond 2 toward 5.5, sink the 6 in hole 3.", allowed, allowed))
        self.assertFalse(check_reply("From diamond 2 toward 5.5, sink the 6 in hole 4.", allowed, allowed))
        self.assertFalse(check_reply("Sink the 6 in hole 3.", allowed, allowed))
        self.assertFalse(check_reply("", allowed, allowed))

    def test_bad_model_reply_falls_back_to_template(self):
        facts = shot_facts(shot((50, 63.5), (100, 63.5)), TABLE)
        self.assertEqual(coached_line(FakeLLM("Hit the 9 hard!"), facts), template_line(facts))
        good = "Gently line up from diamond 2 toward diamond 2 and roll the 6 into hole 3."
        self.assertEqual(coached_line(FakeLLM(good), facts), good)
        self.assertEqual(coached_line(None, facts), template_line(facts))

    def test_explanation_template_uses_planner_reasons(self):
        line = explained_line(None, explain_facts(shot((50, 63.5), (100, 63.5)), 3))
        self.assertIn("the 6", line)
        self.assertIn("31 degrees", line)
        self.assertIn("80 centimeters", line)


class IntentTests(unittest.TestCase):
    def test_malformed_model_output_is_unknown(self):
        for reply in ("", "sure!", "{not json}", '{"intent": "dance"}', None):
            self.assertEqual(parse_intent(reply)["intent"], "unknown")

    def test_valid_output_and_bounds(self):
        self.assertEqual(parse_intent('Here: {"intent": "ball", "ball": 6, "group": null}'),
                         {"intent": "ball", "ball": 6, "group": None})
        self.assertIsNone(parse_intent('{"intent": "ball", "ball": 22, "group": "x"}')["ball"])

    def test_canonical_phrases_pass_existing_parser(self):
        cases = [("best", None, None), ("ball", 6, None), ("ball", 8, None),
                 ("group", None, "stripes"), ("group", None, "solids"), ("best", None, "open")]
        for intent, ball, group in cases:
            phrase = canonical_question(intent, ball, group)
            parsed = parse_question(phrase, "solids")
            if ball:
                self.assertEqual(parsed["ball_id"], ball)
            if group:
                self.assertEqual(parsed["group"], group)
        self.assertIsNone(canonical_question("unknown"))

    def test_quick_follow_ups(self):
        self.assertEqual(quick_intent("why that one?"), "why")
        self.assertEqual(quick_intent("give me another option"), "next")
        self.assertIsNone(quick_intent("which ball should I sink"))


def coach_session(llm):
    CoachSession.llm = llm
    detector = FakeDetector()
    session = CoachSession(FakeCalibration(), detector)
    frame = camera_frame(detector.candidates)
    now = settle(session, frame)
    session.scan(frame, np.ones(frame.shape[:2]), now=now)
    session.assign("1", 0)
    session.assign("2", 1)
    session.confirm()
    return session, now


class CoachSessionTests(unittest.TestCase):
    def tearDown(self):
        CoachSession.llm = None

    def test_supported_question_never_calls_model(self):
        llm = FakeLLM()
        session, now = coach_session(llm)
        self.assertEqual(session.ask("What is my best shot?", now=now)["status"], "ok")
        self.assertEqual(llm.calls, [])

    def test_casual_question_is_interpreted_then_validated(self):
        llm = FakeLLM(intent={"intent": "best", "ball": None, "group": None})
        session, now = coach_session(llm)
        result = session.ask("yo which ball should I sink", now=now)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(llm.calls, [("interpret", "yo which ball should I sink")])

    def test_interpreted_ball_still_checked_against_layout(self):
        llm = FakeLLM(intent={"intent": "ball", "ball": 5, "group": None})
        session, now = coach_session(llm)
        with self.assertRaisesRegex(ValueError, "not on the confirmed table"):
            session.ask("can I make the orange one", now=now)

    def test_why_follow_up_needs_no_model(self):
        llm = FakeLLM()
        session, now = coach_session(llm)
        session.ask(now=now)
        session.ask("why that one?", now=now)
        self.assertTrue(session.pending_explanation)
        self.assertEqual(llm.calls, [])

    def test_unknown_request_raises_help(self):
        session, now = coach_session(FakeLLM())
        with self.assertRaisesRegex(ValueError, "best shot"):
            session.ask("tell me a joke about cats", now=now)

    def test_stale_camera_error_is_not_reinterpreted(self):
        llm = FakeLLM()
        session, now = coach_session(llm)
        with self.assertRaisesRegex(ValueError, "stale"):
            session.ask("yo which ball", now=now + 60)
        self.assertEqual(llm.calls, [])


class FakeVoice:
    def __init__(self):
        self.spoken, self.stops = [], 0

    def say(self, text):
        self.spoken.append(text)

    def stop(self):
        self.stops += 1


class WorkerTests(unittest.TestCase):
    def wait_for(self, condition):
        deadline = time.monotonic() + 2
        while not condition() and time.monotonic() < deadline:
            time.sleep(0.01)

    def test_line_is_spoken(self):
        voice = FakeVoice()
        worker = CoachVoiceWorker(voice)
        worker.submit(lambda llm: "hello")
        self.wait_for(lambda: voice.spoken)
        self.assertEqual(voice.spoken, ["hello"])
        self.assertEqual(worker.last_line, "hello")

    def test_line_computed_after_stop_is_dropped(self):
        voice, release = FakeVoice(), threading.Event()
        worker = CoachVoiceWorker(voice)
        started = threading.Event()

        def slow(llm):
            started.set()
            release.wait(2)
            return "stale advice"

        worker.submit(slow)
        started.wait(2)
        worker.stop()  # The layout changed while the model was wording the line.
        release.set()
        worker.submit(lambda llm: "fresh advice")
        self.wait_for(lambda: voice.spoken)
        self.assertEqual(voice.spoken, ["fresh advice"])


if __name__ == "__main__":
    unittest.main()
