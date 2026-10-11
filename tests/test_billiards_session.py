"""Session checks use synthetic camera frames, never a camera or motor."""

import copy
import unittest

import numpy as np

from billiards_session import PoolSession, SceneGuard


class FakeCalibration:
    image_size = (256, 128)
    polygon = np.array([[0, 0], [255, 0], [255, 127], [0, 127]], np.float32)
    camera_matrix = np.array([[200, 0, 128], [0, 200, 64], [0, 0, 1]], np.float64)
    ball_radius_cm = 2.8575
    table = {"width": 254.0, "height": 127.0, "ball_radius": 2.8575,
             "pocket_width": 11.5}

    def localize_ball(self, pixel, radius_px, depth):
        # A controllable local depth response: +0.01 m moves x by +1 cm.
        # Unlike a whole-frame check, loss at one ball cannot hide behind valid
        # depth elsewhere on the table.
        z = float(depth[round(pixel["y"]), round(pixel["x"])])
        if not np.isfinite(z) or z <= 0:
            raise ValueError("No stereo depth for this candidate.")
        return {"x": float(pixel["x"]) + (z - 1.0) * 100.0,
                "y": float(pixel["y"])}


def candidate(x, y, suggested_id=None):
    return {"pixel": {"x": float(x), "y": float(y)}, "radius_px": 4.0,
            "suggested_id": suggested_id, "source": "synthetic_test"}


class FakeDetector:
    def __init__(self, candidates=None):
        self.candidates = candidates if candidates is not None else [
            candidate(127, 100, 0), candidate(127, 50, 1)]

    def detect(self, frame, polygon):
        return copy.deepcopy(self.candidates)


def camera_frame(candidates=None):
    import cv2
    frame = np.full((128, 256, 3), 50, dtype=np.uint8)
    for item in candidates or FakeDetector().candidates:
        point = item["pixel"]
        cv2.circle(frame, (round(point["x"]), round(point["y"])), 4, (230, 230, 230), -1)
    return frame


def settle(session, frame, start=100.0):
    for now in (start, start + 0.6, start + 1.2):
        session.observe(frame, now, now=now)
    return start + 1.2


def scanned_session(candidates=None):
    detector = FakeDetector(candidates)
    session = PoolSession(FakeCalibration(), detector)
    frame = camera_frame(detector.candidates)
    now = settle(session, frame)
    session.scan(frame, np.ones(frame.shape[:2]), now=now)
    return session, frame, now


def confirmed_session():
    session, frame, now = scanned_session()
    session.assign("1", 0)
    session.assign("2", 1)
    session.confirm()
    session.ask(now=now)
    return session, frame, now


class SceneGuardTests(unittest.TestCase):
    def test_uniform_exposure_change_does_not_look_like_object_motion(self):
        guard = SceneGuard(FakeCalibration.polygon, FakeCalibration.image_size)
        before = np.full((128, 256, 3), 50, dtype=np.uint8)
        after = np.full_like(before, 75)
        self.assertFalse(guard.changed(before, after))

    def test_local_motion_invalidates_even_with_exposure_change(self):
        guard = SceneGuard(FakeCalibration.polygon, FakeCalibration.image_size)
        before = np.full((128, 256, 3), 50, dtype=np.uint8)
        after = np.full_like(before, 75)
        after[30:45, 40:55] = 200
        self.assertTrue(guard.changed(before, after))

    def test_motion_outside_table_mask_is_ignored(self):
        polygon = np.array([[50, 30], [200, 30], [200, 100], [50, 100]], np.float32)
        guard = SceneGuard(polygon, (256, 128))
        before = np.zeros((128, 256, 3), np.uint8)
        after = before.copy()
        after[0:20, 0:20] = 255
        self.assertFalse(guard.changed(before, after))

    def test_frame_size_change_and_tiny_calibration_are_rejected(self):
        guard = SceneGuard(FakeCalibration.polygon, FakeCalibration.image_size)
        self.assertTrue(guard.changed(np.zeros((128, 256, 3), np.uint8),
                                      np.zeros((64, 128, 3), np.uint8)))
        with self.assertRaises(ValueError):
            SceneGuard(np.array([[0, 0], [3, 0], [3, 3], [0, 3]]), (256, 128))


class PoolSessionTests(unittest.TestCase):
    def test_scan_requires_time_and_multiple_fresh_stationary_frames(self):
        session = PoolSession(FakeCalibration(), FakeDetector())
        frame = camera_frame()
        session.observe(frame, 100, now=100)
        with self.assertRaisesRegex(ValueError, "still"):
            session.scan(frame, np.ones((128, 256)), now=100)
        session.observe(frame, 101.1, now=101.1)
        self.assertFalse(session.settled(now=101.1))
        session.observe(frame, 101.2, now=101.2)
        self.assertTrue(session.settled(now=101.2))

    def test_suggestions_never_replace_operator_labels_and_confirmation(self):
        session, _, now = scanned_session()
        self.assertEqual(session.labels, {})
        with self.assertRaisesRegex(ValueError, "Label"):
            session.confirm()
        session.assign("1", 0)
        session.assign("2", 1)
        with self.assertRaisesRegex(ValueError, "press C"):
            session.ask(now=now)
        session.confirm()
        self.assertEqual(session.ask(now=now)["status"], "ok")

    def test_duplicate_ball_identity_is_rejected_without_changing_labels(self):
        session, _, _ = scanned_session()
        session.assign("1", 0)
        with self.assertRaisesRegex(ValueError, "already"):
            session.assign("2", 0)
        self.assertEqual(session.labels, {"1": 0})

    def test_confirm_requires_the_cue_ball(self):
        session, _, _ = scanned_session()
        session.assign("1", 9)
        session.assign("2", 1)
        with self.assertRaisesRegex(ValueError, "cue ball"):
            session.confirm()
        self.assertFalse(session.confirmed)

    def test_ball_motion_clears_plan_labels_and_confirmation(self):
        session, frame, now = confirmed_session()
        revision = session.revision
        moved = frame.copy()
        moved[45:56, 122:133] = 50
        moved[45:56, 152:163] = 230
        session.observe(moved, now + 0.1, now=now + 0.1)
        self.assertIsNone(session.snapshot)
        self.assertIsNone(session.shot)
        self.assertEqual(session.labels, {})
        self.assertFalse(session.confirmed)
        self.assertFalse(session.settled(now=now + 0.1))
        self.assertGreater(session.revision, revision)

    def test_capture_rejection_invalidates_even_when_pixels_match(self):
        session, frame, now = confirmed_session()
        self.assertFalse(session.observe(frame, now + 0.1, accepted=False, now=now + 0.1))
        self.assertIsNone(session.plan)
        self.assertIsNone(session.last_received)
        self.assertFalse(session.confirmed)

    def test_invalid_depth_at_one_ball_clears_advice_despite_unchanged_image(self):
        session, frame, now = confirmed_session()
        revision = session.revision
        depth = np.ones(frame.shape[:2])
        depth[50, 127] = np.nan  # Object ball alone loses stereo correspondence.
        session.observe(frame.copy(), now + 0.1, now=now + 0.1, depth_map=depth)
        self.assertFalse(session.confirmed)
        self.assertIsNone(session.shot)
        self.assertIsNone(session.snapshot)
        self.assertEqual(session.labels, {})
        self.assertFalse(session.settled(now=now + 0.1))
        self.assertGreater(session.revision, revision)
        self.assertIn("Stereo ball measurement changed", session.message)

    def test_metric_shift_over_one_cm_clears_advice_despite_unchanged_image(self):
        session, frame, now = confirmed_session()
        revision = session.revision
        depth = np.ones(frame.shape[:2])
        depth[50, 127] = 1.02  # Fake calibration reports a 2 cm position shift.
        session.observe(frame.copy(), now + 0.1, now=now + 0.1, depth_map=depth)
        self.assertFalse(session.confirmed)
        self.assertIsNone(session.plan)
        self.assertIsNone(session.snapshot)
        self.assertFalse(session.settled(now=now + 0.1))
        self.assertGreater(session.revision, revision)
        self.assertIn("over 1 cm", session.message)

    def test_valid_stable_depth_preserves_confirmation_and_advice(self):
        session, frame, now = confirmed_session()
        revision = session.revision
        original_plan = session.plan
        session.observe(frame.copy(), now + 0.1, now=now + 0.1,
                        depth_map=np.ones(frame.shape[:2]))
        self.assertTrue(session.confirmed)
        self.assertIs(session.plan, original_plan)
        self.assertIsNotNone(session.shot)
        self.assertEqual(session.revision, revision)

    def test_small_metric_jitter_below_one_cm_preserves_advice(self):
        session, frame, now = confirmed_session()
        depth = np.ones(frame.shape[:2])
        depth[50, 127] = 1.005  # 0.5 cm, within the documented tolerance.
        session.observe(frame.copy(), now + 0.1, now=now + 0.1, depth_map=depth)
        self.assertTrue(session.confirmed)
        self.assertIsNotNone(session.shot)

    def test_invalid_stale_and_future_timestamps_are_rejected(self):
        for timestamp in (float("nan"), float("inf"), 95, 105):
            with self.subTest(timestamp=timestamp):
                session = PoolSession(FakeCalibration(), FakeDetector())
                self.assertFalse(session.observe(camera_frame(), timestamp, now=100))
                self.assertFalse(session.settled(now=100))

    def test_capture_gap_and_stale_question_both_clear_advice(self):
        session, frame, now = confirmed_session()
        session.observe(frame, now + 4, now=now + 4)
        self.assertFalse(session.confirmed)
        self.assertIsNone(session.shot)
        session, _, now = confirmed_session()
        with self.assertRaisesRegex(ValueError, "stale"):
            session.ask(now=now + 4)
        self.assertFalse(session.confirmed)
        self.assertIsNone(session.shot)

    def test_editing_id_or_rejecting_candidate_requires_confirmation_again(self):
        for action in (lambda s: s.assign("2", 2), lambda s: s.clear_label("2"),
                       lambda s: s.reject("2")):
            with self.subTest(action=action):
                session, _, now = confirmed_session()
                revision = session.revision
                action(session)
                self.assertIsNone(session.shot)
                self.assertFalse(session.confirmed)
                self.assertGreater(session.revision, revision)
                with self.assertRaisesRegex(ValueError, "press C"):
                    session.ask(now=now)

    def test_missing_ball_does_not_automatically_unlock_eight(self):
        original = [candidate(127, 100, 0), candidate(50, 50, 1), candidate(127, 50, 8)]
        session, frame, now = scanned_session(original)
        for key, number in (("1", 0), ("2", 1), ("3", 8)):
            session.assign(key, number)
        session.confirm()
        self.assertEqual(session.ask(now=now)["eligible_ball_ids"], [1])
        session.detector.candidates = [original[0], original[2]]
        missing = camera_frame(session.detector.candidates)
        later = settle(session, missing, start=now + 0.1)
        session.scan(missing, np.ones(missing.shape[:2]), now=later)
        session.assign("1", 0)
        session.assign("2", 8)
        with self.assertRaisesRegex(ValueError, "press C"):
            session.ask(now=later)
        # Review confirms the physical snapshot, never a game's progress.
        session.confirm()
        self.assertNotIn("pocketed", session.message.lower())
        self.assertEqual(session.ask("best solids", now=later)["eligible_ball_ids"], [])
        self.assertEqual(session.ask("best stripes", now=later)["eligible_ball_ids"], [])
        self.assertEqual(session.ask(now=later)["eligible_ball_ids"], [])
        self.assertEqual(session.ask("shoot the 8 ball", now=later)["eligible_ball_ids"], [8])

    def test_bad_depth_requires_fix_or_reject_before_confirming(self):
        session = PoolSession(FakeCalibration(), FakeDetector())
        frame = camera_frame()
        now = settle(session, frame)
        session.scan(frame, np.full(frame.shape[:2], np.nan), now=now)
        with self.assertRaisesRegex(ValueError, "stereo depth"):
            session.assign("1", 0)
        with self.assertRaisesRegex(ValueError, "stereo depth"):
            session.confirm()

    def test_manual_candidate_requires_valid_depth_and_cannot_duplicate_existing(self):
        session, _, _ = scanned_session()
        with self.assertRaisesRegex(ValueError, "already"):
            session.add_at_pixel(127, 100)
        with self.assertRaisesRegex(ValueError, "inside"):
            session.add_at_pixel(-1, 20)
        session.depth_snapshot[:] = np.nan
        with self.assertRaisesRegex(ValueError, "insufficient"):
            session.add_at_pixel(40, 40)

    def test_successful_ask_advances_revision_to_reject_older_voice_request(self):
        session, _, now = confirmed_session()
        voice_revision = session.revision
        session.ask("What is my best shot?", now=now)
        self.assertGreater(session.revision, voice_revision)

    def test_replayed_or_backward_capture_cannot_count_as_another_frame(self):
        for received in (100.0, 99.9):
            with self.subTest(received=received):
                session = PoolSession(FakeCalibration(), FakeDetector())
                frame = camera_frame()
                session.observe(frame, 100, now=100)
                self.assertFalse(session.observe(frame, received, now=100.6))
                self.assertFalse(session.settled(now=101.2))

    def test_scan_rejects_a_new_moving_frame_not_used_for_settling(self):
        session = PoolSession(FakeCalibration(), FakeDetector())
        frame = camera_frame()
        now = settle(session, frame)
        moved = frame.copy()
        moved[40:60, 80:100] = 230
        with self.assertRaises(ValueError):
            session.scan(moved, np.ones(moved.shape[:2]), now=now)
        self.assertFalse(session.confirmed)
        self.assertIsNone(session.snapshot)


class ExtendedRecommendationTests(unittest.TestCase):
    def test_stripes_then_solids_use_same_snapshot_without_remembering_a_player(self):
        session, _, now = scanned_session([
            candidate(127, 100, 0), candidate(100, 50, 1), candidate(160, 50, 9),
        ])
        for key, number in (("1", 0), ("2", 1), ("3", 9)):
            session.assign(key, number)
        session.confirm()
        original_map = session.virtual_map()
        original_snapshot = session.snapshot

        stripes = session.ask("best stripes", now=now)
        solids = session.ask("best solids", now=now)
        unfiltered = session.ask(now=now)

        self.assertEqual(stripes["eligible_ball_ids"], [9])
        self.assertEqual(solids["eligible_ball_ids"], [1])
        self.assertEqual(unfiltered["eligible_ball_ids"], [1, 9])
        self.assertEqual(session.group, "open")
        self.assertEqual(session.state()["group"], "open")
        # Both groups remain in the map as physical obstacles for every request.
        self.assertEqual(session.virtual_map(), original_map)
        self.assertIs(session.snapshot, original_snapshot)
        self.assertTrue(session.confirmed)
        self.assertNotIn("pocketed", session.message.lower())

    def test_explicit_default_target_filter_survives_a_different_spoken_request(self):
        session, _, now = scanned_session([
            candidate(127, 100, 0), candidate(100, 50, 1), candidate(160, 50, 9),
        ])
        for key, number in (("1", 0), ("2", 1), ("3", 9)):
            session.assign(key, number)
        session.confirm()
        session.set_group("stripes")
        self.assertIn("Target filter", session.message)
        self.assertEqual(session.ask("best solids", now=now)["eligible_ball_ids"], [1])
        self.assertEqual(session.group, "stripes")
        self.assertEqual(session.ask(now=now)["eligible_ball_ids"], [9])

    def test_unreviewed_map_keeps_measured_objects_without_inventing_ball_identities(self):
        session, _, _ = scanned_session()
        mapped = session.virtual_map()
        self.assertEqual(mapped["status"], "review")
        self.assertEqual([(b["x"], b["y"]) for b in mapped["balls"]], [(127., 100.), (127., 50.)])
        self.assertTrue(all(b["id"] is None for b in mapped["balls"]))
        self.assertNotIn("pocketed_ball_ids", mapped)
        self.assertFalse(mapped["complete"])

    def test_failed_measurement_remains_an_unresolved_map_object(self):
        session, _, _ = scanned_session()
        session.candidates[0].pop("position")
        session.candidates[0]["error"] = "No stereo depth"
        mapped = session.virtual_map()
        self.assertEqual(mapped["status"], "invalid")
        self.assertEqual(mapped["unmapped"], [{"key": "1", "reason": "No stereo depth"}])
        self.assertEqual(len(mapped["balls"]), 1)
        self.assertNotIn("pocketed_ball_ids", mapped)
        with self.assertRaisesRegex(ValueError, "No stereo depth"):
            session.confirm()

    def test_quality_and_planner_settings_reach_the_planner_without_mutating_calibration(self):
        session, _, now = confirmed_session()
        session.candidates[0]["confidence"] = 0.72
        session.candidates[0]["position"]["position_uncertainty_cm"] = 0.2
        session.planner_options = {"allow_banks": False, "position_uncertainty_cm": 0.1}
        session.set_pocket_blocked("TM", True)
        state = session.state()
        self.assertEqual(state["balls"][0]["confidence"], 0.72)
        self.assertEqual(state["balls"][0]["position_uncertainty_cm"], 0.2)
        self.assertEqual(state["planner"]["position_uncertainty_cm"], 0.1)
        self.assertNotIn("pockets", session.calibration.table)
        result = session.ask(now=now)
        self.assertTrue(next(p for p in result["pockets"] if p["id"] == "TM")["blocked"])
        self.assertTrue(all(s["pocket_id"] != "TM" for s in result["shots"]))

    def test_block_and_unblock_spoken_pocket_commands_replan(self):
        session, _, now = confirmed_session()
        blocked = session.ask("Block the top middle pocket.", now=now)
        self.assertTrue(all(s["pocket_id"] != "TM" for s in blocked["shots"]))
        reopened = session.ask("Unblock TM", now=now)
        self.assertTrue(any(s["pocket_id"] == "TM" for s in reopened["shots"]))
        self.assertFalse(session.pocket_overrides["TM"])

    def test_alternatives_explanations_and_direct_mode_share_the_current_layout(self):
        session, _, now = confirmed_session()
        revision = session.revision
        result = session.ask("Next shot", now=now)
        self.assertEqual(result["response"], session.shot["instruction"])
        self.assertGreater(session.revision, revision)
        explanation = session.ask("Why this shot?", now=now)
        self.assertIn(session.shot["reasons"][0], explanation["response"])
        result = session.ask("Direct shots only", now=now)
        self.assertTrue(all(s["shot_type"] == "direct" for s in result["shots"]))
        self.assertFalse(session.planner_options["allow_banks"])
        result = session.ask("Allow bank shots", now=now)
        self.assertTrue(session.planner_options["allow_banks"])

    def test_old_or_unreviewed_layout_cannot_be_reused_by_voice_controls(self):
        for command in ("Next shot", "Why this shot?", "Block TM", "Direct shots only"):
            with self.subTest(command=command):
                session, _, now = confirmed_session()
                with self.assertRaisesRegex(ValueError, "stale"):
                    session.ask(command, now=now + 4)
                self.assertIsNone(session.shot)
                session, _, now = scanned_session()
                with self.assertRaisesRegex(ValueError, "press C"):
                    session.ask(command, now=now)

    def test_copying_model_labels_requires_explicit_confirmation(self):
        session, _, now = scanned_session()
        for candidate, number in zip(session.candidates, (0, 1)):
            candidate["model_suggested_id"] = number
        self.assertEqual(session.accept_model_labels(), 2)
        self.assertFalse(session.confirmed)
        with self.assertRaisesRegex(ValueError, "press C"):
            session.ask(now=now)
        session.confirm()
        self.assertEqual(session.ask(now=now)["eligible_ball_ids"], [1])

    def test_copying_model_labels_skips_duplicates_conflicts_and_appearance_only(self):
        session, _, _ = scanned_session()
        self.assertEqual(session.accept_model_labels(), 0)  # Color suggestions are not model IDs.
        for item in session.candidates:
            item["model_suggested_id"] = 1
        self.assertEqual(session.accept_model_labels(), 0)
        session.candidates[0]["model_suggested_id"] = 0
        session.candidates[0]["identity_conflict"] = True
        self.assertEqual(session.accept_model_labels(), 1)
        self.assertNotIn("1", session.labels)
        self.assertEqual(session.labels["2"], 1)

    def test_depth_uncertainty_increase_clears_advice_even_without_position_shift(self):
        session, frame, now = confirmed_session()
        localize = session.calibration.localize_ball

        def worsened(*args):
            return dict(localize(*args), position_uncertainty_cm=2.0)

        session.calibration.localize_ball = worsened
        session.observe(frame, now + 0.1, now=now + 0.1, depth_map=np.ones(frame.shape[:2]))
        self.assertFalse(session.confirmed)
        self.assertIsNone(session.shot)
        self.assertIn("uncertainty", session.message)


if __name__ == "__main__":
    unittest.main()
