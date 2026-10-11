import copy
import json
import math
import unittest

from billiards import (ball_center_is_playable, default_state, normalize_pockets,
                      parse_question, plan_shots, point_segment_distance)


def layout(balls, group="solids"):
    state = default_state()
    state["group"] = group
    state["balls"] = [{"id": number, "x": x, "y": y} for number, x, y in balls]
    return state


def top_middle(state, number=1):
    return next((shot for shot in plan_shots(state)["shots"]
                 if shot["ball_id"] == number and shot["pocket_id"] == "TM"
                 and shot["shot_type"] == "direct"), None)


class BilliardsGeometryTests(unittest.TestCase):
    def test_straight_shot_has_correct_ghost_angle_and_diamond(self):
        state = layout([(0, 127, 100), (1, 127, 50)])
        shot = top_middle(state)
        self.assertIsNotNone(shot)
        self.assertAlmostEqual(shot["ghost"]["x"], 127)
        self.assertAlmostEqual(shot["ghost"]["y"], 55.715)
        self.assertAlmostEqual(shot["aim_angle_deg"], 270)
        self.assertAlmostEqual(shot["cut_angle_deg"], 0)
        self.assertEqual(shot["rail"]["side"], "top")
        self.assertAlmostEqual(shot["rail"]["diamond"], 4)
        self.assertIn("Aiming reference only", shot["rail"]["description"])
        self.assertAlmostEqual(shot["distances"]["cue"], 44.285)

    def test_bottom_shot_angle_uses_clockwise_screen_coordinates(self):
        result = plan_shots(layout([(0, 127, 30), (1, 127, 80)]))
        shot = next(shot for shot in result["shots"] if shot["pocket_id"] == "BM")
        self.assertEqual(shot["aim_angle_deg"], 90)
        self.assertEqual(shot["rail"]["side"], "bottom")

    def test_object_obstacle_blocks_the_pot(self):
        state = layout([(0, 127, 100), (1, 127, 50), (9, 127, 25)])
        self.assertIsNone(top_middle(state))

    def test_cue_obstacle_blocks_the_approach(self):
        state = layout([(0, 127, 100), (1, 127, 50), (9, 127, 75)])
        self.assertIsNone(top_middle(state))

    def test_exact_tangent_obstacle_is_contact_not_clearance(self):
        state = layout([(0, 127, 100), (1, 127, 50), (9, 132.715, 25)])
        self.assertIsNone(top_middle(state))
        state["balls"][-1]["x"] += 0.001
        self.assertIsNotNone(top_middle(state))

    def test_tangent_to_cue_path_is_also_rejected(self):
        state = layout([(0, 127, 100), (1, 127, 50), (9, 132.715, 75)])
        self.assertIsNone(top_middle(state))

    def test_wrong_side_of_target_cannot_make_backwards_pot(self):
        self.assertIsNone(top_middle(layout([(0, 127, 20), (1, 127, 50)])))

    def test_ghost_outside_playable_bounds_is_rejected(self):
        self.assertIsNone(top_middle(layout([(0, 100, 100), (1, 127, 122)])))

    def test_nearly_parallel_side_pocket_approach_hits_pocket_lip(self):
        state = layout([(0, 70, 4), (1, 90, 4)])
        self.assertIsNone(top_middle(state))

    def test_diagonal_corner_pot_is_possible(self):
        result = plan_shots(layout([(0, 100, 100), (1, 50, 50)]))
        shot = next(shot for shot in result["shots"] if shot["pocket_id"] == "TL")
        self.assertAlmostEqual(shot["cut_angle_deg"], 0)
        self.assertAlmostEqual(shot["aim_angle_deg"], 225)

    def test_segment_distance_clamps_to_endpoints(self):
        self.assertAlmostEqual(point_segment_distance((5, 2), (0, 0), (10, 0)), 2)
        self.assertAlmostEqual(point_segment_distance((12, 0), (0, 0), (10, 0)), 2)
        self.assertAlmostEqual(point_segment_distance((3, 4), (0, 0), (0, 0)), 5)


class BilliardsTargetFilterTests(unittest.TestCase):
    def test_open_table_excludes_eight(self):
        result = plan_shots(layout([(0, 127, 100), (8, 127, 50)], "open"))
        self.assertEqual(result["eligible_ball_ids"], [])
        self.assertEqual(result["status"], "no_shot")

    def test_missing_group_balls_never_unlock_eight(self):
        state = layout([(0, 127, 100), (8, 127, 50), (9, 200, 50)])
        result = plan_shots(state)
        self.assertEqual(result["eligible_ball_ids"], [])
        self.assertEqual(result["status"], "no_shot")
        self.assertIn("No observed ball", result["message"])
        self.assertNotIn("pocketed", result["message"])
        state["balls"].append({"id": 1, "x": 50, "y": 50})
        self.assertEqual(plan_shots(state)["eligible_ball_ids"], [1])

    def test_stripes_are_targets_and_solids_are_obstacles(self):
        state = layout([(0, 127, 100), (9, 127, 50), (1, 127, 25)], "stripes")
        result = plan_shots(state)
        self.assertEqual(result["eligible_ball_ids"], [9])
        self.assertIsNone(top_middle(state, 9))

    def test_specific_ball_overrides_default_filter_but_must_be_observed(self):
        state = layout([(0, 127, 100), (1, 127, 50), (9, 200, 50)])
        self.assertEqual(plan_shots(state, "shoot ball 9")["target_ball_ids"], [9])
        with self.assertRaisesRegex(ValueError, "not on"):
            plan_shots(state, "shoot ball 7")
        self.assertEqual(plan_shots(state, "stripes shoot ball 9")["eligible_ball_ids"], [9])
        with self.assertRaisesRegex(ValueError, "target filter"):
            plan_shots(state, "solids shoot ball 9")

    def test_group_is_optional_and_open_targets_all_observed_ordinary_balls(self):
        state = layout([(0, 127, 100), (1, 127, 50), (9, 200, 50), (8, 50, 50)])
        del state["group"]
        result = plan_shots(state)
        self.assertEqual(result["target_ball_ids"], [1, 9])
        self.assertEqual(result["eligible_ball_ids"], result["target_ball_ids"])
        self.assertNotIn("group", state)

    def test_requests_for_each_group_are_independent_without_player_tracking(self):
        state = layout([(0, 127, 100), (1, 127, 50), (9, 200, 50)])
        original = copy.deepcopy(state)
        stripes = plan_shots(state, "best shot for stripes")
        solids = plan_shots(state, "best shot for solids")
        self.assertEqual(stripes["target_ball_ids"], [9])
        self.assertEqual(solids["target_ball_ids"], [1])
        self.assertEqual(plan_shots(state)["target_ball_ids"], [1])
        self.assertEqual(state, original)
        self.assertFalse(stripes["rules_evaluated"])

    def test_explicit_eight_request_evaluates_geometry_with_other_balls_present(self):
        state = layout([(0, 127, 100), (8, 127, 50), (1, 200, 50), (9, 50, 50)])
        for group in ("solids", "stripes", "open"):
            with self.subTest(group=group):
                state["group"] = group
                result = plan_shots(state, "shoot the eight ball")
                self.assertEqual(result["target_ball_ids"], [8])
                self.assertTrue(result["shots"])
                self.assertFalse(result["rules_evaluated"])
                self.assertEqual(result["target_filter"]["ball_id"], 8)

    def test_unrequested_eight_remains_an_obstacle(self):
        state = layout([(0, 127, 100), (1, 127, 50), (8, 127, 25)])
        self.assertEqual(plan_shots(state)["target_ball_ids"], [1])
        self.assertIsNone(top_middle(state))

    def test_ranked_output_is_finite_json_and_does_not_mutate_state(self):
        state = default_state()
        original = copy.deepcopy(state)
        result = plan_shots(state)
        json.dumps(result, allow_nan=False)
        self.assertEqual(state, original)
        scores = [shot["score"] for shot in result["shots"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertTrue(all(0 <= score <= 100 for score in scores))


class BilliardsValidationTests(unittest.TestCase):
    def test_invalid_observations_are_rejected(self):
        base = layout([(0, 127, 100), (1, 127, 50)])
        mutations = [
            lambda s: s["balls"].pop(0),
            lambda s: s["balls"].append(dict(s["balls"][1])),
            lambda s: s["balls"][1].update(x=math.nan),
            lambda s: s["balls"][1].update(y=math.inf),
            lambda s: s["balls"][1].update(x=1),
            lambda s: s["balls"][1].update(x=127, y=99),
            lambda s: s["balls"][1].update(id=True),
            lambda s: s["balls"][1].update(id=1.0),
            lambda s: s["balls"][1].update(id=16),
            lambda s: s["table"].update(ball_radius=0),
            lambda s: s["table"].update(pocket_width=5),
            lambda s: s["table"].update(width=math.inf),
            lambda s: s.update(group="unknown"),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                state = copy.deepcopy(base)
                mutate(state)
                with self.assertRaises(ValueError):
                    plan_shots(state)

    def test_touching_balls_are_valid_layout_not_overlapping(self):
        state = layout([(0, 127, 100), (1, 127, 50), (9, 132.715, 50)])
        self.assertIsInstance(plan_shots(state), dict)

    def test_parser_accepts_examples_and_overrides_group(self):
        self.assertEqual(parse_question("I am solids what is my best shot?", "open"), {"group": "solids"})
        self.assertEqual(parse_question("I'm stripes, aim for ball 10", "solids"), {"group": "stripes", "ball_id": 10})
        self.assertEqual(parse_question("shoot the 3", "solids"), {"group": "solids", "ball_id": 3})
        self.assertEqual(parse_question("where should I aim?", "open"), {"group": "open"})
        self.assertEqual(parse_question("what is the best shot in this position", "open"), {"group": "open"})
        self.assertEqual(parse_question("I'm on solids", "open"), {"group": "solids"})

    def test_parser_resolves_color_with_group_and_checks_number(self):
        self.assertEqual(parse_question("aim at the yellow solid", "open"), {"group": "solids", "ball_id": 1})
        self.assertEqual(parse_question("yellow", "stripes"), {"group": "stripes", "ball_id": 9})
        self.assertEqual(parse_question("black ball", "solids"), {"group": "solids", "ball_id": 8})
        self.assertEqual(parse_question("yellow ball 9", "open"), {"group": "open", "ball_id": 9})
        for question in ("yellow", "yellow ball 2", "red blue", "white ball", "shoot ball -1"):
            with self.subTest(question=question):
                with self.assertRaises(ValueError):
                    parse_question(question, "open")

    def test_parser_rejects_contradictions_and_unsupported_instructions(self):
        for question in ("solids and stripes", "shoot ball 1 or 2", "bank direct the 3",
                         "use left spin", "do not hit ball 8", "shoot ball 30",
                         "hello", "", "shoot 1.5 ball"):
            if question == "":
                self.assertEqual(parse_question(question, "open"), {"group": "open"})
                continue
            with self.subTest(question=question):
                with self.assertRaises(ValueError):
                    parse_question(question, "open")


class ExtendedPlannerTests(unittest.TestCase):
    def test_hanging_side_and_corner_balls_remain_easy_targets(self):
        side = layout([(0, 127, 70), (1, 127, 1)])
        shot = top_middle(side)
        self.assertIsNotNone(shot)
        self.assertEqual(shot["cut_angle_deg"], 0)
        self.assertEqual(shot["distances"]["object"], 1)
        corner = layout([(0, 50, 50), (1, 1, 1)])
        shot = next(s for s in plan_shots(corner)["shots"] if s["id"] == "1-TL")
        self.assertEqual(shot["cut_angle_deg"], 0)

    def test_hanging_opponent_ball_still_blocks_a_pocket(self):
        state = layout([(0, 127, 70), (1, 127, 30), (9, 127, 1)])
        self.assertIsNone(top_middle(state))
        result = plan_shots(state)
        self.assertGreater(result["diagnostics"]["rejected"].get("ball_blocks_path", 0), 0)

    def test_hanging_cue_can_leave_mouth_but_cannot_cut_through_its_jaw(self):
        state = layout([(0, 127, 1), (1, 127, 60)])
        straight = next(s for s in plan_shots(state)["shots"] if s["id"] == "1-BM")
        self.assertEqual(straight["cut_angle_deg"], 0)
        state["balls"][1].update(x=200, y=20)
        result = plan_shots(state)
        self.assertFalse(any(s["id"] == "1-BR" for s in result["shots"]))
        self.assertGreater(result["diagnostics"]["rejected"].get("cue_path_hits_cushion", 0), 0)

    def test_playable_center_helper_uses_custom_mouths_and_excludes_shelf(self):
        table = default_state()["table"]
        for x, y in ((127, 1), (1, 1), (253, 126)):
            self.assertTrue(ball_center_is_playable(table, x, y))
        for x, y in ((1, 50), (80, 1), (127, -0.001), (-0.001, 1), (255, 126)):
            self.assertFalse(ball_center_is_playable(table, x, y))
        table["pockets"] = [{"id": "TM", "x": 117}]
        self.assertTrue(ball_center_is_playable(table, 117, 1))
        self.assertFalse(ball_center_is_playable(table, 127, 1))
        table["pockets"][0]["blocked"] = True
        self.assertTrue(ball_center_is_playable(table, 117, 1))
        with self.assertRaises(ValueError):
            ball_center_is_playable(table, math.nan, 1)

    def test_calibrated_middle_pocket_changes_aim(self):
        state = layout([(0, 119, 100), (1, 119, 50)])
        state["table"]["pockets"] = [{"id": "TM", "x": 119, "width": 12}]
        shot = top_middle(state)
        self.assertEqual(shot["ghost"]["x"], 119)
        self.assertEqual(shot["object_path"][-1], {"x": 119, "y": 0})
        self.assertEqual(shot["cut_angle_deg"], 0)

    def test_marked_pockets_are_never_targets(self):
        state = layout([(0, 127, 100), (1, 127, 50)])
        state["table"]["pockets"] = [{"id": "TM", "blocked": True}]
        result = plan_shots(state)
        self.assertFalse(any(shot["pocket_id"] == "TM" for shot in result["shots"]))
        self.assertEqual(result["diagnostics"]["rejected"]["pocket_marked_blocked"], 1)
        state["table"]["pockets"] = [{"id": key, "blocked": True} for key in ("TL", "TM", "TR", "BL", "BM", "BR")]
        self.assertEqual(plan_shots(state)["status"], "no_shot")

    def test_each_pocket_width_controls_real_jaw_clearance(self):
        state = layout([(0, 127, 100), (1, 127, 50)])
        normal = top_middle(state)
        state["table"]["pockets"] = [{"id": "TM", "width": 6}]
        narrow = top_middle(state)
        self.assertIsNotNone(narrow)
        self.assertLess(narrow["score"], normal["score"])
        state["table"]["position_uncertainty_cm"] = 0.2
        self.assertIsNone(top_middle(state))

    def test_uncertainty_rejects_a_nominally_clear_tight_corridor(self):
        state = layout([(0, 127, 100), (1, 127, 50), (9, 132.915, 25)])
        self.assertIsNotNone(top_middle(state))
        state["balls"][1]["position_uncertainty_cm"] = 0.11
        state["balls"][2]["position_uncertainty_cm"] = 0.11
        self.assertIsNone(top_middle(state))

    def test_confidence_changes_rank_and_error_is_not_double_counted(self):
        state = layout([(0, 127, 100), (1, 127, 50)])
        original = top_middle(state)
        state["balls"][1]["confidence"] = 0.6
        self.assertLess(top_middle(state)["score"], original["score"])
        state["balls"][1]["position_uncertainty_cm"] = 0.3
        margin = top_middle(state)
        state["table"]["position_uncertainty_cm"] = 0.3
        self.assertEqual(top_middle(state)["position_uncertainty_cm"], margin["position_uncertainty_cm"])
        state["planner"] = {"position_uncertainty_cm": 0.2}
        self.assertEqual(top_middle(state)["position_uncertainty_cm"], 0.5)

    def test_bank_reflects_at_ball_center_and_each_leg_clears_blockers(self):
        state = layout([(0, 90, 100), (1, 130, 65)])
        result = plan_shots(state)
        shot = next(s for s in result["shots"] if s["id"] == "1-BR-bank-top")
        a, b, c = [(p["x"], p["y"]) for p in shot["object_path"]]
        self.assertAlmostEqual(b[1], state["table"]["ball_radius"])
        self.assertAlmostEqual((b[0] - a[0]) / (a[1] - b[1]), (c[0] - b[0]) / (c[1] - b[1]), places=6)
        for start, end in ((a, b), (b, c)):
            blocked = copy.deepcopy(state)
            blocked["balls"].append({"id": 9, "x": (start[0] + end[0]) / 2, "y": (start[1] + end[1]) / 2})
            self.assertFalse(any(s["id"] == shot["id"] for s in plan_shots(blocked)["shots"]))

    def test_bank_filter_and_direct_filter_are_explicit(self):
        state = layout([(0, 90, 100), (1, 130, 65)])
        banks = plan_shots(state, "best bank shot")["shots"]
        self.assertTrue(banks)
        self.assertTrue(all(s["shot_type"] == "bank" for s in banks))
        direct = plan_shots(state, "best direct shot")["shots"]
        self.assertTrue(direct)
        self.assertTrue(all(s["shot_type"] == "direct" for s in direct))
        state["planner"] = {"allow_banks": False}
        self.assertTrue(all(s["shot_type"] == "direct" for s in plan_shots(state)["shots"]))
        self.assertEqual(parse_question("bank the 3", "solids"), {"group": "solids", "ball_id": 3, "shot_type": "bank"})

    def test_straight_stun_stops_at_ghost_and_exposes_conditional_model(self):
        shot = top_middle(layout([(0, 127, 100), (1, 127, 50)]))
        self.assertEqual(shot["scratch_risk"]["event"], "stopped")
        self.assertEqual(shot["scratch_risk"]["stop"], shot["ghost"])
        self.assertIn("Conditional", shot["scratch_risk"]["warning"])
        self.assertGreater(shot["power"]["cue_speed_cm_s"], 0)

    def test_tangent_line_to_side_pocket_is_flagged_and_penalized(self):
        radius = default_state()["table"]["ball_radius"]
        ghost_x = 60 + math.sqrt(2) * radius
        state = layout([(0, ghost_x, 110), (1, 60, 60)])
        result = plan_shots(state)
        risky = next(s for s in result["shots"] if s["id"] == "1-TL")
        self.assertAlmostEqual(risky["cut_angle_deg"], 45, places=3)
        self.assertEqual(risky["scratch_risk"]["event"], "scratch")
        self.assertEqual(risky["scratch_risk"]["pocket_id"], "TM")
        self.assertEqual(risky["scratch_risk"]["level"], "high")
        self.assertGreater(risky["score_factors"]["scratch"], 50)
        self.assertNotEqual(result["shots"][0]["id"], risky["id"])

    def test_scaled_geometry_retains_aim_and_bank_type(self):
        state = layout([(0, 90, 100), (1, 130, 65)])
        original = {s["id"]: s for s in plan_shots(state)["shots"]}
        for key in ("width", "height", "ball_radius", "pocket_width"):
            state["table"][key] *= 0.5
        for ball in state["balls"]:
            ball["x"] *= 0.5
            ball["y"] *= 0.5
        scaled = {s["id"]: s for s in plan_shots(state)["shots"]}
        self.assertEqual(set(scaled), set(original))
        for key, shot in scaled.items():
            self.assertEqual(shot["aim_angle_deg"], original[key]["aim_angle_deg"])
            self.assertAlmostEqual(shot["distances"]["object"], original[key]["distances"]["object"] / 2, places=3)

    def test_custom_pocket_and_planner_validation(self):
        base = layout([(0, 127, 100), (1, 127, 50)])
        invalid_pockets = [None, {}, [{"id": "X"}], [{"id": []}], [{"id": "TM", "x": 2}],
                           [{"id": "TL", "x": 1}], [{"id": "TM", "y": 1}],
                           [{"id": "TM", "blocked": "yes"}], [{"id": "TM", "width": 5}],
                           [{"id": "TM"}, {"id": "TM"}], [{"id": "TM", "x": math.nan}]]
        for pockets in invalid_pockets:
            with self.subTest(pockets=pockets):
                state = copy.deepcopy(base)
                state["table"]["pockets"] = pockets
                with self.assertRaises(ValueError):
                    plan_shots(state)
        for field, value in (("allow_banks", "yes"), ("cushion_restitution", 0),
                             ("rolling_deceleration_cm_s2", 0), ("max_cut_angle_deg", math.nan),
                             ("position_uncertainty_cm", -1)):
            state = copy.deepcopy(base)
            state["planner"] = {field: value}
            with self.assertRaises(ValueError):
                plan_shots(state)
        for field, value in (("confidence", 1.1), ("position_uncertainty_cm", math.inf)):
            state = copy.deepcopy(base)
            state["balls"][1][field] = value
            with self.assertRaises(ValueError):
                plan_shots(state)

    def test_normalized_pockets_are_idempotent_and_input_is_not_mutated(self):
        state = default_state()
        state["table"]["pockets"] = [{"id": "TM", "x": 125, "blocked": True}]
        original = copy.deepcopy(state)
        normalized = normalize_pockets(state["table"])
        self.assertEqual(state, original)
        state["table"]["pockets"] = normalized
        self.assertEqual(normalize_pockets(state["table"]), normalized)


if __name__ == "__main__":
    unittest.main()
