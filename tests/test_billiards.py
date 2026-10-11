import copy
import json
import math
import unittest

from billiards import default_state, parse_question, plan_shots, point_segment_distance


def layout(balls, group="solids"):
    state = default_state()
    state["group"] = group
    state["balls"] = [{"id": number, "x": x, "y": y} for number, x, y in balls]
    return state


def top_middle(state, number=1):
    return next((shot for shot in plan_shots(state)["shots"]
                 if shot["ball_id"] == number and shot["pocket_id"] == "TM"), None)


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


class BilliardsRulesTests(unittest.TestCase):
    def test_open_table_excludes_eight(self):
        result = plan_shots(layout([(0, 127, 100), (8, 127, 50)], "open"))
        self.assertEqual(result["eligible_ball_ids"], [])
        self.assertEqual(result["status"], "no_shot")

    def test_eight_available_only_after_own_group_cleared(self):
        state = layout([(0, 127, 100), (8, 127, 50), (9, 200, 50)])
        self.assertEqual(plan_shots(state)["eligible_ball_ids"], [8])
        state["balls"].append({"id": 1, "x": 50, "y": 50})
        self.assertEqual(plan_shots(state)["eligible_ball_ids"], [1])

    def test_stripes_are_legal_and_solids_are_obstacles(self):
        state = layout([(0, 127, 100), (9, 127, 50), (1, 127, 25)], "stripes")
        result = plan_shots(state)
        self.assertEqual(result["eligible_ball_ids"], [9])
        self.assertIsNone(top_middle(state, 9))

    def test_specific_requested_ball_must_be_present_and_legal(self):
        state = layout([(0, 127, 100), (1, 127, 50), (9, 200, 50)])
        with self.assertRaisesRegex(ValueError, "not a legal"):
            plan_shots(state, "shoot ball 9")
        with self.assertRaisesRegex(ValueError, "not on"):
            plan_shots(state, "shoot ball 7")
        self.assertEqual(plan_shots(state, "stripes shoot ball 9")["eligible_ball_ids"], [9])

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
        for question in ("solids and stripes", "shoot ball 1 or 2", "bank the 3",
                         "use left spin", "do not hit ball 8", "shoot ball 30",
                         "hello", "", "shoot 1.5 ball"):
            if question == "":
                self.assertEqual(parse_question(question, "open"), {"group": "open"})
                continue
            with self.subTest(question=question):
                with self.assertRaises(ValueError):
                    parse_question(question, "open")


if __name__ == "__main__":
    unittest.main()
