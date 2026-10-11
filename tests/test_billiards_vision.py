"""Physical-camera proposal and conservative association behavior."""

from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from billiards_vision import BallDetector, CandidateTracker, suggest_ball_id


def scene():
    image = np.full((240, 400, 3), (45, 105, 35), np.uint8)
    return image, np.array([[10, 10], [390, 10], [390, 230], [10, 230]], np.float32)


def candidate(x, y=50, radius=10):
    return dict(pixel={"x": x, "y": y}, radius_px=radius, suggested_id=None)


def test_detects_separate_realistic_ball_colors_without_weights():
    image, polygon = scene()
    balls = [(70, 70, (240, 240, 240)), (160, 120, (10, 10, 10)),
             (270, 80, (10, 20, 235)), (310, 170, (220, 30, 20))]
    for x, y, color in balls:
        cv2.circle(image, (x, y), 12, color, -1)
    found = BallDetector(8, 16).detect(image, polygon)
    assert len(found) == len(balls)
    for x, y, _ in balls:
        assert min(np.hypot(d["pixel"]["x"] - x, d["pixel"]["y"] - y) for d in found) < 2
    assert {d["suggested_id"] for d in found} == {0, 8, 3, 2}
    assert not any(d["identity_confirmed"] for d in found)


def test_cloth_and_objects_outside_table_are_not_balls():
    image, _ = scene()
    polygon = np.array([[30, 30], [200, 30], [200, 200], [30, 200]], np.float32)
    assert BallDetector(8, 16).detect(image, polygon) == []
    cv2.circle(image, (300, 100), 12, (250, 250, 250), -1)
    cv2.rectangle(image, (60, 65), (135, 90), (240, 240, 240), -1)
    assert BallDetector(8, 16).detect(image, polygon) == []


def test_filters_radius_and_image_clipped_ball_candidates():
    image, polygon = scene()
    cv2.circle(image, (80, 70), 4, (250, 250, 250), -1)
    cv2.circle(image, (160, 70), 24, (250, 250, 250), -1)
    cv2.circle(image, (4, 120), 12, (250, 250, 250), -1)
    assert BallDetector(9, 15).detect(image, polygon) == []


def test_color_and_stripe_hints_require_operator_confirmation():
    image, polygon = scene()
    cv2.circle(image, (100, 100), 16, (245, 245, 245), -1)
    stripe = np.zeros(image.shape[:2], np.uint8)
    cv2.circle(stripe, (100, 100), 16, 255, -1)
    stripe[:96] = 0
    stripe[105:] = 0
    image[stripe != 0] = (0, 230, 230)
    assert suggest_ball_id(image, (100, 100), 16) == 9
    found = BallDetector(12, 20).detect(image, polygon)
    assert len(found) == 1 and found[0]["suggested_id"] == 9
    assert found[0]["identity_confirmed"] is False


class Tensor:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class FakeYOLO:
    def __init__(self, label="sports ball"):
        self.label = label

    def predict(self, frame, **kwargs):
        return [SimpleNamespace(names={0: self.label}, boxes=SimpleNamespace(
            xyxy=Tensor([[88, 88, 112, 112]]), cls=Tensor([0]), conf=Tensor([0.95])))]


class BatchYOLO:
    def __init__(self, names, boxes, classes, scores):
        self.result = SimpleNamespace(names=names, boxes=SimpleNamespace(
            xyxy=Tensor(boxes), cls=Tensor(classes), conf=Tensor(scores)))
        self.calls = 0

    def predict(self, frame, **kwargs):
        self.calls += 1
        return [self.result]


def test_yolo_and_circle_proposals_merge_without_inventing_number_labels():
    image, polygon = scene()
    cv2.circle(image, (100, 100), 12, (240, 240, 240), -1)
    found = BallDetector(8, 16, FakeYOLO()).detect(image, polygon)
    assert len(found) == 1
    assert found[0]["source"] == "yolo"
    assert found[0]["suggestion_source"] == "appearance"
    assert found[0]["identity_confirmed"] is False


def test_explicit_custom_model_label_remains_a_suggestion():
    image, polygon = scene()
    cv2.circle(image, (100, 100), 12, (240, 240, 240), -1)
    found = BallDetector(8, 16, FakeYOLO("nine"), class_to_ball_id={"nine": 9}).detect(image, polygon)
    assert len(found) == 1 and found[0]["suggested_id"] == 9
    assert found[0]["suggestion_source"] == "custom_model_label"
    assert found[0]["identity_confirmed"] is False


def test_yolo_nonball_class_is_not_a_proposal():
    image, polygon = scene()
    assert BallDetector(8, 16, FakeYOLO("bottle")).detect(image, polygon) == []


def test_hanging_ball_at_pocket_boundary_is_retained_for_geometry_review():
    image, _ = scene()
    polygon = np.array([[30, 30], [370, 30], [370, 210], [30, 210]], np.float32)
    cv2.circle(image, (200, 31), 12, (240, 240, 240), -1)
    model = BatchYOLO({0: "ball_9"}, [[188, 19, 212, 43]], [0], [0.95])
    found = BallDetector(8, 16, model).detect(image, polygon)
    hanging = next(d for d in found if d["model_suggested_id"] == 9)
    assert hanging["pixel"] == {"x": 200, "y": 31}
    assert hanging["pixel"]["y"] - hanging["radius_px"] < polygon[0, 1]
    assert not hanging["identity_confirmed"]


@pytest.mark.parametrize("y,accepted", [(30, True), (29, False)])
def test_boundary_ball_center_must_still_be_on_or_inside_table(y, accepted):
    image, _ = scene()
    polygon = np.array([[30, 30], [370, 30], [370, 210], [30, 210]], np.float32)
    model = BatchYOLO({0: "ball_9"}, [[188, y - 12, 212, y + 12]], [0], [0.95])
    found = BallDetector(8, 16, model).detect(image, polygon)
    assert bool(found) is accepted


@pytest.mark.parametrize("label,number", [
    ("cue_ball", 0), ("ball_1", 1), ("ball12", 12), ("8-ball", 8),
    ("ball nine", 9), ("striped-13", 13), ("SOLID_3", 3),
    ("fifteen", 15), ("pool ball 4", 4), ("9", 9),
])
def test_canonical_pool_model_names_propose_number_without_mapping(label, number):
    image, polygon = scene()
    found = BallDetector(8, 16, FakeYOLO(label)).detect(image, polygon)
    assert len(found) == 1
    assert found[0]["suggested_id"] == number
    assert found[0]["model_suggested_id"] == number
    assert found[0]["model_suggested_ids"] == [number]
    assert found[0]["model_confidence"] == pytest.approx(0.95)
    assert found[0]["model_identity_confidence"] == pytest.approx(0.95)
    assert found[0]["suggestion_source"] == "model_label"
    assert not found[0]["identity_confirmed"]


@pytest.mark.parametrize("label,group", [("solid", "solids"), ("STRIPED_ball", "stripes")])
def test_group_model_labels_do_not_invent_number(label, group):
    image, polygon = scene()
    found = BallDetector(8, 16, FakeYOLO(label)).detect(image, polygon)
    assert len(found) == 1 and found[0]["group_hint"] == group
    assert found[0]["model_suggested_id"] is None


def test_class_index_is_never_a_ball_number():
    image, polygon = scene()
    model = BatchYOLO({9: "sports ball"}, [[88, 88, 112, 112]], [9], [0.95])
    found = BallDetector(8, 16, model).detect(image, polygon)
    assert len(found) == 1 and found[0]["model_suggested_id"] is None
    assert found[0]["suggestion_source"] == "appearance"
    assert found[0]["suggested_id"] != 9


def test_explicit_mapping_normalizes_names_and_supports_json_indices():
    image, polygon = scene()
    model = BatchYOLO({17: "SPECIAL_nine"}, [[88, 88, 112, 112]], [17], [0.95])
    for mapping in ({"special-NINE": 9}, {17: 9}, {"17": 9}):
        found = BallDetector(8, 16, model, class_to_ball_id=mapping).detect(image, polygon)
        assert found[0]["model_suggested_id"] == 9
        assert found[0]["suggestion_source"] == "custom_model_label"
    # An explicit class name takes precedence over a string index fallback.
    found = BallDetector(8, 16, model, class_to_ball_id={"special nine": 9, "17": 7}).detect(image, polygon)
    assert found[0]["model_suggested_id"] == 9


def test_model_confidence_is_distinct_from_opencv_proposal_quality():
    image, polygon = scene()
    cv2.circle(image, (100, 100), 12, (240, 240, 240), -1)
    model = BatchYOLO({0: "ball_9"}, [[88, 88, 112, 112]], [0], [0.4])
    found = BallDetector(8, 16, model).detect(image, polygon)
    assert len(found) == 1
    assert found[0]["source"] == "opencv_contour"
    assert found[0]["confidence"] > found[0]["model_confidence"]
    assert found[0]["model_confidence"] == pytest.approx(0.4)
    assert found[0]["model_suggested_id"] == 9


@pytest.mark.parametrize("second_label", ["ball_4", "stripes"])
def test_overlapping_conflicting_model_labels_suppress_identity(second_label):
    image, polygon = scene()
    model = BatchYOLO({0: "ball_3", 1: second_label},
                      [[88, 88, 112, 112], [89, 89, 113, 113]], [0, 1], [0.95, 0.4])
    found = BallDetector(8, 16, model).detect(image, polygon)
    assert len(found) == 1
    assert found[0]["identity_conflict"]
    assert found[0]["model_suggested_id"] is None
    assert found[0]["suggested_id"] is None
    assert found[0]["suggestion_source"] == "conflicting_model_labels"
    assert found[0]["model_confidence"] == pytest.approx(0.95)
    assert found[0]["model_identity_confidence"] is None


def test_duplicate_agreeing_labels_keep_numbered_suggestion():
    image, polygon = scene()
    model = BatchYOLO({0: "ball_9", 1: "stripes", 2: "nine"},
                      [[88, 88, 112, 112]] * 3, [0, 1, 2], [0.95, 0.6, 0.4])
    found = BallDetector(8, 16, model).detect(image, polygon)
    assert len(found) == 1 and not found[0]["identity_conflict"]
    assert found[0]["model_suggested_id"] == 9
    assert found[0]["model_suggested_ids"] == [9]


def test_confident_generic_detection_does_not_inflate_number_confidence():
    image, polygon = scene()
    model = BatchYOLO({0: "sports ball", 1: "nine"},
                      [[88, 88, 112, 112]] * 2, [0, 1], [0.99, 0.4])
    found = BallDetector(8, 16, model).detect(image, polygon)
    assert found[0]["model_suggested_id"] == 9
    assert found[0]["model_confidence"] == pytest.approx(0.99)
    assert found[0]["model_identity_confidence"] == pytest.approx(0.4)


@pytest.mark.parametrize("box,cls,score", [
    ([88, 88, 112, 112], 0, 0.19), ([88, 88, 112, 112], 0, 1.01),
    ([88, 88, 112, 112], 0.5, 0.9), ([88, 88, 112, 112], -1, 0.9),
    ([88, 88, 112, 112], 0, float("nan")),
    ([112, 88, 88, 112], 0, 0.9), ([-1, 88, 23, 112], 0, 0.9),
    ([388, 88, 412, 112], 0, 0.9), ([88, 88, float("inf"), 112], 0, 0.9),
])
def test_invalid_or_below_threshold_model_boxes_are_not_candidates(box, cls, score):
    image, polygon = scene()
    model = BatchYOLO({0: "ball_9"}, [box], [cls], [score])
    assert BallDetector(8, 16, model).detect(image, polygon) == []


def test_missing_model_names_only_allow_explicit_index_mapping():
    image, polygon = scene()
    model = BatchYOLO({}, [[88, 88, 112, 112]], [17], [0.95])
    assert BallDetector(8, 16, model).detect(image, polygon) == []
    assert BallDetector(8, 16, model, class_to_ball_id={17: 9}).detect(image, polygon)[0]["model_suggested_id"] == 9


def test_scene_parts_are_unconfirmed_review_hints_and_share_one_inference():
    image, polygon = scene()
    model = BatchYOLO({0: "pocket", 1: "rail", 2: "pool_table", 3: "cue_stick", 4: "ball_8"},
                      [[5, 5, 20, 20], [20, 8, 380, 18], [8, 8, 392, 232],
                       [20, 60, 140, 67], [88, 88, 112, 112]], [0, 1, 2, 3, 4], [0.95] * 5)
    result = BallDetector(8, 16, model).detect_scene(image, polygon)
    assert model.calls == 1
    assert len(result["balls"]) == 1 and result["balls"][0]["model_suggested_id"] == 8
    assert {part["kind"] for part in result["parts"]} == {"pocket", "rail", "table", "cue_stick"}
    assert all(part["geometry_confirmed"] is False for part in result["parts"])
    assert all("blocked" not in part for part in result["parts"])


def test_parts_outside_the_table_are_not_reported():
    image, _ = scene()
    polygon = np.array([[30, 30], [200, 30], [200, 200], [30, 200]], np.float32)
    model = BatchYOLO({0: "pocket"}, [[300, 80, 320, 100]], [0], [0.95])
    assert BallDetector(8, 16, model).detect_scene(image, polygon) == {"balls": [], "parts": []}


@pytest.mark.parametrize("mapping", [{"ball": True}, {"ball": 16}, {"ball": 1.5},
                                      {True: 1}, {-1: 1}, {"": 1}, {"ball_1": 1, "BALL-1": 2}])
def test_invalid_class_mappings_are_rejected(mapping):
    with pytest.raises(ValueError):
        BallDetector(class_to_ball_id=mapping)


def test_nonfinite_or_invalid_camera_inputs_are_rejected():
    image, polygon = scene()
    with pytest.raises(ValueError):
        BallDetector(float("nan"), 20)
    with pytest.raises(ValueError):
        BallDetector(20, 10)
    with pytest.raises(ValueError):
        BallDetector().detect(image.astype(float), polygon)
    polygon[0, 0] = np.nan
    with pytest.raises(ValueError):
        BallDetector().detect(image, polygon)
    with pytest.raises(ValueError):
        suggest_ball_id(image, (10, np.inf), 5)


def test_unique_stationary_association_is_order_independent():
    tracker = CandidateTracker(max_distance_px=5)
    first = tracker.update([candidate(20), candidate(70)])
    second = tracker.update([candidate(71), candidate(19)])
    assert second[0]["key"] == first[1]["key"]
    assert second[1]["key"] == first[0]["key"]
    assert not any(c["ambiguous"] for c in second)


def test_crossing_or_ambiguous_associations_do_not_reuse_identifiers():
    tracker = CandidateTracker(max_distance_px=15)
    first = tracker.update([candidate(40), candidate(60)])
    second = tracker.update([candidate(49), candidate(51)])
    assert {c["key"] for c in first}.isdisjoint(c["key"] for c in second)
    assert all(c["ambiguous"] for c in second)


def test_misses_motion_and_cumulative_drift_retire_identity():
    tracker = CandidateTracker(max_distance_px=5)
    old = tracker.update([candidate(20)])[0]["key"]
    assert tracker.update([candidate(23)])[0]["key"] == old
    moved = tracker.update([candidate(26)])[0]["key"]
    assert moved != old
    assert tracker.update([]) == []
    assert tracker.update([candidate(26)])[0]["key"] != moved
    previous = tracker.update([candidate(26)])[0]["key"]
    assert tracker.update([candidate(100)])[0]["key"] != previous


def test_single_detection_cannot_silently_claim_one_of_two_previous_balls():
    tracker = CandidateTracker(max_distance_px=10)
    first = tracker.update([candidate(40), candidate(55)])
    second = tracker.update([candidate(47)])
    assert second[0]["ambiguous"]
    assert second[0]["key"] not in {c["key"] for c in first}


def test_tracker_does_not_preserve_claimed_identity_or_reuse_reset_keys():
    tracker = CandidateTracker()
    item = dict(candidate(10), identity_confirmed=True)
    first = tracker.update([item])[0]
    assert not first["identity_confirmed"]
    assert item["identity_confirmed"]  # Caller data was not mutated.
    tracker.reset()
    assert tracker.update([item])[0]["key"] != first["key"]
    with pytest.raises(ValueError):
        tracker.update([candidate(np.nan)])
