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


def test_filters_radius_and_clipped_ball_candidates():
    image, polygon = scene()
    cv2.circle(image, (80, 70), 4, (250, 250, 250), -1)
    cv2.circle(image, (160, 70), 24, (250, 250, 250), -1)
    cv2.circle(image, (12, 120), 12, (250, 250, 250), -1)
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
