"""Reference identity, units, scale, and invalidation without camera or models."""

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference_depth import (
    BottleReference, ReferenceDepthCalibrator, adjust_records, bottle_at_pixel,
)


PROFILE = BottleReference("synthetic_test_bottle", 0.2, 0.06)


def record(track_id=7, frame_id=1, depth=1.0, box=None, object_name="bottle"):
    return {"frame_id": frame_id, "track_id": track_id, "object": object_name,
            "depth_m": depth, "bbox_xyxy": box or [80, 40, 140, 240]}


def test_size_reference_scales_other_bottles_without_assigning_them_its_size():
    records = [record(), record(12, depth=2.0, box=[220, 40, 310, 220])]
    calibrator = ReferenceDepthCalibrator(PROFILE)
    adjustment = calibrator.calibrate(records, 7, 1, (640, 480), (500, 500))
    assert adjustment["reference_depth_m"] == pytest.approx(0.5)
    corrected = adjust_records(records, adjustment)
    assert [r["depth_m"] for r in corrected] == pytest.approx([0.5, 1.0])
    assert [r["raw_depth_m"] for r in corrected] == [1.0, 2.0]
    assert "height_m" not in corrected[1]
    assert records[0]["depth_m"] == 1.0  # Raw inputs are preserved.
    json.dumps(corrected, allow_nan=False)


def test_dimensions_alone_never_establish_distance():
    calibrator = ReferenceDepthCalibrator(PROFILE)
    assert calibrator.calibrate([record()], 7, 1, (640, 480)) is None
    assert "Need camera calibration" in calibrator.status
    assert adjust_records([record()], None)[0]["depth_m"] == 1.0


def test_reference_selection_requires_an_unambiguous_tracked_bottle():
    assert bottle_at_pixel([record()], 100, 100) == 7
    assert bottle_at_pixel([record(), record(12)], 100, 100) is None
    assert bottle_at_pixel([record(track_id=None)], 100, 100) is None
    assert bottle_at_pixel([record(object_name="cup")], 100, 100) is None
    assert bottle_at_pixel([record()], 500, 300) is None


def test_known_distance_needs_no_intrinsics_but_invalidates_after_movement():
    calibrator = ReferenceDepthCalibrator(PROFILE, measured_distance_m=0.6)
    adjustment = calibrator.calibrate([record()], 7, 1, (640, 480))
    assert adjustment["scale_factor"] == pytest.approx(0.6)
    assert calibrator.update([record(frame_id=2, depth=1.2)], 2, (640, 480))[
        "scale_factor"] == pytest.approx(0.5)
    moved = record(frame_id=3, box=[100, 40, 165, 248])
    assert calibrator.update([moved], 3, (640, 480)) is None
    assert calibrator.track_id is None


def test_size_reference_updates_as_the_same_bottle_moves_in_depth():
    calibrator = ReferenceDepthCalibrator(PROFILE)
    calibrator.calibrate([record()], 7, 1, (640, 480), (500, 500))
    nearer = record(frame_id=2, depth=0.8, box=[80, 40, 155, 290])
    adjustment = calibrator.update([nearer], 2, (640, 480))
    assert adjustment["reference_depth_m"] == pytest.approx(0.4)
    assert adjustment["scale_factor"] == pytest.approx(0.5)


@pytest.mark.parametrize("bad_record", [
    record(track_id=None), record(track_id=12), record(object_name="cup"),
    record(frame_id=0), record(depth=None), record(depth=float("nan")),
    record(depth=float("inf")), record(depth=0),
    record(box=[0, 40, 65, 248]), record(box=[80, 40, 145, 480]),
    record(box=[80, 40, 200, 100]), record(box=[80, 40, 90, 60]),
])
def test_invalid_reference_never_generates_scale(bad_record):
    calibrator = ReferenceDepthCalibrator(PROFILE)
    assert calibrator.calibrate([bad_record], 7, 1, (640, 480), (500, 500)) is None
    assert calibrator.adjustment is None


@pytest.mark.parametrize("records, frame_id, frame_size", [
    ([record(frame_id=2, track_id=12)], 2, (640, 480)),
    ([], 2, (640, 480)),
    ([record(), record()], 2, (640, 480)),
    ([record()], 1, (640, 480)),
    ([record(frame_id=2)], 2, (1280, 960)),
])
def test_loss_ambiguity_stale_frames_and_resize_clear_calibration(records, frame_id, frame_size):
    calibrator = ReferenceDepthCalibrator(PROFILE)
    calibrator.calibrate([record()], 7, 1, (640, 480), (500, 500))
    assert calibrator.update(records, frame_id, frame_size) is None
    assert calibrator.adjustment is None
    # It cannot silently re-arm when the original bottle reappears.
    assert calibrator.update([record(frame_id=3)], 3, (640, 480)) is None


def test_adjustment_never_applies_to_a_different_frame_or_invents_missing_depth():
    calibrator = ReferenceDepthCalibrator(PROFILE)
    adjustment = calibrator.calibrate([record()], 7, 1, (640, 480), (500, 500))
    assert adjust_records([record(frame_id=2)], adjustment)[0]["depth_m"] == 1.0
    missing = adjust_records([record(depth=None)], adjustment)[0]
    assert missing["depth_m"] is None
    assert missing["raw_depth_m"] is None


def test_zero_and_nonfinite_profile_and_tape_measurements_are_rejected():
    for value in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            BottleReference("invalid", value, 0.06)
        with pytest.raises(ValueError):
            ReferenceDepthCalibrator(PROFILE, value)


def test_profile_is_loaded_with_explicit_meter_units(tmp_path):
    path = tmp_path / "reference.json"
    path.write_text(json.dumps({"name": "example", "height_m": 0.2,
                                "diameter_m": 0.06}), encoding="utf-8")
    assert BottleReference.load(path).height_m == pytest.approx(0.2)
