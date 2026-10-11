"""Numerical object-depth checks without camera, Torch, or model downloads."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from live_yolo_object_depth import depth_label, object_depth_records


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class FakeBoxes:
    def __init__(self, coordinates, classes, ids=None):
        self.xyxy = FakeTensor(coordinates)
        self.cls = FakeTensor(classes)
        self.id = None if ids is None else FakeTensor(ids)
        self.count = len(coordinates)

    def __len__(self):
        return self.count


def result_for(coordinates, classes, ids=None):
    return SimpleNamespace(boxes=FakeBoxes(coordinates, classes, ids),
                           names={0: "bottle", 1: "cup"})


class ObjectDepthTests(unittest.TestCase):
    def test_same_class_objects_keep_separate_ids_and_depths(self):
        depth_map = np.full((100, 200), 9.0, dtype=np.float32)
        depth_map[30:70, 30:70] = 0.3
        depth_map[30:70, 130:170] = 0.5
        result = result_for([[20, 20, 80, 80], [120, 20, 180, 80]],
                            [0, 0], [7, 12])
        records = object_depth_records(result, depth_map, frame_id=42)
        self.assertEqual([r["track_id"] for r in records], [7, 12])
        self.assertEqual([r["object"] for r in records], ["bottle", "bottle"])
        self.assertTrue(all(r["frame_id"] == 42 for r in records))
        self.assertAlmostEqual(records[0]["depth_m"], 0.3, places=6)
        self.assertAlmostEqual(records[1]["depth_m"], 0.5, places=6)

    def test_depth_above_display_range_is_not_clipped(self):
        result = result_for([[10, 10, 90, 90]], [1], [2])
        records = object_depth_records(result, np.full((100, 100), 4.25), 1)
        self.assertEqual(records[0]["depth_m"], 4.25)
        self.assertEqual(depth_label(records[0]),
                         "Depth of cup [ID 2]: 4.250 m (est.)")

    def test_invalid_depth_remains_unknown_and_serializes_to_null(self):
        depth_map = np.full((100, 100), np.nan)
        depth_map[45:50, 45:50] = np.inf
        depth_map[50:55, 45:50] = 0.0
        depth_map[45:50, 50:55] = -1.0
        result = result_for([[20, 20, 80, 80]], [0], [4])
        record = object_depth_records(result, depth_map, 1)[0]
        self.assertIsNone(record["depth_m"])
        self.assertIn("unknown", depth_label(record))
        self.assertIsNone(json.loads(json.dumps(record, allow_nan=False))["depth_m"])

    def test_missing_tracking_id_does_not_invent_an_id(self):
        result = result_for([[10, 10, 90, 90]], [0])
        record = object_depth_records(result, np.ones((100, 100)), 1)[0]
        self.assertIsNone(record["track_id"])
        self.assertEqual(depth_label(record), "Depth of bottle: 1.000 m (est.)")

    def test_empty_detections_do_not_reuse_previous_depths(self):
        result = result_for([], [])
        self.assertEqual(object_depth_records(result, np.ones((100, 100)), 2), [])
        result.boxes = None
        self.assertEqual(object_depth_records(result, np.ones((100, 100)), 3), [])

    def test_too_small_or_outside_box_has_unknown_depth(self):
        result = result_for([[0, 0, 2, 2], [200, 200, 300, 300]], [0, 1], [1, 2])
        records = object_depth_records(result, np.ones((100, 100)), 1)
        self.assertTrue(all(r["depth_m"] is None for r in records))


if __name__ == "__main__":
    unittest.main()
