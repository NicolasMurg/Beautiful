#!/usr/bin/env python3
"""Convert calibrated ball pixel/distance observations into a physical table map.

Unknown ball identities may be mapped for review. --confirmed asserts that all
balls currently on the table are represented and their identities reviewed.
No game history or player assignment is inferred. See TABLE_WORKFLOW.md.
"""

import argparse
import copy
from pathlib import Path
import sys

from billiards_map import map_camera_measurements, planner_state_from_map
from billiards_table import TableCalibration
from table_workflow import (LOCAL, atomic_write, distinct_outputs, json_bytes,
                            png_bytes, read_json, validate_pixel_frame)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-profile", type=Path, default=LOCAL / "table.json")
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--labels", type=Path, help="Reviewed JSON object mapping candidate keys to ball IDs 0..15")
    parser.add_argument("--confirmed", action="store_true", help="Confirm complete current layout and identities")
    parser.add_argument("--output", type=Path, default=LOCAL / "state.json", help="Physical map JSON")
    parser.add_argument("--image", type=Path, help="Optional overhead review image (.png)")
    parser.add_argument("--planner-state", type=Path, help="Optional planner input JSON; requires a complete confirmed map")
    parser.add_argument("--group", choices=("open", "solids", "stripes"), default="open",
                        help="Default target filter for planner export only")
    return parser


def build_physical_state(table_profile, measurement_object, *, labels=None, confirmed=False):
    size = validate_pixel_frame(measurement_object)
    signature = measurement_object["calibration_signature"]
    calibration = TableCalibration.load(table_profile, signature)
    if size != calibration.image_size:
        raise ValueError("Measurement image_size differs from the table calibration.")
    records = measurement_object.get("measurements")
    if not isinstance(records, list):
        raise ValueError("Supply a measurements list from a single stationary capture.")
    if labels is None:
        labels = {}
    if not isinstance(labels, dict):
        raise ValueError("Labels must be an object mapping candidate keys to ball IDs.")
    keys = {record.get("key") for record in records if isinstance(record, dict)
            and isinstance(record.get("key"), str)}
    if any(key not in keys for key in labels):
        raise ValueError("Label keys must refer to candidates in this measurement file.")
    result = map_camera_measurements(
        calibration, records, calibration_signature=signature, labels=labels,
        confirmed=confirmed, observed_at=measurement_object.get("observed_at"))
    result.update(calibration_signature=copy.deepcopy(signature), image_size=list(size),
                  pixel_space="rectified", observation_kind="snapshot")
    return result


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        distinct_outputs([args.table_profile, args.measurements, args.labels],
                         [args.output, args.image, args.planner_state])
        if args.image and args.image.suffix.lower() != ".png":
            raise ValueError("--image must have a .png extension.")
        if args.planner_state and not args.confirmed:
            raise ValueError("--planner-state requires --confirmed after reviewing the complete layout.")
        world = build_physical_state(
            args.table_profile, read_json(args.measurements),
            labels=read_json(args.labels) if args.labels else None, confirmed=args.confirmed)
        failed = world["status"] == "invalid" or (args.confirmed and world["status"] != "confirmed")
        artifacts = [(args.output, json_bytes(world))]
        if args.image:
            from billiards_map_view import draw_table_map
            artifacts.append((args.image, png_bytes(draw_table_map(1200, 700, world))))
        if args.planner_state and not failed:
            state = planner_state_from_map(world, group=args.group)
            state["virtual_map"] = copy.deepcopy(world)
            state["observation"] = {"source": world["source"], "reviewed": True,
                                    "recorded_unix_s": world["observed_at"], "kind": "snapshot"}
            artifacts.append((args.planner_state, json_bytes(state)))
        # Validate and encode every requested artifact before replacing any file.
        for path, content in artifacts:
            atomic_write(path, content)
        print(f"Saved {world['status']} table map: {args.output}\n"
              f"Mapped {len(world['balls'])} balls; unmapped {len(world['unmapped'])}; "
              f"{len(world['ball_separations'])} pair distances.")
        for issue in world["issues"]:
            print(f"Review: {issue}", file=sys.stderr)
        if failed and args.planner_state:
            print("Planner state was not written: the map is not complete and confirmed.", file=sys.stderr)
        return 2 if failed else 0
    except (OSError, UnicodeError, ValueError, TypeError, RuntimeError, OverflowError) as exc:
        print(f"{parser.prog}: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
