"""Plan a reviewed pool layout from a JSON file without cameras or model packages."""

import argparse
import json
from pathlib import Path
import sys

from billiards import parse_question, plan_shots


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Coordinates and distances are centimetres; x points right and y points down.
State JSON requires table and all currently present balls. Optional group is the
default target filter (solids/stripes/open; default open), not player assignment:
  {"table":{"width":254,"height":127,"ball_radius":2.8575,"pocket_width":11.5},
   "group":"solids","balls":[{"id":0,"x":127,"y":100},{"id":1,"x":127,"y":50}]}
0 identifies the cue ball; 1..15 identify numbered balls. Confirm that every ball
currently on the table is represented; no status is assigned to absent balls.
Optional table.pockets overrides: [{"id":"TM","x":127,"width":12,"blocked":true}].
Pocket IDs: TL, TM, TR, BL, BM, BR. Optional ball confidence (0..1) and
position_uncertainty_cm influence ranking and clearance. Optional planner fields
include allow_banks and position_uncertainty_cm. See BILLIARDS.md for details.

Example: python plan_billiards.py --state reviewed.json --confirmed --direct-only
""")
    parser.add_argument("--state", type=Path, required=True,
                        help="JSON file exported from a reviewed layout; no default/sample fallback")
    parser.add_argument("--confirmed", action="store_true", required=True,
                        help="Confirm the complete current layout and all on-table ball identities")
    parser.add_argument("--question", default="What is my best shot?",
                        help="Shot request, e.g. 'best shot for stripes' or 'aim for ball 3' (default: best shot)")
    parser.add_argument("--direct-only", action="store_true",
                        help="Disable one-cushion object-ball bank candidates")
    return parser


def _reject_constant(value):
    raise ValueError(f"State JSON must contain finite numbers, not {value}.")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"State JSON repeats field {key!r}.")
        result[key] = value
    return result


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        # utf-8-sig accepts ordinary UTF-8 and exports carrying a Windows BOM.
        state = json.loads(args.state.read_text(encoding="utf-8-sig"),
                           parse_constant=_reject_constant, object_pairs_hook=_unique_object)
        if not isinstance(state, dict):
            raise ValueError("Supply a table state object.")
        if args.direct_only:
            options = state.get("planner", {})
            if not isinstance(options, dict):
                raise ValueError("Planner settings must be an object.")
            if parse_question(args.question, state.get("group", "open")).get("shot_type") == "bank":
                raise ValueError("A bank-shot request conflicts with --direct-only.")
            state["planner"] = dict(options, allow_banks=False)
        result = plan_shots(state, args.question)
        # Serialize before writing so a nonfinite result cannot produce partial JSON.
        output = json.dumps(result, allow_nan=False, indent=2)
    except (OSError, UnicodeError, ValueError, OverflowError, RecursionError) as exc:
        print(f"{parser.prog}: error: {exc}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
