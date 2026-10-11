"""Exercise the headless planner as a real dependency-free command."""

import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def state():
    return {"table": {"width": 254, "height": 127, "ball_radius": 2.8575, "pocket_width": 11.5},
            "group": "solids", "balls": [{"id": 0, "x": 127, "y": 100},
                                            {"id": 1, "x": 127, "y": 50}]}


def run_cli(*args):
    # -S makes this an actual check that the CLI needs no third-party packages.
    return subprocess.run([sys.executable, "-S", str(ROOT / "plan_billiards.py"), *map(str, args)],
                          cwd=ROOT, capture_output=True, text=True, timeout=15)


def save_state(tmp_path, layout):
    path = tmp_path / "reviewed.json"
    path.write_text(json.dumps(layout), encoding="utf-8")
    return path


def assert_error(result):
    assert result.returncode != 0
    assert result.stdout == ""
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr


def test_confirmed_file_generates_real_planner_result_without_site_packages(tmp_path):
    path = save_state(tmp_path, state())
    original = path.read_bytes()
    result = run_cli("--state", path, "--confirmed")
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    plan = json.loads(result.stdout, parse_constant=lambda value: pytest.fail(f"Nonfinite output {value}"))
    assert plan["status"] == "ok" and plan["eligible_ball_ids"] == [1]
    shot = next(s for s in plan["shots"] if s["pocket_id"] == "TM" and s["shot_type"] == "direct")
    assert shot["ghost"] == {"x": 127.0, "y": 55.715}
    assert path.read_bytes() == original


def test_explicit_confirmation_is_required_even_when_file_claims_confirmed(tmp_path):
    layout = state()
    layout["confirmed"] = True
    result = run_cli("--state", save_state(tmp_path, layout))
    assert_error(result)
    assert "--confirmed" in result.stderr


def test_no_default_layout_is_used():
    result = run_cli("--confirmed")
    assert_error(result)
    assert "--state" in result.stderr


def test_direct_only_disables_banks_without_rewriting_input(tmp_path):
    path = save_state(tmp_path, state())
    result = run_cli("--state", path, "--confirmed", "--direct-only")
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["shots"] and all(shot["shot_type"] == "direct" for shot in plan["shots"])
    assert plan["planner"]["allow_banks"] is False
    assert "planner" not in json.loads(path.read_text())


def test_requested_ball_is_forwarded_and_other_balls_still_block(tmp_path):
    layout = state()
    layout["balls"].extend([{"id": 2, "x": 50, "y": 50}, {"id": 9, "x": 127, "y": 25}])
    result = run_cli("--state", save_state(tmp_path, layout), "--confirmed", "--question", "aim for ball 1")
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["eligible_ball_ids"] == [1]
    assert not any(s["pocket_id"] == "TM" and s["shot_type"] == "direct" for s in plan["shots"])


def test_same_current_layout_supports_either_request_filter_without_assignment(tmp_path):
    layout = state()
    del layout["group"]
    layout["balls"].append({"id": 9, "x": 80, "y": 60})
    path = save_state(tmp_path, layout)
    original = path.read_bytes()
    for requested, eligible in (("solids", [1]), ("stripes", [9])):
        result = run_cli("--state", path, "--confirmed", "--direct-only",
                         "--question", f"best shot for {requested}")
        assert result.returncode == 0, result.stderr
        plan = json.loads(result.stdout)
        assert plan["group"] == requested
        assert plan["eligible_ball_ids"] == eligible
    default_result = run_cli("--state", path, "--confirmed", "--direct-only")
    assert default_result.returncode == 0, default_result.stderr
    assert json.loads(default_result.stdout)["eligible_ball_ids"] == [1, 9]
    assert path.read_bytes() == original


def test_reviewed_pocket_blockage_reaches_real_planner(tmp_path):
    layout = state()
    layout["table"]["pockets"] = [{"id": "TM", "blocked": True}]
    result = run_cli("--state", save_state(tmp_path, layout), "--confirmed")
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert not any(s["pocket_id"] == "TM" for s in plan["shots"])
    assert plan["diagnostics"]["rejected"]["pocket_marked_blocked"] > 0


def test_valid_no_shot_result_is_successful_json(tmp_path):
    layout = state()
    layout["group"] = "open"
    layout["balls"][1]["id"] = 8
    result = run_cli("--state", save_state(tmp_path, layout), "--confirmed")
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["status"] == "no_shot" and plan["shots"] == []


@pytest.mark.parametrize("contents", ["{broken json", "[]", "{}", '{"table":NaN}',
                                      '{"table":Infinity}', '{"group":"solids","group":"stripes"}'])
def test_invalid_json_and_layout_errors_are_concise(tmp_path, contents):
    path = tmp_path / "invalid.json"
    path.write_text(contents, encoding="utf-8")
    assert_error(run_cli("--state", path, "--confirmed"))


def test_missing_file_is_concise_error(tmp_path):
    assert_error(run_cli("--state", tmp_path / "missing.json", "--confirmed"))


def test_invalid_encoding_is_concise_error(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_bytes(b"\xff\xfe")
    assert_error(run_cli("--state", path, "--confirmed"))


def test_windows_utf8_bom_is_accepted(tmp_path):
    path = tmp_path / "windows.json"
    path.write_text(json.dumps(state()), encoding="utf-8-sig")
    assert run_cli("--state", path, "--confirmed").returncode == 0


def test_unsupported_question_and_direct_only_bank_conflict_fail(tmp_path):
    path = save_state(tmp_path, state())
    assert_error(run_cli("--state", path, "--confirmed", "--question", "make a jump shot"))
    result = run_cli("--state", path, "--confirmed", "--direct-only", "--question", "best bank shot")
    assert_error(result)
    assert "conflicts" in result.stderr


def test_direct_only_does_not_silently_replace_malformed_planner_settings(tmp_path):
    layout = state()
    layout["planner"] = []
    assert_error(run_cli("--state", save_state(tmp_path, layout), "--confirmed", "--direct-only"))


def test_out_of_range_integer_is_concise_error(tmp_path):
    layout = state()
    layout["table"]["width"] = 10 ** 400
    assert_error(run_cli("--state", save_state(tmp_path, layout), "--confirmed"))


def test_help_documents_actual_state_schema_without_loading_a_file():
    result = run_cli("--help")
    assert result.returncode == 0 and result.stderr == ""
    assert "--confirmed" in result.stdout and "position_uncertainty_cm" in result.stdout
    assert '"ball_radius"' in result.stdout and "TL, TM, TR, BL, BM, BR" in result.stdout
