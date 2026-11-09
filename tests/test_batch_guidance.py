"""Navigation recommends narrow batch reads without inventing inspected files."""

from copy import deepcopy
import json
import shutil

import pytest

from patchrook_agent.tools import ToolConfig, ToolRegistry, Workspace
from patchrook_agent.tools.base import observation_chars, with_read_guidance
from patchrook_agent.tools.registry import bounded_result


def search_result(paths):
    return {"success": True, "matches": [
        {"path": path, "line_number": 14 + index, "text": "def transform(value):"}
        for index, path in enumerate(paths)
    ], "truncated": False}


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep is required")
def test_search_of_eight_known_paths_suggests_one_narrow_read_files_call(tmp_path):
    for index in range(8):
        (tmp_path / f"stage{index}.py").write_text("# context\n" * 13 + "def transform(value):\n    return value\n")
    tools = ToolRegistry(Workspace(tmp_path))
    result = tools.dispatch("search_code", {"pattern": "def transform"})
    assert len(result["matches"]) == 8
    suggestion = result["suggested_read"]
    assert suggestion["name"] == "read_files"
    assert suggestion["arguments"] == {"files": [
        {"path": f"stage{index}.py", "start_line": 14, "end_line": 19} for index in range(8)
    ]}
    assert "read_files" in result["read_hint"]
    assert observation_chars(result) <= 2000
    # Advice did not execute any follow-up read or claim those files inspected.
    assert tools.last_execution["name"] == "search_code"
    assert "content" not in tools.last_execution["available_output"]


def test_suggestions_deduplicate_paths_and_limit_each_read_to_six_lines():
    result = search_result(["a.py", "a.py", "b.py"])
    projected = with_read_guidance(result, 2000)
    files = projected["suggested_read"]["arguments"]["files"]
    assert files == [
        {"path": "a.py", "start_line": 14, "end_line": 19},
        {"path": "b.py", "start_line": 16, "end_line": 21},
    ]
    assert result == search_result(["a.py", "a.py", "b.py"])


def test_ready_arguments_stop_at_ten_distinct_visible_paths():
    result = with_read_guidance(search_result([f"stage{number}.py" for number in range(14)]), 10000)
    files = result["suggested_read"]["arguments"]["files"]
    assert len(files) == 10 and len({selection["path"] for selection in files}) == 10
    assert [selection["path"] for selection in files] == [match["path"] for match in result["matches"][:10]]


def test_optional_arguments_shrink_before_displacing_any_match_evidence():
    raw = search_result([f"stage{number}.py" for number in range(8)])
    full = with_read_guidance(raw, 10000)
    limit = observation_chars(full) - 150
    result = bounded_result(raw, limit)
    assert result["matches"] == raw["matches"]
    assert not result["truncated"]
    files = result["suggested_read"]["arguments"]["files"]
    assert 2 <= len(files) < 8
    assert {selection["path"] for selection in files} <= {match["path"] for match in result["matches"]}
    assert observation_chars(result) <= limit


def test_final_truncation_discards_stale_arguments_for_unshown_paths():
    raw = search_result([f"stage{number}_{'x' * 80}.py" for number in range(10)])
    raw["returned_count"] = 10
    raw["offset"] = 0
    stale = deepcopy(raw)
    stale["read_hint"] = "Read hidden.py next"
    stale["suggested_read"] = {"name": "read_files", "arguments": {"files": [{"path": "hidden.py"}]}}
    result = bounded_result(stale, 512)
    evidence_only = bounded_result(raw, 512)
    assert result["matches"] == evidence_only["matches"]
    assert 0 < len(result["matches"]) < 10
    assert result["next_offset"] == len(result["matches"])
    assert "hidden.py" not in json.dumps(result)
    suggestion = result.get("suggested_read")
    if suggestion:
        selections = suggestion["arguments"].get("files", [suggestion["arguments"]])
        assert {selection["path"] for selection in selections} <= {match["path"] for match in result["matches"]}
    assert observation_chars(result) <= 512


def test_single_visible_file_uses_read_file_instead_of_fabricating_a_batch():
    result = bounded_result(search_result(["only.py"]), 1000)
    assert result["suggested_read"] == {
        "name": "read_file", "arguments": {"path": "only.py", "start_line": 14, "end_line": 19},
    }


@pytest.mark.parametrize("result", [
    {"success": True, "matches": [], "truncated": False},
    {"success": False, "matches": [{"path": "a.py", "line_number": 1}], "error": {"code": "timeout"}},
    {"success": True, "matches": [{"path": "a.py", "line_number": True}], "truncated": False},
])
def test_empty_failed_or_invalid_search_has_no_invented_read_suggestion(result):
    projected = with_read_guidance(result, 2000)
    assert "suggested_read" not in projected and "read_hint" not in projected


def test_listing_points_to_narrow_batch_reads_without_inventing_line_locations(tmp_path):
    (tmp_path / "a.py").write_text("a = 1\n")
    (tmp_path / "b.py").write_text("b = 2\n")
    tools = ToolRegistry(Workspace(tmp_path, ToolConfig(max_output_chars=512)))
    result = tools.dispatch("list_files", {})
    assert result["entries"] == ["a.py", "b.py"]
    assert "read_files" in result["read_hint"] and "search_code" in result["read_hint"]
    assert "suggested_read" not in result
    assert observation_chars(result) <= 512
