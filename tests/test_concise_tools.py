"""Observation bounds preserve exact evidence and usable continuations."""

import json
import shutil

import pytest

from patchrook_agent.tools import ToolConfig, ToolRegistry, Workspace
from patchrook_agent.tools.registry import bounded_result
from patchrook_agent.tools.tests import _concise_output


def tools(tmp_path, **config):
    registry = ToolRegistry(Workspace(tmp_path, ToolConfig(**config)))
    registry.capture_enabled = True
    return registry


def assert_bounded(observation, limit):
    assert len(json.dumps(observation, ensure_ascii=False, separators=(",", ":"))) <= limit


def test_listing_continuations_cover_every_entry_after_final_observation_bound(tmp_path):
    for directory in ("a", "b"):
        (tmp_path / directory).mkdir()
        for number in range(12):
            (tmp_path / directory / f"{number:02d}_{'x' * 40}.py").write_text("")
    registry = tools(tmp_path, max_output_chars=512)
    seen = []
    offset = 0
    for _ in range(40):
        observation = registry.dispatch("list_files", {"recursive": True, "offset": offset})
        assert_bounded(observation, 512)
        seen.extend(observation["entries"])
        assert observation["returned_count"] == len(observation["entries"])
        if not observation["truncated"]:
            break
        assert observation["next_offset"] > offset
        offset = observation["next_offset"]
    else:
        pytest.fail("listing continuation did not finish")
    expected = ["a/", "b/"] + [f"{directory}/{number:02d}_{'x' * 40}.py"
                                  for directory in ("a", "b") for number in range(12)]
    assert seen == expected
    assert len(set(seen)) == len(seen)


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep is required")
def test_search_pages_beyond_per_file_limit_without_missing_or_repeating_matches(tmp_path):
    for name in ("a.py", "b.py"):
        (tmp_path / name).write_text("needle\n" * 13)
    registry = tools(tmp_path, max_output_chars=512, max_search_results=5)
    found = []
    offset = 0
    for _ in range(30):
        observation = registry.dispatch("search_code", {"pattern": "needle", "offset": offset})
        assert_bounded(observation, 512)
        found.extend((match["path"], match["line_number"]) for match in observation["matches"])
        if not observation["truncated"]:
            break
        assert observation["next_offset"] > offset
        offset = observation["next_offset"]
    else:
        pytest.fail("search continuation did not finish")
    assert found == [(name, line) for name in ("a.py", "b.py") for line in range(1, 14)]


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep is required")
def test_long_search_excerpt_contains_actual_match_and_exact_source(tmp_path):
    source = "a" * 600 + "needle" + "z" * 400
    (tmp_path / "long.py").write_text(source + "\n")
    observation = tools(tmp_path).dispatch("search_code", {"pattern": "needle"})
    match = observation["matches"][0]
    assert match["path"] == "long.py" and match["line_number"] == 1
    assert "needle" in match["text"]
    start = match["column_start"] - 1
    assert source[start:start + len(match["text"])] == match["text"]
    assert match["text_truncated"] and observation["truncated"]


def test_read_continuations_preserve_exact_complete_lines_with_artifact_overhead(tmp_path):
    lines = [f"symbol_{index} = '{'x' * 35}'" for index in range(20)]
    (tmp_path / "source.py").write_text("\n".join(lines) + "\n")
    registry = tools(tmp_path, max_output_chars=512)
    registry.observation_overhead_chars = 100
    collected = []
    start = 1
    for _ in range(25):
        observation = registry.dispatch("read_file", {"path": "source.py", "start_line": start})
        assert_bounded(observation, 412)
        assert not observation.get("partial_line")
        expected = [f"{number}: {lines[number - 1]}"
                    for number in range(start, observation["end_line"] + 1)]
        assert observation["content"] == "\n".join(expected)
        collected.extend(expected)
        if not observation["truncated"]:
            break
        start = observation["next_start_line"]
    assert collected == [f"{number}: {line}" for number, line in enumerate(lines, 1)]


def test_single_long_line_is_explicitly_partial_and_full_selected_text_is_captured(tmp_path):
    source = "UNIQUE_PREFIX = '" + "x" * 5000 + "'"
    (tmp_path / "long.py").write_text(source + "\n")
    registry = tools(tmp_path, max_output_chars=512)
    observation = registry.dispatch("read_file", {"path": "long.py", "start_line": 1, "end_line": 1})
    assert_bounded(observation, 512)
    assert observation["truncated"] and observation["partial_line"]
    assert observation["next_start_line"] == 1
    assert observation["line_chars_returned"] == len(observation["content"]) - 3
    assert ("1: " + source).startswith(observation["content"])
    assert registry.last_execution["available_output"]["content"] == "1: " + source


def test_error_bound_marks_partial_message_without_changing_path_or_error_code():
    path = "deep/" + "x" * 180 + ".py"
    observation = bounded_result({"success": False, "path": path,
                                  "error": {"code": "search_failed", "message": "exact error " * 300}}, 512)
    assert_bounded(observation, 512)
    assert observation["path"] == path
    assert observation["error"]["code"] == "search_failed"
    assert observation["error"]["message_truncated"]
    assert ("exact error " * 300).startswith(observation["error"]["message"])


def test_pytest_observation_keeps_exact_debug_frame_counts_and_full_raw_output(tmp_path):
    (tmp_path / "helper.py").write_text(
        "def calculate():\n    actual = 6\n    raise ValueError('exact failure code 731')\n"
    )
    (tmp_path / "test_example.py").write_text(
        "from helper import calculate\n"
        "def test_broken():\n    print('x' * 12000)\n    calculate()\n"
        "def test_ok():\n    assert True\n"
    )
    registry = tools(tmp_path, allow_tests=True)
    observation = registry.dispatch("run_tests", {"path": "test_example.py"})
    assert_bounded(observation, 2000)
    assert observation["status"] == "failed" and observation["exit_code"] == 1
    assert observation["test_counts"] == {"failed": 1, "passed": 1}
    assert "helper.py:3: in calculate" in observation["stdout"]
    assert "E   ValueError: exact failure code 731" in observation["stdout"]
    assert "raise ValueError('exact failure code 731')" in observation["stdout"]
    assert observation["truncated"]
    available = registry.last_execution["available_output"]
    assert len(available["stdout"]) > 12000
    assert "x" * 12000 in available["stdout"]


def test_oversized_exception_keeps_exact_prefix_and_useful_frame():
    output, omitted = _concise_output(
        "helper.py:3: in calculate\n    raise ValueError(message)\n"
        "E   ValueError: " + "details " * 1000 + "\n1 failed in 0.1s\n", 512)
    assert omitted and len(output) <= 512
    assert "helper.py:3: in calculate" in output
    assert "E   ValueError: details details" in output
    assert "1 failed in 0.1s" in output
