"""A bounded read batch preserves source, failures, and complete raw captures."""

from copy import deepcopy
import json

import pytest

from patchrook_agent.tools import ToolConfig, ToolRegistry, Workspace
from patchrook_agent.tools.base import TOOL_CAPTURE, ToolError, bound_read_files_result
from patchrook_agent.tools.files import read_files


def registry(tmp_path, **config):
    tools = ToolRegistry(Workspace(tmp_path, ToolConfig(**config)))
    tools.capture_enabled = True
    return tools


def size(result):
    return len(json.dumps(result, ensure_ascii=False, separators=(",", ":")))


def test_batch_reads_exact_targeted_numbered_excerpts_in_request_order(tmp_path):
    (tmp_path / "a.py").write_text("first\nsecond\nthird\n")
    (tmp_path / "b.py").write_text("λ exact text\nlast\n")
    result = registry(tmp_path).dispatch("read_files", {"files": [
        {"path": "b.py", "start_line": 1, "end_line": 1},
        {"path": "a.py", "start_line": 2, "end_line": 3},
    ]})
    assert result["success"] and not result["truncated"]
    assert result["requested_count"] == result["returned_count"] == 2
    assert [item["path"] for item in result["files"]] == ["b.py", "a.py"]
    assert [item["content"] for item in result["files"]] == ["1: λ exact text", "2: second\n3: third"]


@pytest.mark.parametrize("overhead", [0, 160])
def test_eight_narrow_source_excerpts_fit_in_one_observation(tmp_path, overhead):
    selections = []
    for number in range(8):
        name = f"stage{number}.py"
        (tmp_path / name).write_text(f"def transform(value):\n    return value + {number}\n")
        selections.append({"path": name, "start_line": 1, "end_line": 2})
    tools = registry(tmp_path)
    tools.observation_overhead_chars = overhead
    result = tools.dispatch("read_files", {"files": selections})
    assert result["returned_count"] == 8 and result["requested_count"] == 8
    assert [item["content"] for item in result["files"]] == [
        f"1: def transform(value):\n2:     return value + {number}" for number in range(8)
    ]
    assert size(result) <= 2000 - overhead


@pytest.mark.parametrize("arguments", [
    {"files": []}, {"files": [{"path": "a.py"}] * 11},
    {"files": "a.py"}, {"files": ["a.py"]}, {"files": [{}]},
    {"files": [{"path": 7}]}, {"files": [{"path": "a.py", "start_line": True}]},
    {"files": [{"path": "a.py", "end_line": "2"}]},
    {"files": [{"path": "a.py", "command": "cat"}]},
    {"files": [{"path": "a.py"}], "recursive": True},
])
def test_batch_arguments_remain_strict_and_bounded(tmp_path, arguments):
    result = registry(tmp_path).dispatch("read_files", arguments)
    assert result["success"] is False and result["error"]["code"] == "invalid_arguments"


def test_batch_exposes_per_file_failures_without_hiding_valid_reads(tmp_path):
    (tmp_path / "a.py").write_text("first\nsecond\n")
    result = registry(tmp_path).dispatch("read_files", {"files": [
        {"path": "a.py", "start_line": 2, "end_line": 2},
        {"path": "missing.py"},
        {"path": "a.py", "start_line": 2, "end_line": 1},
    ]})
    assert result["success"] is False and result["failed_count"] == 2
    assert result["files"][0]["content"] == "2: second"
    assert result["files"][1]["path"] == "missing.py"
    assert result["files"][1]["error"]["code"] == "not_found"
    assert result["files"][2]["error"]["code"] == "invalid_range"


def test_failed_batch_paths_use_safe_canonical_identity_for_later_recovery(tmp_path):
    (tmp_path / "source.py").write_text("first\n")
    result = registry(tmp_path).dispatch("read_files", {"files": [
        {"path": "./missing.py"},
        {"path": "./source.py", "start_line": 3},
        {"path": "../outside.py"},
    ]})
    assert [item["path"] for item in result["files"]] == ["missing.py", "source.py", "../outside.py"]
    assert [item["error"]["code"] for item in result["files"]] == ["not_found", "invalid_range", "outside_workspace"]


def test_batch_keeps_workspace_traversal_and_symlink_restrictions(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "safe.py").write_text("safe\n")
    outside = tmp_path / "outside.py"
    outside.write_text("private source\n")
    (repo / "escape.py").symlink_to(outside)
    result = registry(repo).dispatch("read_files", {"files": [
        {"path": "safe.py"}, {"path": "../outside.py"}, {"path": "escape.py"},
    ]})
    assert not result["success"] and result["failed_count"] == 2
    assert result["files"][0]["content"] == "1: safe"
    assert all(item["error"]["code"] == "outside_workspace" for item in result["files"][1:])
    assert "private source" not in json.dumps(result)


def test_batch_handles_empty_files_and_canonical_paths(tmp_path):
    (tmp_path / "empty.py").write_text("")
    (tmp_path / "alias.py").symlink_to(tmp_path / "empty.py")
    result = registry(tmp_path).dispatch("read_files", {"files": [
        {"path": "empty.py"}, {"path": "alias.py"},
    ]})
    assert result["success"] and result["returned_count"] == 2
    assert all(item["path"] == "empty.py" and item["content"] == "" and item["total_lines"] == 0
               for item in result["files"])


def test_truncated_batch_retains_each_full_selected_capture_separately(tmp_path):
    expected = {}
    for name in ("a.py", "b.py", "c.py"):
        lines = [f"{name}_{number} = '{'x' * 45}'" for number in range(30)]
        (tmp_path / name).write_text("\n".join(lines) + "\n")
        expected[name] = "\n".join(f"{number}: {line}" for number, line in enumerate(lines, 1))
    tools = registry(tmp_path)
    tools.observation_overhead_chars = 160
    result = tools.dispatch("read_files", {"files": [{"path": name} for name in expected]})
    assert result["success"] and result["truncated"] and size(result) <= 1840
    for item in result["files"]:
        assert item["content"] and expected[item["path"]].startswith(item["content"])
        if not item.get("partial_line"):
            assert item["next_start_line"] == item["end_line"] + 1
    captured = tools.last_execution["available_output"]["files"]
    assert [item["path"] for item in captured] == list(expected)
    assert {item["path"]: item["available_output"]["content"] for item in captured} == expected
    assert TOOL_CAPTURE.get() is None


def test_small_observation_omits_whole_file_records_explicitly(tmp_path):
    selections = []
    for number in range(10):
        name = f"stage{number}.py"
        (tmp_path / name).write_text("source_value = '" + "x" * 200 + "'\n")
        selections.append({"path": name})
    tools = registry(tmp_path, max_output_chars=512)
    tools.observation_overhead_chars = 160
    result = tools.dispatch("read_files", {"files": selections})
    assert result["success"] and result["truncated"] and size(result) <= 352
    assert result["omitted_file_count"] + result["returned_count"] == 10
    assert result["next_file_index"] == result["returned_count"]
    assert result["returned_count"] == len(result["files"])
    assert [item["path"] for item in result["files"]] == [item["path"] for item in selections[:len(result["files"])]]
    assert len(tools.last_execution["available_output"]["files"]) == 10


def test_schema_inlines_strict_nested_selection_instead_of_unresolved_refs(tmp_path):
    definition = next(item["function"] for item in registry(tmp_path).definitions()
                      if item["function"]["name"] == "read_files")
    parameters = definition["parameters"]
    selection = parameters["properties"]["files"]["items"]
    assert parameters["additionalProperties"] is False
    assert selection["additionalProperties"] is False and selection["required"] == ["path"]
    assert selection["properties"]["path"]["type"] == "string"
    assert "$ref" not in json.dumps(parameters) and "$defs" not in json.dumps(parameters)


def test_direct_batch_rejects_empty_or_excessive_selections(tmp_path):
    workspace = Workspace(tmp_path)
    for selections in ([], [{"path": "a.py"}] * 11):
        with pytest.raises(ToolError):
            read_files(workspace, selections)


def test_oversized_batch_completes_earlier_requested_ranges_instead_of_showing_comment_prefixes(tmp_path):
    sources = {}
    for index in range(8):
        path = f"module_{index}.py"
        lines = [f"# Context {number}: " + "explanation " * 5 for number in range(8)]
        lines += ["def transform(value):", f"    return value + {index}"]
        (tmp_path / path).write_text("\n".join(lines) + "\n")
        sources[path] = "\n".join(f"{number}: {line}" for number, line in enumerate(lines, 1))
    tools = registry(tmp_path)
    tools.observation_overhead_chars = 160
    result = tools.dispatch("read_files", {"files": [{"path": path} for path in sources]})
    assert size(result) <= 1840 and result["truncated"]
    first = result["files"][0]
    assert first["path"] == "module_0.py" and first["content"] == sources[first["path"]]
    assert "return value + 0" in first["content"] and not first["truncated"]
    assert result["next_file_index"] == result["returned_count"]
    assert result["omitted_file_count"] + result["returned_count"] == 8
    for item in result["files"][:-1]:
        assert item["content"] == sources[item["path"]]
    capture = tools.last_execution["available_output"]["files"]
    assert {item["path"]: item["available_output"]["content"] for item in capture} == sources


def test_completed_prefix_and_one_partial_range_have_independent_truthful_continuations(tmp_path):
    (tmp_path / "first.py").write_text("def first(value):\n    return value + 1\n")
    long_lines = [f"value_{number} = '{'x' * 45}'" for number in range(50)]
    (tmp_path / "second.py").write_text("\n".join(long_lines) + "\n")
    (tmp_path / "third.py").write_text("value = 3\n")
    tools = registry(tmp_path)
    tools.observation_overhead_chars = 160
    result = tools.dispatch("read_files", {"files": [{"path": path} for path in ("first.py", "second.py", "third.py")]})
    assert [item["path"] for item in result["files"]] == ["first.py", "second.py"]
    first, last = result["files"]
    assert not first["truncated"] and first["end_line"] == 2
    assert last["truncated"] and not last.get("partial_line")
    assert last["content"] == "\n".join(f"{number}: {line}" for number, line in enumerate(long_lines[:last["end_line"]], 1))
    assert last["next_start_line"] == last["end_line"] + 1
    assert result["omitted_file_count"] == 1 and result["next_file_index"] == 2
    assert size(result) <= 1840


@pytest.mark.parametrize("limit", [512, 1840, 2000])
def test_batch_fitting_is_idempotent_and_rebounding_preserves_prefix_and_source(tmp_path, limit):
    for number in range(5):
        (tmp_path / f"file{number}.py").write_text("\n".join(f"line_{index} = '{'z' * 35}'" for index in range(12)) + "\n")
    tools = registry(tmp_path)
    tools.dispatch("read_files", {"files": [{"path": f"file{number}.py"} for number in range(5)]})
    originals = [item["result"] for item in tools.last_execution["available_output"]["files"]]
    raw = {"success": True, "files": deepcopy(originals), "requested_count": 5, "returned_count": 5, "truncated": False}
    saved = deepcopy(raw)
    full = bound_read_files_result(raw, 2000)
    projected = bound_read_files_result(full, limit)
    assert projected == bound_read_files_result(raw, limit)
    assert projected == bound_read_files_result(projected, limit)
    assert raw == saved and size(projected) <= limit
    count = projected["returned_count"]
    assert [item["path"] for item in projected["files"]] == [item["path"] for item in originals[:count]]
    assert projected["omitted_file_count"] + count == 5
    assert projected["next_file_index"] == count
    for item in projected["files"]:
        assert item["content"] and originals[int(item["path"][4])]["content"].startswith(item["content"])


def test_omitting_a_failed_tail_record_does_not_turn_a_partial_batch_into_success(tmp_path):
    (tmp_path / "large.py").write_text("\n".join(f"line_{index} = '{'x' * 60}'" for index in range(100)) + "\n")
    tools = registry(tmp_path)
    result = tools.dispatch("read_files", {"files": [{"path": "large.py"}, {"path": "missing.py"}]})
    assert not result["success"] and result["failed_count"] == 1
    assert result["omitted_file_count"] == 1 and result["returned_count"] == 1
    assert result["files"][0]["success"]
    assert tools.last_execution["available_output"]["files"][1]["result"]["error"]["code"] == "not_found"


def test_one_long_line_keeps_a_partial_source_prefix_and_its_capture_at_minimum_budget(tmp_path):
    source = "message = '" + "λ" * 5000 + "'"
    (tmp_path / "line.py").write_text(source + "\n")
    tools = registry(tmp_path, max_output_chars=512)
    tools.observation_overhead_chars = 160
    result = tools.dispatch("read_files", {"files": [{"path": "line.py"}]})
    assert result["returned_count"] == 1 and result["truncated"] and size(result) <= 352
    item = result["files"][0]
    assert item["partial_line"] and item["line_chars_returned"] > 0
    assert item["next_start_line"] == 1 and ("1: " + source).startswith(item["content"])
    assert tools.last_execution["available_output"]["files"][0]["available_output"]["content"] == "1: " + source
