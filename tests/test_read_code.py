"""Lean code inspection selects exact useful source without executing it."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from patchrook_agent.tools import ToolConfig, ToolRegistry, Workspace
from patchrook_agent.tools.base import bound_read_files_result, observation_chars
from patchrook_agent.tools.code import read_code


def registry(tmp_path, *, limit=2000, overhead=160):
    tools = ToolRegistry(Workspace(tmp_path, ToolConfig(max_output_chars=limit)))
    tools.capture_enabled = True
    tools.observation_overhead_chars = overhead
    tools.enable_lean_tools()
    tools.guidance_tools = {item["function"]["name"] for item in tools.definitions()}
    return tools


def read(tools, *files):
    return tools.dispatch("read_code", {"paths": [item["path"] for item in files]})


def selected(tools, *files):
    return read_code(tools.workspace, list(files), observation_limit=1840)


def test_eight_comment_heavy_files_show_complete_two_line_functions_in_one_managed_observation(tmp_path):
    originals = {}
    for index in range(8):
        path = f"part{index}.py"
        source = "# repeated module documentation\n" * 30 + f"def apply(value):\n    return value + {index}\n"
        (tmp_path / path).write_text(source)
        originals[path] = source
    tools = registry(tmp_path)
    result = read(tools, *({"path": path} for path in originals))
    assert result["success"] and result["selection"] == "code"
    assert result["returned_count"] == 8 and observation_chars(result) <= 1840
    assert [item["content"] for item in result["files"]] == [
        f"31: def apply(value):\n32:     return value + {index}" for index in range(8)
    ]
    assert all(item["omitted_ranges"] == [[1, 30]] and not item["truncated"] for item in result["files"])
    captures = tools.last_execution["available_output"]["files"]
    assert {item["path"]: item["available_output"]["source"] for item in captures} == originals
    assert all((tmp_path / path).read_text() == source for path, source in originals.items())


def test_symbol_selection_retains_decorators_docstring_invariants_imports_and_body_without_execution(tmp_path):
    source = (
        "import os\n"
        "# Keep negative values unchanged.\n"
        "@decorate\n"
        "def selected(value):\n"
        "    \"\"\"Return a normalized result.\"\"\"\n"
        "    # The result must preserve the sign.\n"
        "    return value + 1\n"
        "\n"
        "raise RuntimeError('source must never execute')\n"
    )
    (tmp_path / "logic.py").write_text(source)
    result = selected(registry(tmp_path), {"path": "logic.py", "symbol": "selected"})
    item = result["files"][0]
    assert item["start_line"] == 1 and item["end_line"] == 7
    assert item["content"] == "\n".join(f"{number}: {line}" for number, line in enumerate(source.splitlines()[:7], 1))
    assert item["omitted_ranges"] == [[8, 9]] and not item["truncated"]


def test_qualified_method_and_explicit_ranges_work_at_arbitrary_locations(tmp_path):
    source = "# header\n" * 80 + "class Example:\n    def handle(self):\n        return 9\n"
    (tmp_path / "nested.py").write_text(source)
    tools = registry(tmp_path)
    item = selected(tools, {"path": "nested.py", "symbol": "Example.handle"})["files"][0]
    assert "82:     def handle(self):\n83:         return 9" in item["content"]
    ranged = selected(tools, {"path": "nested.py", "start_line": 81, "end_line": 83})["files"][0]
    assert ranged["start_line"] == 81 and ranged["end_line"] == 83
    assert ranged["content"] == "81: class Example:\n82:     def handle(self):\n83:         return 9"


def test_small_importing_module_is_shown_whole_including_every_import(tmp_path):
    lines = [f"from helper{number} import operation as op{number}" for number in range(8)]
    lines += ["", "def run(value):", "    for operation in (op0, op1, op2, op3, op4, op5, op6, op7):",
              "        value = operation(value)", "    return value", "", "RESULT = run(0)"]
    (tmp_path / "entry.py").write_text("\n".join(lines) + "\n")
    item = read(registry(tmp_path), {"path": "entry.py"})["files"][0]
    assert item["start_line"] == 1 and item["end_line"] == len(lines)
    assert item["content"] == "\n".join(f"{number}: {line}" for number, line in enumerate(lines, 1))
    assert not item["truncated"] and "omitted_ranges" not in item


def test_large_module_docstrings_do_not_displace_complete_functions_in_a_batch(tmp_path):
    for index in range(8):
        (tmp_path / f"part{index}.py").write_text('"""' + "module documentation " * 100 + '\"\"\"\n'
                                                 + f"def apply(value):\n    return value + {index}\n")
    result = read(registry(tmp_path), *({"path": f"part{index}.py"} for index in range(8)))
    assert result["returned_count"] == 8 and observation_chars(result) <= 1840
    assert all(not item["truncated"] and "return value +" in item["content"] for item in result["files"])


@pytest.mark.parametrize("name,source", [
    ("notes.txt", "# header\n" * 40 + "useful text\nsecond line\n"),
    ("broken.py", "# header\n" * 40 + "def unfinished(\n    meaningful_tail\n"),
])
def test_text_and_syntax_error_fallbacks_are_small_exact_numbered_ranges(tmp_path, name, source):
    (tmp_path / name).write_text(source)
    result = read(registry(tmp_path), {"path": name})
    item = result["files"][0]
    assert item["success"] and item["start_line"] <= 41 and item["end_line"] == 42
    assert "41: " + source.splitlines()[40] in item["content"]
    assert item["content"] == "\n".join(f"{number}: {source.splitlines()[number - 1]}"
                                           for number in range(item["start_line"], item["end_line"] + 1))


def test_large_selected_function_has_truthful_selected_end_after_final_bounding(tmp_path):
    lines = ["def calculate(value):"] + [f"    value += {number}  # {'x' * 45}" for number in range(80)] + ["    return value"]
    (tmp_path / "large.py").write_text("\n".join(lines) + "\n")
    tools = registry(tmp_path, limit=512)
    result = read(tools, {"path": "large.py"})
    assert result["truncated"] and observation_chars(result) <= 352
    item = result["files"][0]
    assert item["selected_end_line"] == 82 and item["end_line"] < 82
    assert item["next_start_line"] == item["end_line"] + 1
    assert item["content"] == "\n".join(f"{number}: {lines[number - 1]}" for number in range(1, item["end_line"] + 1))
    assert bound_read_files_result(deepcopy(result), 352) == result


def test_per_file_errors_and_workspace_denials_do_not_read_external_source(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "private.py"
    outside.write_text("PRIVATE_SOURCE = True\n")
    (repo / "escape.py").symlink_to(outside)
    (repo / "ok.py").write_text("def good():\n    return 1\n")
    result = selected(registry(repo), {"path": "ok.py"}, {"path": "escape.py"}, {"path": "missing.py"},
                  {"path": "ok.py", "symbol": "absent"})
    assert not result["success"] and result["failed_count"] == 3
    assert result["files"][0]["success"]
    assert [item["error"]["code"] for item in result["files"][1:]] == ["outside_workspace", "not_found", "symbol_not_found"]
    assert "PRIVATE_SOURCE" not in json.dumps(result)


@pytest.mark.parametrize("arguments", [
    {"paths": []}, {"paths": ["a.py"] * 11}, {"paths": [1]}, {"paths": None},
    {"paths": ["a.py"], "command": "cat"},
])
def test_read_code_uses_strict_nested_registry_validation(tmp_path, arguments):
    result = registry(tmp_path).dispatch("read_code", arguments)
    assert not result["success"] and result["error"]["code"] == "invalid_arguments"


def test_invalid_range_or_ambiguous_symbol_is_a_structured_file_error(tmp_path):
    (tmp_path / "same.py").write_text("class A:\n    def method(self):\n        return 1\nclass B:\n    def method(self):\n        return 2\n")
    result = selected(registry(tmp_path), {"path": "same.py", "symbol": "method"},
                  {"path": "same.py", "start_line": 5, "end_line": 2},
                  {"path": "same.py", "symbol": "A", "start_line": 1})
    assert [item["error"]["code"] for item in result["files"]] == ["ambiguous_symbol", "invalid_range", "invalid_selection"]


def test_lean_definitions_are_short_enabled_only_and_standard_catalog_is_unchanged(tmp_path):
    tools = ToolRegistry(Workspace(tmp_path))
    standard = tools.definitions()
    assert "read_code" not in json.dumps(standard)
    lean = tools.definitions(lean=True)
    assert {item["function"]["name"] for item in lean} == {"read_code", "read_file", "search_code", "list_files", "write_file", "edit_file", "run_tests"}
    code_schema = next(item["function"]["parameters"] for item in lean if item["function"]["name"] == "read_code")
    assert code_schema["required"] == ["paths"] and set(code_schema["properties"]) == {"paths"}
    assert code_schema["properties"]["paths"]["items"] == {"type": "string"}
    assert all(len(item["function"]["description"]) < 100 for item in lean)
    restricted = tools.definitions(enabled_tools={"search_code", "list_files"})
    assert {item["function"]["name"] for item in restricted} == {"search_code", "list_files"}
    assert "read_code" not in json.dumps(restricted)
    assert tools.definitions(lean=False) == standard


def test_lean_catalog_keeps_only_actually_attached_optional_helpers(tmp_path):
    tools = ToolRegistry(Workspace(tmp_path))
    assert len(tools.definitions(lean=True)) == 7
    tools.attach_skills(SimpleNamespace(enabled=True, load=lambda **kwargs: {}))
    tools.attach_artifacts(SimpleNamespace(read=lambda **kwargs: {}))
    names = {item["function"]["name"] for item in tools.definitions()}
    assert names == {"read_code", "read_file", "search_code", "list_files", "write_file", "edit_file", "run_tests", "read_skill", "read_artifact"}
    restricted = tools.definitions(enabled_tools={"read_code", "read_skill"})
    assert {item["function"]["name"] for item in restricted} == {"read_code", "read_skill"}


@pytest.mark.parametrize("arguments", [
    {"files": [{"path": "logic.py", "symbol": "does_not_exist", "start_line": -3, "end_line": False}]},
    {"paths": ["logic.py"], "symbol": "does_not_exist"},
    {"paths": [{"path": "logic.py", "symbol": "absent"}]},
])
def test_legacy_or_extra_read_selectors_are_explicitly_normalized_to_automatic_code(tmp_path, arguments):
    original = "# header\n" * 20 + "def operation(value):\n    return value + 1\n"
    (tmp_path / "logic.py").write_text(original)
    tools = registry(tmp_path)
    result = tools.dispatch("read_code", arguments)
    assert result["success"] and result["normalization"] == "used paths; ignored selector hints"
    assert "return value + 1" in result["files"][0]["content"]
    assert observation_chars(result) <= 1840
    assert (tmp_path / "logic.py").read_text() == original


def test_invalid_read_arguments_return_a_bounded_corrected_call_only_for_safe_existing_paths(tmp_path):
    (tmp_path / "logic.py").write_text("value = 1\n")
    tools = registry(tmp_path)
    result = tools.dispatch("read_code", {"paths": "./logic.py"})
    assert not result["success"] and result["error"]["code"] == "invalid_arguments"
    assert result["suggested_call"] == {"name": "read_code", "arguments": {"paths": ["logic.py"]}}
    denied = tools.dispatch("read_code", {"paths": "../outside.py"})
    assert "suggested_call" not in denied
    assert observation_chars(result) <= 1840


def test_read_code_search_guidance_uses_paths_only_and_keeps_exact_ranges_separate(tmp_path):
    tools = registry(tmp_path)
    from patchrook_agent.tools.base import with_read_guidance
    result = with_read_guidance({"success": True, "matches": [
        {"path": "a.py", "line_number": 14, "text": "def operation(value):"},
        {"path": "b.py", "line_number": 30, "text": "def operation(value):"},
    ]}, 1840, enabled_tools=tools.guidance_tools)
    assert result["suggested_read"] == {"name": "read_code", "arguments": {"paths": ["a.py", "b.py"]}}
