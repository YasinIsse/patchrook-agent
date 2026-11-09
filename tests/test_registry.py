import json

import pytest

from patchrook_agent.tools import ToolConfig, ToolRegistry, Workspace


@pytest.fixture
def registry(tmp_path):
    return ToolRegistry(Workspace(tmp_path))


def test_all_seven_tools_have_strict_schemas(registry):
    definitions = registry.definitions()
    assert {item["function"]["name"] for item in definitions} == {
        "list_files", "read_file", "read_files", "search_code", "write_file", "edit_file", "run_tests",
    }
    for item in definitions:
        assert item["type"] == "function"
        assert item["function"]["parameters"]["additionalProperties"] is False


@pytest.mark.parametrize("name,arguments,code", [
    ("bash", {"command": "id"}, "unknown_tool"),
    ("write_file", {"path": "new.py", "content": "x", "command": "id"}, "invalid_arguments"),
    ("read_file", {}, "invalid_arguments"),
    ("read_file", {"path": 1}, "invalid_arguments"),
    ("read_file", {"path": "x", "start_line": True}, "invalid_arguments"),
    ("list_files", {"recursive": "yes"}, "invalid_arguments"),
    ("read_file", "{bad-json", "invalid_arguments"),
    ("read_file", "[]", "invalid_arguments"),
    ("run_tests", {"path": ".", "timeout_seconds": float("nan")}, "invalid_arguments"),
    ("run_tests", {"path": ".", "command": "bash"}, "invalid_arguments"),
    ("run_tests", {"path": "."}, "permission_denied"),
    ("read_file", {"path": "../secret"}, "outside_workspace"),
    ("read_file", {"path": "missing"}, "not_found"),
])
def test_errors_are_observations(registry, name, arguments, code):
    result = registry.dispatch(name, arguments)
    assert result["success"] is False
    assert result["error"]["code"] == code


def test_dispatch_validates_and_executes_json_arguments(registry):
    result = registry.dispatch("write_file", '{"path":"nested/new.py","content":"hello"}')
    assert result["success"]
    assert registry.dispatch("read_file", {"path": "nested/new.py"})["content"] == "1: hello"


def test_result_bound_preserves_valid_json_and_status(tmp_path):
    registry = ToolRegistry(Workspace(tmp_path, ToolConfig(max_output_chars=512)))
    (tmp_path / "huge.py").write_text("x" * 10_000)
    result = registry.dispatch("read_file", {"path": "huge.py"})
    serialized = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
    assert len(serialized) <= 512
    assert result["truncated"] and result["success"]
    assert result["path"] == "huge.py"


def test_expired_deadline_prevents_mutation(registry):
    result = registry.dispatch("write_file", {"path": "x", "content": "x"}, remaining_seconds=0)
    assert result["error"]["code"] == "deadline_exceeded"
    assert not (registry.workspace.root / "x").exists()


def test_unexpected_tool_error_does_not_escape(registry, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("unexpected failure")

    from dataclasses import replace
    registry.tools["read_file"] = replace(registry.tools["read_file"], execute=fail)
    assert registry.dispatch("read_file", {"path": "x"})["error"]["code"] == "execution_error"


def test_workspace_configuration_validation(tmp_path):
    with pytest.raises(ValueError):
        ToolConfig(max_read_lines=0)
    with pytest.raises(ValueError):
        ToolConfig(test_timeout_seconds=float("inf"))
    with pytest.raises(ValueError):
        ToolConfig(max_output_chars=20)
    with pytest.raises(ValueError):
        Workspace(tmp_path / "does-not-exist")
