"""Full available tool evidence stays outside bounded model observations."""

import json

import pytest

from patchrook_agent.observability import EventLogger
from patchrook_agent.tools import ToolConfig, ToolRegistry, Workspace
from patchrook_agent.tools.base import TOOL_CAPTURE, record_tool_output


def registry(tmp_path, **config):
    result = ToolRegistry(Workspace(tmp_path, ToolConfig(**config)))
    result.capture_enabled = True
    return result


def test_read_capture_preserves_full_selected_lines_beyond_observation_limit(tmp_path):
    lines = [f"symbol_{index}: " + "x" * 50 for index in range(1, 201)]
    (tmp_path / "large.py").write_text("\n".join(lines) + "\n")
    tools = registry(tmp_path)
    observed = tools.dispatch("read_file", {"path": "large.py"})
    execution = tools.last_execution
    assert len(json.dumps(observed, separators=(",", ":"))) <= 2000
    assert observed["truncated"]
    expected = "\n".join(f"{index}: {line}" for index, line in enumerate(lines, start=1))
    assert execution["available_output"]["content"] == expected
    assert execution["available_output"]["end_line"] == 200
    assert len(expected) > 10_000
    assert execution["name"] == "read_file" and execution["arguments"] == {"path": "large.py"}
    assert TOOL_CAPTURE.get() is None


def test_process_capture_preserves_noisy_pytest_failure_outside_model_context(tmp_path):
    (tmp_path / "test_noisy.py").write_text(
        "def test_noisy():\n    print('OUTPUT_BEGIN' + 'x' * 50000 + 'OUTPUT_END')\n    assert False\n"
    )
    tools = registry(tmp_path, allow_tests=True)
    observation = tools.dispatch("run_tests", {"path": "test_noisy.py"})
    available = tools.last_execution["available_output"]
    assert observation["status"] == "failed" and observation["exit_code"] == 1
    assert len(json.dumps(observation, separators=(",", ":"))) <= 2000
    assert len(available["stdout"]) > 50_000
    assert "OUTPUT_BEGIN" in available["stdout"] and "OUTPUT_END" in available["stdout"]
    assert "1 failed" in available["stdout"]
    assert available["raw_output_truncated"] is False
    assert available["raw_output_bytes"]["stdout"] == len(available["stdout"].encode("utf-8"))
    assert available["stderr"] == ""


def test_raw_process_capture_has_explicit_byte_ceiling_and_truncation_metadata(tmp_path):
    (tmp_path / "test_noisy.py").write_text(
        "def test_noisy():\n    print('x' * 20000)\n    assert False\n"
    )
    tools = registry(tmp_path, allow_tests=True, max_file_bytes=5120)
    result = tools.dispatch("run_tests", {"path": "test_noisy.py"})
    available = tools.last_execution["available_output"]
    assert result["exit_code"] == 1
    assert len(available["stdout"].encode("utf-8")) <= 5120
    assert available["raw_output_bytes"]["stdout"] > 20_000
    assert available["raw_output_truncated"] is True


@pytest.mark.parametrize("name,arguments,code", [
    ("unknown", {"path": "a.py"}, "unknown_tool"),
    ("read_file", {"path": 7}, "invalid_arguments"),
    ("read_file", '{"path":', "invalid_arguments"),
])
def test_unknown_and_invalid_requests_still_have_execution_metadata(tmp_path, name, arguments, code):
    tools = registry(tmp_path)
    result = tools.dispatch(name, arguments)
    assert result["error"]["code"] == code
    assert tools.last_execution["name"] == name
    assert tools.last_execution["arguments"] == arguments
    assert tools.last_execution["result"]["error"]["code"] == code
    assert tools.last_execution["available_output"] == {}
    assert TOOL_CAPTURE.get() is None


def test_capture_is_per_dispatch_and_restores_context_without_leaking_prior_output(tmp_path):
    (tmp_path / "a.py").write_text("private first observation\n")
    tools = registry(tmp_path)
    tools.dispatch("read_file", {"path": "a.py"})
    saved = tools.last_execution
    tools.dispatch("list_files", {})
    assert tools.last_execution["available_output"] == {}
    record_tool_output(content="unrelated external execution")
    assert saved["available_output"]["content"] == "1: private first observation"
    assert tools.last_execution["available_output"] == {}
    assert TOOL_CAPTURE.get() is None


def test_disabled_capture_does_not_collect_raw_content(tmp_path):
    (tmp_path / "a.py").write_text("x" * 10_000)
    tools = registry(tmp_path)
    tools.capture_enabled = False
    result = tools.dispatch("read_file", {"path": "a.py"})
    assert result["success"] and result["truncated"]
    assert tools.last_execution["available_output"] == {}
    assert TOOL_CAPTURE.get() is None


def test_metadata_only_logs_suppress_available_read_and_process_output(tmp_path):
    logger = EventLogger(tmp_path, "session", log_content=False)
    logger.emit("tool_result", available_output={"content": "private file", "stdout": "private stdout",
                                                "stderr": "private stderr", "raw_output_bytes": {"stdout": 14}},
                result={"success": True, "path": "a.py"})
    saved = logger.path.read_text()
    assert "private file" not in saved and "private stdout" not in saved and "private stderr" not in saved
    assert json.loads(saved)["result"]["success"] is True
