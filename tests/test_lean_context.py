"""Lean context keeps current observed evidence, rather than navigation chronology."""

from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from patchrook_agent.actions import lean_action_schema
from patchrook_agent.agent import AgentState
from patchrook_agent.config import HarnessConfig
from patchrook_agent.context.budget import ContextConfig
from patchrook_agent.context.compaction import _groups
from patchrook_agent.context.lean import LEAN_SYSTEM_PROMPT, WORKING_SET_PREFIX, LeanContext, content_tokens
from patchrook_agent.session import Session
from patchrook_agent.tools import ToolRegistry, Workspace


def history(task="Fix calc.py, preserve its comments, and run the relevant test."):
    return [{"role": "system", "content": LEAN_SYSTEM_PROMPT}, {"role": "user", "content": task}]


def exchange(name, result, arguments=None, identity="call"):
    return [{"role": "assistant", "content": "", "tool_calls": [
        {"id": identity, "function": {"name": name, "arguments": arguments or {}}}]},
        {"role": "tool", "tool_name": name, "tool_call_id": identity, "content": json.dumps(result)}]


def read(path="calc.py", content="1: def calc(a, b):\n2:     return a - b", **more):
    return {"success": True, "path": path, "start_line": 1, "end_line": 2,
            "total_lines": 2, "truncated": False, "content": content, **more}


def facts(prepared):
    message = next(message for message in prepared.messages if message["content"].startswith(WORKING_SET_PREFIX))
    return json.loads(message["content"][len(WORKING_SET_PREFIX):].split("\n", 1)[0])


def text(prepared):
    return "\n".join(message["content"] for message in prepared.messages)


def prepare(messages, **kwargs):
    return LeanContext(ContextConfig()).prepare(messages, definitions=[], response_schema={}, **kwargs)


def test_sustained_repeated_batches_replace_navigation_history_with_exact_source():
    messages = history()
    files = [read(f"module{i}.py", f"1: def transform(value):\n2:     return value + {i}") for i in range(8)]
    messages += exchange("read_files", {"success": True, "files": files}, {"files": [{"path": f["path"]} for f in files]})
    first = prepare(messages)
    for i in range(150):
        messages += exchange("list_files", {"success": True, "entries": ["unused.py"]}, identity=f"list{i}")
        messages += exchange("search_code", {"success": True, "matches": []}, {"pattern": f"unused{i}"}, identity=f"search{i}")
        messages += exchange("read_files", {"success": True, "files": files}, {"files": [{"path": f["path"]} for f in files]}, identity=f"read{i}")
    original = deepcopy(messages)
    after = prepare(messages)
    assert after.safe and after.estimated_tokens == first.estimated_tokens
    assert after.messages == first.messages
    assert messages == after.active_messages == original
    assert len(_groups(after.active_messages)) > 400
    assert "unused149" not in text(after) and "omitted_paths" not in text(after)
    for i in range(8):
        assert f"2:     return value + {i}" in text(after)


def test_read_code_merges_only_displayed_numbered_source_with_prior_ranges():
    messages = history() + exchange("read_file", read(content="1: # comment\n2: def calc(a, b):", end_line=2, total_lines=5))
    messages += exchange("read_code", {"success": True, "files": [
        read(content="3:     return a - b\n4: # incomplete pre", start_line=3, end_line=4,
             total_lines=5, selected_end_line=5, next_start_line=4, truncated=True, partial_line=True)]})
    result = prepare(messages)
    assert result.safe
    assert "1: # comment\n2: def calc(a, b):" in text(result)
    assert "3:     return a - b" in text(result)
    assert facts(result)["latest_action"]["result"]["files"][0]["partial_content"] == "4: # incomplete pre"
    record = facts(result)["latest_action"]["result"]["files"][0]
    assert record["selected_end_line"] == 5 and record["next_start_line"] == 4
    assert "Observed source calc.py lines 3-4" not in text(result)


def test_latest_action_test_failure_and_constraints_are_preserved_exactly_once():
    failure = {"success": False, "status": "failed", "exit_code": 1,
               "stdout": "assert calc(10, 2) == 12\n", "stderr": "ERROR: unexpected subtraction\n"}
    messages = history() + [{"role": "user", "content": "Keep the public API exactly as it is."}]
    messages += exchange("read_file", read())
    messages += exchange("run_tests", failure, {"path": "tests/test_calc.py"})
    result = prepare(messages)
    assert result.safe
    assert messages[1] in result.messages and messages[2] in result.messages
    assert facts(result)["latest_test"]["result"] == failure
    assert facts(result)["latest_action"] == {"tool": "run_tests", "result_ref": "latest_test"}
    assert text(result).count("unexpected subtraction") == 1


def test_unrelated_green_keeps_exact_outstanding_failure_until_controller_resolves_scope():
    failure = {"success": False, "status": "failed", "exit_code": 1,
               "stdout": "AssertionError: expected addition\n", "stderr": ""}
    messages = history() + exchange("run_tests", failure, {"path": "tests/test_bad.py::test_add"})
    messages += exchange("run_tests", {"success": True, "status": "passed", "exit_code": 0},
                         {"path": "tests/test_unrelated.py"})
    result = prepare(messages, state_summary={"outstanding_tests": ["tests/test_bad.py::test_add"]})
    assert result.safe
    assert facts(result)["latest_test"]["result"]["status"] == "passed"
    assert facts(result)["outstanding_test_failures"][0]["result"] == failure
    assert text(result).count("AssertionError: expected addition") == 1
    messages += exchange("run_tests", {"success": True, "status": "passed", "exit_code": 0}, {"path": "tests"})
    resolved = prepare(messages, state_summary={"outstanding_tests": []})
    assert "outstanding_test_failures" not in facts(resolved)
    assert "AssertionError: expected addition" not in text(resolved)


def test_edit_invalidates_old_source_and_old_passing_test_without_inventing_a_fresh_read():
    messages = history() + exchange("read_file", read())
    messages += exchange("run_tests", {"success": True, "status": "passed", "exit_code": 0})
    messages += exchange("edit_file", {"success": True, "modified": True, "path": "calc.py", "replacements": 1},
                         {"path": "calc.py", "old_text": "return a - b", "new_text": "return a + b"})
    result = prepare(messages)
    assert result.safe and "Observed source calc.py" not in text(result)
    current = facts(result)
    assert current["changes"][0]["accepted_delta"] == {"old_text": "return a - b", "new_text": "return a + b"}
    assert current["latest_test"]["after_last_change"] is False
    messages += exchange("run_tests", {"success": True, "status": "passed", "exit_code": 0})
    assert facts(prepare(messages))["latest_test"]["after_last_change"] is True


def test_conflicting_reread_never_splices_unknown_stale_neighbors_into_new_version():
    messages = history() + exchange("read_file", read())
    messages += exchange("read_file", read(content="2:     return a + b", start_line=2))
    result = prepare(messages)
    assert "2:     return a + b" in text(result)
    assert "1: def calc(a, b):" not in text(result)


def test_resume_invalidation_survives_checkpoint_until_new_observed_source():
    messages = history() + exchange("read_file", read())
    messages += exchange("run_tests", {"success": True, "status": "passed", "exit_code": 0})
    lean = LeanContext(ContextConfig())
    lean.invalidate_paths(["calc.py"], history_length=len(messages))
    restored = LeanContext(ContextConfig())
    restored.restore(lean.to_dict())
    result = restored.prepare(messages, definitions=[], response_schema={})
    assert "Observed source calc.py" not in text(result)
    assert facts(result)["stale_source_paths"] == ["calc.py"]
    assert facts(result)["latest_test"]["after_last_change"] is False
    messages += exchange("read_file", read(content="1: def calc(a, b):\n2:     return a + b"))
    fresh = restored.prepare(messages, definitions=[], response_schema={})
    assert "2:     return a + b" in text(fresh)
    assert "stale_source_paths" not in facts(fresh)
    assert facts(fresh)["latest_test"]["after_last_change"] is False
    messages += exchange("run_tests", {"success": True, "status": "passed", "exit_code": 0})
    assert facts(restored.prepare(messages, definitions=[], response_schema={}))["latest_test"]["after_last_change"] is True


def test_budget_counts_model_content_and_native_definitions_not_escaped_http_or_schema():
    messages = [{"role": "user", "content": 'é🙂\\"\n' * 30}]
    byte_count = sum(len(message["content"].encode()) for message in messages)
    assert content_tokens(messages) == (byte_count + 2) // 3 + 16 + 128
    definitions = [{"type": "function", "function": {"name": "read_file", "description": "x" * 600}}]
    assert content_tokens(messages, definitions) >= content_tokens(messages) + 200
    raw = history() + exchange("read_file", read())
    small = LeanContext(ContextConfig()).prepare(raw, definitions=definitions, response_schema={})
    large = LeanContext(ContextConfig()).prepare(raw, definitions=definitions, response_schema={"description": "x" * 100000})
    assert small.safe and small.estimated_tokens == large.estimated_tokens
    native = LeanContext(ContextConfig()).prepare(raw, definitions=definitions)
    assert native.safe and native.estimated_tokens > small.estimated_tokens
    replay = [message for message in native.messages if message.get("role") in {"assistant", "tool"}]
    assert replay[0]["tool_calls"] == raw[-2]["tool_calls"]
    assert replay[1] == raw[-1]


def test_usage_only_increases_conservative_estimate_and_missing_telemetry_is_explicit():
    lean = LeanContext(ContextConfig())
    raw = history() + exchange("read_file", read())
    first = lean.prepare(raw, definitions=[], response_schema={})
    lean.record_usage(100)
    assert lean.correction_tokens == 0
    lean.record_usage(first.estimated_tokens + 200)
    correction = lean.correction_tokens
    assert correction >= 200
    lean.record_usage(100)
    lean.record_usage(99999, prompt_tokens_reported=False)
    lean.record_usage(None)
    after = lean.prepare(raw, definitions=[], response_schema={})
    assert after.estimated_tokens == first.estimated_tokens + correction
    assert lean.observed_calls == 3 and lean.unmeasured_calls == 2
    restored = LeanContext(ContextConfig())
    restored.restore(lean.to_dict())
    assert restored.to_dict() == lean.to_dict()
    assert restored.last_base_estimate is None


def test_old_optional_source_can_leave_working_set_but_latest_exact_read_is_preserved():
    messages = history()
    for i in range(50):
        content = "\n".join(f"{line}: value_{i}_{line} = 'observed exact source'" for line in range(1, 31))
        messages += exchange("read_file", read(f"old{i}.py", content, end_line=30, total_lines=30), identity=str(i))
    messages += exchange("read_code", {"success": True, "files": [read("latest.py")]})
    result = prepare(messages)
    assert result.safe and result.estimated_tokens <= ContextConfig().input_limit
    assert "2:     return a - b" in text(result) and "Observed source latest.py" in text(result)
    assert "Some older source is outside this bounded view" in text(result)
    assert result.tokens_before > result.tokens_after
    assert "old0.py" not in text(result)


def test_irreducible_task_or_current_error_stops_honestly_without_changing_raw_history():
    lean = LeanContext(ContextConfig(context_size=2048, trigger=.5))
    raw = history("Preserve every part of this task: " + "specific constraint " * 600)
    original = deepcopy(raw)
    result = lean.prepare(raw, definitions=[], response_schema={})
    assert not result.safe and result.reason == "context_limit"
    assert result.estimated_tokens > lean.config.input_limit
    assert raw == original == result.active_messages and raw[1] in result.messages
    raw = history() + exchange("run_tests", {"success": False, "status": "failed", "stderr": "exact error " * 1000})
    result = lean.prepare(raw, definitions=[], response_schema={})
    assert not result.safe and facts(result)["latest_test"]["result"]["stderr"] == "exact error " * 1000


@pytest.mark.parametrize("malformed", [
    {"role": "assistant", "content": {}}, {"role": "invalid", "content": "x"},
    {"role": "tool", "tool_name": "read_file", "content": "{}"},
])
def test_malformed_history_fails_before_model_dispatch(malformed):
    result = prepare(history() + [malformed])
    assert not result.safe and result.reason == "invalid_tool_history"


@pytest.mark.parametrize("schema_mode", [False, True])
def test_session_lean_mode_keeps_raw_atomic_history_skips_generic_index_and_records_estimator(tmp_path, schema_mode):
    config = replace(HarnessConfig.preset("C"), workflow_mode="lean", persist=False,
                     checkpoint_enabled=False, logging_enabled=False)
    workspace = Workspace(tmp_path)
    definitions = ToolRegistry(workspace).definitions()
    session = Session(config, workspace, SimpleNamespace(context_size=8192, max_output_tokens=512), tmp_path / "runs", "Fix calc.py.")
    raw = history() + exchange("read_file", read())
    state = AgentState(messages=deepcopy(raw), active_messages=deepcopy(raw))
    state.completion = {"progress": {"next": "edit inspected source"}, "missing": ["change", "tests"],
                        "outstanding_tests": ["tests/test_bad.py"]}
    schema = lean_action_schema(definitions, allow_complete=False) if schema_mode else None
    result = session.prepare(state, definitions, response_schema=schema, protocol_instructions="Use enabled actions.")
    assert result.safe and state.messages == state.active_messages == raw
    assert _groups(state.active_messages)[-1] == raw[-2:]
    assert "Available trusted skills" not in text(result) and "current_plan" not in text(result)
    assert facts(result)["progress"]["current"]["next"] == "edit inspected source"
    assert facts(result)["progress"]["outstanding_tests"] == ["tests/test_bad.py"]
    assert session.metrics.per_step[-1]["context_estimator"] == "lean_content"
    session.record_lean_usage(100)
    assert session.lean.observed_calls == 1
    assert result.messages[-1] == {"role": "system", "content": "Use enabled actions."}


def test_session_loaded_skill_is_included_whole_without_discovery_index(tmp_path):
    config = replace(HarnessConfig.preset("C"), workflow_mode="lean", persist=False,
                     checkpoint_enabled=False, logging_enabled=False)
    workspace = Workspace(tmp_path)
    session = Session(config, workspace, SimpleNamespace(context_size=8192, max_output_tokens=512), tmp_path / "runs", "Fix calc.py.")
    skill = {"role": "system", "content": "Trusted skill document: test/SKILL.md\nPreserve this loaded instruction exactly."}
    session.skills.loaded_messages = lambda: [skill]
    raw = history()
    state = AgentState(messages=deepcopy(raw), active_messages=deepcopy(raw))
    result = session.prepare(state, [], response_schema={})
    assert result.safe and skill in result.messages
    assert "Available trusted skills" not in text(result)
