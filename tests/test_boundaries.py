"""Regressions for evidence and progress at the controller boundary."""

import pytest

from patchrook_agent.agent import Agent
from patchrook_agent.models.ollama import ModelResponse
from patchrook_agent.tools.base import ToolConfig, Workspace
from patchrook_agent.tools.registry import ToolRegistry


def tool_call(name, arguments, call_id=None):
    call = {"type": "function", "function": {"name": name, "arguments": arguments}}
    if call_id is not None:
        call["id"] = call_id
    return call


class ScriptedModel:
    context_size = 100_000
    max_output_tokens = 128

    def __init__(self, messages):
        self.messages = iter(messages)

    def chat(self, **kwargs):
        return ModelResponse(
            message=next(self.messages),
            prompt_tokens=1,
            completion_tokens=1,
            prompt_duration_ns=0,
            completion_duration_ns=0,
            load_duration_ns=0,
            total_duration_ns=0,
        )


@pytest.mark.parametrize("vary_ids,vary_timings", [(True, False), (False, True)])
def test_volatile_fields_do_not_make_repeated_observations_progress(
    tmp_path, monkeypatch, vary_ids, vary_timings,
):
    registry = ToolRegistry(Workspace(tmp_path))
    searches = 0

    def dispatch(name, arguments, **kwargs):
        nonlocal searches
        if name == "edit_file":
            return {"success": False, "error": {"code": "no_match", "message": "No exact match"}}
        searches += 1
        return {
            "success": True,
            "matches": [{"path": "example.py", "line_number": 1, "text": "old"}],
            "duration_seconds": searches / 100 if vary_timings else 0.01,
            "truncated": False,
        }

    monkeypatch.setattr(registry, "dispatch", dispatch)
    messages = []
    for step in range(5):
        messages.append({"role": "assistant", "content": "", "tool_calls": [
            tool_call("edit_file", {"path": "example.py", "old_text": "missing", "new_text": "new"},
                      f"edit-{step}" if vary_ids else None),
            tool_call("search_code", {"pattern": "old"}, f"search-{step}" if vary_ids else None),
        ]})
    state = Agent(model=ScriptedModel(messages), registry=registry,
                  max_steps=5, max_tool_failures=2).run("Fix the example.")
    assert state.stop_reason == "repeated_tool_failures"
    assert state.steps == 3
    assert state.tool_failures == 3


@pytest.mark.parametrize("test_status", ["passed", "failed", "timed_out"])
def test_provisional_final_is_withheld_until_verification_was_reported(
    tmp_path, monkeypatch, test_status,
):
    registry = ToolRegistry(Workspace(tmp_path, ToolConfig(allow_tests=True)))

    def dispatch(name, arguments, **kwargs):
        if name == "edit_file":
            return {"success": True, "path": "example.py", "modified": True}
        return {"success": test_status == "passed", "status": test_status,
                "exit_code": 0 if test_status == "passed" else 1,
                "stdout": "", "stderr": "", "duration_seconds": 0.01}

    monkeypatch.setattr(registry, "dispatch", dispatch)
    model = ScriptedModel([
        {"role": "assistant", "content": "", "tool_calls": [tool_call(
            "edit_file", {"path": "example.py", "old_text": "old", "new_text": "new"},
        )]},
        {"role": "assistant", "content": "Fixed the implementation; all tests passed."},
    ])
    state = Agent(model=model, registry=registry, max_steps=2).run("Fix the example and verify it.")
    assert state.status == "incomplete"
    assert state.stop_reason == "max_steps"
    assert state.verification == test_status
    assert state.final_response == ""


def test_mutation_invalidates_earlier_passing_verification(tmp_path, monkeypatch):
    registry = ToolRegistry(Workspace(tmp_path))

    def dispatch(name, arguments, **kwargs):
        if name == "run_tests":
            return {"success": True, "status": "passed"}
        return {"success": True, "path": "example.py", "modified": True}

    monkeypatch.setattr(registry, "dispatch", dispatch)
    model = ScriptedModel([
        {"role": "assistant", "content": "", "tool_calls": [
            tool_call("run_tests", {"path": "."}),
            tool_call("edit_file", {"path": "example.py", "old_text": "old", "new_text": "new"}),
        ]},
        {"role": "assistant", "content": "Changes complete."},
    ])
    state = Agent(model=model, registry=registry).run("Fix the example.")
    assert state.status == "incomplete"
    assert state.stop_reason == "verification_not_authorized"
    assert state.verification == "not_run"
