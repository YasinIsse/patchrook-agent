"""Read argument mistakes receive a bounded correction, not a false blocker."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from patchrook_agent.actions import lean_action_instructions
from patchrook_agent.agent import Agent
from patchrook_agent.completion import TaskRequirements
from patchrook_agent.config import HarnessConfig
from patchrook_agent.models.ollama import ModelResponse
from patchrook_agent.read_recovery import corrective_read_action
from patchrook_agent.tools import ToolConfig, ToolRegistry, Workspace


class ScriptedModel:
    model = "read-recovery-mechanics"
    context_size = 8192
    max_output_tokens = 512

    def __init__(self, *actions):
        self.actions = iter(actions)
        self.requests = []

    def chat(self, **options):
        self.requests.append(deepcopy(options))
        return ModelResponse(next(self.actions), 0, 0, None, None, None, 0,
                             prompt_tokens_reported=False, completion_tokens_reported=False)


def tool(name, **arguments):
    return {"role": "assistant", "content": json.dumps({"action": "tool", "tool": name, "arguments": arguments})}


def blocked(reason="The requested source could not be inspected."):
    return {"role": "assistant", "content": json.dumps({"action": "blocked", "reason": reason})}


def complete():
    return {"role": "assistant", "content": json.dumps({"action": "complete", "summary":
        "Read calc.py, fixed addition, and observed test_calc.py pass."})}


def make_agent(tmp_path, model, **options):
    root = tmp_path / "repo"
    root.mkdir(exist_ok=True)
    if not (root / "calc.py").exists():
        (root / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (root / "test_calc.py").write_text("from calc import add\ndef test_add():\n    assert add(3, 2) == 5\n")
    config = replace(HarnessConfig.preset("C"), workflow_mode="lean", action_protocol="schema")
    return Agent(model=model, registry=ToolRegistry(Workspace(root, ToolConfig(allow_tests=True))),
                 config=config, runs_dir=tmp_path / "runs", max_steps=20, max_tool_failures=5, **options)


@pytest.fixture(autouse=True)
def no_live_model(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("Read recovery tests must not contact Ollama")
    monkeypatch.setattr("patchrook_agent.models.ollama.httpx.Client", fail)


def invalid_read():
    return tool("read_file", path="calc.py", start_line=99, end_line=100)


def test_recoverable_read_error_then_blocked_gets_one_concrete_retry(tmp_path):
    model = ScriptedModel(invalid_read(), blocked(), tool("read_code", paths=["calc.py"]),
                          tool("edit_file", path="calc.py", old_text="a - b", new_text="a + b"),
                          tool("run_tests", path="test_calc.py"), complete())
    state = make_agent(tmp_path, model).run("Fix addition in calc.py", requirements=TaskRequirements.fix())
    assert state.status == "completed", (state.stop_reason, state.final_response)
    assert state.verification == "passed" and state.modified_files == ["calc.py"]
    assert state.steps == 6 and len(model.requests) == 6
    first_retry_guide = model.requests[1]["messages"][-1]["content"]
    assert '"tool":"read_code","arguments":{"paths":["calc.py"]}' in first_retry_guide
    recovery_request = "\n".join(message["content"] for message in model.requests[2]["messages"])
    assert "read_code" in recovery_request and "calc.py" in recovery_request
    assert "invalid_range" in recovery_request
    results = [json.loads(message["content"]) for message in state.messages if message.get("role") == "tool"]
    assert results[0]["error"]["code"] == "invalid_range"
    assert results[1]["success"] is True


def test_intervening_listing_does_not_hide_recoverable_read_error(tmp_path):
    model = ScriptedModel(invalid_read(), tool("list_files", path="."), blocked(), blocked())
    state = make_agent(tmp_path, model).run("Inspect calc.py", requirements=TaskRequirements.inspect())
    assert state.stop_reason == "model_blocked" and state.steps == 4
    assert len(model.requests) == 4
    recovery_request = "\n".join(message["content"] for message in model.requests[3]["messages"])
    assert "read_code" in recovery_request and "calc.py" in recovery_request


def test_blocked_read_recovery_budget_survives_resume(tmp_path):
    model = ScriptedModel(invalid_read(), blocked(), invalid_read())
    paused = make_agent(tmp_path, model, pause_after_step=3).run(
        "Inspect calc.py", requirements=TaskRequirements.inspect())
    assert paused.stop_reason == "paused" and paused.steps == 3
    resumed_model = ScriptedModel(blocked())
    state = make_agent(tmp_path, resumed_model).resume(paused.session_id, allow_tests=True)
    assert state.stop_reason == "model_blocked" and state.steps == 4
    assert len(resumed_model.requests) == 1


@pytest.mark.parametrize("path,code", [("missing.py", "not_found"), ("../outside.py", "outside_workspace")])
def test_concrete_read_obstacle_stops_without_a_retry(tmp_path, path, code):
    model = ScriptedModel(tool("read_file", path=path), blocked("Required source is unavailable."))
    state = make_agent(tmp_path, model).run("Inspect required source", requirements=TaskRequirements.inspect())
    assert state.stop_reason == "model_blocked" and state.steps == 2
    assert len(model.requests) == 2
    result = next(json.loads(message["content"]) for message in state.messages if message.get("role") == "tool")
    assert result["error"]["code"] == code


def test_permission_obstacle_is_not_converted_to_a_read_retry(tmp_path):
    model = ScriptedModel(tool("run_tests", path="test_calc.py"), blocked("Test execution permission is unavailable."))
    agent = make_agent(tmp_path, model)
    agent.registry.workspace = Workspace(agent.registry.workspace.root, ToolConfig(allow_tests=False))
    state = agent.run("Verify addition", requirements=TaskRequirements.code())
    assert state.stop_reason == "model_blocked" and state.steps == 2
    assert len(model.requests) == 2
    result = next(json.loads(message["content"]) for message in state.messages if message.get("role") == "tool")
    assert result["error"]["code"] == "permission_denied"


def test_new_concrete_obstacle_takes_precedence_over_old_argument_error(tmp_path):
    model = ScriptedModel(invalid_read(), tool("read_file", path="missing.py"), blocked())
    state = make_agent(tmp_path, model).run("Inspect calc.py", requirements=TaskRequirements.inspect())
    assert state.stop_reason == "model_blocked" and state.steps == 3
    assert len(model.requests) == 3


def test_blocked_guidance_does_not_offer_a_canned_reason_to_copy(tmp_path):
    agent = make_agent(tmp_path, ScriptedModel())
    definitions = agent.registry.definitions(lean=True)
    guidance = lean_action_instructions(definitions, False, ["inspect source"])
    assert "obstacle and unfinished work" not in guidance
    assert "obstacle and remaining work" not in guidance


def exchange(name, arguments, result):
    return [{"role": "assistant", "content": "", "tool_calls": [{"function": {
        "name": name, "arguments": arguments}}]}, {"role": "tool", "tool_name": name, "content": json.dumps(result)}]


def argument_error(path=None, code="invalid_arguments"):
    return {"success": False, **({"path": path} if path is not None else {}),
            "error": {"code": code, "message": "Observed argument validation error"}}


def test_corrective_read_uses_observed_confined_paths_without_reading_contents(tmp_path, monkeypatch):
    agent = make_agent(tmp_path, ScriptedModel())
    messages = exchange("read_code", {"paths": ["./calc.py", "test_calc.py", "calc.py"], "unexpected": True},
                        argument_error())
    messages += exchange("list_files", {"path": "."}, {"success": True, "entries": ["calc.py"]})
    original = deepcopy(messages)
    monkeypatch.setattr(type(tmp_path), "read_text", lambda *args, **kwargs: pytest.fail("Hidden source read"))
    assert corrective_read_action(messages, agent.registry.workspace) == {
        "action": "tool", "tool": "read_code", "arguments": {"paths": ["calc.py", "test_calc.py"]}}
    assert messages == original


@pytest.mark.parametrize("name,arguments", [
    ("read_file", {"path": "calc.py", "start_line": 99}),
    ("read_files", {"files": [{"path": "calc.py", "start_line": 99}]}),
    ("read_code", {"files": [{"path": "calc.py", "symbol": "guessed", "start_line": 99}]}),
])
def test_corrective_read_supports_legacy_observed_argument_shapes(tmp_path, name, arguments):
    agent = make_agent(tmp_path, ScriptedModel())
    assert corrective_read_action(exchange(name, json.dumps(arguments), argument_error()), agent.registry.workspace) == {
        "action": "tool", "tool": "read_code", "arguments": {"paths": ["calc.py"]}}


@pytest.mark.parametrize("path", ["../outside.py", "missing.py", "."])
def test_argument_error_without_a_known_existing_file_cannot_offer_a_retry(tmp_path, path):
    agent = make_agent(tmp_path, ScriptedModel())
    assert corrective_read_action(exchange("read_file", {"path": path}, argument_error()), agent.registry.workspace) is None


def test_successful_same_path_read_resolves_only_that_old_argument_error(tmp_path):
    agent = make_agent(tmp_path, ScriptedModel())
    messages = exchange("read_code", {"paths": ["calc.py", "test_calc.py"]}, argument_error())
    messages += exchange("read_file", {"path": "./calc.py"}, {"success": True, "path": "calc.py", "content": "1: value = 1"})
    assert corrective_read_action(messages, agent.registry.workspace)["arguments"] == {"paths": ["test_calc.py"]}
    messages += exchange("read_code", {"paths": ["test_calc.py"]}, {"success": True, "files": [
        {"success": True, "path": "test_calc.py", "content": "1: value = 2"}]})
    assert corrective_read_action(messages, agent.registry.workspace) is None


def test_mixed_batch_concrete_obstacle_takes_precedence_over_argument_retry(tmp_path):
    agent = make_agent(tmp_path, ScriptedModel())
    messages = exchange("read_code", {"paths": ["calc.py", "missing.py"]}, {"success": False, "files": [
        argument_error("calc.py", "invalid_selection"), argument_error("missing.py", "not_found")]})
    assert corrective_read_action(messages, agent.registry.workspace) is None


def test_corrective_read_is_bounded_to_ten_paths(tmp_path):
    workspace = Workspace(tmp_path)
    paths = [f"module_{index}.py" for index in range(12)]
    for path in paths:
        (tmp_path / path).write_text("value = 1")
    action = corrective_read_action(exchange("read_code", {"paths": paths}, argument_error()), workspace)
    assert action["arguments"]["paths"] == paths[:10]


def test_unknown_malformed_read_does_not_recover_an_unrelated_older_read(tmp_path):
    agent = make_agent(tmp_path, ScriptedModel())
    messages = exchange("read_file", {"path": "calc.py"}, argument_error())
    messages += exchange("read_code", {"paths": [42]}, argument_error())
    assert corrective_read_action(messages, agent.registry.workspace) is None


@pytest.mark.parametrize("code", ["permission_denied", "not_found", "outside_workspace", "io_error"])
def test_newer_nonread_filesystem_obstacle_is_not_hidden_by_old_argument_error(tmp_path, code):
    agent = make_agent(tmp_path, ScriptedModel())
    messages = exchange("read_file", {"path": "calc.py"}, argument_error())
    messages += exchange("edit_file", {"path": "other.py"}, argument_error("other.py", code))
    assert corrective_read_action(messages, agent.registry.workspace) is None
