"""Invalid tool identifiers get one correction opportunity without redispatch loops."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from patchrook_agent.agent import Agent
from patchrook_agent.completion import TaskRequirements
from patchrook_agent.config import HarnessConfig
from patchrook_agent.models.ollama import ModelResponse
from patchrook_agent.tools import ToolConfig, ToolRegistry, Workspace


class ScriptedModel:
    model = "tool-repair-mechanics"
    context_size = 8192
    max_output_tokens = 512

    def __init__(self, *messages):
        self.messages = iter(messages)
        self.requests = []

    def chat(self, **options):
        self.requests.append(deepcopy(options))
        return ModelResponse(next(self.messages), 0, 0, None, None, None, 0,
                             prompt_tokens_reported=False, completion_tokens_reported=False)


def action(tool_name, **arguments):
    return {"role": "assistant", "content": json.dumps({"action": "tool", "tool": tool_name, "arguments": arguments})}


def complete():
    return {"role": "assistant", "content": json.dumps({"action": "complete", "summary":
        "Inspected the requested source and observed the required verification."})}


def blocked():
    return {"role": "assistant", "content": json.dumps({"action": "blocked", "reason":
        "The required operation is unavailable."})}


def read():
    return action("read_code", paths=["calc.py"])


def missing_test(**extra):
    return action("run_tests", path="test_missing.py", **extra)


def runner(tmp_path, model, *, allow_tests=True, **options):
    root = tmp_path / "repo"
    root.mkdir(exist_ok=True)
    (root / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    (root / "test_calc.py").write_text("from calc import add\ndef test_add():\n    assert add(3, 2) == 5\n")
    return Agent(model=model, registry=ToolRegistry(Workspace(root, ToolConfig(allow_tests=allow_tests))),
                 config=replace(HarnessConfig.preset("C"), workflow_mode="lean", action_protocol="schema"),
                 runs_dir=tmp_path / "runs", max_steps=20, **options)


def dispatch_spy(agent):
    calls = []
    original = agent.registry.dispatch
    def dispatch(name, arguments, **options):
        calls.append((name, deepcopy(arguments)))
        return original(name, arguments, **options)
    agent.registry.dispatch = dispatch
    return calls


@pytest.fixture(autouse=True)
def no_live_model(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("Tool repair tests must not contact Ollama")
    monkeypatch.setattr("patchrook_agent.models.ollama.httpx.Client", fail)


def test_missing_test_path_immediately_gets_trusted_correction_then_real_root_pytest(tmp_path):
    model = ScriptedModel(read(), missing_test(), action("list_files", path="."), action("run_tests"), complete())
    agent = runner(tmp_path, model)
    calls = dispatch_spy(agent)
    state = agent.run("Verify calc.py", requirements=TaskRequirements.code())
    assert state.status == "completed" and state.verification == "passed"
    assert [args for name, args in calls if name == "run_tests"] == [{"path": "test_missing.py"}, {}]
    messages = model.requests[2]["messages"]
    assert messages[-1]["role"] == "system"
    correction = messages[-1]["content"]
    assert "run_tests" in correction and "discovery" in correction.lower()
    assert "test_missing.py" in "\n".join(message["content"] for message in messages)
    observed = [json.loads(message["content"]) for message in state.messages if message.get("tool_name") == "run_tests"]
    assert observed[0]["error"]["code"] == "not_found"
    assert observed[1]["status"] == "passed" and observed[1]["exit_code"] == 0


@pytest.mark.parametrize("path", ["test_missing.py", ".::test_node"])
def test_identical_invalid_test_request_is_rejected_without_second_dispatch(tmp_path, path):
    request = action("run_tests", path=path)
    model = ScriptedModel(request, deepcopy(request))
    agent = runner(tmp_path, model)
    calls = dispatch_spy(agent)
    state = agent.run("Run the relevant tests", requirements=TaskRequirements(require_tests=True))
    assert state.status == "incomplete" and len(model.requests) == 2
    assert [name for name, _ in calls] == ["run_tests"]
    assert state.verification == "not_run"
    assert state.final_response


def test_unrelated_success_does_not_enable_same_missing_test_retry(tmp_path):
    model = ScriptedModel(missing_test(), read(), missing_test())
    agent = runner(tmp_path, model)
    calls = dispatch_spy(agent)
    state = agent.run("Verify calc.py", requirements=TaskRequirements.code())
    assert state.status == "incomplete" and len(model.requests) == 3
    assert sum(name == "run_tests" for name, _ in calls) == 1


@pytest.mark.parametrize("arguments,allow_tests,code", [
    ({"path": 42}, True, "invalid_arguments"),
    ({"path": "test_calc.py"}, False, "permission_denied"),
    ({"path": "../outside.py"}, True, "outside_workspace"),
])
def test_nonrepairable_test_errors_do_not_automatically_execute_a_replacement(tmp_path, arguments, allow_tests, code):
    model = ScriptedModel(action("run_tests", **arguments), blocked())
    agent = runner(tmp_path, model, allow_tests=allow_tests)
    calls = dispatch_spy(agent)
    state = agent.run("Verify calc.py", requirements=TaskRequirements.code())
    assert state.status == "incomplete" and state.verification == "not_run"
    assert len(model.requests) == 2 and len(calls) == 1
    result = next(json.loads(message["content"]) for message in state.messages if message.get("tool_name") == "run_tests")
    assert result["error"]["code"] == code


@pytest.mark.parametrize("name", ["invented_skill", "invented-skill"])
def test_invalid_skill_gets_grounded_valid_choices_and_allows_a_valid_selection(tmp_path, name):
    model = ScriptedModel(action("read_skill", name=name), action("read_skill", name="debugging"), read(), complete())
    agent = runner(tmp_path, model)
    calls = dispatch_spy(agent)
    state = agent.run("Explain calc.py", requirements=TaskRequirements.inspect())
    assert state.status == "completed" and "debugging" in state.memory["loaded_skills"]
    assert [args["name"] for tool_name, args in calls if tool_name == "read_skill"] == [name, "debugging"]
    messages = model.requests[1]["messages"]
    assert messages[-1]["role"] == "system"
    assert "debugging" in messages[-1]["content"]
    assert name in "\n".join(message["content"] for message in messages)


def test_identical_invalid_skill_is_not_dispatched_twice(tmp_path):
    invalid = action("read_skill", name="invented_skill")
    model = ScriptedModel(invalid, deepcopy(invalid))
    agent = runner(tmp_path, model)
    calls = dispatch_spy(agent)
    state = agent.run("Explain calc.py", requirements=TaskRequirements.inspect())
    assert state.status == "incomplete" and len(model.requests) == 2
    assert [name for name, _ in calls] == ["read_skill"]


def test_invalid_retry_limit_survives_checkpoint_resume(tmp_path):
    first = runner(tmp_path, ScriptedModel(missing_test()), pause_after_step=1)
    paused = first.run("Run the relevant tests", requirements=TaskRequirements(require_tests=True))
    assert paused.stop_reason == "paused" and paused.steps == 1
    model = ScriptedModel(missing_test())
    resumed = runner(tmp_path, model)
    calls = dispatch_spy(resumed)
    # Resume creates its registry from saved settings, so wrap the class method
    # to observe dispatch on that restored registry as well.
    original = ToolRegistry.dispatch
    restored_calls = []
    def dispatch(registry, name, arguments, **options):
        restored_calls.append((name, deepcopy(arguments)))
        return original(registry, name, arguments, **options)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ToolRegistry, "dispatch", dispatch)
        state = resumed.resume(paused.session_id, allow_tests=True)
    assert state.status == "incomplete" and state.steps == 2
    assert len(model.requests) == 1 and not calls and not restored_calls


def test_actual_write_creating_test_allows_legitimate_rerun(tmp_path):
    body = "def test_new():\n    assert 2 + 2 == 4\n"
    model = ScriptedModel(missing_test(), action("write_file", path="test_missing.py", content=body), missing_test(), complete())
    agent = runner(tmp_path, model)
    calls = dispatch_spy(agent)
    state = agent.run("Create the missing test and run it", requirements=TaskRequirements(require_tests=True))
    assert state.status == "completed" and state.verification == "passed"
    assert sum(name == "run_tests" for name, _ in calls) == 2


def test_discovery_of_externally_created_test_allows_legitimate_rerun(tmp_path):
    model = ScriptedModel(missing_test(), action("list_files", path="."), missing_test(), complete())
    agent = runner(tmp_path, model)
    original = agent.registry.dispatch
    test_calls = []
    def dispatch(name, arguments, **options):
        if name == "list_files":
            (agent.registry.workspace.root / "test_missing.py").write_text("def test_new():\n    assert True\n")
        if name == "run_tests":
            test_calls.append(deepcopy(arguments))
        return original(name, arguments, **options)
    agent.registry.dispatch = dispatch
    state = agent.run("Run the relevant tests", requirements=TaskRequirements(require_tests=True))
    assert state.status == "completed" and state.verification == "passed"
    assert len(test_calls) == 2
