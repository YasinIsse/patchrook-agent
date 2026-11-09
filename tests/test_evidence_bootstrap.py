"""Configured baseline checks run before a model and remain ordinary evidence."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest

from patchrook_agent.agent import Agent, AgentState
from patchrook_agent.completion import TaskRequirements
from patchrook_agent.config import HarnessConfig
from patchrook_agent.evidence_bootstrap import run_initial_checks
from patchrook_agent.models.ollama import ModelResponse
from patchrook_agent.tools import ToolConfig, ToolRegistry, Workspace


def action(tool_name, **arguments):
    return {"role": "assistant", "content": json.dumps({"name": tool_name, "arguments": arguments})}


class ScriptedModel:
    model = "offline-baseline-checks"
    context_size = 8192
    max_output_tokens = 512

    def __init__(self, *responses):
        self.responses = iter(responses)
        self.requests = []

    def chat(self, messages, **options):
        self.requests.append(deepcopy(messages))
        return ModelResponse(next(self.responses), 100, 20, 0, 0, 0, 0)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Baseline-check tests must not contact models")
    monkeypatch.setattr("httpx.get", forbidden)
    monkeypatch.setattr("httpx.post", forbidden)
    monkeypatch.setattr("httpx.Client", forbidden)


def make_agent(tmp_path, model, commands, *, initial=True, steps=1):
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / "subject.py").write_text("VALUE = 1\n")
    base = HarnessConfig.preset("C")
    config = replace(base, workflow_mode="evidence", action_protocol="schema",
        evidence_policy="rules", review_mode="off", evidence_initial_checks=initial,
        evidence_test_commands=tuple(tuple(argv) for argv in commands),
        context=replace(base.context, enabled=False), compaction_enabled=False,
        persist=True, checkpoint_enabled=True)
    return Agent(model, registry=ToolRegistry(Workspace(repository, ToolConfig(allow_tests=True))),
        config=config, runs_dir=tmp_path / "runs", max_steps=steps, deadline_seconds=30)


def check_results(state):
    return [json.loads(item["content"]) for item in state.messages
            if item.get("role") == "tool" and item.get("tool_name") == "run_check"]


def recorded_events(state):
    return [json.loads(line) for line in (Path(state.session_dir) / "events.jsonl").read_text().splitlines()]


def test_actual_baseline_assertion_and_every_check_precede_first_model_request(tmp_path):
    commands = [(sys.executable, "-c", "print('preservation passed')"),
                (sys.executable, "-c", "from subject import VALUE; assert VALUE == 2, 'expected VALUE=2'")]
    model = ScriptedModel(action("read_file", path="subject.py"))
    agent = make_agent(tmp_path, model, commands)

    state = agent.run("Fix the value.", requirements=TaskRequirements.fix())

    assert len(model.requests) == 1
    first_prompt = "\n".join(item["content"] for item in model.requests[0])
    assert "expected VALUE=2" in first_prompt
    assert "AssertionError" in first_prompt
    checks = check_results(state)
    assert [item["name"] for item in checks] == ["check_1", "check_2"]
    assert [item["exit_code"] for item in checks] == [0, 1]
    assert all(item["verification_kind"] == "configured_command" for item in checks)
    events = recorded_events(state)
    requests = [item for item in events if item["event"] == "tool_request"]
    assert [item["name"] for item in requests[:2]] == ["run_check", "run_check"]
    assert all(item["automatic"] and item["step"] == 0 for item in requests[:2])
    first_model = next(index for index, item in enumerate(events) if item["event"] == "model_request")
    assert all(events.index(item) < first_model for item in requests[:2])
    assert state.steps == 1 and state.verification == "failed"
    assert state.metrics["evidence_check_results"]["check_2"]["revision"] == 0


def test_baseline_pass_cannot_replace_checks_after_a_guarded_patch(tmp_path):
    commands = [(sys.executable, "-c", "from subject import VALUE; assert VALUE in (1, 2)"),
                (sys.executable, "-c", "from subject import VALUE; assert VALUE == 2")]
    model = ScriptedModel(action("read_file", path="subject.py"),
        action("apply_edits", edits=[{"path": "subject.py", "old_text": "VALUE = 1", "new_text": "VALUE = 2"}]),
        action("run_check", name="check_2"), action("finish"),
        action("run_check", name="check_1"), action("finish"))
    agent = make_agent(tmp_path, model, commands, steps=6)

    state = agent.run("Fix the value.", requirements=TaskRequirements.fix())

    assert state.status == "completed", state.final_response
    assert state.verification == "passed" and state.modified_files == ["subject.py"]
    assert len(model.requests) == 6
    checks = check_results(state)
    assert [item["name"] for item in checks] == ["check_1", "check_2", "check_2", "check_1"]
    finishes = [json.loads(item["content"]) for item in state.messages
                if item.get("role") == "tool" and item.get("tool_name") == "finish"]
    assert finishes[0]["error"]["code"] == "completion_requirements"
    assert finishes[1]["success"] is True


@pytest.mark.parametrize("failure", ["missing_module", "missing_executable", "workspace_change"])
def test_initial_environment_failures_stop_before_any_model_request(tmp_path, failure):
    if failure == "missing_module":
        command = (sys.executable, "-c", "import missing_baseline_dependency_xyz")
    elif failure == "missing_executable":
        command = ("missing-baseline-executable-xyz",)
    else:
        command = (sys.executable, "-c", "from pathlib import Path; Path('subject.py').write_text('VALUE = 9\\n')")
    model = ScriptedModel()
    agent = make_agent(tmp_path, model, [command, (sys.executable, "-c", "print('second check ran')")])

    state = agent.run("Fix the value.", requirements=TaskRequirements.fix())

    assert not model.requests
    assert state.stop_reason == "initial_check_environment_failure"
    assert state.status == "incomplete" and state.steps == 0
    checks = check_results(state)
    assert [item["name"] for item in checks if item.get("name")] == (
        ["check_2"] if failure == "missing_executable" else ["check_1", "check_2"])
    assert "second check ran" in checks[-1]["stdout"]
    if failure == "workspace_change":
        assert checks[0]["error"]["code"] == "workspace_changed_during_check"


def test_initial_checks_default_off_preserves_existing_model_driven_checks(tmp_path):
    model = ScriptedModel(action("read_file", path="subject.py"))
    agent = make_agent(tmp_path, model, [(sys.executable, "-c", "raise AssertionError('not run')")], initial=False)

    state = agent.run("Fix the value.", requirements=TaskRequirements.fix())

    assert len(model.requests) == 1
    assert not check_results(state)
    assert state.verification == "not_run"


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], {}])
def test_initial_check_setting_requires_an_actual_boolean(value):
    with pytest.raises(ValueError, match="evidence_initial_checks must be a boolean"):
        HarnessConfig(evidence_initial_checks=value)


def test_initial_check_setting_is_backward_compatible_and_persisted():
    config = HarnessConfig(evidence_initial_checks=True)
    assert HarnessConfig.from_dict(config.to_dict()).evidence_initial_checks is True
    old = config.to_dict()
    del old["evidence_initial_checks"]
    assert HarnessConfig.from_dict(old).evidence_initial_checks is False




@pytest.mark.parametrize("expired,enabled", [(True, {"run_check"}), (False, set())])
def test_bootstrap_respects_task_deadline_and_enabled_tools(expired, enabled):
    def forbidden(*args, **kwargs):
        pytest.fail("No check may execute after deadline or without tool access")
    runtime = SimpleNamespace(config=SimpleNamespace(evidence_initial_checks=True),
        tools=SimpleNamespace(commands={"check_1": ("python3", "-m", "pytest")}),
        state=AgentState(), enabled=enabled, session=SimpleNamespace(emit=lambda *args, **kwargs: None),
        _perform_action=forbidden)

    stop = run_initial_checks(runtime, time.monotonic() - 1 if expired else time.monotonic() + 10)

    assert stop is True
    assert runtime.state.stop_reason == ("deadline" if expired else "initial_check_environment_failure")
