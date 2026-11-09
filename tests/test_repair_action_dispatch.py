"""Closed diagnostic actions execute before coding and remain durable on resume."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from patchrook_agent.agent import Agent
from patchrook_agent.completion import TaskRequirements
from patchrook_agent.config import HarnessConfig
from patchrook_agent.models.decision import DecisionResult
from patchrook_agent.models.ollama import ModelResponse
from patchrook_agent.tools import ToolConfig, ToolRegistry, Workspace


TASK = "Reproduce the failing tests, fix calc.py and rounding.py using targeted edits, and rerun tests."
DONE = {"role": "assistant", "content": "DONE"}


def edit(path, old, new):
    return tool("edit_file", path=path, old_text=old, new_text=new)


def tool(name, **arguments):
    return {"role": "assistant", "content": "", "tool_calls": [{"function": {
        "name": name, "arguments": arguments}}]}


class Model:
    model = "offline-action-dispatch"
    context_size = 8192
    max_output_tokens = 512

    def __init__(self, *messages, check=None):
        self.script = iter(messages)
        self.requests = []
        self.check = check

    def chat(self, **options):
        self.requests.append(deepcopy(options))
        if self.check:
            self.check(len(self.requests), options)
        return ModelResponse(next(self.script), 0, 0, None, None, None, 0,
                             prompt_tokens_reported=False, completion_tokens_reported=False)


class Decisions:
    def __init__(self, *, path, inspect):
        self.path = path
        self.inspect = inspect
        self.calls = []
        self.selected = None

    def rank(self, context, candidates, **options):
        self.calls.append((context, deepcopy(candidates), options))
        verb = "read lines" if self.inspect else "prepare a targeted patch"
        self.selected = next(candidate.id for candidate in candidates
                             if candidate.description.startswith(self.path + ";") and verb in candidate.description)
        order = (self.selected, *(candidate.id for candidate in candidates if candidate.id != self.selected))
        return DecisionResult(order, {}, called=False)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr("httpx.Client", lambda *args, **kwargs: pytest.fail("No HTTP in action dispatch tests"))


def repository(tmp_path, *, configuration=False):
    root = tmp_path / "repo"
    root.mkdir()
    if configuration:
        (root / "app_config.py").write_text("OFFSET = 1\n")
        (root / "calc.py").write_text("from app_config import OFFSET\n\ndef add(a, b):\n    return a + b + OFFSET\n")
    else:
        (root / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (root / "rounding.py").write_text("def floor_cents(value):\n    return (value + 99) // 100\n")
    (root / "test_calc.py").write_text(
        "from calc import add\nfrom rounding import floor_cents\n\n"
        "def test_add():\n    assert add(3, 2) == 5\n\n"
        "def test_floor():\n    assert floor_cents(151) == 1\n")
    return root


def agent(tmp_path, model, client, *, pause=None, dispatched=None, decision="on"):
    instance = Agent(model, registry=ToolRegistry(Workspace(tmp_path / "repo", ToolConfig(allow_tests=True))),
                     config=replace(HarnessConfig.preset("C"), workflow_mode="coding", decision_model=decision,
                                    review_mode="off"),
                     decision_client=client, runs_dir=tmp_path / "runs",
                     pause_after_step=pause, max_steps=8)
    if dispatched is not None:
        original = instance.registry.dispatch
        def record(name, arguments, **options):
            dispatched.append((name, deepcopy(arguments)))
            return original(name, arguments, **options)
        instance.registry.dispatch = record
    return instance


def changes(configuration=False):
    primary = (edit("app_config.py", "OFFSET = 1", "OFFSET = 0") if configuration
               else edit("calc.py", "a - b", "a + b"))
    return primary, edit("rounding.py", "(value + 99) // 100", "value // 100")


def events(state, name):
    return [entry for line in (Path(state.session_dir) / "events.jsonl").read_text().splitlines()
            if (entry := json.loads(line)).get("event") == name]


@pytest.mark.parametrize("configuration", [False, True], ids=["secondary-helper", "configuration"])
def test_concrete_inspection_executes_one_bounded_read_before_first_coder_call(tmp_path, configuration):
    repository(tmp_path, configuration=configuration)
    expected = "app_config.py" if configuration else "rounding.py"
    dispatched = []
    snapshots = []
    def check(number, options):
        if number == 1:
            reads = [arguments for name, arguments in dispatched if name == "read_file"]
            assert len(reads) == 1
            assert reads[0]["path"] == expected
            assert 1 <= reads[0]["start_line"] <= reads[0]["end_line"]
            assert reads[0]["end_line"] - reads[0]["start_line"] < 24
            snapshots.append(reads[0])
    model = Model(*changes(configuration), DONE, check=check)
    client = Decisions(path=expected, inspect=True)
    task = TASK + (" Restore the configured zero offset in app_config.py." if configuration else "")
    result = agent(tmp_path, model, client, dispatched=dispatched).run(task, requirements=TaskRequirements.fix())
    assert result.status == "completed", result.final_response
    assert result.verification == "passed"
    assert len(client.calls) == 1
    assert len(snapshots) == 1
    action = next(entry for entry in events(result, "decision_action") if entry["action_id"] == client.selected)
    assert action["kind"] == ("inspect_config" if configuration else "inspect_helper")
    outcome = next(entry for entry in events(result, "decision_action_result") if entry["action_id"] == client.selected)
    assert outcome["success"] and outcome["operation"] == "read_file"


def test_prepare_action_does_not_dispatch_a_read_file(tmp_path):
    repository(tmp_path)
    dispatched = []
    def check(number, options):
        assert not [arguments for name, arguments in dispatched if name == "read_file"]
    model = Model(*changes(), DONE, check=check)
    client = Decisions(path="calc.py", inspect=False)
    result = agent(tmp_path, model, client, dispatched=dispatched).run(TASK, requirements=TaskRequirements.fix())
    assert result.status == "completed", result.final_response
    selected = next(entry for entry in events(result, "decision_action") if entry["action_id"] == client.selected)
    assert selected["kind"] == "prepare_patch"
    assert not [arguments for name, arguments in dispatched if name == "read_file"]


def test_applied_inspection_is_not_redispatched_after_resume(tmp_path):
    repository(tmp_path)
    first_dispatches = []
    first_model = Model(tool("read_file", path="calc.py", start_line=1, end_line=2))
    client = Decisions(path="rounding.py", inspect=True)
    first = agent(tmp_path, first_model, client, pause=1, dispatched=first_dispatches).run(
        TASK, requirements=TaskRequirements.fix())
    assert first.stop_reason == "paused", first.final_response
    saved = json.loads((Path(first.session_dir) / "checkpoint.json").read_text())
    assert saved["execution"]["coding"]["repair"]["action_id"] == client.selected
    assert saved["execution"]["coding"]["repair"]["action_status"] == "applied"
    assert len([arguments for name, arguments in first_dispatches
                if name == "read_file" and arguments.get("path") == "rounding.py"]) == 1
    resumed_dispatches = []
    second_model = Model(*changes(), DONE)
    second_client = Decisions(path="rounding.py", inspect=True)
    result = agent(tmp_path, second_model, second_client, dispatched=resumed_dispatches).resume(
        first.session_id, allow_tests=True)
    assert result.status == "completed", result.final_response
    assert second_client.calls == []
    assert not [arguments for name, arguments in resumed_dispatches if name == "read_file"]
    original_actions = [entry for entry in events(result, "decision_action") if entry["action_id"] == client.selected]
    assert len(original_actions) == 1


def test_completion_review_allows_remaining_correction_after_tests_pass(tmp_path):
    root = repository(tmp_path)
    (root / "rounding.py").write_text("def floor_cents(value):\n    return value // 100\n")
    requests = []
    def check(number, options):
        if number == 2:
            text = "\n".join(message.get("content", "") for message in options["messages"])
            assert "Completion review:" in text
            assert "passed" in text
            assert "a * b - 1" in text
            requests.append(text)
    model = Model(edit("calc.py", "a - b", "a * b - 1"),
                  edit("calc.py", "a * b - 1", "a + b"), DONE, check=check)
    result = agent(tmp_path, model, None, decision="off").run(
        "Reproduce the failing tests and implement general integer addition in calc.py. Preserve tests.",
        requirements=TaskRequirements.fix())
    assert result.status == "completed", result.final_response
    assert requests
    assert result.verification == "passed" and result.completion["satisfied"]
    assert (root / "calc.py").read_text() == "def add(a, b):\n    return a + b\n"
    tests = [json.loads(message["content"]) for message in result.messages if message.get("tool_name") == "run_tests"]
    assert [result["status"] for result in tests] == ["failed", "passed", "passed"]
    assert len([message for message in result.messages if message.get("tool_name") == "edit_file"]) == 2

