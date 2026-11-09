"""Lean execution uses real tools and pytest, with no live model inference."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from coding_fixtures import create_fixture, verify_fixture
from patchrook_agent.agent import Agent
from patchrook_agent.completion import TaskRequirements
from patchrook_agent.config import HarnessConfig
from patchrook_agent.models.ollama import ModelResponse
from patchrook_agent.tools import ToolConfig, ToolRegistry, Workspace


class ScriptedModel:
    model = "lean-mechanics"
    context_size = 8192
    max_output_tokens = 512

    def __init__(self, *messages):
        self.messages = iter(messages)
        self.requests = []

    def chat(self, **options):
        self.requests.append(deepcopy(options))
        return ModelResponse(next(self.messages), 0, 0, None, None, None, 0,
                             prompt_tokens_reported=False, completion_tokens_reported=False)


@pytest.fixture(autouse=True)
def no_live_model(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("Lean integration tests must not contact Ollama")
    monkeypatch.setattr("patchrook_agent.models.ollama.httpx.Client", fail)


def action(tool_name, *, protocol="schema", **arguments):
    if protocol == "native":
        return {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": tool_name, "arguments": arguments}}]}
    return {"role": "assistant", "content": json.dumps({"action": "tool", "tool": tool_name, "arguments": arguments})}


def final(protocol="schema"):
    text = "Inspected all requested code, made the intended change, and observed relevant pytest pass."
    return {"role": "assistant", "content": text if protocol == "native" else json.dumps({"action": "complete", "summary": text})}


def make_agent(tmp_path, model, *, protocol="schema", **options):
    root = tmp_path / "repo"
    root.mkdir(exist_ok=True)
    config = replace(HarnessConfig.preset("C"), workflow_mode="lean", action_protocol=protocol)
    return Agent(model=model, registry=ToolRegistry(Workspace(root, ToolConfig(allow_tests=True))),
                 config=config, runs_dir=tmp_path / "runs", max_steps=20, **options)


def can_complete(request):
    return any(branch["properties"]["action"].get("enum") == ["complete"]
               for branch in request["response_format"]["oneOf"])


@pytest.mark.parametrize("protocol", ["native", "schema"])
def test_lean_pipeline_reads_all_operations_and_completes_in_five_scripted_turns(tmp_path, protocol):
    root = tmp_path / "repo"
    root.mkdir()
    task = create_fixture(root, "long-context")
    model = ScriptedModel(
        action("read_code", protocol=protocol, paths=["pipeline/runner.py"]),
        action("read_code", protocol=protocol, paths=[f"pipeline/stage_{i}.py" for i in range(8)]),
        action("edit_file", protocol=protocol, path="pipeline/stage_7.py",
               old_text="return value - 8", new_text="return value + 8"),
        action("run_tests", protocol=protocol, path="tests/test_pipeline.py"), final(protocol),
    )
    state = make_agent(tmp_path, model, protocol=protocol).run(task.instructions, requirements=TaskRequirements.fix())
    acceptance = verify_fixture(root, task, state)
    assert state.status == "completed", (state.stop_reason, state.final_response)
    assert acceptance["passed"], acceptance
    assert state.steps == 5 and state.tool_calls == 4 and state.tool_failures == 0
    assert state.modified_files == ["pipeline/stage_7.py"] and state.verification == "passed"
    batches = [json.loads(m["content"]) for m in state.messages if m.get("tool_name") == "read_code"]
    assert batches[1]["returned_count"] == 8 and not batches[1]["truncated"]
    assert all("def transform" in item["content"] and "return value" in item["content"]
               for item in batches[1]["files"])
    rendered = ["\n".join(m["content"] for m in request["messages"]) for request in model.requests]
    assert all("Historical tool observations" not in text for text in rendered)
    assert all("Available trusted skills" not in text for text in rendered)
    assert all("# Stage 0 note" not in text for text in rendered)
    assert state.metrics["peak_context_utilization"] < .85
    if protocol == "schema":
        assert [can_complete(request) for request in model.requests] == [False, False, False, False, True]
        assert len(json.dumps(model.requests[0]["response_format"])) < 1200


def calculator(tmp_path):
    root = tmp_path / "repo"
    root.mkdir(exist_ok=True)
    (root / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (root / "test_calc.py").write_text("from calc import add\ndef test_add():\n    assert add(3, 2) == 5\n")
    (root / "test_other.py").write_text("def test_other():\n    assert 2 + 2 == 4\n")
    return root


def test_unrelated_green_test_cannot_unlock_completion(tmp_path):
    calculator(tmp_path)
    model = ScriptedModel(action("read_code", paths=["calc.py"]),
                          action("run_tests", path="test_calc.py"),
                          action("run_tests", path="test_other.py"), final())
    state = make_agent(tmp_path, model).run("Fix addition", requirements=TaskRequirements.code())
    assert state.status == "incomplete" and state.stop_reason == "invalid_schema_action"
    assert state.verification == "failed" and not can_complete(model.requests[-1])
    assert any("test_calc.py" in missing for missing in state.completion["missing"])
    context = "\n".join(message["content"] for message in model.requests[-1]["messages"])
    assert "assert 1 == 5" in context and "test_calc.py" in context


def test_failed_scope_survives_edit_until_its_relevant_test_passes(tmp_path):
    calculator(tmp_path)
    model = ScriptedModel(action("read_code", paths=["calc.py"]),
                          action("run_tests", path="test_calc.py"),
                          action("run_tests", path="test_other.py"),
                          action("edit_file", path="calc.py", old_text="a - b", new_text="a + b"),
                          action("run_tests", path="test_calc.py"), final())
    state = make_agent(tmp_path, model).run("Fix addition", requirements=TaskRequirements.fix())
    assert state.status == "completed" and state.verification == "passed"
    assert [can_complete(request) for request in model.requests] == [False] * 5 + [True]


def test_successful_repeated_reads_receive_one_hint_then_stop_and_resume_keeps_limit(tmp_path):
    calculator(tmp_path)
    read = action("read_code", paths=["calc.py"])
    model = ScriptedModel(*[deepcopy(read) for _ in range(6)])
    agent = make_agent(tmp_path, model)
    state = agent.run("Inspect calc.py", requirements=TaskRequirements.inspect())
    assert state.status == "incomplete" and state.stop_reason == "stalled_progress"
    assert state.steps == 5 and len(model.requests) == 5
    events = [json.loads(line) for line in Path(state.session_dir, "events.jsonl").read_text().splitlines()]
    assert sum(event["event"] == "progress_recovery" for event in events) == 1
    assert any("Recent calls added no new evidence" in m["content"] for m in model.requests[3]["messages"])
    resumed_model = ScriptedModel()
    resumed = make_agent(tmp_path, resumed_model).resume(state.session_id, allow_tests=True)
    assert resumed.stop_reason == "stalled_progress" and resumed.steps == 5
    assert not resumed_model.requests


def test_lean_checkpoint_resumes_after_edit_without_repeating_it(tmp_path):
    calculator(tmp_path)
    paused = make_agent(tmp_path, ScriptedModel(
        action("read_code", paths=["calc.py"]),
        action("edit_file", path="calc.py", old_text="a - b", new_text="a + b")), pause_after_step=2).run(
            "Fix addition", requirements=TaskRequirements.fix())
    model = ScriptedModel(action("run_tests", path="test_calc.py"), final())
    # Constructor defaults cannot silently change the saved lean workflow.
    resumed_agent = Agent(model=model, workspace=tmp_path / "repo", runs_dir=tmp_path / "runs")
    state = resumed_agent.resume(paused.session_id, allow_tests=True)
    assert state.status == "completed" and state.verification == "passed"
    assert resumed_agent.config.workflow_mode == "lean"
    assert sum(message.get("tool_name") == "edit_file" for message in state.messages) == 1
    assert [can_complete(request) for request in model.requests] == [False, True]


def test_lean_resume_external_change_discards_old_source_until_reread(tmp_path):
    root = calculator(tmp_path)
    paused = make_agent(tmp_path, ScriptedModel(action("read_code", paths=["calc.py"])),
                        pause_after_step=1).run("Fix addition", requirements=TaskRequirements.fix())
    (root / "calc.py").write_text("def add(a, b):\n    return a * b\n")
    model = ScriptedModel(action("read_code", paths=["calc.py"]),
                          action("edit_file", path="calc.py", old_text="a * b", new_text="a + b"),
                          action("run_tests", path="test_calc.py"), final())
    state = make_agent(tmp_path, model).resume(paused.session_id, allow_tests=True)
    assert state.status == "completed" and state.verification == "passed"
    initial_context = "\n".join(message["content"] for message in model.requests[0]["messages"])
    assert "return a - b" not in initial_context


@pytest.mark.parametrize("name", ["simple-bugfix", "multifile-investigation", "test-diagnosis", "interrupted-resume"])
def test_lean_other_milestone2_tasks_use_real_acceptance(tmp_path, name):
    root = tmp_path / "repo"
    root.mkdir()
    task = create_fixture(root, name)
    requirements = TaskRequirements.fix()
    ending = final()
    if name == "multifile-investigation":
        messages = [action("read_code", paths=["shipping.py", "rates.py"])]
        ending = {"role": "assistant", "content": json.dumps({"action": "complete", "summary":
            "shipping.py quote calls rates.py base_rate: ground 4, express 9; negative weights yield 0."})}
        requirements = TaskRequirements.inspect()
    elif name == "test-diagnosis":
        messages = [action("run_tests", path="tests/test_ids.py"),
                    action("read_code", paths=["ids.py"]),
                    action("edit_file", path="ids.py", old_text="for part in value.split(',')",
                           new_text="for part in value.split(',') if part.strip()"),
                    action("run_tests", path="tests/test_ids.py")]
    else:
        messages = [action("read_code", paths=["math_utils.py"]),
                    action("edit_file", path="math_utils.py", old_text="a - b", new_text="a + b"),
                    action("run_tests", path="tests/test_math_utils.py")]
    if name == "interrupted-resume":
        paused = make_agent(tmp_path, ScriptedModel(*messages[:2]), pause_after_step=2).run(
            task.instructions, requirements=requirements)
        state = make_agent(tmp_path, ScriptedModel(*messages[2:], ending)).resume(paused.session_id, allow_tests=True)
        assert state.metrics["resume_success"] == 1
        assert sum(message.get("tool_name") == "edit_file" for message in state.messages) == 1
    else:
        state = make_agent(tmp_path, ScriptedModel(*messages, ending)).run(task.instructions, requirements=requirements)
    acceptance = verify_fixture(root, task, state)
    assert acceptance["passed"], (state.stop_reason, state.final_response, acceptance)


def test_lean_can_load_requested_skill_without_ordinary_index_overhead(tmp_path):
    calculator(tmp_path)
    model = ScriptedModel(action("read_skill", name="debugging"),
                          action("read_code", paths=["calc.py"]),
                          action("edit_file", path="calc.py", old_text="a - b", new_text="a + b"),
                          action("run_tests", path="test_calc.py"), final())
    state = make_agent(tmp_path, model).run("Use the debugging skill to fix addition.", requirements=TaskRequirements.fix())
    assert state.status == "completed" and state.metrics["skills_loaded"] == 1
    assert state.memory["loaded_skills"] == ["debugging"]


@pytest.mark.parametrize("unknown_tool", ["run_tests", "write_file"])
def test_unknown_interrupted_outcome_cannot_restore_an_old_green_on_read(tmp_path, monkeypatch, unknown_tool):
    root = calculator(tmp_path)
    source = root / "calc.py"
    source.write_text(source.read_text().replace("a - b", "a + b"))
    uncertain = (action("run_tests", path="test_calc.py") if unknown_tool == "run_tests" else
                 action("write_file", path="calc.py", content=source.read_text(), overwrite=True))
    agent = make_agent(tmp_path, ScriptedModel(action("read_code", paths=["calc.py"]),
                                              action("run_tests", path="test_calc.py"), uncertain))
    dispatch = agent.registry.dispatch
    attempts = 0

    def interrupt(name, arguments, **options):
        nonlocal attempts
        if name == unknown_tool:
            attempts += 1
            if attempts == (2 if unknown_tool == "run_tests" else 1):
                raise KeyboardInterrupt
        return dispatch(name, arguments, **options)

    monkeypatch.setattr(agent.registry, "dispatch", interrupt)
    paused = agent.run("Inspect and verify addition", requirements=TaskRequirements.code())
    assert paused.stop_reason == "interrupted" and paused.verification == "passed"
    model = ScriptedModel(action("read_code", paths=["calc.py"]), final())
    state = make_agent(tmp_path, model).resume(paused.session_id, allow_tests=True)
    assert state.stop_reason == "invalid_schema_action" and state.verification == "not_run"
    assert not any(can_complete(request) for request in model.requests)


def test_optional_long_path_hints_cannot_consume_the_request_budget(tmp_path, monkeypatch):
    calculator(tmp_path)
    # Advisory identifiers can approach the tool layer's path bound. They must
    # not displace the task itself or prevent any model request from being sent.
    monkeypatch.setattr("patchrook_agent.agent.pending_local_imports", lambda *args, **kwargs: [
        "a" * 3900 + f"/module_{index}.py" for index in range(8)])
    model = ScriptedModel({"role": "assistant", "content": json.dumps({
        "action": "blocked", "reason": "No source inspection was performed."})})
    state = make_agent(tmp_path, model).run("Inspect available code", requirements=TaskRequirements.inspect())
    assert state.stop_reason == "model_blocked" and len(model.requests) == 1
    assert state.metrics["peak_context_utilization"] < .85
    assert len(model.requests[0]["messages"][-1]["content"]) < 4000

