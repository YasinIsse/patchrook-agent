"""Coding summaries agree with real edits/tests and retain original model text."""

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
    model = "coding-report-mechanics"
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
def no_live_inference(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Coding report tests must not contact Ollama")
    monkeypatch.setattr("patchrook_agent.models.ollama.httpx.Client", forbidden)


def edit(path="calc.py", old_text="a - b", new_text="a + b"):
    return {"role": "assistant", "content": "", "tool_calls": [{"function": {
        "name": "edit_file", "arguments": {"path": path, "old_text": old_text, "new_text": new_text},
    }}]}


def prose(content):
    return {"role": "assistant", "content": content}


def calculator(tmp_path, *, correct=False):
    root = tmp_path / "repo"
    root.mkdir()
    operator = "+" if correct else "-"
    (root / "calc.py").write_text(f"# Preserve this comment.\ndef add(a, b):\n    return a {operator} b\n")
    (root / "test_calc.py").write_text("from calc import add\ndef test_add():\n    assert add(3, 2) == 5\n")
    return root


def agent(tmp_path, root, model, *, max_steps=5):
    config = replace(HarnessConfig.preset("C"), workflow_mode="coding", action_protocol="native")
    return Agent(model=model, registry=ToolRegistry(Workspace(root, ToolConfig(allow_tests=True))),
                 config=config, runs_dir=tmp_path / "runs", max_steps=max_steps)


def last_model_prose(state):
    return next(message["content"] for message in reversed(state.messages)
                if message.get("role") == "assistant" and message.get("content"))


def persisted_report(state):
    return json.loads(Path(state.session_dir, "report.json").read_text())


def test_real_edit_and_passing_tests_override_false_model_no_correction_claim(tmp_path):
    root = calculator(tmp_path)
    false_claim = "All stages were already correct. No correction was needed."
    model = ScriptedModel(edit(), prose(false_claim))
    state = agent(tmp_path, root, model).run("Fix calc.py addition", requirements=TaskRequirements.fix())

    assert state.status == "completed" and state.verification == "passed"
    assert (root / "calc.py").read_text() == "# Preserve this comment.\ndef add(a, b):\n    return a + b\n"
    assert "Changed: calc.py (1 exact edit)." in state.final_response
    assert "Tests: passed" in state.final_response and "1 passed" in state.final_response
    assert "No correction" not in state.final_response and "already correct" not in state.final_response
    assert last_model_prose(state) == false_claim
    assert len(model.requests) == 2  # Reporting itself does not request inference.

    saved = persisted_report(state)
    assert saved["final_response"] == state.final_response
    assert saved["modified_files"] == ["calc.py"] and saved["verification"] == "passed"
    events = [json.loads(line) for line in Path(state.session_dir, "events.jsonl").read_text().splitlines()]
    assert any(event.get("event") == "model_response" and event["message"]["content"] == false_claim
               for event in events)
    assert any(event.get("event") == "execution_summary" and event["summary"] == state.final_response
               for event in events)


def test_false_completion_prose_cannot_turn_real_pytest_failure_into_success(tmp_path):
    root = calculator(tmp_path)
    false_claim = "All tests passed. Everything is complete."
    model = ScriptedModel(edit(new_text="a * b"), prose(false_claim))
    state = agent(tmp_path, root, model, max_steps=2).run("Fix calc.py addition", requirements=TaskRequirements.fix())

    assert state.status == "incomplete" and state.verification == "failed"
    assert state.final_response.startswith("Incomplete.")
    assert "Tests: failed" in state.final_response and "Tests: passed" not in state.final_response
    assert "Remaining:" in state.final_response and "All tests passed" not in state.final_response
    assert last_model_prose(state) == false_claim
    saved = persisted_report(state)
    assert saved["status"] == "incomplete" and saved["verification"] == "failed"
    assert saved["final_response"] == state.final_response


def test_verified_no_change_result_does_not_repeat_model_claimed_edit(tmp_path):
    root = calculator(tmp_path, correct=True)
    original = (root / "calc.py").read_bytes()
    false_claim = "I rewrote calc.py to fix addition."
    model = ScriptedModel(prose(false_claim), prose(false_claim))
    state = agent(tmp_path, root, model).run("Verify calc.py addition", requirements=TaskRequirements.code())

    assert state.status == "completed" and state.verification == "passed"
    assert state.modified_files == [] and (root / "calc.py").read_bytes() == original
    assert "No files changed." in state.final_response and "Changed:" not in state.final_response
    assert "Tests: passed" in state.final_response and "rewrote" not in state.final_response
    assert persisted_report(state)["final_response"] == state.final_response


def test_pipeline_summary_names_every_inspected_stage_after_independent_acceptance(tmp_path):
    root = tmp_path / "repo"
    task = create_fixture(root, "long-context")
    model = ScriptedModel(edit("pipeline/stage_7.py", "return value - 8", "return value + 8"),
                          prose("No correction was needed."))
    state = agent(tmp_path, root, model).run(task.instructions, requirements=TaskRequirements.fix())

    assert verify_fixture(root, task, state)["passed"]
    assert "Changed: pipeline/stage_7.py (1 exact edit)." in state.final_response
    inspected_line = next(line for line in state.final_response.splitlines() if line.startswith("Inspected:"))
    assert "pipeline/runner.py" in inspected_line
    assert all(f"pipeline/stage_{number}.py" in inspected_line for number in range(8))
    assert "Tests: passed" in state.final_response


def test_step_budget_after_verified_patch_reports_incomplete_without_extra_model_call(tmp_path):
    root = calculator(tmp_path)
    model = ScriptedModel(edit())
    state = agent(tmp_path, root, model, max_steps=1).run("Fix calc.py addition", requirements=TaskRequirements.fix())

    assert state.status == "incomplete" and state.stop_reason == "max_steps"
    assert state.verification == "passed" and state.completion["satisfied"]
    assert state.final_response.startswith("Incomplete.")
    assert "Changed: calc.py (1 exact edit)." in state.final_response
    assert "Tests: passed" in state.final_response and "Stopped: max_steps." in state.final_response
    assert len(model.requests) == 1
    saved = persisted_report(state)
    assert saved["status"] == "incomplete" and saved["stop_reason"] == "max_steps"
    assert saved["final_response"] == state.final_response


def test_completed_resume_keeps_factual_summary_without_replaying_model_or_tools(tmp_path):
    root = calculator(tmp_path)
    false_claim = "No code was changed."
    completed = agent(tmp_path, root, ScriptedModel(edit(), prose(false_claim))).run(
        "Fix calc.py addition", requirements=TaskRequirements.fix())
    model = ScriptedModel()
    resumed = Agent(model=model, workspace=root, runs_dir=tmp_path / "runs").resume(
        completed.session_id, allow_tests=True)

    assert resumed.status == "completed" and resumed.final_response == completed.final_response
    assert "Changed: calc.py (1 exact edit)." in resumed.final_response
    assert last_model_prose(resumed) == false_claim and model.requests == []
    assert resumed.tool_calls == completed.tool_calls
    assert persisted_report(resumed)["final_response"] == resumed.final_response


def test_inspection_only_coding_report_preserves_the_semantic_answer_and_factual_evidence(tmp_path):
    root = tmp_path / "repo"
    task = create_fixture(root, "multifile-investigation")
    answer = ("The entry function is shipping.py:quote. It calls rates.py:base_rate. "
              "The ground service rate is 4 and the express rate is 9. "
              "quote clamps negative weights to 0, so a negative weight produces a quote of 0.")
    model = ScriptedModel(prose(answer))
    state = agent(tmp_path, root, model).run(task.instructions, requirements=TaskRequirements.inspect())

    assert state.status == "completed" and state.verification == "not_run"
    assert state.modified_files == []
    assert "No files changed." in state.final_response and "Inspected:" in state.final_response
    assert "shipping.py" in state.final_response and "rates.py" in state.final_response
    assert answer in state.final_response
    assert last_model_prose(state) == answer and len(model.requests) == 1
    assert not any(message.get("tool_name") == "run_tests" for message in state.messages)
    assert all((root / path).read_text() == contents for path, contents in task.files.items())
    acceptance = verify_fixture(root, task, state)
    assert acceptance["passed"], acceptance
    assert persisted_report(state)["final_response"] == state.final_response


def test_diagnosis_report_keeps_observed_baseline_failure_and_post_patch_pass(tmp_path):
    root = tmp_path / "repo"
    task = create_fixture(root, "test-diagnosis")
    model = ScriptedModel(edit("ids.py", "int(part) for part in value.split(',')",
                               "int(part.strip()) for part in value.split(',') if part.strip()"),
                          prose("The tests were always passing; no reproduction was needed."))
    state = agent(tmp_path, root, model).run(task.instructions, requirements=TaskRequirements.fix())

    acceptance = verify_fixture(root, task, state)
    assert acceptance["passed"], acceptance
    assert "Before edits: pytest . failed (1 failed)." in state.final_response
    assert "Tests: passed — pytest . (1 passed)." in state.final_response
    assert "always passing" not in state.final_response
    assert persisted_report(state)["final_response"] == state.final_response
