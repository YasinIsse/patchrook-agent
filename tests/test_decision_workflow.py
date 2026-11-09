"""Decision assistance cannot replace source inspection or observed pytest."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import time

import httpx
import pytest

from patchrook_agent.agent import Agent
from patchrook_agent.completion import TaskRequirements
from patchrook_agent.config import HarnessConfig
from patchrook_agent.decision_assistance import DecisionAssistance
from patchrook_agent.decision_recovery import classify_failure
from patchrook_agent.models.decision import DecisionResult, OllamaDecisionClient
from patchrook_agent.models.ollama import ModelResponse
from patchrook_agent.tools import ToolConfig, ToolRegistry, Workspace

RealHTTPClient = httpx.Client


def edit(old, new):
    return {"role": "assistant", "content": "", "tool_calls": [{"function": {
        "name": "edit_file", "arguments": {"path": "calc.py", "old_text": old, "new_text": new}}}]}


DONE = {"role": "assistant", "content": "DONE"}


class Model:
    model = "test-coding"
    context_size = 8192
    max_output_tokens = 512

    def __init__(self, *responses):
        self.responses = iter(responses)
        self.requests = []

    def chat(self, **options):
        self.requests.append(deepcopy(options))
        return ModelResponse(next(self.responses), 10, 2, None, None, None, 1)


class Decisions:
    def __init__(self, failure=None):
        self.calls = []
        self.failure = failure

    def rank(self, context, candidates, **options):
        self.calls.append((context, candidates, options))
        order = tuple(candidate.id for candidate in reversed(candidates))
        return DecisionResult(order, {item: 1 / len(order) for item in order},
                              self.failure, .01, .008, .002, 20, called=True)


@pytest.fixture(autouse=True)
def no_live(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Automated workflow tests must never contact Ollama")
    monkeypatch.setattr("httpx.Client", forbidden)


def repository(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (root / "helper.py").write_text("def identity(value):\n    return value\n")
    (root / "test_calc.py").write_text("from calc import add\ndef test_add():\n    assert add(3, 2) == 5\n")
    return root


def agent(tmp_path, model, *, decision="on", client=None, pause=None):
    config = replace(HarnessConfig.preset("C"), workflow_mode="coding", decision_model=decision,
                     review_mode="off")
    return Agent(model, registry=ToolRegistry(Workspace(tmp_path / "repo", ToolConfig(allow_tests=True))),
                 config=config, decision_client=client, runs_dir=tmp_path / "runs",
                 pause_after_step=pause, max_steps=5)


def test_disabled_never_constructs_decision_client_and_preserves_original_request(tmp_path, monkeypatch):
    repository(tmp_path)
    def forbidden(*args, **kwargs):
        pytest.fail("Off must not construct a decision client or build the optional index")
    monkeypatch.setattr("patchrook_agent.decision_assistance.OllamaDecisionClient", forbidden)
    monkeypatch.setattr("patchrook_agent.decision_assistance.build_evidence_plan", forbidden)
    injected = Decisions()
    model = Model(edit("a - b", "a + b"), DONE)
    result = agent(tmp_path, model, decision="off", client=injected).run("Fix calc.py", requirements=TaskRequirements.fix())
    assert result.status == "completed" and result.verification == "passed"
    assert injected.calls == [] and result.metrics["decision_calls"] == 0
    assert result.metrics["decision_enabled"] is False
    assert "Suggested investigation" not in json.dumps(model.requests)


def test_enabled_keeps_tools_qwen_options_and_verification_identical(tmp_path):
    root = repository(tmp_path)
    task = "Fix calc.py and preserve helper.py; inspect test_calc.py"
    off = Model(edit("a - b", "a + b"), DONE)
    result_off = agent(tmp_path, off, decision="off").run(task, requirements=TaskRequirements.fix())
    (root / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    on = Model(edit("a - b", "a + b"), DONE)
    client = Decisions()
    result = agent(tmp_path, on, client=client).run(task, requirements=TaskRequirements.fix())
    assert result.status == result_off.status == "completed"
    assert result.verification == result_off.verification == "passed"
    assert client.calls and result.metrics["decision_enabled"]
    assert result.metrics["decision_calls"] == len(client.calls)
    for left, right in zip(off.requests, on.requests):
        assert {key: value for key, value in left.items() if key != "messages"} == {
            key: value for key, value in right.items() if key != "messages"}
    checkpoint = json.loads((Path(result.session_dir) / "checkpoint.json").read_text())
    assert checkpoint["config"]["decision_model"] == "on"
    report = json.loads((Path(result.session_dir) / "report.json").read_text())
    assert report["decision_model"] == "on"
    assert report["metrics"]["decision_calls"] == len(client.calls)


def test_ambiguous_failure_guidance_keeps_exact_error_then_corrects(tmp_path):
    repository(tmp_path)
    model = Model(edit("return a - b", "raise RuntimeError('ambiguous fixture error')"),
                  edit("raise RuntimeError('ambiguous fixture error')", "return a + b"), DONE)
    client = Decisions()
    result = agent(tmp_path, model, client=client).run("Fix calc.py", requirements=TaskRequirements.fix())
    assert result.status == "completed" and result.verification == "passed"
    recoveries = [row for row in client.calls if row[2]["purpose"] == "recovery"]
    # Coding recovery keeps the warm coder resident; Python supplies the
    # deterministic guidance and a concrete repair packet instead of Tev prose.
    assert recoveries == []
    rendered = json.dumps(model.requests[1]["messages"])
    assert "ambiguous fixture error" in rendered and "Suggested investigation" in rendered
    assert "Suggested investigation" not in json.dumps(model.requests[-1]["messages"])


def test_decision_cannot_turn_failed_verification_into_success(tmp_path):
    repository(tmp_path)
    model = Model(edit("return a - b", "raise RuntimeError('still broken')"), DONE, DONE)
    client = Decisions()
    result = agent(tmp_path, model, client=client).run("Fix calc.py", requirements=TaskRequirements.fix())
    assert result.status == "incomplete" and result.verification == "failed"
    assert result.completion["satisfied"] is False


def test_decision_error_falls_back_to_original_workflow(tmp_path):
    repository(tmp_path)
    client = Decisions("timeout")
    model = Model(edit("a - b", "a + b"), DONE)
    result = agent(tmp_path, model, client=client).run("Fix calc.py; inspect helper.py and test_calc.py", requirements=TaskRequirements.fix())
    assert result.status == "completed" and result.verification == "passed"
    assert len(client.calls) == 1 and result.metrics["decision_fallbacks"] == 1


def test_resume_restores_mode_and_avoids_repeating_initial_decisions(tmp_path):
    repository(tmp_path)
    first_client = Decisions()
    first = agent(tmp_path, Model(edit("a - b", "a + b")), client=first_client, pause=1).run(
        "Fix calc.py and preserve helper.py", requirements=TaskRequirements.fix())
    assert first.stop_reason == "paused"
    second_client = Decisions()
    resumed = agent(tmp_path, Model(DONE), decision="off", client=second_client).resume(first.session_id, allow_tests=True)
    assert resumed.status == "completed" and resumed.verification == "passed"
    assert resumed.metrics["decision_enabled"] is True and second_client.calls == []
    assert resumed.metrics["decision_calls"] == first.metrics["decision_calls"]
    first_saved = json.loads((Path(first.session_dir) / "checkpoint.json").read_text())
    assert first_saved["execution"]["decision"]["source_priority"]


@pytest.mark.parametrize("result,category", [
    ({"success": False, "error": {"code": "no_change"}}, "tool"),
    ({"success": False, "status": "timed_out"}, "timeout"),
    ({"success": False, "stdout": "ModuleNotFoundError: No module named example"}, "dependency"),
    ({"success": False, "stderr": "ERROR: usage: unrecognized arguments: bad"}, "configuration"),
    ({"success": False, "stdout": "E   assert 6 == 5"}, "implementation"),
    ({"success": False, "stdout": "SyntaxError: invalid syntax"}, "implementation"),
])
def test_obvious_failures_never_need_a_model(result, category):
    assessment = classify_failure("run_tests", result)
    assert not assessment.ambiguous and assessment.category == category


def test_recovery_calls_bounded_and_repeated_failure_deduplicated(tmp_path):
    root = repository(tmp_path)
    client = Decisions()
    config = replace(HarnessConfig.preset("C"), workflow_mode="coding", decision_model="on")
    controller = DecisionAssistance(config, Model(), Workspace(root), client=client)
    for number in [0, 0, 1, 2, 3]:
        controller.failure("Fix addition", "run_tests", {"success": False, "stdout": f"RuntimeError: {number}", "duration_seconds": time.monotonic()}, 0, time.monotonic() + 60)
    assert len(client.calls) == 2 and controller.recovery_calls == 2
    assert controller.recovery_hint


def test_nested_file_errors_never_call_decision_service(tmp_path):
    root = repository(tmp_path)
    client = Decisions()
    config = replace(HarnessConfig.preset("C"), workflow_mode="coding", decision_model="on")
    controller = DecisionAssistance(config, Model(), Workspace(root), client=client)
    controller.failure("Inspect", "read_code", {"success": False, "files": [
        {"path": "gone.py", "error": {"code": "not_found"}}]}, 0, time.monotonic() + 60)
    assert client.calls == [] and controller.recovery_hint


def test_disappeared_evidence_and_client_exceptions_fall_back(tmp_path, monkeypatch):
    root = repository(tmp_path)
    config = replace(HarnessConfig.preset("C"), workflow_mode="coding", decision_model="on")
    records = []
    controller = DecisionAssistance(config, Model(), Workspace(root), client=Decisions(),
                                    record=lambda purpose, result: records.append(result))
    assert controller.prioritize("Fix", ["gone.py"], time.monotonic() + 60) == ["gone.py"]
    assert records[-1].fallback_reason == "evidence_discovery_error"
    monkeypatch.setattr(controller.client, "rank", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad")))
    assert controller.prioritize("Fix", ["calc.py", "helper.py"], time.monotonic() + 60) == ["calc.py", "helper.py"]
    assert records[-1].fallback_reason == "decision_client_error"


def test_fingerprint_in_actual_request_invalidates_changed_source(tmp_path):
    root = repository(tmp_path)
    config = replace(HarnessConfig.preset("C"), workflow_mode="coding", decision_model="on")
    client = Decisions()
    controller = DecisionAssistance(config, Model(), Workspace(root), client=client)
    controller.prioritize("Fix", ["calc.py", "helper.py"], time.monotonic() + 60)
    before = client.calls[-1][0]
    (root / "calc.py").write_text("def add(a, b):\n    return a * b\n")
    controller.prioritize("Fix", ["calc.py", "helper.py"], time.monotonic() + 60)
    assert client.calls[-1][0] != before


def test_native_initial_decisions_share_phase_and_unload_before_qwen(tmp_path, monkeypatch):
    root = repository(tmp_path)
    (root / "test_helper.py").write_text("from helper import identity\ndef test_identity():\n    assert identity(1) == 1\n")
    events = []
    def handler(request):
        events.append(request.url.path)
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": []})
        payload = json.loads(request.content)
        if request.url.path == "/api/generate":
            return httpx.Response(200, json={"done": True, "done_reason": "unload"})
        assert request.url.path == "/v1/systemone" and payload["keep_alive"] == "1m"
        choices = list(payload["questions"]["rank"]["criteria"])
        return httpx.Response(200, json={"model": "tev1:4b", "answers": {"rank": {
            "type": "choice", "choice": choices[0],
            "probabilities": {item: 1 / len(choices) for item in choices}}}})
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr("httpx.Client", lambda **kwargs: RealHTTPClient(transport=transport, **kwargs))
    class ObservedModel(Model):
        def chat(self, **options):
            events.append("qwen")
            return super().chat(**options)
    model = ObservedModel(edit("a - b", "a + b"), DONE)
    result = agent(tmp_path, model, client=OllamaDecisionClient()).run(
        "Fix calc.py, preserve helper.py, inspect test_calc.py and test_helper.py", requirements=TaskRequirements.fix())
    assert result.status == "completed" and result.verification == "passed"
    assert events == ["/api/ps", "/v1/systemone", "/v1/systemone", "/api/generate", "qwen", "qwen"]
    assert result.metrics["decision_calls"] == 2
    assert result.metrics["decision_management_seconds"] > 0
