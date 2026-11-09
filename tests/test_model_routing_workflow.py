"""Routing uses scripted models; edits, checks and checkpoints remain real.

These assertions establish handoff and verification behavior, not a benefit
from either model. No model endpoint is contacted by this module.
"""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys

import httpx
import pytest

from patchrook_agent.agent import Agent
from patchrook_agent.completion import TaskRequirements
from patchrook_agent.config import HarnessConfig
from patchrook_agent.evidence_workflow import EvidenceWorkflow
from patchrook_agent.models.decision import DecisionResult
from patchrook_agent.models.ollama import ModelResponse, OllamaClient
from patchrook_agent.tools import ToolConfig, ToolRegistry, Workspace
from patchrook_agent.tools.base import ToolError


SMALL = "qwen3:8b"
LARGE = "qwen3:14b"
REAL_HTTP_CLIENT = httpx.Client


def action(tool_name, **arguments):
    return {"role": "assistant", "content": json.dumps({"name": tool_name, "arguments": arguments})}


def read(path="calc.py", **options):
    return action("read_file", path=path, **options)


def edit(old="a - b", new="a + b"):
    return action("apply_edits", edits=[{"path": "calc.py", "old_text": old, "new_text": new}])


def check():
    return action("run_check", name="check_1")


class Coder:
    context_size = 8192
    max_output_tokens = 512
    base_url = "http://offline.invalid:11434"
    temperature = 0.7
    top_p = 0.8
    top_k = 20
    min_p = 0.0

    def __init__(self, name, *steps):
        self.model = name
        self.steps = iter(steps)
        self.requests = []

    def chat(self, messages, **options):
        self.requests.append({"messages": deepcopy(messages), **deepcopy(options)})
        result = next(self.steps, action("finish"))
        if callable(result):
            result = result(messages)
        return ModelResponse(result, 100, 20, 0, 0, 0, 0)


class Router:
    timeout_seconds = 5.0

    def __init__(self, label="small", error=None):
        self.label, self.error, self.requests = label, error, []

    def rank(self, context, candidates, *, purpose, **options):
        self.requests.append({"context": context, "candidates": candidates,
                              "purpose": purpose, **options})
        assert purpose == "model_routing"
        assert {candidate.id for candidate in candidates} == {"small", "large", "uncertain"}
        if self.error:
            raise self.error
        ranking = (self.label, *[label for label in ("small", "large", "uncertain") if label != self.label])
        return DecisionResult(ranking, {label: .9 if label == self.label else .05 for label in ranking},
                              called=True, input_tokens=40)


@pytest.fixture(autouse=True)
def no_model_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Scripted routing test attempted model network access")

    monkeypatch.setattr("httpx.Client", forbidden)


def setup(tmp_path, monkeypatch, small, large, *, router=None, routing="tev", persist=False,
          max_steps=20, pause_after_step=None, commands=None, config_overrides=None, deadline=30):
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    if not (repo / "calc.py").exists():
        (repo / "calc.py").write_text("# retain this contract\ndef add(a, b):\n    return a - b\n")
    if commands is None:
        commands = ((sys.executable, "-c", "from calc import add; assert add(2, 3) == 5"),)
    settings = dict(workflow_mode="evidence", action_protocol="schema", review_mode="off",
                    evidence_policy="rules", model_routing=routing,
                    decision_model="on" if routing == "tev" else "off", escalation_model=LARGE,
                    evidence_test_commands=tuple(commands), persist=persist, checkpoint_enabled=persist)
    settings.update(config_overrides or {})
    config = replace(HarnessConfig.preset("C"), **settings)
    models = {SMALL: small, LARGE: large}
    unloads, loads = [], []

    def make_model(self, name):
        loads.append(name)
        return models[name]

    def unload(self, name, deadline):
        unloads.append(name)

    monkeypatch.setattr(EvidenceWorkflow, "_make_routed_model", make_model)
    monkeypatch.setattr(EvidenceWorkflow, "_unload_model", unload)
    agent = Agent(small, registry=ToolRegistry(Workspace(repo, ToolConfig(allow_tests=True))),
                  config=config, decision_client=router or Router(), runs_dir=tmp_path / "runs",
                  max_steps=max_steps, deadline_seconds=deadline, pause_after_step=pause_after_step)
    return agent, loads, unloads


def text(request):
    return "\n".join(message["content"] for message in request["messages"])


def tool_results(state, name):
    return [json.loads(message["content"]) for message in state.messages
            if message.get("role") == "tool" and message.get("tool_name") == name]


def route(state):
    return state.metrics["model_routing"]


def exposed_names(request):
    if request.get("tools"):
        return {item["function"]["name"] for item in request["tools"]}
    return {option["properties"]["name"]["const"]
            for option in request["response_format"]["oneOf"]}


def test_disabled_routing_never_creates_decision_client_or_switches(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Disabled routing initialized a decision client")

    monkeypatch.setattr("patchrook_agent.evidence_workflow.OllamaDecisionClient", forbidden)
    small, large, router = Coder(SMALL, read(), action("finish")), Coder(LARGE), Router()
    agent, loads, unloads = setup(tmp_path, monkeypatch, small, large, router=router, routing="off")
    state = agent.run("Inspect addition.", requirements=TaskRequirements.inspect())
    assert state.status == "completed"
    assert not router.requests and not large.requests and not loads and not unloads
    assert state.metrics["decision_calls"] == 0


@pytest.mark.parametrize("label", ["small", "uncertain"])
def test_uncertain_and_routine_tasks_stay_on_8b(tmp_path, monkeypatch, label):
    small = Coder(SMALL, read(), edit(), check(), action("finish"))
    large, router = Coder(LARGE), Router(label)
    agent, loads, unloads = setup(tmp_path, monkeypatch, small, large, router=router)
    state = agent.run("Fix addition.", requirements=TaskRequirements.fix())
    assert state.status == "completed" and state.verification == "passed", state.final_response
    assert len(router.requests) == 1 and not large.requests and not loads
    assert route(state)["active_model"] == SMALL
    assert route(state)["initial_route"]["label"] == label
    assert route(state)["repair"]["calls_used"] == 0


def test_router_timeout_falls_back_once_to_8b_and_still_verifies(tmp_path, monkeypatch):
    small = Coder(SMALL, read(), edit(), check(), action("finish"))
    large, router = Coder(LARGE), Router(error=TimeoutError("router timed out"))
    agent, loads, unloads = setup(tmp_path, monkeypatch, small, large, router=router)
    state = agent.run("Fix addition.", requirements=TaskRequirements.fix())
    assert state.status == "completed" and state.verification == "passed", state.final_response
    assert len(router.requests) == 1 and not large.requests and not loads
    assert state.metrics["decision_fallbacks"] >= 1


@pytest.mark.parametrize("include_noop", [False, True])
def test_large_choice_waits_for_source_and_failure_then_returns_after_one_verified_patch(tmp_path, monkeypatch,
                                                                                    include_noop):
    small = Coder(SMALL, read(), check(), action("finish"))
    proposal = edit()
    if include_noop:
        batch = json.loads(proposal["content"])
        batch["arguments"]["edits"].append({"path": "calc.py", "old_text": "a - b", "new_text": "a - b"})
        proposal = {"role": "assistant", "content": json.dumps(batch)}
    large, router = Coder(LARGE, proposal), Router("large")
    agent, loads, unloads = setup(tmp_path, monkeypatch, small, large, router=router)
    state = agent.run("Fix addition for all integers and preserve comments.", requirements=TaskRequirements.fix())
    assert state.status == "completed" and state.verification == "passed", state.final_response
    assert len(small.requests) == 3 and len(large.requests) == 1
    assert loads == [LARGE, SMALL] and unloads == [SMALL, LARGE]
    assert len(router.requests) == 1 and state.steps == 4
    assert exposed_names(large.requests[0]) == {"apply_edits"}
    packet = text(large.requests[0])
    assert "def add(a, b):" in packet and "return a - b" in packet
    assert "AssertionError" in packet
    assert large.context_size == small.context_size == 8192
    assert large.max_output_tokens == small.max_output_tokens == 512
    assert [result["exit_code"] for result in tool_results(state, "run_check")] == [1, 0]
    assert route(state)["active_model"] == SMALL
    assert route(state)["repair"]["status"] == "verified"
    assert route(state)["repair"]["calls_used"] == 1
    assert not state.completion["independent_acceptance_evaluated"]


def test_large_choice_without_failed_check_does_not_start_repair(tmp_path, monkeypatch):
    small, large, router = Coder(SMALL, read(), action("finish")), Coder(LARGE), Router("large")
    agent, loads, unloads = setup(tmp_path, monkeypatch, small, large, router=router)
    state = agent.run("Inspect addition.", requirements=TaskRequirements.inspect())
    assert state.status == "completed"
    assert not large.requests and not loads and not unloads
    assert route(state)["repair"]["calls_used"] == 0


def test_failed_8b_repairs_preserve_patch_and_failure_for_focused_14b(tmp_path, monkeypatch):
    small = Coder(SMALL, check(), read(), edit(new="a * b"), check(),
                  read(), edit(old="a * b", new="a // b"), check(), action("finish"))
    large, router = Coder(LARGE, edit(old="a // b")), Router()
    agent, loads, unloads = setup(tmp_path, monkeypatch, small, large, router=router)
    state = agent.run("Fix addition; reproduce failure and preserve comments.", requirements=TaskRequirements.fix())
    assert state.status == "completed" and state.verification == "passed", state.final_response
    assert len(small.requests) == 8, "The baseline failure must not count as a failed repair"
    assert len(large.requests) == 1 and loads == [LARGE, SMALL]
    packet = text(large.requests[0])
    assert "a // b" in packet and "AssertionError" in packet
    observed = json.loads(large.requests[0]["messages"][-1]["content"].split("\n", 1)[1])
    assert observed["current_patch"]["modified_paths"] == ["calc.py"]
    assert observed["current_patch"]["bytes"] > 0
    assert len(observed["current_patch"]["sha256"]) == 64
    assert "--- a/calc.py" not in packet and "+++ b/calc.py" not in packet
    assert (tmp_path / "repo/calc.py").read_text() == "# retain this contract\ndef add(a, b):\n    return a + b\n"
    assert route(state)["repair"]["calls_used"] == 1
    assert route(state)["repair"]["status"] == "verified"
    assert len(route(state)["switches"]) == 2


def test_old_baseline_failure_cannot_start_repair_after_an_unverified_8b_edit(tmp_path, monkeypatch):
    invalid = {"role": "assistant", "content": "A plan without an enabled action."}

    def forbidden(messages):
        raise AssertionError("A stale baseline failure caused unnecessary 14B inference")

    small = Coder(SMALL, check(), read(), edit(), invalid, invalid, check(), action("finish"))
    large = Coder(LARGE, forbidden)
    agent, loads, unloads = setup(tmp_path, monkeypatch, small, large, router=Router("small"))
    state = agent.run("Reproduce and fix integer addition.", requirements=TaskRequirements.fix())
    assert state.status == "completed" and state.verification == "passed", state.final_response
    assert len(small.requests) == 7 and not large.requests
    assert not loads and not unloads
    assert route(state)["repair"]["calls_used"] == 0
    assert [row["exit_code"] for row in tool_results(state, "run_check")] == [1, 0]


def test_failed_14b_patch_gets_one_retry_with_refreshed_source_and_actual_failure(tmp_path, monkeypatch):
    small = Coder(SMALL, read(), check(), action("finish"))
    large = Coder(LARGE, edit(new="a * b"), edit(old="a * b"))
    agent, loads, unloads = setup(tmp_path, monkeypatch, small, large, router=Router("large"))
    state = agent.run("Fix integer addition and preserve comments.", requirements=TaskRequirements.fix())
    assert state.status == "completed" and state.verification == "passed", state.final_response
    assert len(large.requests) == 2 and len(small.requests) == 3
    assert "return a * b" in text(large.requests[1])
    assert "AssertionError" in text(large.requests[1])
    assert all(exposed_names(request) == {"apply_edits"} for request in large.requests)
    assert [result["exit_code"] for result in tool_results(state, "run_check")] == [1, 1, 0]
    assert route(state)["repair"]["calls_used"] == 2
    assert route(state)["repair"]["status"] == "verified" and loads == [LARGE, SMALL]


def test_second_failed_patch_exhausts_episode_and_does_not_return_or_pingpong(tmp_path, monkeypatch):
    small = Coder(SMALL, read(), check())
    large = Coder(LARGE, edit(new="a * b"), edit(old="a * b", new="a // b"))
    agent, loads, unloads = setup(tmp_path, monkeypatch, small, large, router=Router("large"))
    state = agent.run("Fix addition.", requirements=TaskRequirements.fix())
    assert state.status == "incomplete" and state.verification == "failed"
    assert state.stop_reason == "repair_call_limit"
    assert len(large.requests) == 2 and len(small.requests) == 2
    assert loads == [LARGE] and route(state)["repair"]["status"] == "exhausted"
    assert route(state)["repair"]["calls_used"] == 2


def test_configured_single_repair_call_stops_after_first_failed_patch(tmp_path, monkeypatch):
    small, large = Coder(SMALL, read(), check()), Coder(LARGE, edit(new="a * b"), edit(old="a * b"))
    agent, loads, _ = setup(tmp_path, monkeypatch, small, large, router=Router("large"),
                            config_overrides={"routing_max_repair_calls": 1})
    state = agent.run("Fix addition.", requirements=TaskRequirements.fix())
    assert state.status == "incomplete" and state.stop_reason == "repair_call_limit"
    assert len(large.requests) == 1 and loads == [LARGE]
    assert route(state)["repair"]["calls_used"] == 1
    assert [row["exit_code"] for row in tool_results(state, "run_check")] == [1, 1]


@pytest.mark.parametrize("response", [
    {"role": "assistant", "content": "Here is a plan without an action."},
    read(),
    edit(new="a - b"),
])
def test_invalid_navigation_and_noop_14b_outputs_consume_the_two_call_limit(tmp_path, monkeypatch, response):
    small, large = Coder(SMALL, read(), check()), Coder(LARGE, response, response, response)
    agent, loads, unloads = setup(tmp_path, monkeypatch, small, large, router=Router("large"))
    state = agent.run("Fix addition.", requirements=TaskRequirements.fix())
    assert state.status == "incomplete" and state.verification != "passed"
    assert len(large.requests) == 2 and route(state)["repair"]["calls_used"] == 2
    assert state.stop_reason == "repair_call_limit"
    assert (tmp_path / "repo/calc.py").read_text().endswith("return a - b\n")
    assert len(tool_results(state, "run_check")) == 1, "An unaccepted patch must not manufacture verification"
    assert loads == [LARGE]


def test_all_configured_checks_run_after_each_accepted_repair(tmp_path, monkeypatch):
    commands = ((sys.executable, "-c", "from calc import add; assert add(2, 3) == 5"),
                (sys.executable, "-c", "from calc import double; assert double(7) == 14"))
    small = Coder(SMALL, read(), check(), action("finish"))
    large = Coder(LARGE, edit(), action("apply_edits", edits=[{
        "path": "calc.py", "old_text": "return x - x", "new_text": "return x + x"}]))
    agent, loads, unloads = setup(tmp_path, monkeypatch, small, large, router=Router("large"), commands=commands)
    (tmp_path / "repo/calc.py").write_text("def add(a, b):\n    return a - b\n\ndef double(x):\n    return x - x\n")
    state = agent.run("Fix addition and doubling.", requirements=TaskRequirements.fix())
    assert state.status == "completed" and state.verification == "passed", state.final_response
    checks = tool_results(state, "run_check")
    assert [row["name"] for row in checks] == ["check_1", "check_1", "check_2", "check_1", "check_2"]
    assert [row["exit_code"] for row in checks] == [1, 0, 1, 0, 0]
    assert len(large.requests) == 2
    assert "return a + b" in text(large.requests[1]) and "return x - x" in text(large.requests[1])


def test_focused_call_uses_remaining_task_deadline_without_a_ten_minute_cap(tmp_path, monkeypatch):
    small = Coder(SMALL, read(), check(), action("finish"))
    large = Coder(LARGE, edit())
    agent, _, _ = setup(tmp_path, monkeypatch, small, large, router=Router("large"), deadline=1800)
    state = agent.run("Fix addition.", requirements=TaskRequirements.fix())
    assert state.status == "completed", state.final_response
    assert 1200 < large.requests[0]["timeout_seconds"] <= 1800
    assert route(state)["repair"]["calls_used"] == 1


def test_complete_function_that_cannot_fit_blocks_before_loading_14b(tmp_path, monkeypatch):
    small = Coder(SMALL, read(start_line=1, end_line=2), check())
    large = Coder(LARGE, edit())
    agent, loads, unloads = setup(tmp_path, monkeypatch, small, large, router=Router("large"), persist=True)
    source = "def add(a, b):\n    total = a - b\n" + "    # Preserve this required implementation context.\n" * 500 + "    return total\n"
    (tmp_path / "repo/calc.py").write_text(source)
    state = agent.run("Fix addition while preserving its implementation context.", requirements=TaskRequirements.fix())
    assert state.status == "incomplete" and state.stop_reason == "context_limit", state.final_response
    assert not large.requests and not loads and not unloads
    assert route(state)["repair"]["status"] == "blocked" and route(state)["repair"]["calls_used"] == 0
    assert (tmp_path / "repo/calc.py").read_text() == source


def test_native_focused_repair_uses_patch_only_tools_and_automatic_checks(tmp_path, monkeypatch):
    def native(tool_name, **arguments):
        return {"role": "assistant", "content": "", "tool_calls": [{
            "function": {"name": tool_name, "arguments": arguments}}]}

    small = Coder(SMALL, native("read_file", path="calc.py"), native("run_check", name="check_1"), native("finish"))
    large = Coder(LARGE, native("apply_edits", edits=[{
        "path": "calc.py", "old_text": "a - b", "new_text": "a + b"}]))
    agent, loads, _ = setup(tmp_path, monkeypatch, small, large, router=Router("large"),
                            config_overrides={"action_protocol": "native"})
    state = agent.run("Fix addition.", requirements=TaskRequirements.fix())
    assert state.status == "completed" and state.verification == "passed", state.final_response
    assert len(large.requests) == 1 and loads == [LARGE, SMALL]
    assert exposed_names(large.requests[0]) == {"apply_edits"}
    assert "response_format" not in large.requests[0]
    assert [row["exit_code"] for row in tool_results(state, "run_check")] == [1, 0]


def test_interrupted_repair_generation_is_consumed_before_resume(tmp_path, monkeypatch):
    def interrupted(messages):
        raise KeyboardInterrupt()

    small = Coder(SMALL, read(), check())
    large = Coder(LARGE, interrupted)
    agent, _, _ = setup(tmp_path, monkeypatch, small, large, router=Router("large"), persist=True)
    paused = agent.run("Fix addition.", requirements=TaskRequirements.fix())
    assert paused.stop_reason == "interrupted"
    checkpoint = json.loads((Path(paused.session_dir) / "checkpoint.json").read_text())
    assert checkpoint["execution"]["routing"]["repair"]["calls_used"] == 1
    assert checkpoint["execution"]["routing"]["repair"]["call_inflight"] is True
    assert checkpoint["execution"]["model_settings"]["model"] == LARGE

    resumed_small, resumed_large = Coder(SMALL, action("finish")), Coder(LARGE, edit())
    forbidden_router = Router(error=AssertionError("Resume called Tev"))
    resumed_agent, loads, unloads = setup(tmp_path, monkeypatch, resumed_small, resumed_large,
                                         router=forbidden_router, persist=True)
    resumed_agent.model = resumed_large
    resumed = resumed_agent.resume(paused.session_id, allow_tests=True)
    assert resumed.status == "completed" and resumed.verification == "passed", resumed.final_response
    assert len(resumed_large.requests) == 1 and not forbidden_router.requests
    assert route(resumed)["repair"]["calls_used"] == 2
    assert route(resumed)["active_model"] == SMALL and loads == [SMALL]
    assert route(resumed)["repair"]["call_inflight"] is False
    assert [row["exit_code"] for row in tool_results(resumed, "run_check")] == [1, 1, 0]


def test_interrupted_automatic_checks_resume_before_new_inference_without_replaying_patch(tmp_path, monkeypatch):
    small, large = Coder(SMALL, read(), check()), Coder(LARGE, edit())
    agent, _, _ = setup(tmp_path, monkeypatch, small, large, router=Router("large"), persist=True)
    original_execute = EvidenceWorkflow._execute
    interrupted = False

    def execute(self, state, proposed, deadline):
        nonlocal interrupted
        if (proposed["name"] == "run_check" and self.routing["active_model"] == LARGE
                and not interrupted):
            interrupted = True
            raise KeyboardInterrupt()
        return original_execute(self, state, proposed, deadline)

    monkeypatch.setattr(EvidenceWorkflow, "_execute", execute)
    paused = agent.run("Fix addition.", requirements=TaskRequirements.fix())
    assert paused.stop_reason == "interrupted" and interrupted
    source_after_patch = (tmp_path / "repo/calc.py").read_bytes()
    assert b"return a + b" in source_after_patch
    checkpoint = json.loads((Path(paused.session_dir) / "checkpoint.json").read_text())
    assert checkpoint["execution"]["routing"]["repair"]["pending_checks"]

    resumed_small, resumed_large = Coder(SMALL, action("finish")), Coder(LARGE)
    forbidden_router = Router(error=AssertionError("Resume called Tev"))
    resumed_agent, loads, unloads = setup(tmp_path, monkeypatch, resumed_small, resumed_large,
                                         router=forbidden_router, persist=True)
    resumed_agent.model = resumed_large
    resumed = resumed_agent.resume(paused.session_id, allow_tests=True)
    assert resumed.status == "completed" and resumed.verification == "passed", resumed.final_response
    assert not resumed_large.requests and not forbidden_router.requests
    assert (tmp_path / "repo/calc.py").read_bytes() == source_after_patch
    assert len(tool_results(resumed, "apply_edits")) == 1
    assert route(resumed)["repair"]["calls_used"] == 1 and loads == [SMALL]


def test_resume_after_verified_return_uses_8b_and_does_not_start_second_episode(tmp_path, monkeypatch):
    small, large = Coder(SMALL, read(), check()), Coder(LARGE, edit())
    agent, _, _ = setup(tmp_path, monkeypatch, small, large, router=Router("large"),
                        persist=True, pause_after_step=3, max_steps=6)
    paused = agent.run("Fix addition.", requirements=TaskRequirements.fix())
    assert paused.stop_reason == "paused" and paused.steps == 3
    assert route(paused)["active_model"] == SMALL and route(paused)["repair"]["status"] == "verified"
    resumed_small = Coder(SMALL, read(), check(), action("finish"))
    forbidden_router = Router(error=AssertionError("Resume called Tev"))
    resumed_agent, loads, unloads = setup(tmp_path, monkeypatch, resumed_small, Coder(LARGE),
                                         router=forbidden_router, persist=True)
    resumed = resumed_agent.resume(paused.session_id, allow_tests=True)
    assert resumed.status == "completed" and resumed.verification == "passed", resumed.final_response
    assert not forbidden_router.requests and not loads
    assert route(resumed)["repair"]["calls_used"] == 1 and len(route(resumed)["switches"]) == 2


def test_resume_rechecks_changed_source_before_spending_a_second_repair_call(tmp_path, monkeypatch):
    small, large = Coder(SMALL, read(), check()), Coder(LARGE, edit(new="a * b"))
    agent, _, _ = setup(tmp_path, monkeypatch, small, large, router=Router("large"),
                        persist=True, pause_after_step=3)
    paused = agent.run("Fix addition and preserve comments.", requirements=TaskRequirements.fix())
    assert paused.stop_reason == "paused" and paused.steps == 3
    assert paused.verification == "failed" and route(paused)["active_model"] == LARGE
    assert route(paused)["repair"]["calls_used"] == 1
    assert route(paused)["repair"]["pending_checks"] == []
    source = tmp_path / "repo/calc.py"
    source.write_text(source.read_text().replace("a * b", "a + b"))
    settled_source = source.read_bytes()

    def forbidden(messages):
        raise AssertionError("Resume spent a repair call before checking the changed workspace")

    resumed_small, resumed_large = Coder(SMALL, action("finish")), Coder(LARGE, forbidden)
    forbidden_router = Router(error=AssertionError("Resume called Tev"))
    resumed_agent, loads, unloads = setup(tmp_path, monkeypatch, resumed_small, resumed_large,
                                         router=forbidden_router, persist=True)
    resumed_agent.model = resumed_large
    resumed = resumed_agent.resume(paused.session_id, allow_tests=True)
    assert resumed.status == "completed" and resumed.verification == "passed", resumed.final_response
    assert not resumed_large.requests and not forbidden_router.requests
    assert route(resumed)["repair"]["calls_used"] == 1
    assert route(resumed)["repair"]["status"] == "verified" and route(resumed)["active_model"] == SMALL
    assert loads == [SMALL] and unloads == [LARGE]
    assert source.read_bytes() == settled_source
    assert len(tool_results(resumed, "apply_edits")) == 1
    assert [row["exit_code"] for row in tool_results(resumed, "run_check")] == [1, 1, 0]


def test_later_8b_regression_invalidates_checks_without_starting_another_14b_episode(tmp_path, monkeypatch):
    small = Coder(SMALL, read(), check(), read(), edit(old="a + b", new="a * b"), check(),
                  read(), edit(old="a * b", new="a // b"), check())
    large = Coder(LARGE, edit())
    agent, loads, _ = setup(tmp_path, monkeypatch, small, large, router=Router("large"))
    state = agent.run("Fix addition and verify it.", requirements=TaskRequirements.fix())
    assert state.status == "incomplete" and state.verification == "failed"
    assert state.stop_reason == "routing_retry_limit"
    assert len(large.requests) == 1 and loads == [LARGE, SMALL]
    assert route(state)["repair"]["calls_used"] == 1 and len(route(state)["switches"]) == 2
    assert route(state)["active_model"] == SMALL


def test_source_changed_after_repair_request_cannot_authorize_stale_edit(tmp_path, monkeypatch):
    def externally_changed(messages):
        source = tmp_path / "repo/calc.py"
        source.write_text(source.read_text().replace("a - b", "a / b"))
        return edit()

    small = Coder(SMALL, read(), check())
    large = Coder(LARGE, externally_changed, edit(old="a - b"))
    agent, _, _ = setup(tmp_path, monkeypatch, small, large, router=Router("large"))
    state = agent.run("Fix addition.", requirements=TaskRequirements.fix())
    assert state.status == "incomplete"
    patches = tool_results(state, "apply_edits")
    assert patches and not patches[0]["success"]
    assert not any(patch["success"] for patch in patches)
    assert (tmp_path / "repo/calc.py").read_text().endswith("return a / b\n")
    assert route(state)["repair"]["calls_used"] <= 2


def test_missing_dependency_does_not_start_requested_repair(tmp_path, monkeypatch):
    commands = ((sys.executable, "-c", "import missing_dependency_for_routing_test"),)
    small, large = Coder(SMALL, read(), check()), Coder(LARGE)
    agent, loads, unloads = setup(tmp_path, monkeypatch, small, large, router=Router("large"),
                                commands=commands, max_steps=2)
    state = agent.run("Fix addition.", requirements=TaskRequirements.fix())
    assert state.status == "incomplete" and state.verification == "failed"
    assert not large.requests and not loads
    assert route(state)["repair"]["calls_used"] == 0


def test_interrupted_initial_routing_resumes_without_a_second_tev_call(tmp_path, monkeypatch):
    small, large, router = Coder(SMALL), Coder(LARGE), Router(error=KeyboardInterrupt())
    agent, _, _ = setup(tmp_path, monkeypatch, small, large, router=router, persist=True)
    paused = agent.run("Fix addition.", requirements=TaskRequirements.fix())
    assert paused.stop_reason == "interrupted" and paused.steps == 0
    resumed_small = Coder(SMALL, read(), edit(), check(), action("finish"))
    forbidden_router = Router(error=AssertionError("Resume called Tev"))
    resumed_agent, loads, _ = setup(tmp_path, monkeypatch, resumed_small, Coder(LARGE),
                                   router=forbidden_router, persist=True)
    resumed = resumed_agent.resume(paused.session_id, allow_tests=True)
    assert resumed.status == "completed", resumed.final_response
    assert not forbidden_router.requests and not loads


def test_corrupt_active_model_checkpoint_is_rejected_before_models_or_mutations(tmp_path, monkeypatch):
    small, large = Coder(SMALL, read()), Coder(LARGE)
    agent, _, _ = setup(tmp_path, monkeypatch, small, large, persist=True, pause_after_step=1)
    paused = agent.run("Inspect addition.", requirements=TaskRequirements.inspect())
    checkpoint_path = Path(paused.session_dir) / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["execution"]["routing"]["active_model"] = "unexpected-model:99b"
    checkpoint_path.write_text(json.dumps(checkpoint))
    original = (tmp_path / "repo/calc.py").read_bytes()
    forbidden_router = Router(error=AssertionError("Corrupt checkpoint reached Tev"))
    resumed_small, resumed_large = Coder(SMALL), Coder(LARGE)
    resumed_agent, loads, unloads = setup(tmp_path, monkeypatch, resumed_small, resumed_large,
                                         router=forbidden_router, persist=True)
    with pytest.raises(ValueError):
        resumed_agent.resume(paused.session_id, allow_tests=True)
    assert not resumed_small.requests and not resumed_large.requests and not forbidden_router.requests
    assert not loads and not unloads
    assert (tmp_path / "repo/calc.py").read_bytes() == original


def test_focused_repair_real_http_payloads_unload_14b_before_returning_to_8b(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    config = replace(HarnessConfig.preset("C"), workflow_mode="evidence", action_protocol="schema",
                     review_mode="off", evidence_policy="rules", decision_model="on", model_routing="tev",
                     evidence_test_commands=((sys.executable, "-c", "from calc import add; assert add(2, 3) == 5"),),
                     persist=False, checkpoint_enabled=False)
    steps = {SMALL: iter([read(), check(), action("finish")]), LARGE: iter([edit()])}
    resident = {SMALL, "unrelated:7b"}
    calls, payloads = [], []

    def server(request):
        payload = json.loads(request.content) if request.content else {}
        model, path = payload.get("model"), request.url.path
        calls.append((path, model))
        if path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": name} for name in sorted(resident)]})
        if path == "/api/generate":
            assert payload["keep_alive"] == 0 and model != "unrelated:7b"
            resident.discard(model)
            return httpx.Response(200, json={"done": True, "done_reason": "unload"})
        if path == "/v1/systemone":
            assert model == "tev1:4b" and SMALL not in resident and LARGE not in resident
            return httpx.Response(200, json={"model": model, "answers": {"rank": {
                "type": "choice", "choice": "large", "probabilities": {
                    "small": .05, "large": .9, "uncertain": .05}}}, "usage": {"input_tokens": 80}})
        assert path == "/api/chat" and model in {SMALL, LARGE}
        assert (LARGE if model == SMALL else SMALL) not in resident
        resident.add(model)
        payloads.append(payload)
        return httpx.Response(200, json={"message": next(steps[model]), "prompt_eval_count": 100,
            "eval_count": 20, "load_duration": 7, "total_duration": 1000, "done_reason": "stop"})

    monkeypatch.setattr(httpx, "Client", lambda **options: REAL_HTTP_CLIENT(
        transport=httpx.MockTransport(server), **options))
    agent = Agent(OllamaClient(model=SMALL), registry=ToolRegistry(Workspace(repo, ToolConfig(allow_tests=True))),
                  config=config, runs_dir=tmp_path / "runs", max_steps=8, deadline_seconds=30)
    state = agent.run("Fix addition and verify it.", requirements=TaskRequirements.fix())
    assert state.status == "completed" and state.verification == "passed", state.final_response
    assert [payload["model"] for payload in payloads] == [SMALL, SMALL, LARGE, SMALL]
    assert sum(path == "/v1/systemone" for path, _ in calls) == 1
    assert calls.index(("/api/generate", LARGE)) > calls.index(("/api/chat", LARGE))
    last_base_call = max(index for index, call in enumerate(calls) if call == ("/api/chat", SMALL))
    assert calls.index(("/api/generate", LARGE)) < last_base_call
    assert "unrelated:7b" in resident
    assert all(payload["options"]["num_ctx"] == 8192 and payload["options"]["num_predict"] == 512
               and payload["truncate"] is False and payload["shift"] is False for payload in payloads)
    repair_payload = next(payload for payload in payloads if payload["model"] == LARGE)
    assert {item["properties"]["name"]["const"] for item in repair_payload["format"]["oneOf"]} == {"apply_edits"}
    assert route(state)["model_calls"] == {SMALL: 3, LARGE: 1}
