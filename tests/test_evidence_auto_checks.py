"""Optional post-patch checks use real argv and no external model endpoint."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest

from patchrook_agent.agent import Agent
from patchrook_agent.completion import TaskRequirements
from patchrook_agent.config import HarnessConfig
from patchrook_agent.evidence_workflow import EvidenceWorkflow
from patchrook_agent.evidence_verification import run_base_checks
from patchrook_agent.model_routing import RoutingProgress
from patchrook_agent.models.decision import DecisionResult
from patchrook_agent.models.ollama import ModelResponse
from patchrook_agent.tools import ToolConfig, ToolRegistry, Workspace


def action(name, **arguments):
    return {"role": "assistant", "content": json.dumps({"name": name, "arguments": arguments})}


def edit(old="VALUE = 1", new="VALUE = 2"):
    return action("apply_edits", edits=[{"path": "subject.py", "old_text": old, "new_text": new}])


class Coder:
    context_size = 8192
    max_output_tokens = 512
    def __init__(self, *responses, model="qwen3:8b"):
        self.model, self.responses, self.requests = model, iter(responses), []
    def chat(self, messages, **options):
        self.requests.append(deepcopy(messages))
        return ModelResponse(next(self.responses), 100, 20, 0, 0, 0, 0)


class Router:
    timeout_seconds = 5
    def rank(self, *args, **kwargs):
        return DecisionResult(("uncertain", "small", "large"),
                              {"uncertain": .9, "small": .05, "large": .05}, called=True)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Post-patch checks must not contact model endpoints")
    monkeypatch.setattr("httpx.Client", forbidden)
    monkeypatch.setattr("httpx.get", forbidden)
    monkeypatch.setattr("httpx.post", forbidden)


def make_agent(tmp_path, model, *, automatic=True, commands=None, max_steps=8,
               pause_after_step=None, deadline=30, routed=False):
    root = tmp_path / "repository"
    root.mkdir(exist_ok=True)
    source = root / "subject.py"
    if not source.exists():
        source.write_text("VALUE = 1\n")
    if commands is None:
        commands = ((sys.executable, "-c", "from subject import VALUE; assert VALUE == 2, 'expected VALUE=2'"),
                    (sys.executable, "-c", "print('preservation passed')"))
    base = HarnessConfig.preset("C")
    config = replace(base, workflow_mode="evidence", action_protocol="schema",
        evidence_auto_check_edits=automatic, evidence_test_commands=tuple(commands),
        model_routing="tev" if routed else "off", decision_model="on" if routed else "off",
        review_mode="off", context=replace(base.context, enabled=False), compaction_enabled=False,
        persist=True, checkpoint_enabled=True)
    return Agent(model, registry=ToolRegistry(Workspace(root, ToolConfig(allow_tests=True))),
                 config=config, runs_dir=tmp_path / "runs", max_steps=max_steps,
                 deadline_seconds=deadline, pause_after_step=pause_after_step,
                 decision_client=Router() if routed else None)


def results(state, name="run_check"):
    return [json.loads(message["content"]) for message in state.messages
            if message.get("role") == "tool" and message.get("tool_name") == name]


def events(state):
    return [json.loads(line) for line in (Path(state.session_dir) / "events.jsonl").read_text().splitlines()]


def test_default_off_preserves_explicit_model_control(tmp_path):
    assert HarnessConfig().evidence_auto_check_edits is False
    coder = Coder(action("read_file", path="subject.py"), edit(), action("finish"))
    state = make_agent(tmp_path, coder, automatic=False, max_steps=3).run(
        "Fix the value.", requirements=TaskRequirements.fix())
    assert results(state) == [] and state.verification == "not_run"
    assert state.status == "incomplete"


def test_all_trusted_checks_follow_patch_before_the_next_model_request(tmp_path):
    coder = Coder(action("read_file", path="subject.py"), edit(), action("finish"))
    state = make_agent(tmp_path, coder, max_steps=3).run("Fix the value.", requirements=TaskRequirements.fix())
    assert state.status == "completed" and state.verification == "passed"
    assert len(coder.requests) == 3 and len(results(state)) == 2
    assert "preservation passed" in "\n".join(message["content"] for message in coder.requests[-1])
    recorded = events(state)
    checks = [item for item in recorded if item["event"] == "tool_request" and item["name"] == "run_check"]
    assert all(item["automatic"] and item["step"] == 2 for item in checks)
    assert all(item["verification_kind"] == "configured_command" for item in results(state))
    assert all(item["revision"] == 1 for item in state.metrics["evidence_check_results"].values())


def test_a_partial_accepted_patch_retains_the_exact_failed_assertion(tmp_path):
    coder = Coder(action("read_file", path="subject.py"), edit(new="VALUE = 3"), action("finish"))
    state = make_agent(tmp_path, coder, max_steps=3).run("Fix the value.", requirements=TaskRequirements.fix())
    assert state.status == "incomplete" and state.verification == "failed"
    assert [item["success"] for item in results(state)] == [False, True]
    assert "expected VALUE=2" in "\n".join(message["content"] for message in coder.requests[-1])


def test_baseline_check_records_are_replaced_for_the_new_revision(tmp_path):
    coder = Coder(action("read_file", path="subject.py"), edit(), action("finish"))
    agent = make_agent(tmp_path, coder, max_steps=3)
    agent.config = replace(agent.config, evidence_initial_checks=True)
    state = agent.run("Fix the value.", requirements=TaskRequirements.fix())
    assert state.status == "completed" and state.verification == "passed"
    assert [item["success"] for item in results(state)] == [False, True, True, True]
    assert all(item["revision"] == 1 for item in state.metrics["evidence_check_results"].values())


@pytest.mark.parametrize("proposal", [edit(new="VALUE = 1"), edit(old="missing")])
def test_rejected_or_noop_patch_does_not_run_automatic_checks(tmp_path, proposal):
    coder = Coder(action("read_file", path="subject.py"), proposal)
    state = make_agent(tmp_path, coder, max_steps=2).run("Fix the value.", requirements=TaskRequirements.fix())
    assert results(state) == [] and state.modified_files == []


def test_checks_still_run_after_the_last_permitted_model_generation(tmp_path):
    coder = Coder(action("read_file", path="subject.py"), edit())
    state = make_agent(tmp_path, coder, max_steps=2).run("Fix the value.", requirements=TaskRequirements.fix())
    assert state.stop_reason == "max_steps" and state.verification == "passed"
    assert len(coder.requests) == 2 and len(results(state)) == 2


def test_actual_infrastructure_failure_stops_before_another_model_request(tmp_path):
    commands = ((sys.executable, "-c", "import missing_optional_test_dependency"),
                (sys.executable, "-c", "print('remaining trusted check ran')"))
    coder = Coder(action("read_file", path="subject.py"), edit())
    state = make_agent(tmp_path, coder, commands=commands).run("Fix the value.", requirements=TaskRequirements.fix())
    assert state.stop_reason == "base_check_environment_failure" and len(coder.requests) == 2
    assert len(results(state)) == 2 and results(state)[1]["success"] is True


def test_check_that_changes_workspace_is_not_validation_success(tmp_path):
    commands = ((sys.executable, "-c", "from pathlib import Path; Path('subject.py').write_text('VALUE = 3\\n')"),
                (sys.executable, "-c", "from subject import VALUE; assert VALUE == 3"))
    coder = Coder(action("read_file", path="subject.py"), edit())
    state = make_agent(tmp_path, coder, commands=commands).run("Fix the value.", requirements=TaskRequirements.fix())
    assert state.stop_reason == "base_check_environment_failure"
    assert results(state)[0]["error"]["code"] == "workspace_changed_during_check"
    assert state.status == "incomplete"


def test_existing_deadline_bounds_checks_without_another_model_call(tmp_path):
    coder = Coder(action("read_file", path="subject.py"), edit())
    state = make_agent(tmp_path, coder, deadline=.4,
        commands=((sys.executable, "-c", "import time; time.sleep(5)"),)).run(
            "Fix the value.", requirements=TaskRequirements.fix())
    assert state.stop_reason == "deadline" and len(coder.requests) == 2


def test_two_real_failed_base_patches_trigger_existing_focused_repair(tmp_path, monkeypatch):
    small = Coder(action("read_file", path="subject.py"), edit(new="VALUE = 2"),
        action("read_file", path="subject.py"), edit(old="VALUE = 2", new="VALUE = 3"), action("finish"))
    large = Coder(edit(old="VALUE = 3", new="VALUE = 4"), model="qwen3:14b")
    monkeypatch.setattr(EvidenceWorkflow, "_make_routed_model", lambda self, name: large if name == large.model else small)
    monkeypatch.setattr(EvidenceWorkflow, "_unload_model", lambda *args: None)
    agent = make_agent(tmp_path, small, routed=True,
        commands=((sys.executable, "-c", "from subject import VALUE; assert VALUE == 4, 'expected VALUE=4'"),))
    state = agent.run("Fix the value.", requirements=TaskRequirements.fix())
    assert state.status == "completed" and state.verification == "passed", state.final_response
    assert len(large.requests) == 1 and len(small.requests) == 5
    assert [item["success"] for item in results(state)] == [False, False, True]
    assert state.metrics["model_routing"]["switches"][0]["reason"] == "failed_repairs"
    assert state.metrics["model_routing"]["repair"]["calls_used"] == 1


def test_resume_refreshes_missing_checks_before_model_without_replaying_the_patch(tmp_path, monkeypatch):
    original = EvidenceWorkflow._perform_action
    interrupted = False
    def interrupt_after_first_check(self, state, proposed, deadline, **options):
        nonlocal interrupted
        result = original(self, state, proposed, deadline, **options)
        if proposed["name"] == "run_check" and self.revision == 1 and not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        return result
    monkeypatch.setattr(EvidenceWorkflow, "_perform_action", interrupt_after_first_check)
    first = Coder(action("read_file", path="subject.py"), edit())
    paused = make_agent(tmp_path, first).run("Fix the value.", requirements=TaskRequirements.fix())
    assert paused.stop_reason == "interrupted" and len(results(paused)) == 1
    monkeypatch.setattr(EvidenceWorkflow, "_perform_action", original)
    following = Coder(action("read_file", path="subject.py"), action("finish"))
    resumed = make_agent(tmp_path, following).resume(paused.session_id, allow_tests=True)
    assert resumed.status == "completed" and resumed.verification == "passed", resumed.final_response
    assert len(results(resumed, "apply_edits")) == 1
    assert len(results(resumed)) == 3, "Resume deliberately invalidates earlier verification"
    assert "preservation passed" in "\n".join(message["content"] for message in following.requests[0])


@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_config_flag_is_strict_boolean(value):
    with pytest.raises(ValueError, match="evidence_auto_check_edits"):
        replace(HarnessConfig(), evidence_auto_check_edits=value)


def test_auto_check_setting_is_opt_in_and_config_round_trips():
    assert HarnessConfig().evidence_auto_check_edits is False
    config = HarnessConfig(evidence_auto_check_edits=True)
    assert HarnessConfig.from_dict(config.to_dict()).evidence_auto_check_edits is True


def test_automatic_check_count_cannot_spend_model_progress_counters():
    config = replace(HarnessConfig.preset("C"), workflow_mode="evidence", model_routing="tev",
                     decision_model="on", evidence_auto_check_edits=True)
    progress = RoutingProgress()
    failure = {"success": False, "exit_code": 1, "stdout": "AssertionError", "verification_kind": "configured_command"}
    commands = {f"check_{number}": ("python3", "-c", "pass") for number in range(16)}
    for name in commands:
        progress.observe("run_check", {"name": name}, failure, 1, config)
    progress.invalid_responses = progress.repeated_actions = 1
    reasons = []
    value = SimpleNamespace(config=config, revision=1, tools=SimpleNamespace(commands=commands),
        routing={"active_model": "qwen3:8b"}, routing_progress=progress,
        state=SimpleNamespace(steps=1), enabled={"run_check"}, checks={},
        session=SimpleNamespace(emit=lambda *args, **kwargs: None),
        _fingerprint=lambda deadline: "a" * 64, _update_state=lambda state: None,
        _verification=lambda: "failed", _maybe_start_repair=lambda deadline: False)
    def perform(state, proposed, deadline, **options):
        value.checks[proposed["arguments"]["name"]] = {"revision": 1, "snapshot": "a" * 64, "passed": False}
        return dict(failure)
    value._perform_action = perform
    value._routing_signal = lambda reason, deadline: reasons.append(reason) or False
    assert not run_base_checks(value, time.monotonic() + 10)
    assert reasons == [None]
    assert progress.invalid_responses == progress.repeated_actions == 1
    assert len(progress.failed_repairs) == 16
    assert RoutingProgress.from_dict(progress.to_dict()).to_dict() == progress.to_dict()
