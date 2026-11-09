"""Independent end-to-end state guards for the evidence coding runtime.

Models are scripted here; named checks are real subprocesses against disposable
repositories. These tests establish runtime bookkeeping, not model capability.
"""

from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import py_compile
import re
import sys

import pytest

from patchrook_agent.agent import Agent
from patchrook_agent.completion import TaskRequirements
from patchrook_agent.config import HarnessConfig
from patchrook_agent.models.ollama import ModelResponse
from patchrook_agent.tools import ToolConfig, ToolRegistry, Workspace


def action(tool_name, **arguments):
    return {"role": "assistant", "content": json.dumps({"name": tool_name, "arguments": arguments})}


def read(path="calc.py", **options):
    return action("read_file", path=path, **options)


def edit(path="calc.py", old="a - b", new="a + b"):
    return {"path": path, "old_text": old, "new_text": new}


class ScriptedModel:
    model = "offline-evidence-guard-tests"
    context_size = 8192
    max_output_tokens = 512

    def __init__(self, *steps):
        self.steps = iter(steps)
        self.requests = []

    def chat(self, messages, **options):
        self.requests.append({"messages": deepcopy(messages), **deepcopy(options)})
        result = next(self.steps, action("finish"))
        if callable(result):
            result = result(messages)
        return ModelResponse(result, 100, 20, 0, 0, 0, 0)


def make_agent(tmp_path, model, *, commands=None, policy="rules", max_steps=None,
               persist=False, pause_after_step=None, config_overrides=None):
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    if commands is None:
        commands = ((sys.executable, "-c", "from calc import add; assert add(2, 3) == 5"),)
    config = replace(HarnessConfig.preset("C"), workflow_mode="evidence", action_protocol="schema",
                     evidence_policy=policy, evidence_test_commands=tuple(tuple(cmd) for cmd in commands),
                     review_mode="off", persist=persist, checkpoint_enabled=persist,
                     **(config_overrides or {}))
    return Agent(model, registry=ToolRegistry(Workspace(repo, ToolConfig(allow_tests=True))),
                 config=config, runs_dir=tmp_path / "runs", max_steps=max_steps or 20,
                 deadline_seconds=30, pause_after_step=pause_after_step)


def observations(state, name):
    return [json.loads(message["content"]) for message in state.messages
            if message.get("role") == "tool" and message.get("tool_name") == name]


def request_text(request):
    return "\n".join(item["content"] for item in request["messages"])


def make_calc(repo):
    (repo / "calc.py").write_text("# retain this comment\ndef add(a, b):\n    return a - b\n")


def test_real_multi_file_batch_requires_every_configured_check_then_explicit_finish(tmp_path):
    model = ScriptedModel(read(), read("other.py"),
        action("apply_edits", edits=[edit(), edit("other.py", "x - x", "x + x")]),
        action("run_check", name="check_1"), action("finish"),
        action("run_check", name="check_2"), action("finish"))
    commands = ((sys.executable, "-c", "from calc import add; assert add(2, 3) == 5"),
                (sys.executable, "-c", "from other import double; assert double(7) == 14"))
    agent = make_agent(tmp_path, model, commands=commands, max_steps=7)
    repo = agent.registry.workspace.root
    make_calc(repo)
    (repo / "other.py").write_text("def double(x):\n    return x - x\n")
    state = agent.run("Fix addition and doubling; preserve comments.", requirements=TaskRequirements.fix())
    assert state.status == "completed", state.final_response
    assert state.verification == "passed"
    assert state.modified_files == ["calc.py", "other.py"]
    assert observations(state, "finish")[0]["error"]["code"] == "completion_requirements"
    assert [record["exit_code"] for record in observations(state, "run_check")] == [0, 0]
    assert (repo / "calc.py").read_text().startswith("# retain this comment\n")
    assert state.metrics["evidence_revision"] == 1
    assert not state.completion["independent_acceptance_evaluated"]


@pytest.mark.parametrize("prose", ["DONE", "All changes are complete; tests passed.", ""])
def test_model_prose_cannot_complete_or_fabricate_checks(tmp_path, prose):
    model = ScriptedModel({"role": "assistant", "content": prose}, action("finish"))
    agent = make_agent(tmp_path, model, max_steps=2)
    make_calc(agent.registry.workspace.root)
    state = agent.run("Fix addition.", requirements=TaskRequirements.fix())
    assert state.status == "incomplete"
    assert state.verification == "not_run"
    assert not state.modified_files
    assert not observations(state, "run_check")
    assert observations(state, "finish")[0]["error"]["code"] == "completion_requirements"


def test_source_read_earlier_but_omitted_from_latest_request_cannot_authorize_patch(tmp_path):
    steps = [read(f"noise{number}.py") for number in range(8)]
    steps.append(action("apply_edits", edits=[edit("noise0.py", "VALUE_0 = 1", "VALUE_0 = 2")]))
    model = ScriptedModel(*steps)
    agent = make_agent(tmp_path, model, policy="window", max_steps=len(steps))
    repo = agent.registry.workspace.root
    for number in range(8):
        (repo / f"noise{number}.py").write_text("# " + ("irrelevant context " * 60) + f"\nVALUE_{number} = 1\n")
    state = agent.run("Adjust the requested behavior and run configured checks.", requirements=TaskRequirements.fix())
    assert len(model.requests) == len(steps), state.final_response
    assert "VALUE_0 = 1" not in request_text(model.requests[-1])
    result = observations(state, "apply_edits")[-1]
    assert result["error"]["code"] == "source_not_presented"
    assert (repo / "noise0.py").read_text().endswith("VALUE_0 = 1\n")
    assert state.status == "incomplete"


def test_requested_archived_evidence_reaches_next_request_without_granting_edit_authority(tmp_path):
    def fetch_old_card(messages):
        first_read_request = request_text(model.requests[1])
        match = re.search(r"Requested source \(data\) (ev_[0-9a-f]+): noise0.py", first_read_request)
        assert match, "First read did not have a retrievable source card"
        return action("read_evidence", id=match.group(1), max_chars=1800)

    model = ScriptedModel(*[read(f"noise{number}.py") for number in range(8)], fetch_old_card,
        action("apply_edits", edits=[edit("noise0.py", "VALUE_0 = 1", "VALUE_0 = 2")]))
    agent = make_agent(tmp_path, model, policy="window", max_steps=10)
    repo = agent.registry.workspace.root
    for number in range(8):
        (repo / f"noise{number}.py").write_text("# " + ("irrelevant context " * 60) + f"\nVALUE_{number} = 1\n")
    state = agent.run("Inspect archived facts and preserve source until freshly read.", requirements=TaskRequirements.fix())
    assert "VALUE_0 = 1" not in request_text(model.requests[8])
    assert "VALUE_0 = 1" in request_text(model.requests[9])
    result = observations(state, "read_evidence")[0]
    assert result["success"] and result["edit_authorization"] is False
    assert observations(state, "apply_edits")[0]["error"]["code"] == "source_not_presented"
    assert (repo / "noise0.py").read_text().endswith("VALUE_0 = 1\n")


def test_external_edit_after_successful_check_invalidates_finish(tmp_path):
    def change_after_check(messages):
        target = tmp_path / "repo" / "unrelated.py"
        previous = target.stat()
        target.write_text("external = 2\n")
        # Same size and restored mtime must not make old verification reusable.
        os.utime(target, ns=(previous.st_atime_ns, previous.st_mtime_ns))
        return action("finish")

    model = ScriptedModel(read(), action("apply_edits", edits=[edit()]),
                          action("run_check", name="check_1"), change_after_check)
    agent = make_agent(tmp_path, model, max_steps=4)
    make_calc(agent.registry.workspace.root)
    (agent.registry.workspace.root / "unrelated.py").write_text("external = 1\n")
    state = agent.run("Fix addition.", requirements=TaskRequirements.fix())
    assert observations(state, "run_check")[0]["exit_code"] == 0
    assert observations(state, "finish")[0]["error"]["code"] == "workspace_changed"
    assert state.status == "incomplete" and state.verification == "not_run"


def test_a_check_that_mutates_source_cannot_verify_its_own_output(tmp_path):
    command = (sys.executable, "-c", "from pathlib import Path; Path('calc.py').write_text('value = 999\\n')")
    model = ScriptedModel(read(), action("run_check", name="check_1"), action("finish"))
    agent = make_agent(tmp_path, model, commands=(command,), max_steps=3)
    make_calc(agent.registry.workspace.root)
    state = agent.run("Verify existing behavior.", requirements=TaskRequirements.code())
    check = observations(state, "run_check")[0]
    assert check["exit_code"] == 0
    assert not check["success"] and check["error"]["code"] == "workspace_changed_during_check"
    assert state.status == "incomplete" and state.verification == "failed"


def test_nonzero_check_cannot_pass_by_printing_success_claims(tmp_path):
    command = (sys.executable, "-c", "import sys; print('100 passed in 0.01s; ALL CHECKS PASSED'); sys.exit(7)")
    model = ScriptedModel(read(), action("run_check", name="check_1"), action("finish"))
    agent = make_agent(tmp_path, model, commands=(command,), max_steps=3)
    make_calc(agent.registry.workspace.root)
    state = agent.run("Inspect and verify the implementation.", requirements=TaskRequirements.code())
    assert observations(state, "run_check")[0]["exit_code"] == 7
    assert state.verification == "failed" and state.status == "incomplete"


def test_new_patch_invalidates_a_previous_successful_check(tmp_path):
    model = ScriptedModel(read(), action("apply_edits", edits=[edit()]),
        action("run_check", name="check_1"), read(),
        action("apply_edits", edits=[edit(old="a + b", new="b + a")]), action("finish"),
        action("run_check", name="check_1"), action("finish"))
    agent = make_agent(tmp_path, model, max_steps=8)
    make_calc(agent.registry.workspace.root)
    state = agent.run("Fix addition.", requirements=TaskRequirements.fix())
    assert observations(state, "finish")[0]["error"]["code"] == "completion_requirements"
    assert state.status == "completed" and state.verification == "passed"
    assert state.metrics["evidence_revision"] == 2
    assert len(observations(state, "run_check")) == 2


def test_failed_baseline_then_same_size_patch_is_checked_against_current_source(tmp_path):
    model = ScriptedModel(action("run_check", name="check_1"), read(),
        action("apply_edits", edits=[edit()]), action("run_check", name="check_1"), action("finish"))
    agent = make_agent(tmp_path, model, max_steps=5)
    make_calc(agent.registry.workspace.root)
    state = agent.run("Reproduce and fix addition.", requirements=TaskRequirements.fix())
    checks = observations(state, "run_check")
    assert checks[0]["exit_code"] != 0
    assert checks[1]["exit_code"] == 0
    assert state.status == "completed" and state.verification == "passed"


def test_existing_timestamp_bytecode_cache_cannot_hide_same_size_source_patch(tmp_path):
    original_timestamp = 1_750_000_000

    def retain_original_mtime_then_check(messages):
        os.utime(tmp_path / "repo/calc.py", (original_timestamp, original_timestamp))
        return action("run_check", name="check_1")

    model = ScriptedModel(action("run_check", name="check_1"), read(),
        action("apply_edits", edits=[edit()]), retain_original_mtime_then_check, action("finish"))
    agent = make_agent(tmp_path, model, max_steps=5)
    source = agent.registry.workspace.root / "calc.py"
    make_calc(agent.registry.workspace.root)
    os.utime(source, (original_timestamp, original_timestamp))
    py_compile.compile(str(source), doraise=True)
    state = agent.run("Reproduce and fix addition.", requirements=TaskRequirements.fix())
    checks = observations(state, "run_check")
    assert checks[0]["exit_code"] != 0
    assert checks[1]["exit_code"] == 0, "Verification imported stale preexisting bytecode after an exact same-size patch"
    assert state.status == "completed" and state.verification == "passed"


def test_latest_failed_check_witness_survives_unrelated_read_pressure(tmp_path):
    witness = "ORIGINAL_FAILURE_WITNESS_assert_invoice_total"
    command = (sys.executable, "-c", f"import sys; print({witness!r}); sys.exit(1)")
    steps = [action("run_check", name="check_1"), *[read(f"noise{number}.py") for number in range(8)],
             action("finish")]
    model = ScriptedModel(*steps)
    agent = make_agent(tmp_path, model, commands=(command,), policy="window", max_steps=len(steps))
    repo = agent.registry.workspace.root
    for number in range(8):
        (repo / f"noise{number}.py").write_text("# " + ("unrelated data " * 65) + f"\nVALUE_{number} = 1\n")
    state = agent.run("Repair behavior using actual failed checks.", requirements=TaskRequirements.fix())
    assert observations(state, "run_check")[0]["exit_code"] == 1
    assert len(model.requests) == len(steps), state.final_response
    assert witness in request_text(model.requests[-1]), "Latest failing check was evicted by unrelated successful reads"
    assert state.verification == "failed"


def test_resume_requires_fresh_inspection_and_reverification(tmp_path):
    initial = ScriptedModel(read(), action("apply_edits", edits=[edit()]), action("run_check", name="check_1"))
    agent = make_agent(tmp_path, initial, persist=True, pause_after_step=3)
    make_calc(agent.registry.workspace.root)
    paused = agent.run("Fix addition.", requirements=TaskRequirements.fix())
    assert paused.stop_reason == "paused" and paused.verification == "passed"
    resumed_model = ScriptedModel(action("finish"), read(), action("run_check", name="check_1"), action("finish"))
    resumed = make_agent(tmp_path, resumed_model, persist=True).resume(paused.session_id, allow_tests=True)
    finish_results = observations(resumed, "finish")
    assert finish_results[0]["error"]["code"] == "completion_requirements"
    assert resumed.status == "completed"
    assert len(observations(resumed, "run_check")) == 2
    assert resumed.metrics["resume_success"] == 1
    events = [json.loads(line) for line in (Path(resumed.session_dir) / "events.jsonl").read_text().splitlines()]
    resumed_event = next(item for item in events if item["event"] == "session_resume")
    assert resumed_event["checks_invalidated"] and resumed_event["source_authority_reset"]


def test_uncertain_pending_mutation_is_not_replayed_on_resume(tmp_path):
    initial = ScriptedModel(read())
    agent = make_agent(tmp_path, initial, persist=True, pause_after_step=1)
    make_calc(agent.registry.workspace.root)
    paused = agent.run("Fix addition.", requirements=TaskRequirements.fix())
    checkpoint_path = Path(paused.session_dir) / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["execution"]["pending_operation"] = {"name": "apply_edits", "arguments": {"edits": [edit()]}}
    checkpoint["execution"]["evidence"]["originals"] = {"calc.py": (agent.registry.workspace.root / "calc.py").read_text()}
    checkpoint_path.write_text(json.dumps(checkpoint))
    (agent.registry.workspace.root / "calc.py").write_text("# retain this comment\ndef add(a, b):\n    return a + b\n")
    model = ScriptedModel(read(), action("run_check", name="check_1"), action("finish"))
    resumed = make_agent(tmp_path, model, persist=True).resume(paused.session_id, allow_tests=True)
    assert resumed.status == "completed", resumed.final_response
    assert not observations(resumed, "apply_edits")
    assert any("not replayed" in note for note in resumed.recovery_notes)


@pytest.mark.parametrize("location,value", [
    (("execution", "evidence", "revision"), -1),
    (("execution", "evidence", "revision"), "1"),
    (("execution", "evidence", "originals"), []),
    (("execution", "evidence", "originals"), {"calc.py": 42}),
    (("execution", "evidence", "recent"), "not a message collection"),
    (("execution", "inference_attempts"), -1),
    (("execution", "enabled_tools"), ["unavailable_tool"]),
    (("state", "steps"), -1),
])
def test_corrupt_checkpoint_is_rejected_before_model_or_workspace_execution(tmp_path, location, value):
    initial = ScriptedModel(read())
    agent = make_agent(tmp_path, initial, persist=True, pause_after_step=1)
    source = agent.registry.workspace.root / "calc.py"
    make_calc(agent.registry.workspace.root)
    original = source.read_bytes()
    paused = agent.run("Inspect addition.", requirements=TaskRequirements.inspect())
    path = Path(paused.session_dir) / "checkpoint.json"
    checkpoint = json.loads(path.read_text())
    target = checkpoint
    for component in location[:-1]:
        target = target[component]
    target[location[-1]] = value
    path.write_text(json.dumps(checkpoint))

    def forbidden(messages):
        raise AssertionError("Corrupt checkpoint reached model inference")

    model = ScriptedModel(forbidden)
    resumed = make_agent(tmp_path, model, persist=True)
    with pytest.raises(ValueError):
        resumed.resume(paused.session_id, allow_tests=True)
    assert not model.requests
    assert source.read_bytes() == original


@pytest.mark.parametrize("target", ["../outside.py", "alias/outside.py", ".git/config"])
def test_model_cannot_mutate_paths_outside_eligible_source(tmp_path, target):
    model = ScriptedModel(action("apply_edits", edits=[{"path": target, "content": "clobbered"}]))
    agent = make_agent(tmp_path, model, max_steps=1)
    repo = agent.registry.workspace.root
    outside = tmp_path / "outside.py"
    outside.write_text("original outside\n")
    (repo / "alias").symlink_to(tmp_path, target_is_directory=True)
    (repo / ".git").mkdir()
    (repo / ".git/config").write_text("original config\n")
    state = agent.run("Create a new source file.", requirements=TaskRequirements.fix())
    assert not observations(state, "apply_edits")[0]["success"]
    assert outside.read_text() == "original outside\n"
    assert (repo / ".git/config").read_text() == "original config\n"
    assert state.status == "incomplete"


def test_disabled_classifier_is_never_initialized_and_no_review_phase_is_exposed(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Disabled decision model was initialized")

    monkeypatch.setattr("patchrook_agent.evidence_workflow.OllamaDecisionClient", forbidden)
    model = ScriptedModel(read(), action("finish"))
    agent = make_agent(tmp_path, model, max_steps=2)
    make_calc(agent.registry.workspace.root)
    state = agent.run("Inspect addition.", requirements=TaskRequirements.inspect())
    assert state.status == "completed"
    assert not state.metrics["decision_enabled"]
    assert state.metrics["evidence_decision_calls"] == 0
    assert not state.metrics["proposal_review_enabled"]
    for request in model.requests:
        assert request["tools"] is None and request["response_format"]
        text = request_text(request)
        assert "propose_next_steps" not in text and "request_review" not in text
        names = {option["properties"]["name"]["const"] for option in request["response_format"]["oneOf"]}
        assert "apply_edits" in names and "finish" in names
