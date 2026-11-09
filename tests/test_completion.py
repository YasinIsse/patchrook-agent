"""Observed work, bounded continuation, and durable completion requirements."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from patchrook_agent.agent import Agent
from patchrook_agent.completion import TaskRequirements, inspection_observed
from patchrook_agent.config import HarnessConfig
from patchrook_agent.models.ollama import ModelResponse
from patchrook_agent.tools import ToolConfig, ToolRegistry, Workspace


def action(name, **arguments):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": name, "arguments": arguments}}]}


def final(text="Done."):
    return {"role": "assistant", "content": text}


class Model:
    context_size = 8192
    max_output_tokens = 512

    def __init__(self, *messages):
        self.messages = iter(messages)
        self.requests = []

    def chat(self, **options):
        self.requests.append(deepcopy(options))
        return ModelResponse(next(self.messages), 1, 1, None, None, None, 1)


def agent(tmp_path, *messages, allow_tests=True, **options):
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    return Agent(Model(*messages), registry=ToolRegistry(Workspace(repo, ToolConfig(allow_tests=allow_tests))),
                 config=HarnessConfig.preset("C"), runs_dir=tmp_path / "runs", **options)


def fixture(tmp_path, *, broken=True):
    repo = tmp_path / "repo"
    repo.mkdir()
    repo.joinpath("calc.py").write_text("def add(a, b):\n    return a " + ("-" if broken else "+") + " b\n")
    repo.joinpath("test_calc.py").write_text("from calc import add\ndef test_add():\n    assert add(3, 2) == 5\n")


def test_plan_only_final_cannot_complete_and_gets_only_one_recovery(tmp_path):
    runner = agent(tmp_path, final("I will read and fix calc.py."), final("I will test it next."), final())
    state = runner.run("Fix calc.py", requirements=TaskRequirements.fix())
    assert state.status == "incomplete" and state.stop_reason == "completion_requirements_unmet"
    assert state.steps == 2 and state.tool_calls == 0 and state.verification == "not_run"
    assert state.final_response.startswith("Incomplete:")
    assert len(state.completion["missing"]) == 3
    assert sum("Task incomplete:" in message.get("content", "") for message in state.messages) == 1


def test_premature_final_recovers_into_real_patch_and_observed_tests(tmp_path):
    fixture(tmp_path)
    runner = agent(tmp_path, final("I will inspect calc.py."), action("read_file", path="calc.py"),
                   action("edit_file", path="calc.py", old_text="a - b", new_text="a + b"),
                   action("run_tests", path="test_calc.py"), final("Fixed addition; pytest passed."))
    state = runner.run("Fix calc.py and verify", requirements=TaskRequirements.fix())
    assert state.status == "completed" and state.verification == "passed"
    assert state.modified_files == ["calc.py"] and state.completion["satisfied"]
    assert (tmp_path / "repo/calc.py").read_text() == "def add(a, b):\n    return a + b\n"
    report = json.loads(Path(state.session_dir, "report.json").read_text())
    assert report["completion"] == state.completion


def test_already_correct_code_completes_without_gratuitous_edit(tmp_path):
    fixture(tmp_path, broken=False)
    state = agent(tmp_path, action("read_file", path="calc.py"), action("run_tests", path="test_calc.py"),
                  final("Already correct; observed tests passed.")).run("Check addition", requirements=TaskRequirements.code())
    assert state.status == "completed" and state.verification == "passed" and not state.modified_files
    assert state.completion["satisfied"]


def test_strict_fix_cannot_substitute_passing_tests_for_requested_patch(tmp_path):
    fixture(tmp_path, broken=False)
    state = agent(tmp_path, action("read_file", path="calc.py"), action("run_tests", path="test_calc.py"),
                  final(), final()).run("Make requested change", requirements=TaskRequirements.fix())
    assert state.status == "incomplete" and state.stop_reason == "completion_requirements_unmet"
    assert state.completion["missing"] == ["make the requested code change"]


def test_inspection_mode_needs_no_tests_or_edits(tmp_path):
    fixture(tmp_path)
    state = agent(tmp_path, action("read_file", path="calc.py"), final("Addition subtracts."),
                  allow_tests=False).run("Inspect addition", requirements=TaskRequirements.inspect())
    assert state.status == "completed" and state.verification == "not_run"
    assert not state.modified_files and state.tool_calls == 1


def test_listing_names_and_claiming_tests_is_not_completion_evidence(tmp_path):
    fixture(tmp_path)
    state = agent(tmp_path, action("list_files"), final("Inspected; tests passed."), final("Done.")).run(
        "Inspect code", requirements=TaskRequirements.inspect())
    assert state.stop_reason == "completion_requirements_unmet"
    assert not state.completion["evidence"]["inspected"] and state.verification == "not_run"


@pytest.mark.parametrize("allow_tests", [False, True])
def test_required_omitted_tests_use_existing_authorized_fallback(tmp_path, allow_tests):
    fixture(tmp_path, broken=False)
    state = agent(tmp_path, action("read_file", path="calc.py"), final("Already correct."), final("Verified."),
                  allow_tests=allow_tests).run("Check code", requirements=TaskRequirements.code())
    assert state.status == ("completed" if allow_tests else "incomplete")
    assert state.verification == ("passed" if allow_tests else "not_run")
    assert state.tool_calls == (2 if allow_tests else 1)
    if not allow_tests:
        assert state.stop_reason == "verification_not_authorized"


def test_verification_before_patch_does_not_satisfy_fix(tmp_path):
    fixture(tmp_path, broken=False)
    state = agent(tmp_path, action("read_file", path="calc.py"), action("run_tests", path="test_calc.py"),
                  action("edit_file", path="calc.py", old_text="a + b", new_text="a - b"),
                  final("Tests passed."), final("Done.")).run("Change addition", requirements=TaskRequirements.fix())
    assert state.status == "incomplete" and state.verification == "failed"
    assert state.stop_reason == "tests_failed" and not state.completion["satisfied"]


def test_requirements_and_spent_completion_recovery_survive_resume(tmp_path):
    fixture(tmp_path)
    paused = agent(tmp_path, final("I will inspect."), action("read_file", path="calc.py"), pause_after_step=2).run(
        "Fix calc.py", requirements=TaskRequirements.fix())
    assert paused.stop_reason == "paused"
    resumed = agent(tmp_path, final("I will edit next."), final()).resume(paused.session_id, allow_tests=True)
    assert resumed.status == "incomplete" and resumed.stop_reason == "completion_requirements_unmet"
    assert resumed.steps == 3
    assert resumed.completion["requirements"] == TaskRequirements.fix().to_dict()


@pytest.mark.parametrize("requirements", [{"require_inspection": True},
                                         {"require_inspection": 1, "require_change": False, "require_tests": False},
                                         {"require_inspection": True, "require_change": False, "require_tests": False, "extra": True}])
def test_malformed_saved_requirements_are_rejected(tmp_path, requirements):
    fixture(tmp_path)
    paused = agent(tmp_path, action("read_file", path="calc.py"), pause_after_step=1).run(
        "Inspect", requirements=TaskRequirements.inspect())
    checkpoint = Path(paused.session_dir, "checkpoint.json")
    data = json.loads(checkpoint.read_text())
    data["execution"]["task_requirements"] = requirements
    checkpoint.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="requirements"):
        agent(tmp_path, final()).resume(paused.session_id)


def test_visible_partial_batch_counts_but_hidden_capture_does_not():
    def tool(name, result):
        return {"role": "tool", "tool_name": name, "content": json.dumps(result)}
    visible = {"success": True, "path": "calc.py", "content": "1: return 1"}
    assert inspection_observed([tool("read_files", {"success": False, "files": [visible, {"success": False}]})])
    assert not inspection_observed([tool("read_file", {"success": True, "path": "calc.py", "truncated": True})])
    assert not inspection_observed([tool("read_skill", visible)])
    assert not inspection_observed([tool("read_file", {**visible, "success": False})])


def test_three_repeated_empty_searches_stop_successful_tool_loop(tmp_path):
    fixture(tmp_path)
    search = action("search_code", pattern=r"stage\d+\.", mode="regex")
    state = agent(tmp_path, search, search, search, final()).run("Fix calc.py", requirements=TaskRequirements.fix())
    assert state.stop_reason == "stalled_search" and state.status == "incomplete"
    assert state.steps == 3 and state.tool_calls == 3 and state.tool_failures == 0
    assert sum("no matches twice" in m.get("content", "").lower() for m in state.messages if m["role"] == "system") == 1


def test_empty_search_recovery_can_read_known_path_and_finish(tmp_path):
    fixture(tmp_path)
    search = action("search_code", pattern="not_found_here")
    state = agent(tmp_path, search, search, action("read_file", path="calc.py"), final("Addition subtracts.")).run(
        "Inspect calc.py", requirements=TaskRequirements.inspect())
    assert state.status == "completed" and state.steps == 4 and not state.modified_files


def test_search_counter_survives_pause_resume(tmp_path):
    fixture(tmp_path)
    search = action("search_code", pattern="not_found_here")
    paused = agent(tmp_path, search, search, pause_after_step=2).run("Inspect calc.py", requirements=TaskRequirements.inspect())
    state = agent(tmp_path, search, final()).resume(paused.session_id)
    assert state.stop_reason == "stalled_search" and state.steps == 3


@pytest.mark.parametrize("empty_count", [2, 3])
def test_interruption_after_durable_search_result_applies_gate_before_inference(tmp_path, monkeypatch, empty_count):
    fixture(tmp_path)
    search = action("search_code", pattern="not_found_here")
    first = agent(tmp_path, *[search] * empty_count)
    checkpoint = first._checkpoint
    interrupted = False

    def interrupt_after_result(state):
        nonlocal interrupted
        checkpoint(state)
        if not interrupted and first._pending is None and first._search_progress.empty_count == empty_count:
            interrupted = True
            raise KeyboardInterrupt

    monkeypatch.setattr(first, "_checkpoint", interrupt_after_result)
    paused = first.run("Inspect calc.py", requirements=TaskRequirements.inspect())
    assert paused.stop_reason == "interrupted"
    second = agent(tmp_path, action("read_file", path="calc.py"), final())
    state = second.resume(paused.session_id)
    if empty_count == 2:
        assert state.status == "completed"
        assert any("no matches twice" in m.get("content", "") for m in second.model.requests[0]["messages"])
    else:
        assert state.stop_reason == "stalled_search" and state.status == "incomplete"
        assert not second.model.requests and state.steps == 3


def test_external_changed_source_resets_search_stall_and_requires_reread(tmp_path):
    fixture(tmp_path)
    search = action("search_code", pattern="not_found_here")
    paused = agent(tmp_path, action("read_file", path="calc.py"), search, search, search).run(
        "Inspect calc.py", requirements=TaskRequirements.inspect())
    assert paused.stop_reason == "stalled_search"
    source = tmp_path / "repo/calc.py"
    source.write_text(source.read_text() + "# external edit\n")
    second = agent(tmp_path, action("read_file", path="calc.py"), final())
    state = second.resume(paused.session_id)
    assert state.status == "completed" and len(second.model.requests) == 2
