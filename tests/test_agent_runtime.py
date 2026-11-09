"""Integration regressions for persistent context, observed memory and recovery.

These models are deterministic unit-test fixtures, not performance evidence.
"""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from patchrook_agent.agent import Agent
from patchrook_agent.config import HarnessConfig
from patchrook_agent.context import ContextConfig
from patchrook_agent.context.compaction import SUMMARY_PREFIX
from patchrook_agent.models.ollama import ModelResponse
from patchrook_agent.tools import ToolConfig, ToolRegistry, Workspace


SOURCE = "def add_numbers(a, b):\n    return a - b\n\ndef multiply_numbers(a, b):\n    return a * b\n"
FIXED = SOURCE.replace("return a - b", "return a + b")
OBJECTIVE = "Fix addition without changing multiplication or public function signatures."


def call(tool_name, **arguments):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": tool_name, "arguments": arguments}},
    ]}


def final(text="The observed work is complete."):
    return {"role": "assistant", "content": text}


class SequenceModel:
    model = "unit-test-qwen"
    context_size = 4096
    max_output_tokens = 512

    def __init__(self, *messages, context_size=4096):
        self.context_size = context_size
        self.messages = iter(messages)
        self.requests = []

    def chat(self, **options):
        self.requests.append(deepcopy(options))
        return ModelResponse(next(self.messages), 11, 7, 0, 0, 0, 10_000_000)


@pytest.fixture
def harness(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "calc.py").write_text(SOURCE, encoding="utf-8")
    (repo / "test_calc.py").write_text(
        "from calc import add_numbers\n\ndef test_add():\n    assert add_numbers(3, 2) == 5\n",
        encoding="utf-8",
    )
    skills = tmp_path / "trusted-skills"
    (skills / "debugging").mkdir(parents=True)
    (skills / "debugging" / "SKILL.md").write_text(
        "---\nname: debugging\ndescription: Debug observed failures.\n"
        "activation: Use for failing tests.\n---\n"
        "SKILL_BODY_ONLY_AFTER_LOAD. Inspect actual failures and verify the correction.\n",
        encoding="utf-8",
    )
    return repo, tmp_path / "runs", skills


def config_for(harness, variant="C", **overrides):
    return replace(HarnessConfig.preset(variant), trusted_skills_dir=str(harness[2]), **overrides)


def make_agent(harness, model, *, variant="C", config=None, **options):
    return Agent(model, registry=ToolRegistry(Workspace(harness[0], ToolConfig(allow_tests=True))),
                 runs_dir=harness[1], config=config or config_for(harness, variant), **options)


def tool_results(state, name):
    return [json.loads(message["content"]) for message in state.messages
            if message.get("role") == "tool" and message.get("tool_name") == name]


def events(state):
    return [json.loads(line) for line in (Path(state.session_dir) / "events.jsonl").read_text().splitlines()]


def test_c_compaction_keeps_raw_history_and_full_available_tool_evidence(harness):
    repo, _, _ = harness
    names = []
    for index in range(12):
        name = f"part_{index}.py"
        names.append(name)
        (repo / name).write_text(("# inspected code " + "x" * 100 + "\n") * 100
                                + f"# RAW_END_FILE_{index}\n", encoding="utf-8")
    model = SequenceModel(*(call("read_file", path=name) for name in names), final(), context_size=12000)
    config = config_for(harness, context=ContextConfig(context_size=12000, trigger=0.60))
    state = make_agent(harness, model, config=config, max_steps=15).run(OBJECTIVE)

    assert state.status == "completed"
    assert state.metrics["compaction_count"] > 0
    assert len(state.active_messages) < len(state.messages)
    assert state.messages[1] == {"role": "user", "content": OBJECTIVE}
    assert len(tool_results(state, "read_file")) == 12
    assert any(message.get("content", "").startswith(SUMMARY_PREFIX) for message in state.active_messages)
    recorded = events(state)
    assert sum(event["event"] == "model_response" for event in recorded) == 13
    assert sum(event["event"] == "model_request" for event in recorded) == 13
    assert any("RAW_END_FILE_0" in event.get("available_output", {}).get("content", "")
               for event in recorded if event["event"] == "tool_result")
    assert all("RAW_END_FILE_0" not in result.get("content", "") for result in tool_results(state, "read_file"))
    assert state.metrics["actual_input_tokens"] == 13 * 11
    assert state.metrics["actual_output_tokens"] == 13 * 7
    assert state.memory["objective"] == OBJECTIVE
    assert set(names) <= set(state.memory["relevant_files"])


def test_pause_resume_preserves_objective_metrics_and_does_not_repeat_edit(harness):
    first_model = SequenceModel(call("read_file", path="calc.py"), call(
        "edit_file", path="calc.py", old_text="return a - b", new_text="return a + b",
    ))
    paused = make_agent(harness, first_model, pause_after_step=2).run(OBJECTIVE)
    assert paused.stop_reason == "paused" and paused.steps == 2
    assert paused.memory["verification"] == "not_run"
    assert (harness[0] / "calc.py").read_text() == FIXED

    model = SequenceModel(call("run_tests", path="test_calc.py"), final("Addition was fixed; pytest passed."))
    state = make_agent(harness, model).resume(paused.session_id, allow_tests=True)
    assert state.status == "completed" and state.verification == "passed"
    assert state.steps == 4 and state.tool_calls == 3
    assert state.metrics["resume_attempts"] == state.metrics["resume_success"] == 1
    assert state.metrics["actual_input_tokens"] == 44
    assert len(tool_results(state, "edit_file")) == 1
    assert tool_results(state, "run_tests")[-1]["exit_code"] == 0
    assert (harness[0] / "calc.py").read_text() == FIXED
    assert state.memory["objective"] == OBJECTIVE
    assert state.memory["modified_files"] == ["calc.py"]
    assert state.memory["verification"] == "passed"
    progress = (Path(state.session_dir) / "TASKS.md").read_text()
    assert OBJECTIVE in progress and "calc.py" in progress and "passed" in progress
    checkpoint = json.loads((Path(state.session_dir) / "checkpoint.json").read_text())
    assert checkpoint["memory"]["verification"] == "passed"
    assert checkpoint["state"]["verification"] == "passed"


def test_interrupted_mutation_is_inspected_instead_of_automatically_replayed(harness, monkeypatch):
    first = make_agent(harness, SequenceModel(call("read_file", path="calc.py"), call(
        "edit_file", path="calc.py", old_text="return a - b", new_text="return a + b",
    )))
    dispatch = first.registry.dispatch

    def interrupted(name, arguments, **options):
        result = dispatch(name, arguments, **options)
        if name == "edit_file":
            raise KeyboardInterrupt
        return result

    monkeypatch.setattr(first.registry, "dispatch", interrupted)
    paused = first.run(OBJECTIVE)
    assert paused.stop_reason == "interrupted"
    assert (harness[0] / "calc.py").read_text() == FIXED

    resumed_model = SequenceModel(call("read_file", path="calc.py"),
                                  call("run_tests", path="test_calc.py"), final())
    state = make_agent(harness, resumed_model).resume(paused.session_id, allow_tests=True)
    assert state.status == "completed" and state.verification == "passed"
    assert any("Unknown outcome" in note and "no automatic replay" in note for note in state.recovery_notes)
    edits = tool_results(state, "edit_file")
    assert len(edits) == 1 and edits[0]["error"]["code"] == "operation_interrupted"
    assert (harness[0] / "calc.py").read_text() == FIXED
    assert sum(event["event"] == "tool_request" and event.get("name") == "edit_file" for event in events(state)) == 1


@pytest.mark.parametrize("alias", ["dot", "absolute", "symlink"])
def test_external_changes_require_reread_even_when_model_uses_path_alias(harness, alias):
    repo, _, _ = harness
    paused = make_agent(harness, SequenceModel(call("read_file", path="calc.py")),
                        pause_after_step=1).run(OBJECTIVE)
    external = SOURCE.replace("return a - b", "return a - b  # externally changed")
    (repo / "calc.py").write_text(external, encoding="utf-8")
    if alias == "absolute":
        path = str(repo / "calc.py")
    elif alias == "symlink":
        (repo / "alias.py").symlink_to("calc.py")
        path = "alias.py"
    else:
        path = "./calc.py"
    model = SequenceModel(call("edit_file", path=path, old_text="return a - b", new_text="return a + b"),
                          call("read_file", path=path),
                          call("edit_file", path=path, old_text="return a - b", new_text="return a + b"),
                          call("run_tests", path="test_calc.py"), final())
    state = make_agent(harness, model).resume(paused.session_id, allow_tests=True)
    edits = tool_results(state, "edit_file")
    assert edits[0]["success"] is False and edits[0]["error"]["code"] == "workspace_changed"
    assert edits[1]["success"] is True
    assert state.status == "completed" and state.verification == "passed"
    assert (repo / "calc.py").read_text() == external.replace("return a - b", "return a + b")


@pytest.mark.parametrize("variant,tool_count", [("A", 7), ("B", 8), ("C", 9)])
def test_ablations_control_model_skill_exposure(harness, variant, tool_count):
    model = SequenceModel(final("No changes requested."))
    state = make_agent(harness, model, variant=variant).run("Inspect available capabilities.")
    assert state.status == "completed"
    assert len(model.requests[0]["tools"]) == tool_count
    initial = "\n".join(message.get("content", "") for message in model.requests[0]["messages"])
    assert ("Available trusted skills" in initial) is (variant == "C")
    assert "SKILL_BODY_ONLY_AFTER_LOAD" not in initial
    assert bool(state.memory) is (variant == "C")


def test_skills_and_memory_still_work_when_compaction_is_disabled_independently(harness):
    config = config_for(harness, compaction_enabled=False)
    model = SequenceModel(call("read_skill", name="debugging"), final("Skill consulted; no changes needed."))
    state = make_agent(harness, model, config=config).run("Investigate testing practice.")
    assert state.status == "completed"
    first = "\n".join(message.get("content", "") for message in model.requests[0]["messages"])
    second = "\n".join(message.get("content", "") for message in model.requests[1]["messages"])
    assert "Available trusted skills" in first
    assert "Working memory" in first
    assert "SKILL_BODY_ONLY_AFTER_LOAD" not in first
    assert second.count("SKILL_BODY_ONLY_AFTER_LOAD") == 1
    assert state.memory["loaded_skills"] == ["debugging"]
    assert state.metrics["compaction_count"] == 0


@pytest.mark.parametrize("corruption", ["active_objective", "missing_raw_objective"])
def test_resume_rejects_checkpoint_with_missing_or_conflicting_objective(harness, corruption):
    paused = make_agent(harness, SequenceModel(call("read_file", path="calc.py")),
                        pause_after_step=1).run(OBJECTIVE)
    path = Path(paused.session_dir) / "checkpoint.json"
    checkpoint = json.loads(path.read_text())
    if corruption == "active_objective":
        checkpoint["state"]["active_messages"][1]["content"] = "Change every public API instead."
    else:
        checkpoint["state"]["messages"] = checkpoint["state"]["messages"][:1]
    path.write_text(json.dumps(checkpoint), encoding="utf-8")
    model = SequenceModel(final())
    with pytest.raises(ValueError):
        make_agent(harness, model).resume(paused.session_id, allow_tests=True)
    assert not model.requests


def test_resume_rejects_workspace_path_replaced_with_symlink(harness):
    paused = make_agent(harness, SequenceModel(call("read_file", path="calc.py")),
                        pause_after_step=1).run(OBJECTIVE)
    repo = harness[0]
    original = repo.with_name("original-repo")
    repo.rename(original)
    replacement = repo.with_name("replacement-repo")
    replacement.mkdir()
    (replacement / "calc.py").write_text(SOURCE, encoding="utf-8")
    repo.symlink_to(replacement, target_is_directory=True)
    model = SequenceModel(final())
    with pytest.raises(ValueError):
        make_agent(harness, model).resume(paused.session_id, allow_tests=True)
    assert not model.requests
