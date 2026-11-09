"""Recovery directives reach the model as trusted, current instructions."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from patchrook_agent.agent import Agent
from patchrook_agent.completion import TaskRequirements
from patchrook_agent.config import HarnessConfig
from patchrook_agent.models.ollama import ModelResponse
from patchrook_agent.tools import ToolConfig, ToolRegistry, Workspace


class ScriptedModel:
    model = "recovery-guidance-mechanics"
    context_size = 8192
    max_output_tokens = 512

    def __init__(self, *messages, protocol="schema"):
        self.messages = iter(messages)
        self.requests = []
        self.protocol = protocol

    def chat(self, **options):
        self.requests.append(deepcopy(options))
        message = next(self.messages)
        if self.protocol == "native":
            envelope = json.loads(message["content"])
            message = ({"role": "assistant", "content": "", "tool_calls": [{"function": {
                "name": envelope["tool"], "arguments": envelope["arguments"]}}]}
                if envelope["action"] == "tool" else {"role": "assistant", "content": envelope["summary"]})
        return ModelResponse(message, 0, 0, None, None, None, 0,
                             prompt_tokens_reported=False, completion_tokens_reported=False)


def action(name, **arguments):
    return {"role": "assistant", "content": json.dumps({"action": "tool", "tool": name, "arguments": arguments})}


def finish():
    return {"role": "assistant", "content": json.dumps({"action": "complete", "summary":
        "Inspected the requested source and reported observed changes and verification."})}


def blocked():
    return {"role": "assistant", "content": json.dumps({"action": "blocked", "reason":
        "The attempted source read could not be completed."})}


def read():
    return action("read_code", paths=["calc.py"])


def edit():
    return action("edit_file", path="calc.py", old_text="a - b", new_text="a + b")


def verify():
    return action("run_tests", path="test_calc.py")


def runner(tmp_path, model, **options):
    root = tmp_path / "repo"
    root.mkdir(exist_ok=True)
    if not (root / "calc.py").exists():
        (root / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (root / "test_calc.py").write_text("from calc import add\ndef test_add():\n    assert add(3, 2) == 5\n")
    return Agent(model=model, registry=ToolRegistry(Workspace(root, ToolConfig(allow_tests=True))),
                 config=replace(HarnessConfig.preset("C"), workflow_mode="lean", action_protocol=model.protocol),
                 runs_dir=tmp_path / "runs", max_steps=20, **options)


@pytest.fixture(autouse=True)
def no_live_inference(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("Guidance tests must not contact Ollama")
    monkeypatch.setattr("patchrook_agent.models.ollama.httpx.Client", fail)


def events(state, event="progress_recovery"):
    return [record for line in Path(state.session_dir, "events.jsonl").read_text().splitlines()
            if (record := json.loads(line))["event"] == event]


def trusted_recovery(state, model, index, event="progress_recovery"):
    records = events(state, event)
    assert len(records) == 1
    guidance = records[0]["guidance"]
    messages = model.requests[index]["messages"]
    assert messages[-1]["role"] == "system"
    assert guidance in messages[-1]["content"]
    assert all(guidance not in message["content"] for message in messages if message["role"] == "user")
    return guidance


def test_stalled_inspection_is_directed_to_read_source_before_required_patch(tmp_path):
    listing = action("list_files", path=".")
    model = ScriptedModel(listing, deepcopy(listing), deepcopy(listing), read(), edit(), verify(), finish())
    state = runner(tmp_path, model).run("Fix addition in calc.py", requirements=TaskRequirements.fix())
    assert state.status == "completed" and state.verification == "passed"
    hint = trusted_recovery(state, model, 3)
    assert "read_code" in hint or "read_file" in hint
    assert "inspect" in hint.lower() or "read" in hint.lower()


@pytest.mark.parametrize("protocol", ["schema", "native"])
def test_after_inspection_stall_directs_requested_supported_code_change(tmp_path, protocol):
    model = ScriptedModel(read(), read(), read(), edit(), verify(), finish(), protocol=protocol)
    state = runner(tmp_path, model).run("Fix addition in calc.py", requirements=TaskRequirements.fix())
    assert state.status == "completed" and state.modified_files == ["calc.py"]
    hint = trusted_recovery(state, model, 3)
    assert "edit_file" in hint or "code change" in hint
    # Recovery is guidance. The model still provides and validates the patch.
    edits = [message for message in state.messages if message.get("tool_name") == "edit_file"]
    assert len(edits) == 1


def test_after_observed_change_stall_directs_verification(tmp_path):
    model = ScriptedModel(read(), edit(), read(), read(), read(), verify(), finish())
    state = runner(tmp_path, model).run("Fix addition in calc.py", requirements=TaskRequirements.fix())
    assert state.status == "completed" and state.verification == "passed"
    hint = trusted_recovery(state, model, 5)
    assert "run_tests" in hint


def test_observed_failure_guidance_names_the_outstanding_test_scope(tmp_path):
    model = ScriptedModel(read(), verify(), read(), read(), edit(), verify(), finish())
    state = runner(tmp_path, model).run("Fix addition in calc.py", requirements=TaskRequirements.code())
    assert state.status == "completed" and state.verification == "passed"
    hint = trusted_recovery(state, model, 4)
    assert "test_calc.py" in hint
    assert "fail" in hint.lower() or "investigat" in hint.lower()


def test_read_only_recovery_does_not_demand_an_edit_or_test(tmp_path):
    model = ScriptedModel(read(), read(), read(), finish())
    state = runner(tmp_path, model).run("Explain calc.py", requirements=TaskRequirements.inspect())
    assert state.status == "completed" and not state.modified_files
    hint = trusted_recovery(state, model, 3)
    assert "report" in hint.lower() or "complete" in hint.lower()
    assert "make the requested code change" not in hint
    assert "run_tests" not in hint


def test_pending_recovery_is_restored_in_system_tail_after_resume(tmp_path):
    class InterruptOnRecovery(ScriptedModel):
        def chat(self, **options):
            if len(self.requests) == 3:
                self.requests.append(deepcopy(options))
                raise KeyboardInterrupt
            return super().chat(**options)

    interrupted_model = InterruptOnRecovery(read(), read(), read())
    paused = runner(tmp_path, interrupted_model).run("Explain calc.py", requirements=TaskRequirements.inspect())
    assert paused.stop_reason == "interrupted" and paused.steps == 3
    saved = json.loads(Path(paused.session_dir, "checkpoint.json").read_text())
    assert saved["execution"]["lean_recovery_hint"]
    assert len(interrupted_model.requests) == 4
    model = ScriptedModel(finish())
    state = runner(tmp_path, model).resume(paused.session_id, allow_tests=True)
    assert state.status == "completed" and len(model.requests) == 1
    trusted_recovery(state, model, 0)


def test_corrective_read_recovery_is_an_instruction_and_remains_bounded(tmp_path):
    invalid = action("read_file", path="calc.py", start_line=99, end_line=100)
    model = ScriptedModel(invalid, blocked(), blocked())
    state = runner(tmp_path, model).run("Explain calc.py", requirements=TaskRequirements.inspect())
    assert state.stop_reason == "model_blocked" and len(model.requests) == 3
    hint = trusted_recovery(state, model, 2, event="read_argument_recovery")
    assert "read_code" in hint and "calc.py" in hint


def test_new_missing_file_blocker_is_not_overridden_by_generic_recovery(tmp_path):
    model = ScriptedModel(action("read_file", path="missing.py"), blocked())
    state = runner(tmp_path, model).run("Inspect missing.py", requirements=TaskRequirements.inspect())
    assert state.stop_reason == "model_blocked" and len(model.requests) == 2
    assert not events(state) and not events(state, "read_argument_recovery")
