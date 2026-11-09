"""Offline recovery of a directory-list request aimed at an actual source file."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import pytest

from patchrook_agent.agent import Agent, AgentState
from patchrook_agent.completion import TaskRequirements
from patchrook_agent.config import HarnessConfig
from patchrook_agent.evidence_workflow import EvidenceWorkflow
from patchrook_agent.models.ollama import ModelResponse
from patchrook_agent.tools import ToolConfig, ToolRegistry, Workspace


def action(tool_name, **arguments):
    return {"role": "assistant", "content": json.dumps({"name": tool_name, "arguments": arguments})}


class ScriptedModel:
    model = "offline-tool-recovery"
    context_size = 8192
    max_output_tokens = 512

    def __init__(self, *actions):
        self.actions = iter(actions)
        self.requests = []

    def chat(self, messages, **options):
        self.requests.append(deepcopy(messages))
        response = next(self.actions)
        if callable(response):
            response = response(messages)
        return ModelResponse(response, 100, 20, 0, 0, 0, 0)


@pytest.fixture(autouse=True)
def no_model_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Tool recovery tests must not contact models or network services")
    monkeypatch.setattr("httpx.get", forbidden)
    monkeypatch.setattr("httpx.post", forbidden)
    monkeypatch.setattr("httpx.Client", forbidden)


def make_agent(tmp_path, model, *, max_steps=4):
    repo = tmp_path / "repo"
    repo.mkdir()
    config = replace(HarnessConfig.preset("C"), workflow_mode="evidence",
        action_protocol="schema", evidence_policy="rules", review_mode="off",
        evidence_test_commands=((sys.executable, "-c", "from subject import VALUE; assert VALUE == 2"),),
        persist=True, checkpoint_enabled=True)
    return Agent(model, registry=ToolRegistry(Workspace(repo, ToolConfig(allow_tests=True))),
                 config=config, runs_dir=tmp_path / "runs", max_steps=max_steps, deadline_seconds=30)


def observations(state, tool):
    return [json.loads(item["content"]) for item in state.messages
            if item.get("role") == "tool" and item.get("tool_name") == tool]


def events(state):
    return [json.loads(line) for line in (Path(state.session_dir) / "events.jsonl").read_text().splitlines()]


def test_misplaced_list_reads_source_once_then_permits_a_verified_guarded_patch(tmp_path):
    model = ScriptedModel(action("list_files", path="subject.py"),
        action("apply_edits", edits=[{"path": "subject.py", "old_text": "VALUE = 1", "new_text": "VALUE = 2"}]),
        action("run_check", name="check_1"), action("finish"))
    agent = make_agent(tmp_path, model)
    (agent.registry.workspace.root / "subject.py").write_text("# preserve this comment\nVALUE = 1\n")

    state = agent.run("Fix the value and preserve the comment.", requirements=TaskRequirements.fix())

    assert state.status == "completed", state.final_response
    assert state.verification == "passed" and state.modified_files == ["subject.py"]
    assert len(model.requests) == state.steps == 4
    assert state.tool_calls == 5 and state.tool_failures == 1
    assert observations(state, "list_files")[0]["error"]["code"] == "not_directory"
    reads = observations(state, "read_file")
    assert len(reads) == 1 and reads[0]["success"]
    assert reads[0]["source_sha256"]
    next_prompt = "\n".join(item["content"] for item in model.requests[1])
    assert "Requested source (data)" in next_prompt and "VALUE = 1" in next_prompt
    recorded = events(state)
    read_requests = [e for e in recorded if e["event"] == "tool_request" and e["name"] == "read_file"]
    assert len(read_requests) == 1 and read_requests[0]["automatic"] is True
    assert read_requests[0]["arguments"] == {"path": "subject.py", "start_line": 1, "end_line": 40}
    assert len([e for e in recorded if e["event"] == "tool_recovery"]) == 1
    assert (agent.registry.workspace.root / "subject.py").read_text() == "# preserve this comment\nVALUE = 2\n"


def test_recovered_read_does_not_bypass_source_hash_freshness(tmp_path):
    target = None
    def external_change(messages):
        assert "VALUE = 1" in "\n".join(item["content"] for item in messages)
        target.write_text("VALUE = 9\n")
        return action("apply_edits", edits=[{"path": "subject.py", "old_text": "VALUE = 1", "new_text": "VALUE = 2"}])
    model = ScriptedModel(action("list_files", path="subject.py"), external_change)
    agent = make_agent(tmp_path, model, max_steps=2)
    target = agent.registry.workspace.root / "subject.py"
    target.write_text("VALUE = 1\n")

    state = agent.run("Fix the value.", requirements=TaskRequirements.fix())

    assert len(model.requests) == 2
    assert observations(state, "apply_edits")[0]["error"]["code"] == "stale_source"
    assert target.read_text() == "VALUE = 9\n"
    assert state.status != "completed"


@pytest.fixture
def runtime(tmp_path):
    agent = make_agent(tmp_path, ScriptedModel())
    worker = EvidenceWorkflow(agent)
    worker.requirements = TaskRequirements.fix()
    worker._setup("Inspect the source.")
    worker.definitions = worker._definitions()
    worker.enabled = {item["function"]["name"] for item in worker.definitions}
    worker.state = AgentState()
    worker.session.bind(worker.state)
    yield worker
    if worker.session.store:
        worker.session.store.release()


@pytest.mark.parametrize("kind", ["missing", "directory", "symlink", "protected", "absolute", "parent",
                                 "disabled_read", "wrong_error", "wrong_tool"])
def test_recovery_only_targets_guarded_existing_files(runtime, tmp_path, kind):
    root = runtime.workspace.root
    (root / "subject.py").write_text("VALUE = 1\n")
    path = "subject.py"
    name = "list_files"
    code = "not_directory"
    if kind == "missing":
        path = "missing.py"
    elif kind == "directory":
        (root / "directory").mkdir()
        path = "directory"
    elif kind == "symlink":
        (root / "linked.py").symlink_to(root / "subject.py")
        path = "linked.py"
    elif kind == "protected":
        (root / ".git").mkdir()
        (root / ".git/config").write_text("secret\n")
        path = ".git/config"
    elif kind == "absolute":
        path = str(root / "subject.py")
    elif kind == "parent":
        path = "../repo/subject.py"
    elif kind == "disabled_read":
        runtime.enabled.remove("read_file")
    elif kind == "wrong_error":
        code = "not_found"
    elif kind == "wrong_tool":
        name = "apply_edits"
    proposed = {"name": name, "arguments": {"path": path}}
    failed = {"success": False, "error": {"code": code, "message": "Observed action failure"}}

    actual_action, result = runtime._recover_read_action(runtime.state, proposed, failed, time.monotonic() + 10)

    assert actual_action == proposed and result == failed
    assert runtime.state.tool_calls == 0
    assert not runtime.direct
    assert not runtime.tools._presented


def test_binary_file_recovery_cannot_create_source_or_edit_authority(runtime):
    (runtime.workspace.root / "image.bin").write_bytes(b"\x00\xff\x00")
    proposed = {"name": "list_files", "arguments": {"path": "image.bin"}}
    failed = {"success": False, "error": {"code": "not_directory", "message": "Expected a directory"}}

    _, result = runtime._recover_read_action(runtime.state, proposed, failed, time.monotonic() + 10)

    assert result["success"] is False
    assert not runtime.direct
    assert not runtime.tools._presented
    assert not [card for card in runtime.evidence.cards(fresh=False) if card.kind == "source"]
