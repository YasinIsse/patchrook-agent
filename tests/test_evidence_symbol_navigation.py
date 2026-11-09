"""A broad mention remains visible while navigation locates a Python declaration."""

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
    model = "offline-symbol-navigation"
    context_size = 8192
    max_output_tokens = 512
    def __init__(self, *actions):
        self.actions = iter(actions)
        self.requests = []
    def chat(self, messages, **kwargs):
        self.requests.append(deepcopy(messages))
        response = next(self.actions)
        if callable(response):
            response = response(messages)
        return ModelResponse(response, 100, 20, 0, 0, 0, 0)


@pytest.fixture(autouse=True)
def no_model_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Symbol navigation tests must not contact a model")
    monkeypatch.setattr("httpx.get", forbidden)
    monkeypatch.setattr("httpx.post", forbidden)
    monkeypatch.setattr("httpx.Client", forbidden)


def make_agent(tmp_path, model, *, max_steps=4):
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "src/pkg/core.py"
    source.parent.mkdir(parents=True)
    source.write_text("class Core:\n    value = 1\n")
    (root / "HISTORY.md").write_text("Core was introduced in an old release.\n")
    config = replace(HarnessConfig.preset("C"), workflow_mode="evidence", action_protocol="schema",
        evidence_policy="rules", review_mode="off", persist=True, checkpoint_enabled=True,
        evidence_test_commands=((sys.executable, "-c", "from src.pkg.core import Core; assert Core.value == 2"),))
    return Agent(model, registry=ToolRegistry(Workspace(root, ToolConfig(allow_tests=True))),
        config=config, runs_dir=tmp_path / "runs", max_steps=max_steps, deadline_seconds=30)


def observations(state, tool):
    return [json.loads(message["content"]) for message in state.messages
            if message.get("role") == "tool" and message.get("tool_name") == tool]


def test_docs_mention_preserved_and_automatic_definition_read_allows_guarded_patch(tmp_path):
    model = ScriptedModel(action("search_code", pattern="Core", path=".", max_results=1, mode="literal"),
        action("apply_edits", edits=[{"path": "src/pkg/core.py", "old_text": "value = 1", "new_text": "value = 2"}]),
        action("run_check", name="check_1"), action("finish"))
    agent = make_agent(tmp_path, model)
    state = agent.run("Change Core.value to 2 and verify it.", requirements=TaskRequirements.fix())
    assert state.status == "completed", state.final_response
    assert state.verification == "passed" and state.modified_files == ["src/pkg/core.py"]
    searches = observations(state, "search_code")
    assert searches[0]["matches"][0]["path"] == "HISTORY.md"
    assert searches[0]["next_offset"] == 1
    assert searches[1]["matches"][0]["path"] == "src/pkg/core.py"
    reads = observations(state, "read_file")
    assert len(reads) == 1 and reads[0]["source_sha256"]
    assert "class Core:" in "\n".join(m["content"] for m in model.requests[1])
    events = [json.loads(line) for line in (Path(state.session_dir) / "events.jsonl").read_text().splitlines()]
    recovery = [e for e in events if e["event"] == "navigation_symbol_definition"]
    assert len(recovery) == 1 and recovery[0]["edit_authorization"] is False
    actual = [e for e in events if e["event"] == "tool_request"]
    assert [(e["name"], e["automatic"]) for e in actual[:3]] == [("search_code", False), ("search_code", True), ("read_file", True)]
    assert (agent.registry.workspace.root / "HISTORY.md").read_text() == "Core was introduced in an old release.\n"


def test_declaration_recovery_still_rejects_source_changed_after_presentation(tmp_path):
    target = None
    def mutate(messages):
        assert "class Core:" in "\n".join(m["content"] for m in messages)
        target.write_text("class Core:\n    value = 9\n")
        return action("apply_edits", edits=[{"path": "src/pkg/core.py", "old_text": "value = 1", "new_text": "value = 2"}])
    model = ScriptedModel(action("search_code", pattern="Core", path=".", max_results=1), mutate)
    agent = make_agent(tmp_path, model, max_steps=2)
    target = agent.registry.workspace.root / "src/pkg/core.py"
    state = agent.run("Change Core.value to 2.", requirements=TaskRequirements.fix())
    assert observations(state, "apply_edits")[0]["error"]["code"] == "stale_source"
    assert target.read_text() == "class Core:\n    value = 9\n"


@pytest.fixture
def runtime(tmp_path):
    agent = make_agent(tmp_path, ScriptedModel())
    worker = EvidenceWorkflow(agent)
    worker.requirements = TaskRequirements.fix()
    worker._setup("Change Core.value to 2.")
    worker.definitions = worker._definitions()
    worker.enabled = {item["function"]["name"] for item in worker.definitions}
    worker.state = AgentState()
    worker.session.bind(worker.state)
    worker.index.refresh()
    yield worker
    if worker.session.store:
        worker.session.store.release()


@pytest.mark.parametrize("kind", ["regex", "offset", "nonidentifier", "unmentioned", "already_definition",
                                 "disabled_read", "disabled_search", "explicit_document", "no_matches", "failed"])
def test_intentional_scope_transport_and_existing_definition_are_not_rewritten(runtime, kind):
    args = {"path": ".", "pattern": "Core", "mode": "literal", "max_results": 1}
    result = {"success": True, "matches": [{"path": "HISTORY.md", "line_number": 1, "text": "Core was introduced."}]}
    if kind == "regex": args["mode"] = "regex"
    if kind == "offset": args["offset"] = 1
    if kind == "nonidentifier": args["pattern"] = "Core was introduced"
    if kind == "unmentioned": args["pattern"] = "OtherClass"
    if kind == "already_definition": result["matches"] = [{"path": "src/pkg/core.py", "line_number": 1, "text": "class Core:"}]
    if kind == "disabled_read": runtime.enabled.remove("read_file")
    if kind == "disabled_search": runtime.enabled.remove("search_code")
    if kind == "explicit_document": args["path"] = "HISTORY.md"
    if kind == "no_matches": result["matches"] = []
    if kind == "failed": result = {"success": False, "error": {"code": "timeout"}}
    proposed = {"name": "search_code", "arguments": args}
    recovered, observed = runtime._recover_symbol_search(runtime.state, proposed, result, time.monotonic() + 10)
    assert recovered == proposed and observed == result
    assert runtime.state.tool_calls == 0


def test_no_declaration_does_not_replace_original_mentions_or_claim_source(runtime):
    target = runtime.workspace.root / "src/pkg/core.py"
    target.write_text("Core = object()\n")
    proposed = {"name": "search_code", "arguments": {"path": ".", "pattern": "Core", "max_results": 1}}
    result = {"success": True, "matches": [{"path": "HISTORY.md", "line_number": 1, "text": "Core was introduced."}], "next_offset": 1}
    recovered, observed = runtime._recover_symbol_search(runtime.state, proposed, result, time.monotonic() + 10)
    assert recovered == proposed and observed == result
    assert runtime.state.tool_calls == 1
    assert not runtime.direct and not runtime.tools._presented


def test_searching_document_does_not_expand_old_document_blocks(runtime):
    result = runtime._perform_action(runtime.state, {"name": "search_code", "arguments": {
        "path": ".", "pattern": "Core", "mode": "literal", "max_results": 1}}, time.monotonic() + 10)
    assert result["matches"][0]["path"] == "HISTORY.md"
    assert not [card for card in runtime.evidence.cards(fresh=False)
                if card.kind == "source" and card.path == "HISTORY.md"]


def test_explicit_document_read_still_pins_exact_requested_lines(runtime):
    result = runtime._perform_action(runtime.state, {"name": "read_file", "arguments": {
        "path": "HISTORY.md", "start_line": 1, "end_line": 1}}, time.monotonic() + 10)
    assert result["success"]
    assert len(runtime.direct) == 1
    card = runtime.evidence.get(runtime.direct[0])
    assert card.path == "HISTORY.md" and card.text == "Core was introduced in an old release.\n"


def test_test_source_stays_eligible_for_automatic_expansion(runtime):
    target = runtime.workspace.root / "tests/test_core.py"
    target.parent.mkdir()
    target.write_text("def test_core():\n    assert True\n")
    runtime.index.refresh()
    runtime._expand(["tests/test_core.py"], time.monotonic() + 10)
    assert any(card.path == "tests/test_core.py" and card.kind == "source"
               for card in runtime.evidence.cards(fresh=False))
