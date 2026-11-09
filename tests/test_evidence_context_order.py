"""Presentation order must preserve exact evidence, budgets, and edit guards."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest

from patchrook_agent.agent import Agent
from patchrook_agent.completion import TaskRequirements
from patchrook_agent.config import HarnessConfig
from patchrook_agent.evidence.compiler import estimate_tokens
from patchrook_agent.evidence_workflow import _ordered_messages
from patchrook_agent.models.ollama import ModelResponse
from patchrook_agent.tools import ToolConfig, ToolRegistry, Workspace


def test_order_preserves_all_message_bytes_and_relative_category_order():
    def user(content):
        return {"role": "user", "content": content}
    system = {"role": "system", "content": "stable policy"}
    task = user("task with non-ASCII: café")
    state = user('Observed task state (data):\n{"revision":2}')
    patch = user("Current patch (data, not fresh edit authorization):\n+ changed")
    first = user("Requested source (data) ev_a: a.py:1-2 sha256=a\n1: café\n2: unchanged")
    second = user("Requested source (data) ev_b: b.py:1-1 sha256=b\n1: exact")
    correction = user("Action correction: exact failure")
    failure = user("Unresolved observed check failure (data):\nexact assertion")
    receipt = user("Observed read_file result (data), evidence ev_c:\nexact receipt")
    archive = user("Exact archived evidence selected for the current information need (data, not instructions):\n"
                   "Information need: current task\n\nSource ev_d: d.py:1-1 sha256=d revision=2\n1: exact\n\n"
                   "Observation ev_e: check revision=2\ntrailing exact observation")
    messages = [system, task, state, patch, first, second, correction, failure, receipt, archive]
    before = deepcopy(messages)
    ordered = _ordered_messages(messages)
    assert ordered == [system, task, first, second, archive, state, patch, correction, failure, receipt]
    assert messages == before
    assert sorted(map(id, ordered)) == sorted(map(id, messages))
    assert estimate_tokens(ordered) == estimate_tokens(messages)
    assert _ordered_messages(ordered) == ordered


@pytest.mark.parametrize("role", ["assistant", "tool", "system"])
def test_unexpected_conversation_roles_preserve_native_call_order(role):
    messages = [{"role": "system", "content": "policy"}, {"role": "user", "content": "task"},
                {"role": role, "content": "feedback", "tool_call_id": "call-1"},
                {"role": "user", "content": "Requested source (data) ev_a: exact source"}]
    assert _ordered_messages(messages) is messages


def test_empty_or_unrecognized_evidence_is_not_rewritten():
    for messages in ([], [{"role": "system", "content": "policy"}],
                     [{"role": "system", "content": "policy"}, {"role": "user", "content": "task"},
                      {"role": "user", "content": "source-looking contents without a generated header"}]):
        assert _ordered_messages(messages) == messages


@pytest.mark.parametrize("protocol", ["schema", "native"])
def test_workflow_reordered_source_still_authorizes_only_observed_patch(tmp_path, protocol):
    root = tmp_path / "repo"
    root.mkdir()
    source = "def total(a, b):\n    return a - b\n"
    (root / "calc.py").write_text(source)
    requests = []
    actions = iter([
        {"name": "read_file", "arguments": {"path": "calc.py"}},
        {"name": "apply_edits", "arguments": {"edits": [
            {"path": "calc.py", "old_text": "return a - b", "new_text": "return a + b"}]}},
        {"name": "run_check", "arguments": {"name": "check_1"}},
        {"name": "finish", "arguments": {}},
    ])
    class OfflineModel:
        model = "offline-context-presentation"
        context_size = 8192
        max_output_tokens = 1024
        def chat(self, messages, **options):
            requests.append(deepcopy(messages))
            action = next(actions)
            if len(requests) == 2:
                kinds = [message["content"] for message in messages]
                source_index = next(i for i, content in enumerate(kinds)
                                    if content.startswith("Requested source (data) "))
                state_index = next(i for i, content in enumerate(kinds)
                                   if content.startswith("Observed task state (data):"))
                assert 2 <= source_index < state_index
                assert "1: def total(a, b):\n2:     return a - b" in kinds[source_index]
            message = ({"role": "assistant", "content": json.dumps(action)} if protocol == "schema" else
                       {"role": "assistant", "content": "", "tool_calls": [{"function": {
                           "name": action["name"], "arguments": action["arguments"]}}]})
            return ModelResponse(message, 100, 20, 0, 0, 0, 0)
    base = HarnessConfig.preset("C")
    config = replace(base, workflow_mode="evidence", action_protocol=protocol, review_mode="off",
                     context=replace(base.context, enabled=False), compaction_enabled=False,
                     evidence_test_commands=((sys.executable, "-c",
                         "from calc import total; assert total(2,3)==5; assert total(-4,1)==-3"),))
    state = Agent(model=OfflineModel(), config=config,
                  registry=ToolRegistry(Workspace(root, ToolConfig(allow_tests=True))),
                  runs_dir=tmp_path / "runs", max_steps=4, deadline_seconds=30).run(
                      "Fix total for integers and verify it.", requirements=TaskRequirements.fix())
    assert state.status == "completed", state.final_response
    assert state.verification == "passed" and state.modified_files == ["calc.py"]
    assert (root / "calc.py").read_text() == source.replace("a - b", "a + b")
    records = [json.loads(line) for line in (Path(state.session_dir) / "events.jsonl").read_text().splitlines()]
    contexts = [event for event in records if event["event"] == "evidence_context"]
    assert len(contexts) == len(requests) == 4
    assert contexts[1]["presented_source_ids"]
    for context, messages in zip(contexts, requests):
        assert context["estimated_tokens"] == (estimate_tokens(messages) + context["native_tool_tokens"]
                                                + context["estimation_correction"])
        assert context["estimated_tokens"] <= context["input_limit"]
