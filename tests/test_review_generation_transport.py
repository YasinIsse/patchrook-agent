"""Proposal JSON uses its own wire format while coding retains native tools."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import httpx
import pytest

from patchrook_agent.agent import Agent
from patchrook_agent.completion import TaskRequirements
from patchrook_agent.config import HarnessConfig
from patchrook_agent.context.compaction import _groups
from patchrook_agent.context.lean import content_tokens
from patchrook_agent.models.decision import DecisionResult
from patchrook_agent.models.ollama import OllamaClient
from patchrook_agent.proposal_generation import generation_schema
from patchrook_agent.tools import ToolConfig, ToolRegistry, Workspace


TASK = "Reproduce the failing tests and fix general integer addition in calc.py. Preserve existing tests and comments."
IDEAS = {"approaches": [
    {"approach": "Replace return a - b with return a + b for general integer addition.",
     "expected": "Both positive and negative integer operands produce their exact sum."},
    {"approach": "Reverse the subtraction operands to test an operand-order explanation.",
     "expected": "The observed examples reveal whether reversed subtraction implements addition."},
]}


def tool(name, **arguments):
    return {"function": {"name": name, "arguments": arguments}}


def action(*calls):
    return {"role": "assistant", "content": "", "tool_calls": list(calls)}


class Decisions:
    model = "offline-proposal-choice"

    def __init__(self):
        self.calls = []

    def rank(self, context, candidates, **options):
        assert options["purpose"] == "proposal_review"
        assert {"p1", "p2", "none"} <= {item.id for item in candidates}
        self.calls.append((context, deepcopy(candidates), deepcopy(options)))
        return DecisionResult(("p1", *(item.id for item in candidates if item.id != "p1")), {}, called=False)


class Server:
    """A local mocked HTTP boundary; it never starts an Ollama process."""

    def __init__(self, *, native_side_call=False):
        self.requests = []
        self.generated = 0
        self.native_side_call = native_side_call
        self.patched = False
        self.raw_generation = json.dumps(IDEAS, separators=(",", ":"))

    def __call__(self, request):
        assert request.method == "POST" and str(request.url) == "http://localhost:11434/api/chat"
        payload = json.loads(request.content)
        self.requests.append(payload)
        if len(self.requests) == 1:
            message = action(tool("request_review", path="calc.py", reason=(
                "Compare changing the arithmetic operator with reversing operands before selecting a correction.")))
        elif "format" in payload:
            self.generated += 1
            message = {"role": "assistant", "content": self.raw_generation}
            if self.native_side_call and self.generated == 1:
                message["tool_calls"] = [tool("edit_file", path="calc.py", old_text="a - b", new_text="999")]
        elif not self.patched:
            self.patched = True
            message = action(tool("edit_file", path="calc.py", old_text="a - b", new_text="a + b"))
        else:
            message = {"role": "assistant", "content": "DONE"}
        return httpx.Response(200, json={
            "message": message, "prompt_eval_count": 1000, "eval_count": 80,
            "total_duration": 1_000_000, "done_reason": "stop",
        })


def mock_transport(monkeypatch, server):
    original_client = httpx.Client
    def client(**options):
        return original_client(transport=httpx.MockTransport(server), **options)
    monkeypatch.setattr(httpx, "Client", client)
    monkeypatch.setattr(httpx, "get", lambda *args, **kwargs: pytest.fail("No model discovery HTTP is permitted"))


def repository(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "calc.py").write_text("# Preserve this comment.\ndef add(a, b):\n    return a - b\n")
    (root / "test_calc.py").write_text(
        "from calc import add\n\ndef test_positive():\n    assert add(3, 2) == 5\n\n"
        "def test_negative():\n    assert add(-3, 2) == -1\n")
    return root


def agent(tmp_path, client, *, pause=None):
    return Agent(OllamaClient(model="qwen3:8b", context_size=8192, max_output_tokens=512),
                 registry=ToolRegistry(Workspace(tmp_path / "repo", ToolConfig(allow_tests=True))),
                 config=replace(HarnessConfig.preset("C"), workflow_mode="coding", action_protocol="native",
                                review_mode="adaptive", decision_model="on"),
                 decision_client=client, runs_dir=tmp_path / "runs", max_steps=8,
                 deadline_seconds=60, pause_after_step=pause)


def events(state, name):
    path = Path(state.session_dir) / "events.jsonl"
    return [event for line in path.read_text().splitlines()
            if (event := json.loads(line)).get("event") == name]


def test_proposal_generation_http_format_is_small_closed_and_has_no_native_tools(tmp_path, monkeypatch):
    repository(tmp_path)
    server = Server()
    mock_transport(monkeypatch, server)
    instance = agent(tmp_path, Decisions())
    result = instance.run(TASK, requirements=TaskRequirements.fix())
    assert result.status == "completed" and result.verification == "passed", result.final_response
    assert instance.config.action_protocol == "native"
    requests = server.requests
    assert len(requests) == 4
    proposal = requests[1]
    assert proposal["format"] == generation_schema() and "tools" not in proposal
    schema = proposal["format"]
    assert schema["required"] == ["approaches"] and schema["additionalProperties"] is False
    approaches = schema["properties"]["approaches"]
    assert approaches["minItems"] == approaches["maxItems"] == 2
    assert set(approaches["items"]["properties"]) == {"approach", "expected"}
    assert set(approaches["items"]["required"]) == {"approach", "expected"}
    assert approaches["items"]["additionalProperties"] is False
    assert all(set(message) == {"role", "content"} for message in proposal["messages"])
    assert all(message["role"] in {"system", "user"} for message in proposal["messages"])
    assert not any("Current transport uses one JSON action" in message["content"] for message in proposal["messages"])
    assert proposal["think"] is False and proposal["stream"] is False
    assert proposal["truncate"] is False and proposal["shift"] is False
    assert proposal["options"]["num_predict"] == 512
    for request in (requests[0], *requests[2:]):
        assert "format" not in request and "tools" in request
        assert "edit_file" in {item["function"]["name"] for item in request["tools"]}
    prepared = next(event for event in events(result, "model_request") if event["step"] == 2)
    assert prepared["estimated_tokens"] == content_tokens(proposal["messages"], [])
    assert prepared["estimated_tokens"] <= 4000


def test_raw_generation_and_compiled_native_receipt_keep_distinct_provenance(tmp_path, monkeypatch):
    root = repository(tmp_path)
    server = Server()
    mock_transport(monkeypatch, server)
    client = Decisions()
    result = agent(tmp_path, client).run(TASK, requirements=TaskRequirements.fix())
    assert result.status == "completed" and result.verification == "passed", result.final_response
    response = next(event for event in events(result, "model_response") if event["step"] == 2)
    assert response["message"] == {"role": "assistant", "content": server.raw_generation}
    proposals = [call["function"]["arguments"] for message in result.messages
                 for call in message.get("tool_calls", [])
                 if call.get("function", {}).get("name") == "propose_next_steps"]
    assert len(proposals) == 1
    compiled = proposals[0]["proposals"]
    assert [item["id"] for item in compiled[:2]] == ["p1", "p2"]
    assert all(item["kind"] == "implement_change" and item["path"] == "calc.py" for item in compiled[:2])
    assert [{key: item[key] for key in ("approach", "expected")} for item in compiled[:2]] == IDEAS["approaches"]
    assert len(compiled) <= 3
    assert all(item["kind"] == "inspect_source" and item["path"] == "calc.py" for item in compiled[2:])
    evidence = next(event for event in events(result, "model_request") if event["step"] == 2)["proposal_evidence"]
    assert all(item["evidence_refs"] and all(ref in evidence for ref in item["evidence_refs"]) for item in compiled)
    assert all(any(evidence[ref]["kind"] == "source" and evidence[ref]["path"] == "calc.py"
                   for ref in item["evidence_refs"]) for item in compiled)
    raw_messages = [message for message in result.messages
                    if message.get("role") == "assistant" and message.get("content") == server.raw_generation]
    assert len(raw_messages) == 1 and not raw_messages[0].get("tool_calls")
    assert raw_messages[0]["_proposal_response"] == response["message"]
    compiled_messages = [message for message in result.messages if any(
        call.get("function", {}).get("name") == "propose_next_steps" for call in message.get("tool_calls", []))]
    assert len(compiled_messages) == 1 and compiled_messages[0]["content"] == ""
    receipts = [json.loads(message["content"]) for message in result.messages
                if message.get("tool_name") == "propose_next_steps"]
    assert len(receipts) == 1 and receipts[0]["success"]
    _groups(result.messages)  # Synthetic provenance must retain complete call/result groups.
    assert len(client.calls) == 1
    assert (root / "calc.py").read_text() == "# Preserve this comment.\ndef add(a, b):\n    return a + b\n"


def test_proposal_generation_never_executes_unrequested_native_side_calls(tmp_path, monkeypatch):
    root = repository(tmp_path)
    server = Server(native_side_call=True)
    mock_transport(monkeypatch, server)
    result = agent(tmp_path, Decisions()).run(TASK, requirements=TaskRequirements.fix())
    assert result.status == "completed" and result.verification == "passed", result.final_response
    edits = [event for event in events(result, "tool_request") if event.get("name") == "edit_file"]
    assert len(edits) == 1 and edits[0]["arguments"]["new_text"] == "a + b"
    assert not any(call.get("function", {}).get("name") == "edit_file"
                   and call["function"].get("arguments", {}).get("new_text") == "999"
                   for message in result.messages for call in message.get("tool_calls", []))
    assert "999" not in (root / "calc.py").read_text()
    _groups(result.messages)


def test_selected_generated_proposal_resumes_as_native_coding_without_reselection(tmp_path, monkeypatch):
    repository(tmp_path)
    server = Server()
    mock_transport(monkeypatch, server)
    first_client = Decisions()
    first = agent(tmp_path, first_client, pause=2).run(TASK, requirements=TaskRequirements.fix())
    assert first.stop_reason == "paused" and len(first_client.calls) == 1
    saved = json.loads((Path(first.session_dir) / "checkpoint.json").read_text())
    assert saved["config"]["action_protocol"] == "native"
    assert saved["memory"]["task_state"]["selected"]["id"] == "p1"
    second_client = Decisions()
    result = agent(tmp_path, second_client).resume(first.session_id, allow_tests=True)
    assert result.status == "completed" and result.verification == "passed", result.final_response
    assert second_client.calls == [] and len(server.requests) == 4
    assert sum("format" in request for request in server.requests) == 1
    assert all("tools" in request and "format" not in request for request in server.requests[2:])
