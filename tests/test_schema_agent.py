"""Schema actions use the existing execution boundary and observed completion."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import httpx
import pytest

from patchrook_agent.agent import Agent
from patchrook_agent.completion import TaskRequirements
from patchrook_agent.config import HarnessConfig
from patchrook_agent.models.ollama import ModelResponse, OllamaClient
from patchrook_agent.tools import ToolConfig, ToolRegistry, Workspace


def tool(name, **arguments):
    return {"role": "assistant", "content": json.dumps({"action": "tool", "tool": name, "arguments": arguments})}


def complete():
    return {"role": "assistant", "content": json.dumps({"action": "complete", "summary": "Observed work and verification complete."})}


def blocked():
    return {"role": "assistant", "content": json.dumps({"action": "blocked", "reason": "The required dependency is unavailable."})}


class Model:
    context_size = 8192
    max_output_tokens = 512

    def __init__(self, *messages):
        self.messages = iter(messages)
        self.requests = []

    def chat(self, **options):
        self.requests.append(deepcopy(options))
        return ModelResponse(next(self.messages), 3, 2, None, None, None, 1)


def runner(tmp_path, *messages, allow_tests=True, **options):
    root = tmp_path / "repo"
    root.mkdir(exist_ok=True)
    config = replace(HarnessConfig.preset("C"), action_protocol="schema")
    return Agent(Model(*messages), registry=ToolRegistry(Workspace(root, ToolConfig(allow_tests=allow_tests))),
                 config=config, runs_dir=tmp_path / "runs", **options)


def fixture(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (root / "test_calc.py").write_text("from calc import add\ndef test_add():\n    assert add(3, 2) == 5\n")


def can_complete(request):
    return any(branch["properties"]["action"].get("enum") == ["complete"]
               for branch in request["response_format"]["oneOf"])


def test_schema_repair_gates_finish_and_preserves_raw_response_evidence(tmp_path):
    fixture(tmp_path)
    agent = runner(tmp_path, tool("read_file", path="calc.py"),
                   tool("edit_file", path="calc.py", old_text="a - b", new_text="a + b"),
                   tool("run_tests", path="test_calc.py"), complete())
    state = agent.run("Fix addition", requirements=TaskRequirements.fix())
    assert state.status == "completed" and state.verification == "passed"
    assert [can_complete(request) for request in agent.model.requests] == [False, False, False, True]
    for request in agent.model.requests:
        assert request["tools"] is None
        assert all("tool_calls" not in message and message["role"] != "tool" for message in request["messages"])
    assert sum(message["role"] == "tool" for message in state.messages) == 3
    events = [json.loads(line) for line in Path(state.session_dir, "events.jsonl").read_text().splitlines()]
    responses = [event for event in events if event["event"] == "model_response"]
    assert json.loads(responses[0]["message"]["content"])["action"] == "tool"
    requests = [event for event in events if event["event"] == "model_request"]
    assert requests[0]["format"] and requests[0]["tools"] == []


@pytest.mark.parametrize("message", [complete(), {"role": "assistant", "content": "I will read calc.py next."},
                                     {"role": "assistant", "content": '{}', "tool_calls": []}])
def test_premature_complete_prose_and_mixed_calls_are_rejected(tmp_path, message):
    agent = runner(tmp_path, message)
    state = agent.run("Fix code", requirements=TaskRequirements.fix())
    assert state.status == "incomplete" and state.stop_reason == "invalid_schema_action"
    assert state.tool_calls == 0 and state.steps == 1


def test_blocked_is_always_incomplete_and_does_not_execute_tools(tmp_path):
    state = runner(tmp_path, blocked()).run("Fix code", requirements=TaskRequirements.fix())
    assert state.status == "incomplete" and state.stop_reason == "model_blocked"
    assert state.tool_calls == 0 and "dependency" in state.final_response


def test_registry_remains_authority_for_types_and_test_permissions(tmp_path):
    fixture(tmp_path)
    agent = runner(tmp_path, tool("read_file", path=42), tool("run_tests", path="test_calc.py"), blocked(), allow_tests=False)
    state = agent.run("Fix code", requirements=TaskRequirements.fix())
    results = [json.loads(m["content"]) for m in state.messages if m["role"] == "tool"]
    assert [r["error"]["code"] for r in results] == ["invalid_arguments", "permission_denied"]
    assert state.tool_failures == 2 and state.verification == "not_run"
    assert state.status == "incomplete"


def test_inspection_profile_can_finish_without_edits_or_tests(tmp_path):
    fixture(tmp_path)
    agent = runner(tmp_path, tool("read_file", path="calc.py"), complete(), allow_tests=False)
    state = agent.run("Inspect addition", requirements=TaskRequirements.inspect())
    assert state.status == "completed" and state.verification == "not_run" and not state.modified_files
    assert [can_complete(r) for r in agent.model.requests] == [False, True]


def test_schema_api_defaults_to_coding_requirements(tmp_path):
    agent = runner(tmp_path, complete())
    state = agent.run("Check code")
    assert state.stop_reason == "invalid_schema_action" and not can_complete(agent.model.requests[0])
    assert state.completion["requirements"] == TaskRequirements.code().to_dict()


def test_schema_checkpoint_restores_protocol_and_finish_gate(tmp_path):
    fixture(tmp_path)
    first = runner(tmp_path, tool("read_file", path="calc.py"),
                   tool("edit_file", path="calc.py", old_text="a - b", new_text="a + b"), pause_after_step=2)
    paused = first.run("Fix addition", requirements=TaskRequirements.fix())
    # A native-configured controller restores the saved schema protocol.
    model = Model(tool("run_tests", path="test_calc.py"), complete())
    second = Agent(model, workspace=tmp_path / "repo", runs_dir=tmp_path / "runs")
    resumed = second.resume(paused.session_id, allow_tests=True)
    assert resumed.status == "completed" and resumed.verification == "passed"
    assert second.config.action_protocol == "schema"
    assert [can_complete(r) for r in model.requests] == [False, True]
    assert sum(m.get("tool_name") == "edit_file" for m in resumed.messages) == 1


def test_schema_resume_revalidates_external_changes_before_finishing(tmp_path):
    fixture(tmp_path)
    paused = runner(tmp_path, tool("read_file", path="calc.py"), pause_after_step=1).run(
        "Inspect", requirements=TaskRequirements.inspect())
    source = tmp_path / "repo/calc.py"
    source.write_text(source.read_text() + "# changed externally\n")
    second = runner(tmp_path, tool("read_file", path="calc.py"), complete())
    state = second.resume(paused.session_id)
    assert state.status == "completed"
    assert [can_complete(r) for r in second.model.requests] == [False, True]


def test_resume_comment_prefix_cannot_revalidate_changed_source(tmp_path):
    fixture(tmp_path)
    source = tmp_path / "repo/calc.py"
    source.write_text(("# Preserve this contextual note while inspecting the implementation.\n" * 40)
                      + source.read_text())
    paused = runner(tmp_path, tool("read_file", path="calc.py", start_line=41, end_line=42),
                    pause_after_step=1).run("Fix calc.py", requirements=TaskRequirements.fix())
    source.write_text(source.read_text() + "# changed externally\n")
    agent = runner(tmp_path,
                   tool("read_file", path="calc.py", start_line=1, end_line=100),
                   tool("edit_file", path="calc.py", old_text="a - b", new_text="a + b"),
                   tool("read_file", path="calc.py", start_line=41, end_line=42),
                   tool("edit_file", path="calc.py", old_text="a - b", new_text="a + b"),
                   tool("run_tests", path="test_calc.py"), complete())
    state = agent.resume(paused.session_id, allow_tests=True)
    edits = [json.loads(message["content"]) for message in state.messages
             if message.get("tool_name") == "edit_file"]
    assert edits[0]["error"]["code"] == "workspace_changed"
    assert edits[1]["success"] and state.status == "completed"
    assert state.verification == "passed" and not agent._revalidate_paths


def test_schema_guidance_recovers_partial_and_omitted_reads(tmp_path):
    fixture(tmp_path)
    root = tmp_path / "repo"
    names = ["alpha", "beta", "gamma", "delta"]
    for name in names:
        (root / f"{name}.py").write_text("# Context for this implementation.\n" * 80
                                         + "def operation():\n    return 1\n")
    (root / "entry.py").write_text("\n".join(f"import {name}" for name in names) + "\n")
    agent = runner(tmp_path, tool("read_file", path="entry.py"),
                   tool("read_files", files=[{"path": f"{name}.py", "start_line": 1, "end_line": 100}
                                             for name in names]), blocked())
    state = agent.run("Trace entry.py", requirements=TaskRequirements.inspect())
    broad = next(json.loads(message["content"]) for message in state.messages
                 if message.get("tool_name") == "read_files")
    assert broad["truncated"] and broad["omitted_file_count"]
    prefix = broad["files"][0]
    assert prefix["truncated"] and "def operation" not in prefix["content"]
    guidance = agent.model.requests[-1]["messages"][-1]["content"]
    assert '"path":"alpha.py","start_line":' + str(prefix["next_start_line"]) in guidance
    assert '"path":"beta.py","start_line":1' in guidance
    assert "these hints do not count as inspection" in guidance
    assert state.memory["inspection"]["files"]["alpha.py"]["pending_ranges"]
    assert state.memory["inspection"]["files"]["beta.py"]["visible_ranges"] == []


def test_http_schema_request_has_format_and_no_native_tools(monkeypatch):
    observed = []
    original = httpx.Client

    def handler(request):
        observed.append(json.loads(request.content))
        return httpx.Response(200, json={"message": blocked()})

    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))
    schema = {"type": "object", "properties": {"action": {"enum": ["blocked"]}}}
    response = OllamaClient().chat([], response_format=schema)
    assert observed[0]["format"] == schema and "tools" not in observed[0]
    assert response.message == blocked()
    with pytest.raises(ValueError, match="cannot be combined"):
        OllamaClient().chat([], tools=[{"type": "function"}], response_format=schema)
    assert len(observed) == 1
