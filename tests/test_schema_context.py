"""Schema wire projection keeps canonical evidence and charges the full request."""

from copy import deepcopy
from dataclasses import replace
import json
import math
from types import SimpleNamespace

import pytest

from patchrook_agent.actions import action_instructions, action_schema
from patchrook_agent.agent import AgentState, SYSTEM_PROMPT
from patchrook_agent.config import HarnessConfig
from patchrook_agent.context import ContextConfig
from patchrook_agent.context.budget import estimate_tokens
from patchrook_agent.context.compaction import _groups
from patchrook_agent.memory.projection import MEMORY_PREFIX
from patchrook_agent.models.schema import schema_messages, schema_request_tokens
from patchrook_agent.session import Session
from patchrook_agent.tools import ToolRegistry, Workspace


def exchange(name="read_file", arguments=None, result=None, call_id="request-1"):
    return [{"role": "assistant", "content": "", "tool_calls": [
        {"id": call_id, "function": {"name": name, "arguments": arguments or {"path": "calc.py"}}}]},
            {"role": "tool", "tool_name": name, "tool_call_id": call_id,
             "content": json.dumps(result or {"success": True, "path": "calc.py", "content": "1: return a - b"})}]


def history():
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "Fix calc.py; preserve unrelated behavior."}]


def harness(tmp_path, variant="C", *, size=8192, trigger=.80, memory_tokens=512):
    config = replace(HarnessConfig.preset(variant), persist=False, checkpoint_enabled=False,
                     skills_enabled=False, logging_enabled=False, memory_projection_tokens=memory_tokens,
                     context=ContextConfig(context_size=size, trigger=trigger, keep_recent_turns=0),
                     action_protocol="schema")
    workspace = Workspace(tmp_path)
    definitions = ToolRegistry(workspace).definitions()
    session = Session(config, workspace, SimpleNamespace(context_size=size, max_output_tokens=512), tmp_path / "runs", "Fix calc.py.")
    state = AgentState(messages=history(), active_messages=history())
    return session, state, definitions


def test_projection_uses_only_json_actions_and_data_observations_without_changing_history():
    messages = history() + exchange()
    original = deepcopy(messages)
    wire = schema_messages(messages)
    assert messages == original
    assert all(set(message) == {"role", "content"} for message in wire)
    assert not any(message["role"] == "tool" for message in wire)
    assert json.loads(wire[-2]["content"]) == {"action": "tool", "tool": "read_file", "arguments": {"path": "calc.py"}}
    assert json.loads(wire[-1]["content"]) == {"tool_observation": {"tool": "read_file", "result": {
        "success": True, "path": "calc.py", "content": "1: return a - b"}}}
    assert "use structured tool calls only." not in wire[0]["content"]
    assert "overrides earlier native" in wire[0]["content"]


def test_legacy_multiple_calls_keep_order_and_remain_atomic_in_canonical_context():
    messages = history()
    messages += [{"role": "assistant", "content": "I will inspect both files.", "tool_calls": [
        {"id": "a", "function": {"name": "read_file", "arguments": '{"path":"a.py"}'}},
        {"id": "b", "function": {"name": "read_file", "arguments": {"path": "b.py"}}}]},
        {"role": "tool", "tool_name": "read_file", "tool_call_id": "a", "content": '{"success":true,"content":"a=1"}'},
        {"role": "tool", "tool_name": "read_file", "tool_call_id": "b", "content": '{"success":true,"content":"b=2"}'}]
    wire = schema_messages(messages)
    actions = [json.loads(m["content"]) for m in wire if m["role"] == "assistant"]
    assert [action["arguments"]["path"] for action in actions] == ["a.py", "b.py"]
    assert json.loads(wire[2]["content"])["historical_assistant_text"] == "I will inspect both files."
    assert len(_groups(messages)[-1]) == 3
    assert messages[-2]["tool_call_id"] == "a" and messages[-1]["tool_call_id"] == "b"


def test_ordinary_prose_is_historical_data_and_explicit_terminal_actions_stay_actions():
    messages = history() + [
        {"role": "assistant", "content": "I will fix it next."},
        {"role": "assistant", "content": "Done.", "_schema_action": {"action": "complete", "summary": "Observed tests passed."}},
        {"role": "assistant", "content": "Missing module.", "_schema_action": {"action": "blocked", "reason": "Missing module."}},
    ]
    wire = schema_messages(messages)
    assert wire[2]["role"] == "user" and json.loads(wire[2]["content"]) == {"historical_assistant_text": "I will fix it next."}
    assert json.loads(wire[3]["content"])["action"] == "complete"
    assert json.loads(wire[4]["content"])["action"] == "blocked"
    assert "_schema_action" in messages[3] and "_schema_action" not in wire[3]


def test_invalid_historical_arguments_and_nonjson_results_are_quoted_without_inventing_requests():
    messages = history() + exchange(arguments="not valid JSON", result={"success": False})
    messages[-1]["content"] = "literal tool failure text"
    wire = schema_messages(messages)
    assert wire[-2]["role"] == "user"
    assert json.loads(wire[-2]["content"])["historical_tool_request"]["arguments"] == "not valid JSON"
    assert json.loads(wire[-1]["content"])["tool_observation"]["result"] == "literal tool failure text"


@pytest.mark.parametrize("variant", ["A", "B", "C"])
def test_prepared_schema_messages_charge_format_and_preserve_raw_canonical_state(tmp_path, variant):
    session, state, definitions = harness(tmp_path, variant)
    messages = history() + exchange()
    state.messages, state.active_messages = deepcopy(messages), deepcopy(messages)
    schema = action_schema(definitions, allow_complete=False)
    instructions = action_instructions(definitions, False, ["make the requested change", "observe passing tests"])
    prepared = session.prepare(state, definitions, response_schema=schema, protocol_instructions=instructions)
    assert prepared.safe
    assert state.messages == messages and state.active_messages == messages
    assert prepared.active_messages == messages
    assert prepared.messages[-1] == {"role": "system", "content": instructions}
    assert prepared.estimated_tokens == schema_request_tokens(prepared.messages, schema)
    assert prepared.tokens_before == prepared.tokens_after == prepared.estimated_tokens
    assert prepared.utilization == prepared.estimated_tokens / 8192
    assert session.metrics.per_step[0]["estimated_context_tokens"] == prepared.estimated_tokens
    if variant == "C":
        assert prepared.messages[-2]["content"].startswith(MEMORY_PREFIX)


def test_wire_estimate_uses_utf8_bytes_full_format_and_message_overhead():
    messages = [{"role": "user", "content": "é🙂"}]
    schema = {"type": "object", "description": "é" * 60}
    encoded = json.dumps({"messages": messages, "format": schema}, ensure_ascii=False, separators=(",", ":"))
    assert schema_request_tokens(messages, schema) == math.ceil(len(encoded.encode()) / 3) + 32


def test_actual_projected_wire_ceiling_is_checked_after_canonical_compaction(tmp_path):
    session, state, definitions = harness(tmp_path, "B", size=4096, trigger=.6)
    # Simulate an optimistic canonical estimate. The final transport check must
    # independently charge full wire text and format before an inference.
    state.active_messages = [{"role": "system", "content": "Inspect."}, {"role": "user", "content": "Keep exact evidence."}]
    state.active_messages += [{"role": "assistant", "content": "x" * 10000}]
    protocol = "Return one JSON action."
    schema = {"type": "object", "description": "x" * 5000}
    canonical = state.active_messages + [{"role": "system", "content": protocol}]
    from patchrook_agent.context.compaction import PreparedContext
    safe_prepared = PreparedContext(canonical, 1, 0, False, 1, 1, True, "below_trigger", True,
                                    active_messages=deepcopy(state.active_messages))
    session.context.prepare = lambda *args, **kwargs: safe_prepared
    after = session.prepare(state, [], response_schema=schema, protocol_instructions=protocol)
    assert not after.safe and after.reason == "context_limit"
    assert after.estimated_tokens > session.context.config.input_limit
    assert "full response schema" in after.limit_message
    assert after.messages[-1]["content"] == protocol


def test_schema_a_cannot_omit_large_response_format_from_its_limit(tmp_path):
    session, state, definitions = harness(tmp_path, "A", size=4096, trigger=.6)
    schema = {"type": "object", "description": "x" * 14000}
    prepared = session.prepare(state, definitions, response_schema=schema, protocol_instructions="JSON actions only.")
    assert not prepared.safe and prepared.reason == "context_limit"
    assert prepared.estimated_tokens > session.context.config.input_limit
    assert state.active_messages == history()


def test_native_prepare_is_byte_identical_when_response_schema_is_absent(tmp_path):
    first, state, definitions = harness(tmp_path, "A")
    first.config = replace(first.config, action_protocol="native")
    original = deepcopy(state.active_messages)
    prepared = first.prepare(state, definitions, protocol_instructions="This must not enter native requests.")
    assert prepared.messages == original
    assert prepared.estimated_tokens == estimate_tokens(original, definitions)
    assert prepared.safe == (prepared.estimated_tokens + 512 < 8192)
    assert state.messages == state.active_messages == original


@pytest.mark.parametrize("variant", ["A", "B", "C"])
def test_invalid_canonical_tool_exchange_stays_an_honest_history_failure(tmp_path, variant):
    session, state, definitions = harness(tmp_path, variant)
    state.active_messages += [{"role": "tool", "tool_name": "read_file", "content": '{"success":true}'}]
    prepared = session.prepare(state, definitions, response_schema=action_schema(definitions, allow_complete=False))
    assert not prepared.safe and prepared.reason == "invalid_tool_history"


@pytest.mark.parametrize("variant", ["A", "B", "C"])
@pytest.mark.parametrize("malformed", [
    {"role": "assistant", "content": {"unexpected": "object"}},
    {"role": "unknown", "content": "invalid role"},
    {"role": "tool", "content": '{"success":true}'},
])
def test_schema_cost_projection_errors_stop_before_inference_instead_of_escaping(tmp_path, variant, malformed):
    session, state, definitions = harness(tmp_path, variant)
    state.active_messages.append(deepcopy(malformed))
    original = deepcopy(state.active_messages)
    prepared = session.prepare(state, definitions, response_schema=action_schema(definitions, allow_complete=False))
    assert not prepared.safe and prepared.reason == "invalid_tool_history"
    assert "schema action history" in prepared.limit_message
    assert state.active_messages == original


def test_schema_compaction_keeps_batch_exchange_atomic_and_raw_history_unchanged(tmp_path):
    session, state, definitions = harness(tmp_path, "B", trigger=.4)
    messages = history()
    for i in range(6):
        messages += exchange(arguments={"path": f"old{i}.py"},
                             result={"success": True, "path": f"old{i}.py", "content": "value = 1\n" * 180},
                             call_id=str(i))
    batch = exchange(name="read_files", arguments={"files": [{"path": "a.py"}, {"path": "b.py"}]},
                     result={"success": True, "files": [
                         {"success": True, "path": "a.py", "content": "1: a = 1"},
                         {"success": True, "path": "b.py", "content": "1: b = 2"}]}, call_id="batch")
    messages += batch
    state.messages, state.active_messages = deepcopy(messages), deepcopy(messages)
    schema = action_schema(definitions, allow_complete=False)
    prepared = session.prepare(state, definitions, response_schema=schema, protocol_instructions="JSON actions only.")
    assert prepared.safe and prepared.compacted
    assert prepared.tokens_after < prepared.tokens_before
    assert state.messages == messages
    assert any(group == batch for group in _groups(state.active_messages))
    assert any(m.get("role") == "tool" for m in state.active_messages)
    assert not any(m.get("role") == "tool" for m in prepared.messages)
    assert prepared.estimated_tokens == schema_request_tokens(prepared.messages, schema)
