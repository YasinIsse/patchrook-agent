"""Repair packets keep exact observed evidence small and cache-stable."""

from copy import deepcopy
import json

import pytest

from source_views import request_evidence
from patchrook_agent.context.budget import ContextConfig
from patchrook_agent.context.lean import LeanContext, WORKING_SET_PREFIX
from patchrook_agent.tools import ToolConfig, ToolRegistry, Workspace


def append(raw, name, arguments, result):
    identity = f"repair_{len(raw)}"
    raw.extend([
        {"role": "assistant", "content": "", "tool_calls": [{"id": identity, "type": "function",
         "function": {"name": name, "arguments": arguments}}]},
        {"role": "tool", "tool_name": name, "tool_call_id": identity,
         "content": json.dumps(result, separators=(",", ":"))},
    ])


@pytest.fixture
def observed(tmp_path):
    files = {
        "amount.py": "def adjusted(unit, quantity, returned):\n    subtotal = unit * quantity\n"
                     "    returned_total = unit * returned\n    return subtotal + returned_total\n",
        "tests/test_amount.py": "from amount import adjusted\n\ndef test_returns():\n    assert adjusted(100, 3, 1) == 200\n",
        "unrelated.py": "# Remains available in raw history\n" + "\n".join(
            f"setting_{number} = 'unrelated evidence {number}'" for number in range(160)) + "\n",
    }
    for path, content in files.items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    registry = ToolRegistry(Workspace(tmp_path, ToolConfig(max_output_chars=12000)))
    registry.enable_lean_tools()
    raw = [{"role": "system", "content": "original policy"},
           {"role": "user", "content": "Fix returned merchandise, preserving unrelated behavior."}]
    for path in files:
        args = {"path": path}
        append(raw, "read_file", args, registry.dispatch("read_file", args))
    failure = {"success": False, "status": "failed", "exit_code": 1,
               "stdout": "tests/test_amount.py:4: AssertionError: assert 400 == 200", "stderr": ""}
    append(raw, "run_tests", {"path": "."}, failure)
    packet = {"id": "amount", "fingerprint": "source_version_1", "failure_ids": ["tests/test_amount.py::test_returns"],
              "narrow": False, "source_ranges": [
                  {"path": "amount.py", "start_line": 1, "end_line": 4},
                  {"path": "tests/test_amount.py", "start_line": 1, "end_line": 4}]}
    definitions = registry.definitions(enabled_tools={"edit_file", "read_file"})
    return registry, raw, packet, definitions


def prepare(lean, raw, packet, definitions):
    return lean.prepare(raw, definitions=definitions, compact=True,
                        source_paths=["amount.py", "tests/test_amount.py", "unrelated.py"],
                        state_summary={"verification": "failed", "outstanding_tests": ["."]},
                        repair_packet=packet)


def working(result):
    return next(message["content"] for message in result.messages
                if message.get("content", "").startswith(WORKING_SET_PREFIX))


def prefix(result):
    return working(result).split("\nObserved state (data):\n", 1)[0]


def test_packet_prefix_stays_identical_across_unchanged_native_rereads(observed):
    registry, raw, packet, definitions = observed
    lean = LeanContext(ContextConfig(), workspace=registry.workspace)
    first = prepare(lean, raw, packet, definitions)
    assert first.safe and first.reason == "repair_working_set"
    assert first.estimated_tokens < 3000
    first_prefix = prefix(first)
    assert "4:     return subtotal + returned_total" in first_prefix
    for start, end in ((1, 2), (3, 4), (1, 4)):
        args = {"path": "amount.py", "start_line": start, "end_line": end}
        append(raw, "read_file", args, registry.dispatch("read_file", args))
        canonical = deepcopy(raw)
        result = prepare(lean, raw, packet, definitions)
        assert result.safe
        assert prefix(result) == first_prefix
        assert result.messages[-2:] == raw[-2:]
        assert result.active_messages == canonical == raw
        assert "assert 400 == 200" in working(result)


def test_packet_does_not_refill_spare_room_with_unrelated_source(observed):
    registry, raw, packet, definitions = observed
    result = prepare(LeanContext(ContextConfig(), workspace=registry.workspace), raw, packet, definitions)
    assert result.safe
    assert "unrelated evidence" not in json.dumps(result.messages)
    view = request_evidence(result.messages)
    assert set(view["presented_paths"]) == {"amount.py", "tests/test_amount.py"}


def test_selected_function_identity_is_preserved_as_data(observed):
    registry, raw, packet, definitions = observed
    packet.update(selected_path="amount.py", selected_symbol="adjusted")
    result = prepare(LeanContext(ContextConfig(), workspace=registry.workspace), raw, packet, definitions)
    facts = json.loads(working(result).split("\nObserved state (data):\n", 1)[1].split("\n", 1)[0])
    assert result.safe
    assert facts["repair"]["selected_path"] == "amount.py"
    assert facts["repair"]["selected_symbol"] == "adjusted"


def test_packet_selectors_cannot_load_or_invent_unobserved_source(observed, monkeypatch):
    registry, raw, packet, definitions = observed
    packet["source_ranges"].append({"path": "secret.py", "start_line": 1, "end_line": 10})
    packet["source_ranges"][0]["end_line"] = 12
    def forbidden(*args, **kwargs):
        pytest.fail("A repair selector must never read the filesystem")
    monkeypatch.setattr(type(registry.workspace.root), "read_text", forbidden)
    result = prepare(LeanContext(ContextConfig(), workspace=registry.workspace), raw, packet, definitions)
    assert result.safe
    assert "secret.py" not in request_evidence(result.messages)["presented_paths"]
    assert "missing_source_ranges" in working(result)
    assert "Observed source amount.py lines 1-4:" in prefix(result)
    assert "Observed source amount.py lines 1-12:" not in prefix(result)


def test_stale_packet_ranges_do_not_restore_invalidated_source(observed):
    registry, raw, packet, definitions = observed
    lean = LeanContext(ContextConfig(), workspace=registry.workspace)
    lean.invalidate_paths(["amount.py"], history_length=len(raw))
    saved = lean.to_dict()
    restored = LeanContext(lean.config, workspace=registry.workspace)
    restored.restore(saved)
    result = prepare(restored, raw, packet, definitions)
    assert result.safe
    assert "amount.py" not in request_evidence(result.messages)["presented_paths"]
    assert "missing_source_ranges" in working(result)


def test_current_modified_source_is_protected_even_outside_old_packet(observed):
    registry, raw, packet, definitions = observed
    packet["source_ranges"] = [packet["source_ranges"][1]]
    args = {"path": "amount.py", "old_text": "return subtotal + returned_total",
            "new_text": "return subtotal - returned_total"}
    append(raw, "edit_file", args, registry.dispatch("edit_file", args))
    args = {"path": "amount.py"}
    append(raw, "read_file", args, registry.dispatch("read_file", args))
    append(raw, "run_tests", {"path": "."}, {"success": True, "status": "passed", "exit_code": 0})
    result = prepare(LeanContext(ContextConfig(), workspace=registry.workspace), raw, packet, definitions)
    assert result.safe
    assert "4:     return subtotal - returned_total" in prefix(result)
    assert "4:     return subtotal + returned_total" not in prefix(result)


def test_large_native_failure_is_preserved_and_fails_repair_ceiling(observed):
    registry, raw, packet, definitions = observed
    failure = {"success": False, "status": "failed", "exit_code": 1,
               "stdout": "exact failure evidence\n" * 650, "stderr": ""}
    append(raw, "run_tests", {"path": "."}, failure)
    result = prepare(LeanContext(ContextConfig(), workspace=registry.workspace), raw, packet, definitions)
    assert not result.safe and result.reason == "context_limit"
    assert result.messages[-2:] == raw[-2:]
    assert "repair ceiling is 4000" in result.limit_message


def test_small_configured_input_limit_still_wins_over_repair_ceiling(observed):
    registry, raw, packet, definitions = observed
    config = ContextConfig(context_size=4096, trigger=.60, hard_limit=.75)
    failure = {"success": False, "status": "failed", "exit_code": 1,
               "stdout": "exact failure evidence\n" * 420, "stderr": ""}
    append(raw, "run_tests", {"path": "."}, failure)
    result = prepare(LeanContext(config, workspace=registry.workspace), raw, packet, definitions)
    assert not result.safe
    assert f"repair ceiling is {config.input_limit}" in result.limit_message


def test_partial_read_prefix_never_becomes_an_exact_packet_line(observed):
    registry, raw, packet, definitions = observed
    # A contradictory newer observation invalidates the old neighboring lines;
    # an explicitly partial line does not become exact source evidence.
    append(raw, "read_file", {"path": "amount.py"}, {
        "success": True, "path": "amount.py", "start_line": 1, "end_line": 2,
        "content": "1: def adjusted(unit, quantity, returned):\n2:     subtotal = unit *",
        "partial_line": True, "truncated": True})
    lean = LeanContext(ContextConfig(), workspace=registry.workspace)
    lean.invalidate_paths(["amount.py"], history_length=len(raw) - 2)
    result = prepare(lean, raw, packet, definitions)
    assert result.safe
    visible = request_evidence(result.messages)["source_views"]["amount.py"]["visible_ranges"]
    assert visible == [[1, 1]]
    assert "missing_source_ranges" in working(result)


def test_oversized_optional_packet_can_shrink_without_unrelated_refill(observed):
    registry, raw, packet, definitions = observed
    packet["source_ranges"].append({"path": "unrelated.py", "start_line": 1, "end_line": 161})
    result = prepare(LeanContext(ContextConfig(), workspace=registry.workspace), raw, packet, definitions)
    assert result.safe and result.estimated_tokens <= 4000
    assert "4:     return subtotal + returned_total" in prefix(result)
    assert "Some older source is outside this bounded view" in working(result)


def test_two_explicit_middle_reads_stay_visible_after_later_native_feedback(observed):
    registry, raw, packet, definitions = observed
    lean = LeanContext(ContextConfig(), workspace=registry.workspace)
    packet["requested_ranges"] = []
    for start, end in ((40, 43), (80, 82)):
        args = {"path": "unrelated.py", "start_line": start, "end_line": end}
        append(raw, "read_file", args, registry.dispatch("read_file", args))
        packet["requested_ranges"].append(args)
    result = prepare(lean, raw, packet, definitions)
    assert result.safe
    view = request_evidence([{"role": "user", "content": working(result)}])
    ranges = view["source_views"]["unrelated.py"]["visible_ranges"]
    assert ranges == [[40, 43], [80, 82]]
    assert result.messages[-2:] == raw[-2:]
    saved_prefix = prefix(result)
    args = {"path": "unrelated.py", "start_line": 80, "end_line": 82}
    append(raw, "read_file", args, registry.dispatch("read_file", args))
    assert prefix(prepare(lean, raw, packet, definitions)) == saved_prefix


def test_requested_range_cannot_authorize_source_that_was_never_observed(observed):
    registry, raw, packet, definitions = observed
    packet["requested_ranges"] = [{"path": "secret.py", "start_line": 1, "end_line": 10}]
    result = prepare(LeanContext(ContextConfig(), workspace=registry.workspace), raw, packet, definitions)
    assert result.safe
    assert "secret.py" not in request_evidence(result.messages)["presented_paths"]
    assert "missing_source_ranges" in working(result)


def test_oversized_protected_requested_evidence_is_not_silently_trimmed(observed):
    registry, raw, packet, definitions = observed
    packet["requested_ranges"] = []
    for number in range(4):
        path = f"details_{number}.py"
        (registry.workspace.root / path).write_text("\n".join(
            f"detail_{line} = '{'observed evidence ' * 12}'" for line in range(24)) + "\n")
        args = {"path": path, "start_line": 1, "end_line": 24}
        append(raw, "read_file", args, registry.dispatch("read_file", args))
        packet["requested_ranges"].append(args)
    result = prepare(LeanContext(ContextConfig(), workspace=registry.workspace), raw, packet, definitions)
    assert not result.safe and result.reason == "context_limit"
    view = request_evidence([{"role": "user", "content": working(result)}])
    for number in range(4):
        assert view["source_views"][f"details_{number}.py"]["visible_ranges"] == [[1, 24]]
    assert result.messages[-2:] == raw[-2:]
