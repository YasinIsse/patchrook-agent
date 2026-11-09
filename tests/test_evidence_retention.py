"""Retain bounded exact focus evidence without weakening native exchanges."""

from copy import deepcopy
import json

import pytest

from source_views import request_evidence
from invoice_fixtures import CORRECTIONS, RELEVANT_PATHS, create_fixture
from patchrook_agent.coding import CODING_SYSTEM_PROMPT
from patchrook_agent.context.budget import ContextConfig
from patchrook_agent.context.lean import LeanContext, WORKING_SET_PREFIX, _retained_blocks, _render
from patchrook_agent.tools import ToolConfig, ToolRegistry, Workspace


FOCUS = ["checkout.py", "promotions.py", "tests/test_checkout.py", "adjustments.py"]
PRIORITY = [*FOCUS, "shipping.py", "loyalty.py", "receipts.py", "taxes.py", "audit.py"]


def append(raw, name, arguments, result):
    identity = f"retention_{len(raw)}"
    raw.extend([
        {"role": "assistant", "content": "", "tool_calls": [{"id": identity, "type": "function",
         "function": {"name": name, "arguments": arguments}}]},
        {"role": "tool", "tool_name": name, "tool_call_id": identity,
         "content": json.dumps(result, ensure_ascii=False, separators=(",", ":"))},
    ])


@pytest.fixture
def observed(tmp_path):
    root = tmp_path / "repo"
    task = create_fixture(root)
    registry = ToolRegistry(Workspace(root, ToolConfig(allow_tests=True, max_output_chars=12000)))
    registry.lean_mode = True
    registry.code_context = False
    registry.enable_lean_tools()
    raw = [{"role": "system", "content": CODING_SYSTEM_PROMPT},
           {"role": "user", "content": task.instructions}]
    for paths in (["audit.py", "checkout.py", "promotions.py"],
                  ["tests/test_checkout.py", "adjustments.py", "shipping.py"],
                  ["loyalty.py", "receipts.py", "taxes.py"]):
        arguments = {"paths": paths}
        append(raw, "read_code", arguments, registry.dispatch("read_code", arguments))
    failure = registry.dispatch("run_tests", {"path": "."})
    assert failure["status"] == "failed" and failure["test_counts"]["failed"] == 2
    append(raw, "run_tests", {"path": "."}, failure)
    definitions = registry.definitions(enabled_tools={"read_code", "read_file", "edit_file", "write_file"})
    return root, task, registry, raw, failure, definitions


def prepared(lean, raw, definitions, *, retained=FOCUS, priority=PRIORITY):
    return lean.prepare(raw, definitions=definitions, compact=True,
                        system_prompt=CODING_SYSTEM_PROMPT,
                        protocol_instructions="Use exact observed source and test failures to correct the implementation.",
                        state_summary={"verification": "failed", "outstanding_tests": ["."]},
                        source_paths=PRIORITY, source_priority=priority,
                        retained_source_paths=retained)


def working(result):
    return next(message["content"] for message in result.messages
                if message.get("content", "").startswith(WORKING_SET_PREFIX))


def assert_bug_lines_visible(result, task):
    view = request_evidence(result.messages, list(task.files), expected_sources=task.files)
    for path in RELEVANT_PATHS:
        old = CORRECTIONS[path][0]
        number = task.files[path][:task.files[path].index(old)].count("\n") + 1
        assert any(start <= number <= end for start, end in view["source_views"][path]["visible_ranges"]), view


def test_controlled_read_loop_keeps_both_faulty_operations_and_exact_errors(observed):
    root, task, registry, raw, failure, definitions = observed
    lean = LeanContext(ContextConfig(context_size=8192), workspace=registry.workspace)
    reads = [("adjustments.py", 2, 47), ("taxes.py", 2, 45), ("shipping.py", 2, 45),
             ("taxes.py", 2, 45), ("shipping.py", 2, 45)]
    for request in [None, *reads]:
        if request is not None:
            path, first, last = request
            arguments = {"path": path, "start_line": first, "end_line": last}
            append(raw, "read_file", arguments, registry.dispatch("read_file", arguments))
        original = deepcopy(raw)
        result = prepared(lean, raw, definitions)
        assert result.safe and result.estimated_tokens <= lean.config.input_limit
        assert_bug_lines_visible(result, task)
        assert result.messages[-3:-1] == raw[-2:]
        assert result.active_messages == original == raw
        if request is not None:
            facts = json.loads(working(result).split("\nObserved state (data):\n", 1)[1].split("\n", 1)[0])
            assert facts["latest_test"]["result"] == failure


def test_small_focus_excerpts_improve_retention_even_with_old_unranked_prefix(observed):
    root, task, registry, raw, failure, definitions = observed
    arguments = {"path": "taxes.py", "start_line": 2, "end_line": 45}
    append(raw, "read_file", arguments, registry.dispatch("read_file", arguments))
    old_priority = ["audit.py", "checkout.py", "promotions.py", "tests/test_checkout.py", "adjustments.py"]
    legacy = prepared(LeanContext(ContextConfig(), workspace=registry.workspace), raw, definitions,
                      retained=None, priority=old_priority)
    retained = prepared(LeanContext(ContextConfig(), workspace=registry.workspace), raw, definitions,
                        priority=old_priority)
    assert legacy.safe and retained.safe
    assert "adjustments.py" not in request_evidence(legacy.messages)["presented_paths"]
    assert_bug_lines_visible(retained, task)
    assert legacy.messages[-3:-1] == retained.messages[-3:-1] == raw[-2:]


def test_focus_does_not_read_hidden_source_or_invent_unobserved_lines(observed, monkeypatch):
    root, task, registry, raw, failure, definitions = observed
    (root / "hidden.py").write_text("secret = 'must not enter the prompt'\n")
    def forbidden(*args, **kwargs):
        pytest.fail("Retention must not read hidden files")
    monkeypatch.setattr(type(root), "read_text", forbidden)
    result = prepared(LeanContext(ContextConfig(), workspace=registry.workspace), raw, definitions,
                      retained=["hidden.py", "adjustments.py"])
    assert result.safe
    assert "must not enter the prompt" not in json.dumps(result.messages)
    assert "hidden.py" not in request_evidence(result.messages)["presented_paths"]


def test_added_helper_does_not_move_the_observed_primary_function_tail():
    lines = {index: line for index, line in enumerate([
        "def calculate(value):", *[f"    check_{index} = value" for index in range(40)],
        "    return value - 1", "", "def unrelated(value):",
        *[f"    validation_{index} = value" for index in range(40)], "    return value",
    ], 1)}
    blocks = _retained_blocks("module.py", lines, 0)
    visible = {number: line for block in blocks for number, line in block["lines"]}
    assert len(visible) <= 24
    assert visible[1] == "def calculate(value):"
    assert visible[42] == "    return value - 1"
    assert "def unrelated(value):" not in visible.values()
    assert all(lines[number] == line for number, line in visible.items())


def test_focus_excerpts_keep_stable_prefix_across_unrelated_reads(observed):
    root, task, registry, raw, failure, definitions = observed
    lean = LeanContext(ContextConfig(), workspace=registry.workspace)
    first = prepared(lean, raw, definitions)
    anchors = []
    for path in FOCUS:
        lines = task.files[path].splitlines()
        # The actual initial read_code selections begin at each first function;
        # use their observed lines, not file contents outside those selections.
        records = [json.loads(message["content"]) for message in raw if message.get("tool_name") == "read_code"]
        record = next(item for result in records for item in result["files"] if item["path"] == path)
        from patchrook_agent.inspection import visible_lines
        anchors += _retained_blocks(path, visible_lines(record), 0)
    prefix = _render({}, anchors, False, source_first=True, source_priority=FOCUS).split("\nObserved state (data):\n")[0]
    assert working(first).startswith(prefix)
    for path in ("taxes.py", "shipping.py"):
        arguments = {"path": path, "start_line": 2, "end_line": 45}
        append(raw, "read_file", arguments, registry.dispatch("read_file", arguments))
        assert working(prepared(lean, raw, definitions)).startswith(prefix)


def test_latest_oversized_error_cannot_be_dropped_to_make_room_for_focus(observed):
    root, task, registry, raw, failure, definitions = observed
    large = {**failure, "stderr": "Exact protected error\n" * 1600}
    append(raw, "run_tests", {"path": "."}, large)
    lean = LeanContext(ContextConfig(), workspace=registry.workspace)
    result = prepared(lean, raw, definitions)
    assert not result.safe and result.reason == "context_limit"
    assert result.messages[-3:-1] == raw[-2:]
    assert json.loads(result.messages[-2]["content"]) == large


def test_retention_uses_fresh_observations_after_mutation_and_restore(observed):
    root, task, registry, raw, failure, definitions = observed
    lean = LeanContext(ContextConfig(), workspace=registry.workspace)
    old, new = CORRECTIONS["adjustments.py"]
    arguments = {"path": "adjustments.py", "old_text": old, "new_text": new}
    append(raw, "edit_file", arguments, registry.dispatch("edit_file", arguments))
    arguments = {"paths": ["adjustments.py"]}
    append(raw, "read_code", arguments, registry.dispatch("read_code", arguments))
    current = prepared(lean, raw, definitions)
    assert current.safe
    assert f"47:     {new}" in json.dumps(current.messages)
    assert f"47:     {old}" not in json.dumps(current.messages)
    restored = LeanContext(lean.config, workspace=registry.workspace)
    restored.restore(lean.to_dict())
    assert prepared(restored, raw, definitions).messages == current.messages


def test_external_source_invalidation_prevents_old_focus_reappearing(observed):
    root, task, registry, raw, failure, definitions = observed
    lean = LeanContext(ContextConfig(), workspace=registry.workspace)
    lean.invalidate_paths(["adjustments.py"], history_length=len(raw))
    result = prepared(lean, raw, definitions)
    assert result.safe
    assert "adjustments.py" not in request_evidence(result.messages)["presented_paths"]
