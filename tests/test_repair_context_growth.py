"""Keep multi-edit repair and semantic completion review within their budget."""

from copy import deepcopy
import json

import pytest

from source_views import request_evidence
from invoice_fixtures import CORRECTIONS, create_fixture
from patchrook_agent.agent import Agent, AgentState
from patchrook_agent.coding import CODING_SYSTEM_PROMPT
from patchrook_agent.context.budget import ContextConfig
from patchrook_agent.context.lean import LeanContext, WORKING_SET_PREFIX
from patchrook_agent.repair import build_repair_plan
from patchrook_agent.tools import ToolConfig, ToolRegistry, Workspace


def append(raw, name, arguments, result):
    identity = f"growth_{len(raw)}"
    raw.extend([
        {"role": "assistant", "content": "", "tool_calls": [{"id": identity, "type": "function",
         "function": {"name": name, "arguments": arguments}}]},
        {"role": "tool", "tool_name": name, "tool_call_id": identity,
         "content": json.dumps(result, separators=(",", ":"))},
    ])


def dispatch(registry, raw, name, arguments):
    result = registry.dispatch(name, arguments)
    append(raw, name, arguments, result)
    return result


def tools(registry):
    return registry.definitions(enabled_tools={"edit_file", "write_file", "read_file"})


def working(prepared):
    return next(message["content"] for message in prepared.messages
                if message.get("content", "").startswith(WORKING_SET_PREFIX))


def facts(prepared):
    return json.loads(working(prepared).split("\nObserved state (data):\n", 1)[1].split("\n", 1)[0])


def checkout_history(tmp_path, *, wrong_discount):
    task = create_fixture(tmp_path)
    registry = ToolRegistry(Workspace(tmp_path, ToolConfig(allow_tests=True, max_output_chars=12000)))
    registry.enable_lean_tools()
    registry.code_context = False
    raw = [{"role": "system", "content": CODING_SYSTEM_PROMPT},
           {"role": "user", "content": task.instructions}]
    for paths in (["audit.py", "checkout.py", "loyalty.py"],
                  ["shipping.py", "adjustments.py", "promotions.py"],
                  ["taxes.py", "receipts.py", "tests/test_checkout.py"]):
        dispatch(registry, raw, "read_code", {"paths": paths})
    assert dispatch(registry, raw, "run_tests", {"path": "."})["status"] == "failed"
    for path in ("adjustments.py", "promotions.py"):
        old, new = CORRECTIONS[path]
        if path == "promotions.py" and wrong_discount:
            new = old + " - 1"
        result = dispatch(registry, raw, "edit_file", {"path": path, "old_text": old, "new_text": new})
        assert result["success"] and result["modified"]
        dispatch(registry, raw, "read_code", {"paths": [path]})
        final_tests = dispatch(registry, raw, "run_tests", {"path": "."})
    return task, registry, raw, final_tests


def test_two_changed_helpers_do_not_exhaust_repair_and_correction_still_passes(tmp_path):
    task, registry, raw, failure = checkout_history(tmp_path, wrong_discount=True)
    assert failure["test_counts"] == {"failed": 1, "passed": 2}
    assert "assert 6601 == 6600" in failure["stdout"]
    plan = build_repair_plan(raw, task.instructions, revision=2)
    original = deepcopy(raw)
    prepared = LeanContext(ContextConfig(), workspace=registry.workspace).prepare(
        raw, definitions=tools(registry), system_prompt=CODING_SYSTEM_PROMPT, repair_packet=plan.packet(),
        state_summary={"verification": "failed", "outstanding_tests": ["."],
                       "modified_files": ["adjustments.py", "promotions.py"]},
        source_paths=list(task.files))
    assert prepared.safe and prepared.reason == "repair_working_set"
    assert prepared.estimated_tokens < 3200
    assert prepared.messages[-2:] == raw[-2:]
    assert prepared.active_messages == original == raw
    view = request_evidence(prepared.messages)
    assert sum(end - start + 1 for start, end in view["source_views"]["adjustments.py"]["visible_ranges"]) <= 12
    old, new = CORRECTIONS["promotions.py"]
    arguments = {"path": "promotions.py", "old_text": old + " - 1", "new_text": new}
    call = {"function": {"name": "edit_file", "arguments": arguments}}
    # The compact request must still authorize this exact current one-line
    # correction through the production source-before-edit guard.
    assert Agent(model=object(), registry=registry)._coding_edit_error(AgentState(), call, prepared.messages) == ""
    assert dispatch(registry, raw, "edit_file", arguments)["success"]
    assert dispatch(registry, raw, "run_tests", {"path": "."})["test_counts"] == {"passed": 3}


def test_passing_changes_use_compact_semantic_review_with_exact_verification(tmp_path):
    task, registry, raw, passed = checkout_history(tmp_path, wrong_discount=False)
    assert passed["status"] == "passed"
    lean = LeanContext(ContextConfig(), workspace=registry.workspace)
    original = deepcopy(raw)
    options = dict(definitions=tools(registry), system_prompt=CODING_SYSTEM_PROMPT,
                   compact=True, source_paths=list(task.files),
                   state_summary={"verification": "passed", "outstanding_tests": [],
                                  "modified_files": ["adjustments.py", "promotions.py"]})
    broad = lean.prepare(raw, **options)
    review = lean.prepare(raw, completion_review=True, **options)
    assert review.safe and review.reason == "completion_review"
    assert review.estimated_tokens < 2600 < broad.estimated_tokens
    assert review.active_messages == original == raw
    assert review.messages[-2:] == raw[-2:]
    assert raw[1] in review.messages
    assert "3 passed" in json.dumps(review.messages)
    assert len(facts(review)["accepted_changes"]) == 2
    assert {item["path"] for item in facts(review)["accepted_changes"]} == {"adjustments.py", "promotions.py"}
    view = request_evidence(review.messages)
    assert set(view["presented_paths"]) == {"adjustments.py", "promotions.py"}
    for path, (_, corrected) in CORRECTIONS.items():
        assert corrected in working(review)


def test_review_keeps_multiple_accepted_edits_on_same_file_with_omitted_count(tmp_path):
    (tmp_path / "value.py").write_text("value = 0\n")
    registry = ToolRegistry(Workspace(tmp_path))
    raw = [{"role": "system", "content": "policy"}, {"role": "user", "content": "Update the value."}]
    dispatch(registry, raw, "read_file", {"path": "value.py"})
    for number in range(9):
        dispatch(registry, raw, "edit_file", {"path": "value.py", "old_text": f"value = {number}",
                                               "new_text": f"value = {number + 1}"})
        dispatch(registry, raw, "read_file", {"path": "value.py"})
    append(raw, "run_tests", {"path": "."}, {"success": True, "status": "passed", "exit_code": 0})
    prepared = LeanContext(ContextConfig()).prepare(raw, definitions=tools(registry), completion_review=True)
    assert prepared.safe
    data = facts(prepared)
    assert len(data["accepted_changes"]) == 8 and data["older_accepted_changes"] == 1
    assert data["accepted_changes"][0]["accepted_delta"]["old_text"] == "value = 1"
    assert data["accepted_changes"][-1]["accepted_delta"]["new_text"] == "value = 9"


def test_accepted_delta_without_fresh_read_never_becomes_current_review_source(tmp_path):
    (tmp_path / "value.py").write_text("value = 1\n")
    registry = ToolRegistry(Workspace(tmp_path))
    raw = [{"role": "system", "content": "policy"}, {"role": "user", "content": "Update value."}]
    dispatch(registry, raw, "read_file", {"path": "value.py"})
    dispatch(registry, raw, "edit_file", {"path": "value.py", "old_text": "value = 1", "new_text": "value = 2"})
    append(raw, "run_tests", {"path": "."}, {"success": True, "status": "passed", "exit_code": 0})
    prepared = LeanContext(ContextConfig()).prepare(raw, definitions=tools(registry), completion_review=True)
    assert prepared.safe
    assert "value.py" not in request_evidence(prepared.messages)["presented_paths"]
    assert facts(prepared)["accepted_changes"][0]["accepted_delta"]["new_text"] == "value = 2"
    call = {"function": {"name": "edit_file", "arguments": {
        "path": "value.py", "old_text": "value = 2", "new_text": "value = 3"}}}
    assert Agent(model=object(), registry=registry)._coding_edit_error(AgentState(), call, prepared.messages)


def test_deletion_protects_observed_neighbors_and_write_fallback_stays_small(tmp_path):
    content = "\n".join(f"value_{number} = {number}" for number in range(80)) + "\n"
    (tmp_path / "value.py").write_text(content)
    registry = ToolRegistry(Workspace(tmp_path, ToolConfig(max_output_chars=12000)))
    raw = [{"role": "system", "content": "policy"}, {"role": "user", "content": "Update value files."}]
    dispatch(registry, raw, "read_file", {"path": "value.py"})
    dispatch(registry, raw, "edit_file", {"path": "value.py", "old_text": "value_40 = 40\n", "new_text": ""})
    dispatch(registry, raw, "read_file", {"path": "value.py"})
    dispatch(registry, raw, "write_file", {"path": "new.py", "content": content})
    dispatch(registry, raw, "read_file", {"path": "new.py"})
    append(raw, "run_tests", {"path": "."}, {"success": True, "status": "passed", "exit_code": 0})
    prepared = LeanContext(ContextConfig()).prepare(raw, definitions=tools(registry), completion_review=True)
    assert prepared.safe
    view = request_evidence(prepared.messages)
    assert sum(end - start + 1 for start, end in view["source_views"]["new.py"]["visible_ranges"]) <= 12
    assert "41: value_41 = 41" in working(prepared)
    assert "41: value_40 = 40" not in working(prepared)


def test_ambiguous_new_text_keeps_the_observed_edit_location(tmp_path):
    content = "def early(value):\n    return value\n\n" + "\n" * 30 + "def later(value):\n    return value + 1\n"
    (tmp_path / "value.py").write_text(content)
    registry = ToolRegistry(Workspace(tmp_path))
    raw = [{"role": "system", "content": "policy"}, {"role": "user", "content": "Correct the later helper."}]
    dispatch(registry, raw, "read_file", {"path": "value.py"})
    dispatch(registry, raw, "edit_file", {"path": "value.py", "old_text": "return value + 1", "new_text": "return value"})
    dispatch(registry, raw, "read_file", {"path": "value.py"})
    append(raw, "run_tests", {"path": "."}, {"success": True, "status": "passed", "exit_code": 0})
    prepared = LeanContext(ContextConfig()).prepare(raw, definitions=tools(registry), completion_review=True)
    assert prepared.safe
    # The identical earlier return cannot displace the actual later edit.
    assert "35:     return value" in working(prepared)
    visible = request_evidence(prepared.messages)["source_views"]["value.py"]["visible_ranges"]
    assert sum(end - start + 1 for start, end in visible) <= 12


def test_optional_packet_source_repacks_before_rejecting_large_native_evidence(tmp_path):
    (tmp_path / "extra.py").write_text("\n".join(f"value_{n} = '{'details ' * 6}'" for n in range(150)) + "\n")
    registry = ToolRegistry(Workspace(tmp_path, ToolConfig(max_output_chars=12000)))
    raw = [{"role": "system", "content": "policy"}, {"role": "user", "content": "Fix a value."}]
    dispatch(registry, raw, "read_file", {"path": "extra.py"})
    packet = {"id": "work", "fingerprint": "version", "failure_ids": [], "narrow": False,
              "source_ranges": [{"path": "extra.py", "start_line": 1, "end_line": 150}]}
    lean = LeanContext(ContextConfig())
    append(raw, "run_tests", {"path": "."}, {"success": False, "status": "failed", "exit_code": 1,
                                                "stdout": "assert value == expected"})
    small = lean.prepare(raw, definitions=tools(registry), repair_packet=packet)
    failure = {"success": False, "status": "failed", "exit_code": 1, "stdout": "exact failure\n" * 500}
    append(raw, "run_tests", {"path": "."}, failure)
    large = lean.prepare(raw, definitions=tools(registry), repair_packet=packet)
    assert large.safe and large.estimated_tokens <= 4000
    assert large.messages[-2:] == raw[-2:]
    first_source = request_evidence([message for message in small.messages if message["role"] == "user"])
    last_source = request_evidence([message for message in large.messages if message["role"] == "user"])
    first_count = first_source["source_views"]["extra.py"]["visible_line_count"]
    last_count = last_source["source_views"].get("extra.py", {}).get("visible_line_count", 0)
    assert last_count < first_count


def test_review_still_rejects_irreducible_native_overflow(tmp_path):
    registry = ToolRegistry(Workspace(tmp_path))
    raw = [{"role": "system", "content": "policy"}, {"role": "user", "content": "Review the task."}]
    append(raw, "run_tests", {"path": "."}, {"success": True, "status": "passed", "exit_code": 0,
                                                "stdout": "exact passing output\n" * 800})
    result = LeanContext(ContextConfig()).prepare(raw, definitions=tools(registry), completion_review=True)
    assert not result.safe and result.reason == "context_limit"
    assert result.messages[-2:] == raw[-2:]


def test_two_review_reads_keep_acquired_middle_ranges_across_native_feedback(tmp_path):
    (tmp_path / "details.py").write_text("\n".join(f"value_{number} = {number}" for number in range(90)) + "\n")
    registry = ToolRegistry(Workspace(tmp_path, ToolConfig(max_output_chars=12000)))
    raw = [{"role": "system", "content": "policy"}, {"role": "user", "content": "Review all requested behavior."}]
    dispatch(registry, raw, "read_file", {"path": "details.py"})
    append(raw, "run_tests", {"path": "."}, {"success": True, "status": "passed", "exit_code": 0})
    lean = LeanContext(ContextConfig())
    selectors = []
    for first, last in ((30, 33), (70, 73)):
        selector = {"path": "details.py", "start_line": first, "end_line": last}
        dispatch(registry, raw, "read_file", selector)
        selectors.append(selector)
    original = deepcopy(raw)
    prepared = lean.prepare(raw, definitions=tools(registry), completion_review=True,
                            review_source_ranges=selectors)
    assert prepared.safe and prepared.reason == "completion_review"
    assert "repair" not in facts(prepared)
    view = request_evidence([{"role": "user", "content": working(prepared)}])
    assert view["source_views"]["details.py"]["visible_ranges"] == [[30, 33], [70, 73]]
    assert prepared.messages[-2:] == raw[-2:]
    assert prepared.active_messages == original == raw
    stable_prefix = working(prepared).split("\nObserved state (data):\n", 1)[0]
    dispatch(registry, raw, "read_file", selectors[-1])
    repeated = lean.prepare(raw, definitions=tools(registry), completion_review=True,
                            review_source_ranges=selectors)
    assert working(repeated).split("\nObserved state (data):\n", 1)[0] == stable_prefix


def test_unobserved_review_selector_reports_missing_without_granting_source(tmp_path):
    (tmp_path / "hidden.py").write_text("secret = 1\n")
    registry = ToolRegistry(Workspace(tmp_path))
    raw = [{"role": "system", "content": "policy"}, {"role": "user", "content": "Review the changes."}]
    append(raw, "run_tests", {"path": "."}, {"success": True, "status": "passed", "exit_code": 0})
    selector = {"path": "hidden.py", "start_line": 1, "end_line": 1}
    prepared = LeanContext(ContextConfig()).prepare(raw, definitions=tools(registry),
        completion_review=True, review_source_ranges=[selector])
    assert prepared.safe
    assert "repair" not in facts(prepared)
    assert facts(prepared)["review"]["missing_source_ranges"] == [selector]
    assert "hidden.py" not in request_evidence(prepared.messages)["presented_paths"]
    call = {"function": {"name": "edit_file", "arguments": {
        "path": "hidden.py", "old_text": "secret = 1", "new_text": "secret = 2"}}}
    assert Agent(model=object(), registry=registry)._coding_edit_error(AgentState(), call, prepared.messages)


@pytest.mark.parametrize("selectors", [
    {},
    [{"path": "value.py", "start_line": 1, "end_line": 25}],
    [{"path": "value.py", "start_line": True, "end_line": 2}],
    [{"path": "value.py", "start_line": 5, "end_line": 4}],
    [{"path": f"file_{number}.py", "start_line": 1, "end_line": 2} for number in range(5)],
])
def test_review_selectors_enforce_the_same_bounded_read_contract(selectors):
    raw = [{"role": "system", "content": "policy"}, {"role": "user", "content": "Review."}]
    result = LeanContext(ContextConfig()).prepare(raw, definitions=[], completion_review=True,
                                                 review_source_ranges=selectors)
    assert not result.safe and result.reason == "invalid_tool_history"


def test_review_selectors_cannot_enable_review_implicitly():
    raw = [{"role": "system", "content": "policy"}, {"role": "user", "content": "Work."}]
    result = LeanContext(ContextConfig()).prepare(raw, definitions=[], review_source_ranges=[])
    assert not result.safe and result.reason == "invalid_tool_history"
