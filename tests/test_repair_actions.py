"""Factual decision cards and closed Python-owned diagnostic actions."""

from dataclasses import asdict
from dataclasses import replace
import json
import re
from pathlib import Path
import time

import pytest

from invoice_fixtures import CORRECTIONS, create_fixture
from patchrook_agent.coding import CODING_SYSTEM_PROMPT
from patchrook_agent.config import HarnessConfig
from patchrook_agent.decision_assistance import DecisionAssistance
from patchrook_agent.models.decision import DecisionResult
from patchrook_agent.repair import RepairItem, RepairPlan, build_repair_plan
from patchrook_agent.repair_actions import RepairAction, action_for_choice, build_repair_actions
from patchrook_agent.tools import ToolConfig, ToolRegistry, Workspace


def append(raw, name, arguments, result):
    identity = f"card_{len(raw)}"
    raw.extend([
        {"role": "assistant", "content": "", "tool_calls": [{"id": identity, "type": "function",
         "function": {"name": name, "arguments": arguments}}]},
        {"role": "tool", "tool_name": name, "tool_call_id": identity,
         "content": json.dumps(result)},
    ])


@pytest.fixture
def observed(tmp_path):
    task = create_fixture(tmp_path / "repo")
    registry = ToolRegistry(Workspace(tmp_path / "repo", ToolConfig(allow_tests=True, max_output_chars=12000)))
    registry.lean_mode = True
    registry.code_context = False
    registry.enable_lean_tools()
    raw = [{"role": "system", "content": CODING_SYSTEM_PROMPT},
           {"role": "user", "content": task.instructions}]
    paths = list(task.files)
    for index in range(0, len(paths), 3):
        arguments = {"paths": paths[index:index + 3]}
        append(raw, "read_code", arguments, registry.dispatch("read_code", arguments))
    append(raw, "run_tests", {"path": "."}, registry.dispatch("run_tests", {"path": "."}))
    return task, registry, raw, build_repair_plan(raw, task.instructions)


def test_cards_expose_actual_operations_assertions_and_caller_bindings(observed):
    task, _, _, plan = observed
    for path in ("adjustments.py", "promotions.py"):
        item = next(item for item in plan.items if item.path == path)
        assert CORRECTIONS[path][0].strip() in item.evidence_card
        assert "Test tests/test_checkout.py:" in item.evidence_card
        assert "assert " in item.evidence_card
        assert "Caller checkout.py:" in item.evidence_card
        assert item.symbol + "(" in item.evidence_card
        assert 3 <= len(re.findall(r"(?m)^\d+: ", item.evidence_card)) <= 8
        assert len(item.evidence_card) <= 700


def test_context_has_short_factual_cards_without_banner_or_opaque_version(observed):
    task, _, _, plan = observed
    context = plan.decision_context(task.instructions)
    assert len(context) <= 2400
    assert plan.fingerprint not in context
    assert "=== FAILURES ===" not in context
    assert "2 failed, 1 passed in" not in context
    assert "return subtotal + returned_total" in context
    assert "(merchandise_cents * discount_percent + 99) // 100" in context
    assert "assert invoice" in context
    # Candidate IDs remain necessary to bind a choice to its immutable action.
    assert all(item.id in context for item in plan.items[:4])


def test_actual_action_shortlist_retains_faults_assertions_and_call_bindings(observed):
    task, _, _, plan = observed
    item_ids = [action.item_id for action in build_repair_actions(plan)]
    for shortlist in (item_ids, [*item_ids, plan.items[3].id]):
        context = plan.decision_context(task.instructions, items=shortlist)
        assert len(context) <= 2400
        assert "return subtotal + returned_total" in context
        assert "(merchandise_cents * discount_percent + 99) // 100" in context
        assert 'assert invoice["adjusted_cents"] == 2002' in context
        assert "Caller checkout.py:29:" in context
        assert "Caller checkout.py:30:" in context


def test_card_context_whitelists_known_selected_items_and_preserves_whole_lines(observed):
    task, _, _, plan = observed
    selected = [plan.items[1].id, plan.items[0], "fabricated", plan.items[1]]
    context = plan.decision_context(task.instructions * 20, items=selected)
    assert len(context) <= 2400
    assert plan.items[0].id in context and plan.items[1].id in context
    assert plan.items[2].id not in context
    assert "fabricated" not in context
    assert context.count("Implementation (observed):") == 2


def test_cards_never_read_files_and_refresh_only_from_current_observations(observed, monkeypatch):
    task, registry, raw, original = observed
    old, new = CORRECTIONS["adjustments.py"]
    arguments = {"path": "adjustments.py", "old_text": old, "new_text": new}
    append(raw, "edit_file", arguments, registry.dispatch("edit_file", arguments))
    arguments = {"paths": ["adjustments.py"]}
    append(raw, "read_code", arguments, registry.dispatch("read_code", arguments))
    def forbidden(*args, **kwargs):
        pytest.fail("Decision cards must not read repository files")
    monkeypatch.setattr(Path, "read_text", forbidden)
    current = build_repair_plan(raw, task.instructions, revision=1)
    card = next(item.evidence_card for item in current.items if item.path == "adjustments.py")
    assert old.strip() not in card
    assert new.strip() in card
    assert original.fingerprint != current.fingerprint
    assert len(current.decision_context(task.instructions)) <= 2400


def test_manual_old_work_items_have_explicit_missing_excerpt_fallback():
    item = RepairItem("a", "calc.py", "calculate", "calc.py; calculate", (("calc.py", 1, 2),), ("test_sum",))
    plan = RepairPlan("a" * 64, "assert calculate(3) == 4", (item,))
    context = plan.decision_context("Fix")
    assert "No implementation excerpt supplied." in context
    assert "assert calculate(3) == 4" in context
    assert len(context) <= 2400


def test_decision_card_does_not_clip_a_long_source_line():
    raw = [{"role": "user", "content": "Fix"}]
    source = "def calculate(value):\n    intermediate = value + 1\n    return " + "very_long_name_" * 80
    append(raw, "read_file", {"path": "calc.py"}, {
        "success": True, "path": "calc.py", "start_line": 1, "end_line": 3,
        "content": "\n".join(f"{number}: {line}" for number, line in enumerate(source.splitlines(), 1))})
    append(raw, "run_tests", {"path": "."}, {
        "success": False, "status": "failed", "exit_code": 1,
        "stdout": "calc.py:3: error\nNameError: name was not defined", "stderr": ""})
    plan = build_repair_plan(raw, "Fix the missing name")
    assert "very_long_name_" not in plan.items[0].evidence_card
    assert "intermediate = value + 1" in plan.items[0].evidence_card


def test_actions_are_closed_stable_bounded_and_json_serializable(observed):
    _, _, _, plan = observed
    actions = build_repair_actions(plan)
    assert 2 <= len(actions) <= 6
    assert actions == build_repair_actions(plan)
    assert actions[0].kind == "prepare_patch"
    assert actions[0].id == plan.items[0].id
    assert len({action.id for action in actions}) == len(actions)
    assert all(re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,47}", action.id) for action in actions)
    assert json.loads(json.dumps([asdict(action) for action in actions])) == [asdict(action) for action in actions]
    inspections = [action for action in actions if action.kind.startswith("inspect_")]
    assert inspections
    assert len({tuple(action.arguments.items()) for action in inspections}) == len(inspections)
    for action in inspections:
        assert 1 <= action.start_line <= action.end_line
        assert action.end_line - action.start_line < 24
        assert action.path in {item.path for item in plan.items}
        assert set(action.arguments) == {"path", "start_line", "end_line"}
    assert all(action.arguments == {} for action in actions if action.kind == "prepare_patch")


def test_abstention_maps_to_a_specific_omitted_range_not_generic_advice(observed):
    _, _, _, plan = observed
    action = action_for_choice(plan, "insufficient_evidence")
    item = plan.items[0]
    assert action.id == "insufficient_evidence"
    assert action.kind == "inspect_helper"
    assert action.item_id == item.id
    assert action.path == item.path == "adjustments.py"
    represented = {number for path, first, last in item.ranges if path == item.path
                   for number in range(first, last + 1)}
    assert any(number not in represented for number in range(action.start_line, action.end_line + 1))
    assert action.arguments == {"path": "adjustments.py", "start_line": 44, "end_line": 47}


def test_unknown_choice_cannot_supply_paths_arguments_or_shell(observed):
    _, _, _, plan = observed
    for choice in ("../../secret", "read_file(path='/etc/passwd')", {"path": "/etc/passwd"}, None):
        action = action_for_choice(plan, choice)
        assert action == build_repair_actions(plan)[0]
        assert action.kind == "prepare_patch"
        assert action.path == plan.items[0].path
        assert action.arguments == {}


def test_observed_configuration_creates_a_real_bounded_config_inspection():
    primary = RepairItem("work", "loader.py", "load", "loader.py; load",
                         (("loader.py", 1, 3),), ("test_load",), evidence_range=(1, 3))
    config = RepairItem("configuration", "app_config.py", "<observed source>", "app_config.py; configuration",
                        (("app_config.py", 7, 9),), ("test_load",))
    plan = RepairPlan("b" * 64, "Configuration error", (primary, config))
    actions = build_repair_actions(plan)
    reads = [action for action in actions if action.kind == "inspect_config"]
    assert len(reads) == 1
    assert reads[0].arguments == {"path": "app_config.py", "start_line": 7, "end_line": 9}
    assert all(action.path != "unobserved_config.py" for action in actions)


def test_single_item_legacy_and_empty_plans_remain_safe():
    item = RepairItem("work", "calc.py", "add", "calc.py; add", (("calc.py", 4, 5),), ("test_sum",))
    plan = RepairPlan("c" * 64, "assertion failed", (item,))
    actions = build_repair_actions(plan)
    assert len(actions) == 2
    assert actions[0].kind == "prepare_patch"
    assert action_for_choice(plan, "insufficient_evidence").arguments == {
        "path": "calc.py", "start_line": 4, "end_line": 5}
    empty = RepairPlan("d" * 64, "", ())
    assert build_repair_actions(empty) == ()
    assert action_for_choice(empty, "arbitrary") is None


def test_disabled_inspection_actions_cannot_be_offered_or_resolved():
    plan = RepairPlan("e" * 64, "assertion failed", (
        RepairItem("first", "calc.py", "add", "calc.py; add", (("calc.py", 1, 2),), ("test_add",)),
        RepairItem("second", "app_config.py", "<observed source>", "app_config.py; config",
                   (("app_config.py", 1, 2),), ("test_config",)),
    ))
    enabled = build_repair_actions(plan)
    disabled = build_repair_actions(plan, allow_inspection=False)
    assert len(disabled) == 2 and all(action.kind == "prepare_patch" for action in disabled)
    for action in enabled:
        if action.kind.startswith("inspect_"):
            resolved = action_for_choice(plan, action.id, allow_inspection=False)
            assert resolved == disabled[0]
            assert resolved.arguments == {}
    with pytest.raises(ValueError, match="boolean"):
        build_repair_actions(plan, allow_inspection="false")


def test_disabled_inspection_filters_cached_choices_without_another_decision(tmp_path):
    plan = RepairPlan("f" * 64, "assertion failed", (
        RepairItem("first", "calc.py", "add", "calc.py; add", (("calc.py", 1, 2),), ("test_add",)),
        RepairItem("second", "rounding.py", "floor", "rounding.py; floor",
                   (("rounding.py", 1, 2),), ("test_floor",)),
    ))
    class Client:
        def __init__(self):
            self.calls = []
        def rank(self, context, candidates, **options):
            self.calls.append(tuple(candidate.id for candidate in candidates))
            order = ("insufficient_evidence", *(candidate.id for candidate in candidates
                                                if candidate.id != "insufficient_evidence"))
            return DecisionResult(order, {}, called=False)
    client = Client()
    config = replace(HarnessConfig.preset("C"), workflow_mode="coding", decision_model="on")
    controller = DecisionAssistance(config, object(), Workspace(tmp_path), client=client)
    assert controller.select_repair("Fix", plan, time.monotonic() + 60) == "insufficient_evidence"
    assert len(client.calls) == 1
    assert controller.select_repair("Fix", plan, time.monotonic() + 60, allow_inspection=False) == "first"
    assert len(client.calls) == 1
    restored = DecisionAssistance(config, object(), Workspace(tmp_path), client=client)
    restored.restore(controller.to_dict())
    assert restored.select_repair("Fix", plan, time.monotonic() + 60, allow_inspection=False) == "first"
    assert len(client.calls) == 1
    fresh = DecisionAssistance(config, object(), Workspace(tmp_path), client=client)
    assert fresh.select_repair("Fix", plan, time.monotonic() + 60, allow_inspection=False) == "first"
    assert client.calls[-1] == ("first", "second")


@pytest.mark.parametrize("changes", [
    {"id": "shell;cat"}, {"item_id": "../secret"}, {"path": "../secret"},
    {"path": "/etc/passwd"}, {"start_line": 0}, {"end_line": 100},
    {"start_line": True}, {"kind": "execute_shell"},
])
def test_invalid_action_manifests_are_rejected(changes):
    values = {"id": "read_1", "kind": "inspect_helper", "item_id": "work", "path": "calc.py",
              "start_line": 2, "end_line": 5, "description": "Read observed implementation"}
    with pytest.raises(ValueError):
        RepairAction(**{**values, **changes})

