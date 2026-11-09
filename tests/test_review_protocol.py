"""Proposal review is bounded, grounded in supplied facts, and fails closed."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import time

import pytest

from patchrook_agent.config import HarnessConfig
from patchrook_agent.decision_assistance import DecisionAssistance
from patchrook_agent.models.decision import DecisionResult
from patchrook_agent.review_protocol import (
    MAX_REVIEW_CALLS, evidence_paths, parse_proposals, proposal_definition, proposal_digest,
    proposal_description, review_context,
)
from patchrook_agent.tools import Workspace
from patchrook_agent.task_state import TaskState
from invoice_fixtures import create_fixture


@pytest.fixture
def evidence():
    return {
        "e1": {"kind": "source", "path": "calc.py", "start_line": 1, "end_line": 3,
               "text": "1: def add(a, b):\n2:     total = a - b\n3:     return total"},
        "e2": {"kind": "failure", "path": ".",
               "text": "test_calc.py:4: in test_add\n    assert add(3, 2) == 5\nE   assert 1 == 5\n1 failed in 0.01s"},
        "e3": {"kind": "known_path", "path": "settings.py", "text": "Known observed repository path"},
        "e4": {"kind": "new_file", "path": "new_helper.py", "text": "The file does not exist."},
    }


@pytest.fixture
def packet():
    return {"goal": "Restore integer addition", "reason": "The last approach failed the addition assertion.",
            "proposals": [
                {"id": "p1", "kind": "implement_change", "path": "calc.py",
                 "approach": "Replace subtraction with integer addition.",
                 "expected": "add(3, 2) returns 5.", "evidence_refs": ["e1", "e2"]},
                {"id": "p2", "kind": "inspect_source", "path": "calc.py",
                 "start_line": 1, "end_line": 3,
                 "approach": "Read the complete helper before selecting another operation.",
                 "expected": "Confirm whether a missing guard affects the result.", "evidence_refs": ["e1"]},
            ]}


def test_definition_is_virtual_closed_and_bounded_to_known_paths_and_evidence(evidence):
    original = deepcopy(evidence)
    definition = proposal_definition(evidence, allowed_paths=evidence_paths(evidence))
    assert definition["function"]["name"] == "propose_next_steps"
    schema = definition["function"]["parameters"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"goal", "reason", "proposals"}
    assert schema["properties"]["proposals"]["minItems"] == 2
    assert schema["properties"]["proposals"]["maxItems"] == 3
    item = schema["properties"]["proposals"]["items"]
    assert item["type"] == "object" and item["additionalProperties"] is False
    assert set(item["required"]) == {"id", "kind", "path", "approach", "expected", "evidence_refs"}
    assert item["properties"]["path"]["enum"] == ["calc.py", "settings.py", "new_helper.py"]
    assert item["properties"]["evidence_refs"]["items"]["enum"] == list(evidence)
    assert item["properties"]["kind"]["enum"] == ["implement_change", "inspect_source"]
    assert "start_line" not in item["required"] and "end_line" not in item["required"]
    assert "oneOf" not in json.dumps(definition) and "anyOf" not in json.dumps(definition)
    assert evidence == original


def test_valid_packet_is_canonical_and_does_not_mutate_arguments(packet, evidence):
    packet["goal"] = " Restore integer addition "
    original = deepcopy(packet)
    parsed = parse_proposals(json.dumps(packet), evidence, evidence_paths(evidence))
    assert parsed["goal"] == "Restore integer addition"
    assert packet == original
    assert set(parsed) == {"goal", "reason", "proposals"}
    assert parsed["proposals"][0]["id"] == "p1"
    assert all(len(proposal_description(proposal)) <= 240 for proposal in parsed["proposals"])


@pytest.mark.parametrize("change", [
    {"tool": "execute_shell"}, {"kind": "run_tests"}, {"path": "../secret.py"},
    {"path": "/etc/passwd"}, {"path": "./calc.py"}, {"path": "unknown.py"},
    {"id": "none"}, {"id": "NONE"}, {"id": "bad;id"}, {"id": "p" * 33},
    {"approach": ""}, {"approach": "x" * 241}, {"expected": "x" * 161},
    {"evidence_refs": []}, {"evidence_refs": ["e99"]},
    {"evidence_refs": ["e1", "e1"]}, {"evidence_refs": ["e1", "e2", "e3", "e4"]},
    {"evidence_refs": ["e2"]}, {"start_line": 1},
])
def test_invalid_implementation_proposals_are_rejected(packet, evidence, change):
    packet["proposals"][0].update(change)
    with pytest.raises(ValueError):
        parse_proposals(packet, evidence, evidence_paths(evidence))


def test_known_path_is_enough_for_inspection_but_does_not_authorize_implementation(packet, evidence):
    packet["proposals"][1].update(path="settings.py", evidence_refs=["e3"])
    assert parse_proposals(packet, evidence, evidence_paths(evidence))["proposals"][1]["path"] == "settings.py"
    packet["proposals"][0].update(path="settings.py", evidence_refs=["e3"])
    with pytest.raises(ValueError, match="actual source"):
        parse_proposals(packet, evidence, evidence_paths(evidence))


def test_explicit_new_file_observation_can_ground_an_implementation(packet, evidence):
    packet["proposals"][0].update(path="new_helper.py", evidence_refs=["e4"])
    parsed = parse_proposals(packet, evidence, evidence_paths(evidence))
    assert parsed["proposals"][0]["path"] == "new_helper.py"


def test_empty_metadata_cannot_masquerade_as_observed_source(packet, evidence):
    evidence["e1"]["text"] = ""
    with pytest.raises(ValueError, match="actual source"):
        parse_proposals(packet, evidence, evidence_paths(evidence))


def test_inspection_limit_is_24_lines_and_disabled_tools_are_not_proposed(packet, evidence):
    packet["proposals"][1].update(start_line=10, end_line=33)
    assert parse_proposals(packet, evidence, evidence_paths(evidence))
    packet["proposals"][1]["end_line"] = 34
    with pytest.raises(ValueError, match="24"):
        parse_proposals(packet, evidence, evidence_paths(evidence))
    packet["proposals"][1].update(start_line=True, end_line=3)
    with pytest.raises(ValueError):
        parse_proposals(packet, evidence, evidence_paths(evidence))
    packet["proposals"][1].update(start_line=1, end_line=3)
    with pytest.raises(ValueError, match="disabled"):
        parse_proposals(packet, evidence, evidence_paths(evidence), allow_inspection=False)
    schema = proposal_definition(evidence, allowed_paths=evidence_paths(evidence), allow_inspection=False)
    item = schema["function"]["parameters"]["properties"]["proposals"]["items"]
    assert item["properties"]["kind"]["enum"] == ["implement_change"]
    assert "start_line" not in item["properties"] and "end_line" not in item["properties"]


def test_duplicate_alternatives_and_duplicate_json_keys_are_rejected(packet, evidence):
    duplicate = {**packet["proposals"][0], "id": "p2",
                 "approach": "replace subtraction with INTEGER addition!"}
    packet["proposals"][1] = duplicate
    with pytest.raises(ValueError, match="distinct"):
        parse_proposals(packet, evidence, evidence_paths(evidence))
    text = '{"goal":"first","goal":"second","reason":"r","proposals":[]}'
    with pytest.raises(ValueError, match="valid JSON"):
        parse_proposals(text, evidence, evidence_paths(evidence))


def test_captured_review_failure_reports_both_missing_refs_and_43_line_read():
    # Reproduces the proposal arguments in m3-review-tev1.json without making
    # this regression depend on retained reports or any model inference.
    arguments = {
        "reason": "Controlled review diagnostic: compare grounded next approaches for reproduced failures",
        "goal": "adjustments.py",
        "proposals": [
            {"approach": "Investigate the calculate_adjusted_merchandise function to identify why returned units are not reducing merchandise before percentage offers.",
             "end_line": 44,
             "expected": "Understand the logic for calculating adjusted merchandise and identify the root cause of the failed test.",
             "id": "i1", "kind": "inspect_source", "path": "adjustments.py", "start_line": 2},
            {"approach": "Analyze the failing tests to understand the expected behavior for returned units and fractional discounts.",
             "end_line": 13,
             "expected": "Identify the exact calculation expected by the tests and how the current implementation deviates from it.",
             "id": "i2", "kind": "inspect_source", "path": "tests/test_checkout.py", "start_line": 3},
        ],
    }
    evidence = {
        "e1": {"kind": "source", "path": "adjustments.py", "text": "47:     return subtotal + returned_total"},
        "e2": {"kind": "source", "path": "tests/test_checkout.py", "text": '5:     assert invoice["adjusted_cents"] == 2002'},
    }
    original = deepcopy(arguments)
    with pytest.raises(ValueError) as rejected:
        parse_proposals(arguments, evidence, evidence_paths(evidence))
    message = str(rejected.value)
    assert "proposals[0] (i1): missing required fields: 'evidence_refs'" in message
    assert "proposals[1] (i2): missing required fields: 'evidence_refs'" in message
    assert "range 2-44 spans 43 lines; maximum is 24 lines" in message
    assert len(message) <= 1400
    assert arguments == original  # The harness must never invent citations.
    arguments["proposals"][0].update(evidence_refs=["e1"], start_line=43, end_line=47)
    arguments["proposals"][1]["evidence_refs"] = ["e2"]
    accepted = parse_proposals(arguments, evidence, evidence_paths(evidence))
    assert [proposal["id"] for proposal in accepted["proposals"]] == ["i1", "i2"]
    assert accepted["proposals"][0]["start_line"] == 43


def test_structural_errors_name_unsupported_keys_kind_and_missing_wrapper_fields(packet, evidence):
    packet.pop("reason")
    packet["shell"] = "arbitrary"
    packet["proposals"][0].update(kind="run_tests", arguments={"path": "."})
    packet["proposals"][0].pop("evidence_refs")
    with pytest.raises(ValueError) as rejected:
        parse_proposals(packet, evidence, evidence_paths(evidence))
    message = str(rejected.value)
    assert "packet: missing required fields: 'reason'" in message
    assert "packet: unsupported keys: 'shell'" in message
    assert "proposals[0] (p1): unsupported keys: 'arguments'" in message
    assert "unsupported kind 'run_tests'; use inspect_source or implement_change" in message
    assert "missing required fields: 'evidence_refs'" in message


@pytest.mark.parametrize("change", [{"kind": ["inspect_source"]}, {"kind": {"tool": "shell"}},
                                    {"path": {"file": "calc.py"}}])
def test_malformed_native_field_types_are_actionable_value_errors(packet, evidence, change):
    packet["proposals"][1].update(change)
    with pytest.raises(ValueError, match="proposals\\[1\\]"):
        parse_proposals(packet, evidence, evidence_paths(evidence))


def test_operator_alternatives_are_not_collapsed_by_prose_normalization(packet, evidence):
    packet["proposals"] = [
        {**packet["proposals"][0], "id": "p1", "approach": "Use a + b."},
        {**packet["proposals"][0], "id": "p2", "approach": "Use a - b."},
    ]
    parsed = parse_proposals(packet, evidence, evidence_paths(evidence))
    assert len(parsed["proposals"]) == 2


def test_model_must_propose_two_but_filtered_single_candidate_can_be_reviewed(packet, evidence):
    packet["proposals"] = packet["proposals"][:1]
    with pytest.raises(ValueError, match="2 to three"):
        parse_proposals(packet, evidence, evidence_paths(evidence))
    assert parse_proposals(packet, evidence, evidence_paths(evidence), min_proposals=1)
    assert "p1" in review_context("Fix", packet, evidence, [])


def test_review_context_balances_proposals_actual_code_assertion_and_failed_history(packet, evidence):
    canonical = parse_proposals(packet, evidence, evidence_paths(evidence))
    history = [{"source_digest": "a" * 64, "proposal": {"approach": "Multiply the inputs."},
                "failure": "assert 6 == 5", "duration_seconds": 1.0}]
    context = review_context("Fix integer addition", canonical, evidence, history)
    assert len(context) <= 2400
    assert "total = a - b" in context
    assert "assert add(3, 2) == 5" in context
    assert "Replace subtraction with integer addition." in context
    assert "Read the complete helper" in context
    assert "Multiply the inputs." in context
    assert "a" * 64 not in context


def test_task_ledger_history_preserves_both_failed_formulas_and_counterexamples(tmp_path, packet, evidence):
    original = "discount_cents = (merchandise_cents * discount_percent + 99) // 100"
    minus_one = original + " - 1"
    ledger = TaskState(active_goal="Repair percentage discount rounding")
    def failure(amounts):
        return {"success": False, "status": "failed", "exit_code": 1,
                "stdout": "E   assert " + amounts + "\nFAILED tests/test_checkout.py::test_invoice",
                "stderr": "", "test_counts": {"failed": 1, "passed": 2}}
    ledger.observe_action("run_tests", {}, failure("4951 == 4952"), step=0, source_digest="A")
    ledger.observe_action("edit_file", {"path": "promotions.py", "old_text": original, "new_text": minus_one},
                          {"success": True, "modified": True, "path": "promotions.py"}, step=1)
    ledger.observe_action("run_tests", {}, failure("6601 == 6600"), step=2, source_digest="B")
    ledger.observe_action("edit_file", {"path": "promotions.py", "old_text": minus_one, "new_text": original},
                          {"success": True, "modified": True, "path": "promotions.py"}, step=3)
    ledger.observe_action("run_tests", {}, failure("4951 == 4952"), step=4, source_digest="A")
    assert len(ledger.failed_states) == 2
    evidence["e1"] = {"kind": "source", "path": "promotions.py", "start_line": 40, "end_line": 40,
                       "text": "40:     " + original}
    evidence["e2"]["text"] = "assert merchandise == 4952\nE   assert 4951 == 4952"
    for proposal in packet["proposals"]:
        proposal["path"] = "promotions.py"
    packet["proposals"][0]["approach"] = "Use integer floor division for the computed discount."
    packet["proposals"][0]["expected"] = "Fractional discounts retain the remaining cent."
    packet["proposals"][1]["approach"] = "Inspect the offer cap and early return before changing arithmetic."
    history = {"failed_states": ledger.failed_states[-4:]}
    task = create_fixture(tmp_path / "invoice").instructions
    context = review_context(task, packet, evidence, history)
    assert len(context) <= 2400
    assert "percentage discounts must round down" in context.split("\nGoal (proposal):", 1)[0]
    assert all(proposal["approach"] in context for proposal in packet["proposals"])
    assert original + "\nFailed:" in context
    assert minus_one + "\nFailed:" in context
    assert "6601 == 6600" in context and "4951 == 4952" in context
    assert "source_digest" not in context


def test_three_proposal_context_has_complete_lines_from_both_helpers_and_failure(packet, evidence):
    evidence["e5"] = {"kind": "source", "path": "rounding.py", "start_line": 1, "end_line": 2,
                      "text": "1: def amount(m, p):\n2:     return (m * p + 99) // 100"}
    packet["proposals"].append({"id": "p3", "kind": "implement_change", "path": "rounding.py",
                                "approach": "Compute the percentage discount using integer floor division.",
                                "expected": "Fractional cents are retained in merchandise.", "evidence_refs": ["e5", "e2"]})
    canonical = parse_proposals(packet, evidence, evidence_paths(evidence))
    context = review_context("Fix both invoice helpers", canonical, evidence, [])
    assert len(context) <= 2400
    assert "total = a - b" in context
    assert "return (m * p + 99) // 100" in context
    assert "assert add(3, 2) == 5" in context
    assert all(proposal["approach"] in context for proposal in canonical["proposals"])


def test_evidence_validation_never_uses_hidden_files(packet, evidence, monkeypatch):
    monkeypatch.setattr(Path, "read_text", lambda *args, **kwargs: pytest.fail("No hidden source reads"))
    assert parse_proposals(packet, evidence, evidence_paths(evidence))
    assert review_context("Fix", packet, evidence, [])


def test_verification_evidence_is_supported_without_becoming_an_allowed_dot_path(packet, evidence):
    evidence["e2"] = {"kind": "verification", "path": ".", "text": "3 passed in 0.02s"}
    assert "." not in evidence_paths(evidence)
    assert parse_proposals(packet, evidence, evidence_paths(evidence))
    assert "3 passed" in review_context("Complete the remaining requirement", packet, evidence, [])


class Client:
    model = "tev1:4b"

    def __init__(self, selected="p1", *, result=None):
        self.selected = selected
        self.result = result
        self.calls = []

    def rank(self, context, candidates, **options):
        self.calls.append((context, deepcopy(candidates), options))
        if self.result is not None:
            return self.result
        ids = [candidate.id for candidate in candidates]
        return DecisionResult((self.selected, *(value for value in ids if value != self.selected)), {}, called=True)


def controller(tmp_path, client, *, records=None, events=None):
    config = replace(HarnessConfig.preset("C"), workflow_mode="coding", decision_model="on")
    return DecisionAssistance(config, object(), Workspace(tmp_path), client=client,
                              record=lambda purpose, result: records.append((purpose, result)) if records is not None else None,
                              emit=lambda event, **values: events.append((event, values)) if events is not None else None)


@pytest.fixture(autouse=True)
def no_http(monkeypatch):
    monkeypatch.setattr("httpx.Client", lambda *args, **kwargs: pytest.fail("No model calls in proposal tests"))


def test_review_selects_known_id_with_explicit_none_and_its_own_call_budget(tmp_path, packet, evidence):
    client = Client()
    records, events = [], []
    review = controller(tmp_path, client, records=records, events=events)
    review.repair_calls = 2
    assert review.select_proposal("Fix", packet, evidence, [], time.monotonic() + 60) == "p1"
    assert review.review_calls == 1 and review.repair_calls == 2
    assert {candidate.id for candidate in client.calls[0][1]} == {"p1", "p2", "none"}
    assert client.calls[0][2]["purpose"] == "proposal_review"
    assert records[0][0] == "proposal_review"
    assert events[-1][1]["selected"] == "p1"


def test_legitimate_none_is_recorded_cached_and_restored(tmp_path, packet, evidence):
    client = Client("none")
    records, events = [], []
    review = controller(tmp_path, client, records=records, events=events)
    assert review.select_proposal("Fix", packet, evidence, [], time.monotonic() + 60) == "none"
    assert records[0][1].called is True
    assert events[-1][1]["reason"] == "none_selected"
    saved = review.to_dict()
    restored = controller(tmp_path, client, events=events)
    restored.restore(saved)
    assert restored.select_proposal("Fix", packet, evidence, [], time.monotonic() + 60) == "none"
    assert len(client.calls) == 1
    assert events[-1][1]["cached"] is True


@pytest.mark.parametrize("ranking", [("unknown", "p1", "p2"), ("p1",), ("p1", "p1", "none"), ()])
def test_malformed_review_rankings_cannot_approve_a_candidate(tmp_path, packet, evidence, ranking):
    client = Client(result=DecisionResult(ranking, {}, called=True))
    events = []
    review = controller(tmp_path, client, events=events)
    assert review.select_proposal("Fix", packet, evidence, [], time.monotonic() + 60) == "none"
    assert events[-1][1]["reason"] == "invalid_proposal_ranking"
    assert review.review_choices == {}


def test_fallback_or_unavailable_review_is_not_silent_approval(tmp_path, packet, evidence):
    client = Client(result=DecisionResult(("p1", "p2", "none"), {}, "timeout", called=True))
    events = []
    review = controller(tmp_path, client, events=events)
    assert review.select_proposal("Fix", packet, evidence, [], time.monotonic() + 60) == "none"
    assert review.unavailable
    assert review.review_choices == {}
    assert review.select_proposal("Fix", packet, evidence, [], time.monotonic() + 60) == "none"
    assert len(client.calls) == 1
    assert events[-1][1]["reason"] == "review_unavailable"


def test_malformed_optional_client_result_cannot_approve_a_candidate(tmp_path, packet, evidence):
    class Malformed:
        def rank(self, *args, **kwargs):
            return {"ranking": ["p1", "p2", "none"]}
    events = []
    review = controller(tmp_path, Malformed(), events=events)
    assert review.select_proposal("Fix", packet, evidence, [], time.monotonic() + 60) == "none"
    assert review.unavailable
    assert events[-1][1]["reason"] == "invalid_proposal_result"


def test_invalid_packet_or_exhausted_deadline_does_not_query_decider(tmp_path, packet, evidence):
    client = Client()
    review = controller(tmp_path, client)
    invalid = {**packet, "shell": "arbitrary"}
    assert review.select_proposal("Fix", invalid, evidence, [], time.monotonic() + 60) == "none"
    assert review.select_proposal("Fix", packet, evidence, [], time.monotonic() - 1) == "none"
    assert client.calls == [] and review.review_calls == 0


def test_review_memo_ignores_measurement_noise_but_invalidates_changed_source(tmp_path, packet, evidence):
    client = Client()
    review = controller(tmp_path, client)
    history = [{"approach": "Multiply", "failure": "assert 6 == 5", "duration_seconds": 1.0}]
    assert review.select_proposal("Fix", packet, evidence, history, time.monotonic() + 60) == "p1"
    evidence["e2"]["text"] = evidence["e2"]["text"].replace("0.01s", "0.99s")
    history[0]["duration_seconds"] = 9000
    assert review.select_proposal("Fix", packet, evidence, history, time.monotonic() + 60) == "p1"
    assert len(client.calls) == 1
    evidence["e1"]["text"] = evidence["e1"]["text"].replace("a - b", "a * b")
    assert review.select_proposal("Fix", packet, evidence, history, time.monotonic() + 60) == "p1"
    assert len(client.calls) == 2


def test_review_budget_is_eight_and_old_checkpoints_restore_defaults(tmp_path, packet, evidence):
    client = Client()
    review = controller(tmp_path, client)
    legacy = review.to_dict()
    legacy.pop("review_calls")
    legacy.pop("review_choices")
    review.restore(legacy)
    assert review.review_calls == 0 and review.review_choices == {}
    for index in range(MAX_REVIEW_CALLS):
        packet["reason"] = f"Unresolved attempt {index}"
        assert review.select_proposal("Fix", packet, evidence, [], time.monotonic() + 60) == "p1"
    packet["reason"] = "Another unresolved attempt"
    assert review.select_proposal("Fix", packet, evidence, [], time.monotonic() + 60) == "none"
    assert len(client.calls) == MAX_REVIEW_CALLS


def test_single_filtered_candidate_still_requires_review_against_none(tmp_path, packet, evidence):
    packet["proposals"] = packet["proposals"][:1]
    client = Client("none")
    review = controller(tmp_path, client)
    assert review.select_proposal("Fix", packet, evidence, [], time.monotonic() + 60) == "none"
    assert {candidate.id for candidate in client.calls[0][1]} == {"p1", "none"}


@pytest.mark.parametrize("changes", [
    {"review_calls": True}, {"review_calls": 9}, {"review_choices": {"bad": "p1"}},
    {"review_choices": {"a" * 64: "unknown;unsafe"}},
])
def test_invalid_review_checkpoints_are_rejected(tmp_path, changes):
    review = controller(tmp_path, Client())
    with pytest.raises(ValueError, match="proposal review"):
        review.restore({**review.to_dict(), **changes})

