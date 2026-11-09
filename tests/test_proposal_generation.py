"""Constrained proposal text cannot supply execution scope or provenance."""

from copy import deepcopy
import json

import pytest

from patchrook_agent.proposal_generation import (
    compile_generation, generation_schema, parse_generation,
)


@pytest.fixture
def approaches():
    return {"approaches": [
        {"approach": "Return subtotal - returned_total.",
         "expected": "Returning one of three 1001-cent units leaves 2002 cents."},
        {"approach": "Calculate unit_cents * (quantity - returned_quantity).",
         "expected": "Use retained units while preserving integer cents."},
    ]}


@pytest.fixture
def evidence():
    return {
        "e1": {"kind": "source", "path": "adjustments.py", "start_line": 47, "end_line": 47,
               "text": "47:     return subtotal + returned_total"},
        "e2": {"kind": "failure", "path": ".", "text": "E   assert 4004 == 2002"},
        "e3": {"kind": "source", "path": "other.py", "text": "1: unrelated = 0"},
        "e4": {"kind": "known_path", "path": "settings.py", "text": "Observed repository path"},
        "e5": {"kind": "new_file", "path": "new_helper.py", "text": "File does not exist."},
    }


def compile_packet(content, evidence, **overrides):
    return compile_generation(content, target=overrides.get("target", "adjustments.py"),
                              evidence=evidence, goal=overrides.get("goal", "Correct returned merchandise"),
                              reason=overrides.get("reason", "The reproduced return assertion failed."))


def test_schema_is_flat_closed_and_requires_only_model_judgment():
    schema = generation_schema()
    assert schema["type"] == "object" and schema["additionalProperties"] is False
    assert schema["required"] == ["approaches"]
    alternatives = schema["properties"]["approaches"]
    assert alternatives["minItems"] == alternatives["maxItems"] == 2
    item = alternatives["items"]
    assert item["type"] == "object" and item["additionalProperties"] is False
    assert set(item["properties"]) == set(item["required"]) == {"approach", "expected"}
    assert item["properties"]["approach"] == {"type": "string", "minLength": 1, "maxLength": 240}
    assert item["properties"]["expected"] == {"type": "string", "minLength": 1, "maxLength": 160}
    assert all(token not in json.dumps(schema) for token in ("oneOf", "anyOf", "evidence_refs", "start_line"))
    schema["properties"].clear()
    assert generation_schema()["properties"]  # No mutable global schema leaks.


def test_valid_generation_canonicalizes_without_changing_inputs(approaches):
    approaches["approaches"][0]["approach"] = "  Return subtotal - returned_total.  "
    original = deepcopy(approaches)
    parsed = parse_generation(json.dumps(approaches))
    assert parsed["approaches"][0]["approach"] == "Return subtotal - returned_total."
    assert approaches == original
    assert parsed == parse_generation(approaches)
    parsed["approaches"][0]["expected"] = "Changed externally"
    assert approaches == original


@pytest.mark.parametrize("content", [
    "", "```json\n{}\n```", '{"approaches":[]} trailing prose', "[]", "null", "true",
    '{"approaches":[],"approaches":[]}',
    '{"approaches":[{"approach":"a","approach":"b","expected":"x"},{"approach":"c","expected":"y"}]}',
    "[" * 1500, " " * 3001,
])
def test_non_json_duplicate_keys_and_overlarge_output_fail_closed(content):
    with pytest.raises(ValueError):
        parse_generation(content)


@pytest.mark.parametrize("content", [
    {}, {"approaches": []}, {"approaches": {}}, {"approaches": [None, None]},
    {"approaches": [{"approach": "a", "expected": "b"}]},
    {"approaches": [{"approach": "a", "expected": "b"}] * 3},
    {"approaches": [{"approach": "a", "expected": "b"}, {"approach": "c"}]},
])
def test_missing_fields_wrong_types_and_wrong_candidate_counts_fail(content):
    with pytest.raises(ValueError):
        parse_generation(content)


@pytest.mark.parametrize("extra", ["id", "kind", "path", "evidence_refs", "start_line", "end_line", "tool", "arguments"])
def test_generation_cannot_supply_paths_refs_tools_or_coordinates(approaches, extra):
    approaches["approaches"][0][extra] = "model supplied"
    with pytest.raises(ValueError, match="unsupported keys"):
        parse_generation(approaches)
    approaches["approaches"][0].pop(extra)
    approaches[extra] = "model supplied"
    with pytest.raises(ValueError, match="unsupported keys"):
        parse_generation(approaches)


@pytest.mark.parametrize("field,value", [
    ("approach", ""), ("approach", "   "), ("approach", "a" * 241),
    ("expected", ""), ("expected", "\x00"), ("expected", "e" * 161),
    ("approach", True), ("approach", {}), ("expected", None),
])
def test_text_bounds_are_enforced(approaches, field, value):
    approaches["approaches"][0][field] = value
    with pytest.raises(ValueError, match=field):
        parse_generation(approaches)


def test_prose_variants_are_not_distinct_but_operators_are(approaches):
    approaches["approaches"][1]["approach"] = " return   SUBTOTAL - returned_total! "
    with pytest.raises(ValueError, match="distinct"):
        parse_generation(approaches)
    approaches["approaches"][1]["approach"] = "Return subtotal + returned_total."
    assert len(parse_generation(approaches)["approaches"]) == 2
    approaches["approaches"][0]["approach"] = "discount_cents = merchandise_cents * percent // 100"
    approaches["approaches"][1]["approach"] = "discount_cents = merchandise_cents * percent / 100"
    assert len(parse_generation(approaches)["approaches"]) == 2


def test_compilation_binds_scope_and_provenance_to_supplied_target(approaches, evidence):
    original_content, original_evidence = deepcopy(approaches), deepcopy(evidence)
    packet = compile_packet(json.dumps(approaches), evidence)
    assert set(packet) == {"goal", "reason", "proposals"}
    assert [item["id"] for item in packet["proposals"]] == ["p1", "p2"]
    assert all(item["kind"] == "implement_change" and item["path"] == "adjustments.py"
               and item["evidence_refs"] == ["e1", "e2"] for item in packet["proposals"])
    assert all(set(item) == {"id", "kind", "path", "approach", "expected", "evidence_refs"}
               for item in packet["proposals"])
    assert approaches == original_content and evidence == original_evidence
    packet["proposals"][0]["evidence_refs"].append("e3")
    assert packet["proposals"][1]["evidence_refs"] == ["e1", "e2"]


def test_compilation_uses_actual_new_file_observation_or_source_without_failure(approaches, evidence):
    packet = compile_packet(approaches, evidence, target="new_helper.py")
    assert all(item["path"] == "new_helper.py" and item["evidence_refs"] == ["e5", "e2"]
               for item in packet["proposals"])
    evidence.pop("e2")
    assert compile_packet(approaches, evidence)["proposals"][0]["evidence_refs"] == ["e1"]


def test_compilation_stably_selects_only_real_nonempty_source_and_failure_cards(approaches, evidence):
    evidence["e1"]["text"] = " "
    evidence["e2"]["text"] = ""
    evidence["e10"] = {"kind": "failure", "path": ".", "text": "E assert 12 == 4"}
    evidence["e6"] = {"kind": "source", "path": "adjustments.py", "text": "47: return current_value"}
    evidence["e7"] = {"kind": "failure", "path": ".", "text": "E assert 4 == 2"}
    forward = compile_packet(approaches, evidence)
    backward = compile_packet(approaches, dict(reversed(list(evidence.items()))))
    assert forward == backward
    assert forward["proposals"][0]["evidence_refs"] == ["e6", "e7"]


@pytest.mark.parametrize("target", ["../secret.py", "/etc/passwd", "./adjustments.py", "a/../adjustments.py",
                                      "a\\b.py", ".", "unknown.py", "settings.py", "other.py\x00", None])
def test_unknown_metadata_only_and_noncanonical_targets_cannot_authorize_change(approaches, evidence, target):
    with pytest.raises(ValueError, match="target"):
        compile_packet(approaches, evidence, target=target)


def test_source_mutation_or_absent_current_source_cannot_use_unrelated_evidence(approaches, evidence):
    assert compile_packet(approaches, evidence)
    evidence["e1"]["path"] = "renamed.py"
    with pytest.raises(ValueError, match="actual source"):
        compile_packet(approaches, evidence)
    evidence["e1"]["path"] = "adjustments.py"
    evidence["e1"]["text"] = ""
    with pytest.raises(ValueError, match="actual source"):
        compile_packet(approaches, evidence)
    evidence["e1"].update(kind="known_path", text="Only metadata remains after invalidation")
    with pytest.raises(ValueError, match="actual source"):
        compile_packet(approaches, evidence)


@pytest.mark.parametrize("bad_evidence", [
    {}, [], {"invented_id": {"kind": "source", "path": "adjustments.py", "text": "x"}},
    {"e1": {"kind": "source", "path": "../secret.py", "text": "x"}},
    {"e1": {"kind": [], "path": "adjustments.py", "text": "x"}},
    {"e1": {"kind": "source", "path": "adjustments.py", "text": None}},
])
def test_malformed_provenance_fails_closed(approaches, bad_evidence):
    with pytest.raises(ValueError):
        compile_packet(approaches, bad_evidence)


def test_existing_compiled_packet_prose_and_output_budgets_still_apply(approaches, evidence):
    approaches["approaches"] = [
        {"approach": "a" * 240, "expected": "x" * 160},
        {"approach": "b" * 240, "expected": "y" * 160},
    ]
    assert parse_generation(approaches)
    with pytest.raises(ValueError, match="prose"):
        compile_packet(approaches, evidence, goal="g" * 180, reason="r" * 200)
    evidence["e1"]["path"] = "a" * 1024
    with pytest.raises(ValueError, match="output budget"):
        compile_packet({"approaches": [{"approach": "a", "expected": "x"},
                                               {"approach": "b", "expected": "y"}]},
                       evidence, target="a" * 1024)


def test_compile_uses_only_passed_observations_without_filesystem_reads(approaches, evidence, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Compilation must not read hidden filesystem evidence")
    monkeypatch.setattr("builtins.open", forbidden)
    assert compile_packet(approaches, evidence)["proposals"][0]["evidence_refs"] == ["e1", "e2"]


def test_v4_native_proposals_with_missing_ranges_are_not_the_new_wire_protocol():
    # Captured m3-review-v4-proposal shape. The minimal protocol does not
    # repair these calls by inventing ranges or silently stripping authority.
    captured = {
        "goal": "adjustments.py",
        "reason": "Controlled review diagnostic: compare grounded next approaches for reproduced failures",
        "proposals": [
            {"approach": "Inspect the calculate_adjusted_merchandise function to understand how returned units are handled and how the adjusted_cents is calculated.",
             "evidence_refs": ["e1", "e4"],
             "expected": "Identify the logic for handling returned units and the calculation of adjusted_cents.",
             "id": "i1", "kind": "inspect_source", "path": "adjustments.py"},
            {"approach": "Review the test cases to understand the expected behavior for returned units and fractional discounts.",
             "evidence_refs": ["e3", "e4"],
             "expected": "Understand the expected outcomes for the tests and identify discrepancies.",
             "id": "i2", "kind": "inspect_source", "path": "tests/test_checkout.py"},
        ],
    }
    with pytest.raises(ValueError, match="missing required fields: approaches"):
        parse_generation(json.dumps(captured))
    envelope = {"name": "propose_next_steps", "arguments": captured}
    with pytest.raises(ValueError):
        parse_generation(json.dumps(envelope))
