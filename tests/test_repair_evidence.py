"""Repair routing uses versioned observations, never repository answer keys."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from patchrook_agent.repair import build_repair_plan


def append(raw, name, arguments, result):
    identity = f"repair_{len(raw)}"
    raw.extend([
        {"role": "assistant", "content": "", "tool_calls": [{"id": identity, "type": "function",
         "function": {"name": name, "arguments": arguments}}]},
        {"role": "tool", "tool_name": name, "tool_call_id": identity,
         "content": json.dumps(result)},
    ])


def read(raw, path, source, *, start=1, **metadata):
    lines = source.splitlines()
    result = {"success": True, "path": path, "start_line": start,
              "end_line": start + len(lines) - 1,
              "content": "\n".join(f"{number}: {line}" for number, line in enumerate(lines, start)),
              **metadata}
    append(raw, "read_file", {"path": path, "start_line": start, "end_line": result["end_line"]}, result)


def fail(raw, *, path=".", elapsed="0.01", prefix=""):
    result = {"success": False, "status": "failed", "exit_code": 1,
              "stdout": f"{prefix}tests/test_price.py:4: in test_total\n"
                        "    assert settle(10, 2) == 8\nE   assert 12 == 8\n"
                        "FAILED tests/test_price.py::test_total\n"
                        f"1 failed in {elapsed}s", "stderr": "", "duration_seconds": float(elapsed),
              "test_counts": {"failed": 1}}
    append(raw, "run_tests", {"path": path}, result)


@pytest.fixture
def observed():
    raw = [{"role": "system", "content": "Coding"},
           {"role": "user", "content": "Reproduce failures and fix the price adjustment."}]
    read(raw, "adjustment.py", "def unrelated(value):\n    return value\n\ndef settle(amount, refund):\n    return amount + refund")
    read(raw, "tests/test_price.py", "from adjustment import settle\n\ndef test_total():\n    assert settle(10, 2) == 8")
    fail(raw)
    return raw


def test_failed_test_localizes_called_function_without_first_function_assumption(observed):
    plan = build_repair_plan(observed, "Fix the observed failure")
    assert plan is not None
    assert [(item.path, item.symbol) for item in plan.items] == [("adjustment.py", "settle")]
    item = plan.items[0]
    assert item.evidence_range == (4, 5)
    assert item.failure_ids == ("tests/test_price.py::test_total",)
    assert ("adjustment.py", 4, 5) in item.ranges
    assert ("tests/test_price.py", 3, 4) in item.ranges
    assert "12 == 8" in plan.failure_excerpt
    assert "unrelated" not in item.description


def test_packet_is_json_safe_and_never_invents_unobserved_lines(observed):
    plan = build_repair_plan(observed, "Fix")
    packet = plan.packet()
    assert json.loads(json.dumps(packet)) == packet
    assert set(packet) == {"id", "fingerprint", "source_ranges", "failure_ids", "narrow"}
    assert packet["narrow"] is False
    assert plan.packet(narrow=True)["narrow"] is True
    assert all(record["path"] in {"adjustment.py", "tests/test_price.py"} for record in packet["source_ranges"])
    with pytest.raises(ValueError, match="Unknown"):
        plan.packet("fabricated")
    with pytest.raises(ValueError, match="boolean"):
        plan.packet(narrow="yes")


def test_repeated_overlapping_reads_and_measurement_noise_do_not_reroute(observed):
    original = deepcopy(observed)
    first = build_repair_plan(observed, "Fix")
    read(observed, "adjustment.py", "    return amount + refund", start=5)
    fail(observed, elapsed="9.81")
    second = build_repair_plan(observed, "Fix")
    assert first.fingerprint == second.fingerprint
    assert first.items == second.items
    assert build_repair_plan(observed, "Fix", revision=1).fingerprint != first.fingerprint
    assert original[:2] == observed[:2]


def test_mutation_requires_fresh_observation_before_implementation_returns(observed):
    first = build_repair_plan(observed, "Fix")
    append(observed, "edit_file", {"path": "adjustment.py"},
           {"success": True, "modified": True, "path": "adjustment.py"})
    assert build_repair_plan(observed, "Fix") is None
    read(observed, "adjustment.py", "def settle(amount, refund):\n    return amount - refund", start=4)
    current = build_repair_plan(observed, "Fix")
    assert current.fingerprint != first.fingerprint
    assert current.items[0].evidence_range == (4, 5)


def test_conflicting_reread_does_not_join_unknown_stale_neighbors(observed):
    first = build_repair_plan(observed, "Fix")
    read(observed, "adjustment.py", "    return amount - refund", start=5)
    second = build_repair_plan(observed, "Fix")
    assert second.fingerprint != first.fingerprint
    item = next(item for item in second.items if item.path == "adjustment.py")
    assert item.evidence_range is None
    assert all(start == end == 5 for path, start, end in item.ranges if path == "adjustment.py")


def test_external_source_boundaries_hide_old_observations(observed):
    assert build_repair_plan(observed, "Fix", stale_before={"adjustment.py": len(observed)}) is None
    boundary = len(observed)
    read(observed, "adjustment.py", "def settle(amount, refund):\n    return amount - refund", start=4)
    assert build_repair_plan(observed, "Fix", stale_before={"adjustment.py": boundary}).items[0].symbol == "settle"


def test_partial_line_is_not_exact_repair_source(observed):
    read(observed, "fragment.py", "def broken(value):\n    return value +", partial_line=True)
    plan = build_repair_plan(observed, "Fix")
    fragment = next(item for item in plan.items if item.path == "fragment.py")
    assert all(end == 1 for path, _, end in fragment.ranges if path == "fragment.py")


def test_gapped_python_and_non_python_stay_observed_only_evidence_choices(observed):
    read(observed, "other.py", "def other(value):", start=3)
    read(observed, "other.py", "    return value", start=8)
    read(observed, "policy.sql", "SELECT price + adjustment FROM prices;", start=17)
    plan = build_repair_plan(observed, "Fix")
    other = next(item for item in plan.items if item.path == "other.py")
    assert other.evidence_range is None
    assert {(start, end) for path, start, end in other.ranges if path == "other.py"} == {(3, 3), (8, 8)}
    sql = next(item for item in plan.items if item.path == "policy.sql")
    assert ("policy.sql", 17, 17) in sql.ranges


def test_unrelated_green_tests_do_not_erase_observed_failure(observed):
    append(observed, "run_tests", {"path": "tests/test_elsewhere.py"},
           {"success": True, "status": "passed", "exit_code": 0, "stdout": "1 passed", "stderr": ""})
    assert build_repair_plan(observed, "Fix") is not None
    append(observed, "run_tests", {"path": "."},
           {"success": True, "status": "passed", "exit_code": 0, "stdout": "1 passed", "stderr": ""})
    assert build_repair_plan(observed, "Fix") is None


def test_ambiguous_function_names_are_not_treated_as_unique_call_edges(observed):
    read(observed, "collision.py", "def settle(amount, refund):\n    return amount * refund")
    plan = build_repair_plan(observed, "Fix")
    assert {item.path for item in plan.items} >= {"adjustment.py", "collision.py"}


def test_source_tail_and_call_sites_survive_bounded_packet_selection():
    raw = [{"role": "user", "content": "Fix"}]
    body = "def dispatch(value):\n" + "\n".join(f"    check_{i} = value" for i in range(50))
    body += "\n    adjusted = settle(value, 2)\n    return adjusted"
    read(raw, "dispatcher.py", body)
    read(raw, "adjustment.py", "def settle(amount, refund):\n    return amount + refund")
    read(raw, "tests/test_price.py", "from dispatcher import dispatch\n\ndef test_total():\n    assert dispatch(10) == 8")
    fail(raw)
    plan = build_repair_plan(raw, "Fix")
    item = next(item for item in plan.items if item.symbol == "dispatch")
    assert sum(end - start + 1 for _, start, end in item.ranges) <= 48
    assert any(path == "dispatcher.py" and start <= 52 <= end for path, start, end in item.ranges)
    assert any(path == "dispatcher.py" and start <= 53 <= end for path, start, end in item.ranges)


def test_plan_is_bounded_and_does_not_read_files(observed, monkeypatch):
    for index in range(20):
        read(observed, f"extra_{index}.py", f"def extra_{index}(value):\n    return value")
    def forbidden(*args, **kwargs):
        pytest.fail("Repair planning must not consult hidden source")
    monkeypatch.setattr(Path, "read_text", forbidden)
    plan = build_repair_plan(observed, "Fix")
    assert 1 <= len(plan.items) <= 8
    assert len(plan.failure_excerpt) <= 1400
    assert all(len(item.description) <= 240 for item in plan.items)
    assert all(sum(end - start + 1 for _, start, end in item.ranges) <= 48 for item in plan.items)


def test_invalid_or_incomplete_observations_do_not_create_a_plan(observed):
    assert build_repair_plan(observed[:-1], "Fix") is None
    with pytest.raises(ValueError, match="inputs"):
        build_repair_plan(observed, "Fix", revision=True)


def test_decision_fingerprint_normalizes_only_ephemeral_fixture_prefix(observed):
    first = deepcopy(observed)
    second = deepcopy(observed)
    fail(first, prefix="/tmp/patchrook-m3-repair-abc123/", elapsed="0.21")
    fail(second, prefix="/tmp/patchrook-m3-repair-xyz789/", elapsed="0.89")
    a, b = build_repair_plan(first, "Fix"), build_repair_plan(second, "Fix")
    assert a.fingerprint == b.fingerprint
    assert "12 == 8" in a.failure_excerpt
    assert "<workspace>/tests/test_price.py" in a.failure_excerpt


def test_header_only_read_expansion_keeps_existing_completed_function(observed):
    first = build_repair_plan(observed, "Fix")
    read(observed, "adjustment.py", "\ndef next_helper(value):", start=6)
    second = build_repair_plan(observed, "Fix")
    assert first.fingerprint == second.fingerprint
    assert first.items == second.items

