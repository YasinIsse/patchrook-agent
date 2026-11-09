"""Initial Tev state contains current observations, never packet/repair authority."""

from copy import deepcopy
import json
import sys
import time

import pytest

from patchrook_agent.evidence_routing import initial_routing_context
from patchrook_agent.models.decision import DecisionResult
from patchrook_agent.tools.base import ToolError
from test_evidence_symbol_bootstrap import runtime_factory


TASK = "Implement `recover` on `Handler`, returning complete partial results."


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr("httpx.Client", lambda *a, **k: pytest.fail("No network or model calls"))


def make(runtime_factory, *, failure="raise AssertionError('missing nested partial fields')", task=TASK):
    runtime = runtime_factory(task, {"subject.py": "class Handler:\n    def recover(self, data):\n        pass\n"})
    runtime.tools.commands["check_1"] = [sys.executable, "-c", failure]
    runtime.tools.commands["check_2"] = [sys.executable, "-c", "pass"]
    deadline = time.monotonic() + 10
    for name in runtime.tools.commands:
        runtime._perform_action(runtime.state, {"name": "run_check", "arguments": {"name": name}},
                                deadline, automatic=True)
    runtime._perform_action(runtime.state, {"name": "read_file", "arguments": {
        "path": "subject.py", "start_line": 1, "end_line": 3}}, deadline, automatic=True)
    return runtime


def state(context):
    text = context.split("\nObserved current evidence (data):\n", 1)[1]
    return json.loads(text.split("\nObserved repository inventory (data):\n", 1)[0])


def context(runtime):
    return initial_routing_context(runtime, time.monotonic() + 10)


def test_current_configured_failures_and_fresh_direct_source_are_exposed(runtime_factory):
    runtime = make(runtime_factory)
    text = context(runtime)
    summary = state(text)
    assert summary["checks"] == {"check_1": "code_failure", "check_2": "passed"}
    assert summary["current_check_count"] == 2 and summary["checks_omitted"] == 0
    assert "missing nested partial fields" in summary["failure_excerpt"]
    assert summary["fresh_source_views"] == 1
    assert summary["source_views"] == [{"path": "subject.py", "lines": [1, 3]}]
    assert summary["repair_packet"] == "not_evaluated" and summary["edit_authorization"] is False
    assert TASK in text and len(text) <= 2300
    assert not runtime.tools._presented, "Routing observations do not authorize edits"


@pytest.mark.parametrize("change", ["revision", "workspace", "argv", "timeout"])
def test_stale_or_mismatched_results_cannot_enter_routing_state(runtime_factory, change):
    runtime = make(runtime_factory)
    if change == "revision":
        runtime.revision += 1
    elif change == "workspace":
        (runtime.workspace.root / "subject.py").write_text("EXTERNAL = True\n")
    else:
        result = next(row for row in runtime.session.memory.test_results if row["name"] == "check_1")
        if change == "argv":
            result["argv"] = [sys.executable, "-c", "unconfigured"]
        else:
            result["timed_out"] = True
    text = context(runtime)
    assert "missing nested partial fields" not in text
    assert "check_1" not in state(text)["checks"]


def test_hard_infrastructure_failure_is_not_presented_as_code_repair_evidence(runtime_factory):
    runtime = make(runtime_factory, failure="raise ImportError('missing dependency')")
    summary = state(context(runtime))
    assert summary["checks"]["check_1"] == "infrastructure_failure"
    assert "failure_excerpt" not in summary
    assert summary["repair_packet"] == "not_evaluated"


def test_hash_stale_oracle_and_expansion_cards_are_not_direct_source_facts(runtime_factory):
    runtime = make(runtime_factory)
    for origin in ("dependency", "diagnostic_oracle"):
        card = runtime.evidence.observe_source("subject.py", 1, 3, revision=0, origin=origin)
        runtime.direct.append(card.id)
    summary = state(context(runtime))
    assert summary["fresh_source_views"] == 1
    (runtime.workspace.root / "subject.py").write_text("EXTERNAL = True\n")
    summary = state(context(runtime))
    assert summary["fresh_source_views"] == 0 and not summary.get("source_views")


@pytest.mark.parametrize("task", ["x" * 1800, "x" * 2400, "λ" * 1800])
def test_task_prefix_preserved_and_added_json_is_complete_within_original_cap(runtime_factory, task):
    runtime = make(runtime_factory, task=task)
    runtime.evidence.add_observation("inventory", '"' * 2000, revision=0)
    text = context(runtime)
    assert text.startswith("Task (data):\n" + task[:1800])
    assert len(text) <= 2300
    summary = state(text)
    assert summary["checks"]["check_1"] == "code_failure"


def test_unavailable_freshness_does_not_forward_cached_check_failures(runtime_factory, monkeypatch):
    runtime = make(runtime_factory)
    def incomplete(deadline):
        raise ToolError("inventory_incomplete", "not current")
    monkeypatch.setattr(runtime, "_fingerprint", incomplete)
    summary = state(context(runtime))
    assert summary["freshness"] == "unconfirmed"
    assert summary["checks"] == {} and summary["fresh_source_views"] == 0
    assert "failure_excerpt" not in summary


def test_readonly_summary_does_not_run_tools_prepare_packet_or_mutate_execution(runtime_factory, monkeypatch):
    runtime = make(runtime_factory)
    original = deepcopy((runtime.direct, runtime.checks, runtime.session.memory.test_results,
                         runtime.config.to_dict(), runtime.state.messages))
    monkeypatch.setattr(runtime.tools, "execute", lambda *a, **k: pytest.fail("No extra tools or checks"))
    monkeypatch.setattr(runtime, "_prepare_repair", lambda *a, **k: pytest.fail("No packet prequalification"))
    context(runtime)
    assert (runtime.direct, runtime.checks, runtime.session.memory.test_results,
            runtime.config.to_dict(), runtime.state.messages) == original


def test_initial_router_candidates_budget_label_and_one_call_semantics_stay_unchanged(runtime_factory):
    runtime = make(runtime_factory)
    requests = []
    class Router:
        def rank(self, text, candidates, **options):
            requests.append((text, candidates, options))
            return DecisionResult(("uncertain", "small", "large"),
                                  {"uncertain": .4, "small": .36, "large": .24}, called=True)
    runtime.routing = {"initial_decided": False, "initial_route": None, "repair": {"status": "idle", "reason": ""}}
    runtime.routing_client = Router()
    config = deepcopy(runtime.config.to_dict())
    runtime._initial_route(time.monotonic() + 300)
    runtime._initial_route(time.monotonic() + 300)
    assert len(requests) == 1
    text, candidates, options = requests[0]
    assert [item.id for item in candidates] == ["small", "large", "uncertain"]
    assert options == {"purpose": "model_routing", "timeout_seconds": runtime.config.decision_timeout_seconds}
    assert runtime.routing["initial_route"]["label"] == "uncertain"
    assert runtime.routing["repair"]["status"] == "idle"
    assert runtime.config.to_dict() == config and len(text) <= 2300
    assert "seed_ancestry" not in text and "parent_repair_calls_used" not in text


def test_readonly_preparation_cannot_spend_a_stale_remaining_routing_budget(runtime_factory, monkeypatch):
    runtime = make(runtime_factory)
    runtime.routing = {"initial_decided": False, "initial_route": None, "repair": {"status": "idle"}}
    class Router:
        def rank(self, *args, **kwargs):
            pytest.fail("No Tev call after the task deadline")
    runtime.routing_client = Router()
    now = [time.monotonic()]
    deadline = now[0] + 5
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    def exhausted(actual_deadline):
        assert actual_deadline == deadline
        now[0] = deadline + 1
        raise ToolError("deadline_exceeded", "fingerprint preparation used remaining budget")
    monkeypatch.setattr(runtime, "_fingerprint", exhausted)
    runtime._initial_route(deadline)
    assert runtime.routing["initial_decided"] is True
    assert runtime.routing["initial_route"]["fallback_reason"] == "task_budget"
    assert runtime.routing["initial_route"]["called"] is False
