"""Exact V5 worker failures: pytest setup exit4 plus matching summary-only canary."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import time

import pytest

from test_code_import_provenance import runtime
from patchrook_agent.model_routing import CodeFailureContext, infrastructure_failure, reported_code_exception


FIXTURE = json.loads((Path(__file__).with_name("fixtures") / "import_error_provenance.json").read_text())
TRACE = FIXTURE["after_patch"]["check_1"]
SUMMARY = FIXTURE["after_patch"]["check_2"]
CONTEXT = CodeFailureContext("ev_" + "a" * 24, "/app", ("src/cattrs/__init__.py",),
                             "check_1", tuple(TRACE["argv"]))


@pytest.fixture(autouse=True)
def no_models_or_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Provenance fixtures cannot use models or network")
    monkeypatch.setattr("httpx.Client", forbidden)
    monkeypatch.setattr("httpx.get", forbidden)
    monkeypatch.setattr("httpx.post", forbidden)


def text(result):
    return str(result.get("stdout", "")) + "\n" + str(result.get("stderr", ""))


def test_exact_v5_pytest_exit4_requires_provenance_and_summary_identity_matches():
    assert TRACE["exit_code"] == 4 and infrastructure_failure(TRACE)
    assert not infrastructure_failure(TRACE, code_failure_context=CONTEXT)
    assert reported_code_exception(text(TRACE)) == reported_code_exception(text(SUMMARY))
    own = replace(CONTEXT, check_name="check_2", trusted_argv=tuple(SUMMARY["argv"]))
    assert infrastructure_failure(SUMMARY, code_failure_context=own)
    corroborated = replace(own, corroborated_exception=reported_code_exception(text(TRACE)),
                           traceback_evidence_id="ev_" + "b" * 24)
    assert not infrastructure_failure(SUMMARY, code_failure_context=corroborated)


@pytest.mark.parametrize("extra", ["usage: pytest [options]\n", "ERROR: unrecognized arguments: --bad\n",
    "pytest: error: invalid usage\n", "MemoryError\n", "PermissionError\n", "ConnectionError\n"])
def test_exit4_usage_and_hard_errors_remain_infrastructure(extra):
    result = {**TRACE, "stderr": TRACE["stderr"] + extra}
    assert infrastructure_failure(result, code_failure_context=CONTEXT)


def test_exit4_alone_and_non_pytest_exit4_are_not_excused():
    assert infrastructure_failure({**TRACE, "stdout": "", "stderr": ""}, code_failure_context=CONTEXT)
    result = {**TRACE, "argv": ["python3", "arbitrary.py"]}
    assert infrastructure_failure(result, code_failure_context=replace(CONTEXT, trusted_argv=tuple(result["argv"])))


@pytest.fixture
def scenario(runtime, monkeypatch):
    target = runtime.workspace.root / "src/cattrs/__init__.py"
    target.parent.mkdir(parents=True)
    target.write_text("ORIGINAL = True\n")
    commands = {name: tuple(result["argv"]) for name, result in FIXTURE["baseline"].items()}
    runtime.tools.commands = commands
    runtime.config = replace(runtime.config, evidence_test_commands=tuple(commands.values()),
                             evidence_auto_check_edits=True)
    runtime.definitions = runtime._definitions()
    def frozen_worker_output(name, **options):
        return deepcopy(FIXTURE["baseline" if runtime.revision == 0 else "after_patch"][name])
    monkeypatch.setattr(runtime.tools, "_run_check", frozen_worker_output)
    def check(name):
        return runtime._perform_action(runtime.state, {"name": "run_check", "arguments": {"name": name}},
                                       time.monotonic() + 10)
    for name in commands:
        check(name)
    source = runtime.evidence.observe_source("src/cattrs/__init__.py", origin="direct")
    runtime.direct = [source.id]
    runtime._prepare(runtime.state, runtime.definitions, time.monotonic() + 10)
    result = runtime._perform_action(runtime.state, {"name": "apply_edits", "arguments": {"edits": [{
        "path": "src/cattrs/__init__.py", "old_text": "ORIGINAL = True\n",
        "new_text": "from .converters import PartialResult\n"}]}}, time.monotonic() + 10)
    assert result["success"]
    current = {name: check(name) for name in commands}
    return runtime, current


def test_exact_worker_outputs_derive_independent_current_corrob_proof(scenario):
    runtime, current = scenario
    proof = runtime._code_failure_context("check_2", current["check_2"], time.monotonic() + 10)
    assert proof.traceback_evidence_id and proof.corroborated_exception == reported_code_exception(text(TRACE))
    for name, result in current.items():
        assert not runtime._check_infrastructure_failure(name, result, time.monotonic() + 10)


@pytest.mark.parametrize("kind", ["different_exception", "missing_message", "conflicting_summaries", "truncated",
                                  "wrong_argv", "source_write", "unrelated_frame"])
def test_canary_summary_conflicts_and_unsafe_results_remain_blocked(scenario, kind):
    runtime, current = scenario
    result = deepcopy(current["check_2"])
    if kind == "different_exception":
        result["stderr"] = result["stderr"].replace("PartialResult", "DifferentSymbol")
    elif kind == "missing_message":
        result["stderr"] = "FAIL public canary: ImportError:\n"
    elif kind == "conflicting_summaries":
        result["stderr"] += "FAIL public canary: NameError: conflicting\n"
    elif kind == "truncated":
        result["truncated"] = True
    elif kind == "wrong_argv":
        result["argv"] += ["untrusted"]
    elif kind == "source_write":
        result["error"] = {"code": "workspace_changed_during_check"}
    else:
        result["stderr"] = '  File "/outside/package.py", line 2, in <module>\n' + result["stderr"]
    assert runtime._check_infrastructure_failure("check_2", result, time.monotonic() + 10)


@pytest.mark.parametrize("kind", ["stale_revision", "stale_snapshot", "other_wrong_argv", "missing_own_baseline"])
def test_corroboration_requires_current_other_check_and_own_healthy_baseline(scenario, monkeypatch, kind):
    runtime, current = scenario
    if kind in {"stale_revision", "stale_snapshot"}:
        record = dict(runtime.checks["check_1"])
        record["revision" if kind == "stale_revision" else "snapshot"] = 0 if kind == "stale_revision" else "stale"
        monkeypatch.setitem(runtime.checks, "check_1", record)
    elif kind == "other_wrong_argv":
        monkeypatch.setitem(runtime.tools.commands, "check_1", ("pytest", "untrusted"))
    else:
        original = runtime.evidence.cards
        def without_own_baseline(**options):
            return [card for card in original(**options) if not (
                card.kind == "run_check" and card.revision == 0 and
                json.loads(card.text).get("result", {}).get("name") == "check_2")]
        monkeypatch.setattr(runtime.evidence, "cards", without_own_baseline)
    assert runtime._check_infrastructure_failure("check_2", current["check_2"], time.monotonic() + 10)


@pytest.mark.parametrize("order", [["check_1", "check_2"], ["check_2", "check_1"]])
def test_all_checks_requalify_remaining_repair_call_independent_of_order(scenario, order):
    runtime, _ = scenario
    runtime.checks.clear(); runtime.snapshot = None
    runtime.routing = {"repair": {"status": "active", "pending_checks": list(order), "calls_used": 1}}
    assert runtime._run_repair_checks(time.monotonic() + 10) is False
    assert runtime.routing["repair"]["calls_used"] == 1 and runtime.routing["repair"]["status"] == "active"
    assert runtime.state.stop_reason != "repair_environment_failure"
