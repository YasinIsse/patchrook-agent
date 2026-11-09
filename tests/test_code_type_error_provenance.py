"""Import-time TypeError is repairable only with existing changed-code proof."""

from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time

import pytest

from patchrook_agent.model_routing import CodeFailureContext, infrastructure_failure
from test_code_import_provenance import check, edit, runtime


FIXTURE = json.loads((Path(__file__).with_name("fixtures") / "type_error_provenance.json").read_text())
TRACE = FIXTURE["after_patch"]["check_1"]
CONTEXT = CodeFailureContext("ev_" + "a" * 24, "/app", ("src/cattrs/converters.py",),
                             "check_1", tuple(TRACE["argv"]))


@pytest.fixture(autouse=True)
def no_models_and_offline_pytest(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("TypeError provenance tests must not call a model or network")
    monkeypatch.setattr("httpx.Client", forbidden)
    monkeypatch.setattr("httpx.get", forbidden)
    monkeypatch.setattr("httpx.post", forbidden)
    monkeypatch.setattr("patchrook_agent.models.ollama.OllamaClient.chat", forbidden)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, (
        os.environ.get("PYTHONPATH"), str(Path(pytest.__file__).parent.parent)))))


def test_exact_v17_type_error_is_code_only_with_unchanged_existing_provenance_gates():
    assert TRACE["exit_code"] == 4 and "TypeError:" in TRACE["stderr"]
    assert infrastructure_failure(TRACE)
    assert not infrastructure_failure(TRACE, code_failure_context=CONTEXT)
    assert infrastructure_failure(TRACE, code_failure_context=replace(CONTEXT, modified_sources=()))
    assert infrastructure_failure(TRACE, code_failure_context=replace(CONTEXT, baseline_evidence_id="unverified"))


@pytest.mark.parametrize("extra", ["usage: pytest [options]\n", "ERROR: unrecognized arguments: --bad\n",
    "MemoryError: allocation failed\n", "PermissionError: denied\n", "ConnectionError: offline\n"])
def test_hard_infrastructure_and_usage_still_win_over_changed_frame(extra):
    assert infrastructure_failure({**TRACE, "stderr": TRACE["stderr"] + extra}, code_failure_context=CONTEXT)


@pytest.mark.parametrize("change", [
    {"timed_out": True}, {"truncated": True}, {"error": {"code": "workspace_changed_during_check"}},
    {"argv": ["pytest", "untrusted.py"]}, {"exit_code": 3}, {"exit_code": 5}, {"exit_code": 137},
])
def test_type_error_does_not_bypass_result_integrity_or_hard_exit_status(change):
    assert infrastructure_failure({**TRACE, **change}, code_failure_context=CONTEXT)


def prepare(runtime, *, dependency_body="self.ok = True"):
    root = runtime.workspace.root
    (root / "dependency.py").write_text(
        "class Dispatch:\n    def __init__(self, fallback, converter):\n        " + dependency_body + "\n")
    (root / "subject.py").write_text("from dependency import Dispatch\n\ndef value():\n    return 1\n")
    (root / "test_subject.py").write_text("import subject\n\ndef test_value():\n    assert subject.value() == 1\n")
    runtime.tools.commands["check_1"] = (sys.executable, "-m", "pytest", "-q", "test_subject.py")


def break_callsite(runtime, call="Dispatch()"):
    target = runtime.workspace.root / "subject.py"
    before = target.read_text()
    after = before.replace("\ndef value():", "\nclass Subject:\n    dispatch = " + call + "\n\ndef value():")
    assert edit(runtime, before, after)["success"]


def test_real_healthy_baseline_and_accepted_changed_callsite_qualify_type_error(runtime):
    prepare(runtime)
    assert check(runtime)["success"]
    break_callsite(runtime)
    failure = check(runtime)
    assert failure["exit_code"] == 2 and "TypeError:" in failure["stdout"]
    assert infrastructure_failure(failure)
    context = runtime._code_failure_context("check_1", failure, time.monotonic() + 10)
    assert isinstance(context, CodeFailureContext) and context.modified_sources == ("subject.py",)
    assert not runtime._check_infrastructure_failure("check_1", failure, time.monotonic() + 10)
    assert runtime.evidence.get(context.baseline_evidence_id, fresh=False).revision == 0
    (runtime.workspace.root / "subject.py").write_text("# stale after check\n")
    assert runtime._code_failure_context("check_1", failure, time.monotonic() + 10) is None


@pytest.mark.parametrize("baseline", ["missing", "broken"])
def test_missing_or_broken_baseline_cannot_qualify_type_error(runtime, baseline):
    prepare(runtime)
    target = runtime.workspace.root / "subject.py"
    if baseline == "broken":
        target.write_text(target.read_text() + "\nDispatch()\n")
        assert infrastructure_failure(check(runtime))
    break_callsite(runtime)
    failure = check(runtime)
    assert infrastructure_failure(failure)
    assert runtime._code_failure_context("check_1", failure, time.monotonic() + 10) is None


def test_unchanged_dependency_innermost_frame_does_not_become_code_proof(runtime):
    prepare(runtime, dependency_body="raise TypeError('dependency compatibility failure')")
    assert check(runtime)["success"]
    break_callsite(runtime, "Dispatch(None, None)")
    failure = check(runtime)
    assert failure["exit_code"] == 2 and "dependency compatibility failure" in failure["stdout"]
    assert infrastructure_failure(failure)
    assert runtime._check_infrastructure_failure("check_1", failure, time.monotonic() + 10)


@pytest.mark.parametrize("calls_used", [1, 2])
def test_remaining_call_is_never_refunded_or_extended_by_type_error_reclassification(runtime, calls_used):
    prepare(runtime)
    assert check(runtime)["success"]
    break_callsite(runtime)
    runtime.routing = {"repair": {"status": "active", "pending_checks": ["check_1"], "calls_used": calls_used}}
    stopped = runtime._run_repair_checks(time.monotonic() + 10)
    assert runtime.routing["repair"]["calls_used"] == calls_used
    assert stopped is (calls_used == 2)
    assert runtime.routing["repair"]["status"] == ("exhausted" if calls_used == 2 else "active")
    assert runtime.state.stop_reason == ("repair_call_limit" if calls_used == 2 else "max_steps")


def test_exact_v17_pytest_and_public_setup_outputs_remain_failed_not_verified():
    for name, result in FIXTURE["baseline"].items():
        assert not infrastructure_failure(result)
    summary = FIXTURE["after_patch"]["check_2"]
    assert summary["success"] is False
    assert json.loads(summary["stdout"])["blocked"] == 18
    assert not infrastructure_failure(summary)
