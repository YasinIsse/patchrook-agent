"""Narrow import-failure routing provenance; models and live workspaces untouched."""

from dataclasses import replace
from pathlib import Path
import sys
import time

import pytest

from patchrook_agent.agent import Agent, AgentState
from patchrook_agent.completion import TaskRequirements
from patchrook_agent.config import HarnessConfig
from patchrook_agent.evidence_workflow import EVIDENCE_SYSTEM_PROMPT, EvidenceWorkflow
from patchrook_agent.model_routing import CodeFailureContext, RoutingProgress, infrastructure_failure
from patchrook_agent.tools import ToolConfig, ToolRegistry, Workspace


COMMAND = ("python3", "-m", "pytest")
CONTEXT = CodeFailureContext("ev_" + "a" * 24, "/app", ("src/subject.py",), "check_1", COMMAND)


def failure(trace='  File "/app/src/subject.py", line 2, in <module>\n    from missing import thing\nModuleNotFoundError: No module named missing\n', **changes):
    return {"success": False, "verification_kind": "configured_command", "name": "check_1",
        "argv": list(COMMAND), "exit_code": 2, "timed_out": False, "stdout": trace, "stderr": "", **changes}


@pytest.fixture(autouse=True)
def no_models_or_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Import provenance tests must not run models or network requests")
    monkeypatch.setattr("httpx.Client", forbidden)
    monkeypatch.setattr("httpx.get", forbidden)
    monkeypatch.setattr("httpx.post", forbidden)
    monkeypatch.setattr("patchrook_agent.models.ollama.OllamaClient.chat", forbidden)


@pytest.mark.parametrize("trace", [
    'Traceback (most recent call last):\n  File "/app/src/subject.py", line 2, in <module>\nImportError: cannot import name thing\n',
    'Traceback:\nsrc/subject.py:2: in <module>\n    from missing import thing\nE   ModuleNotFoundError: No module named missing\n',
    '/app/src/subject.py:2: in <module>\nE   ImportError: cannot import name thing\n',
])
def test_modified_repo_traceback_with_verified_context_is_code_failure(trace):
    result = failure(trace)
    assert infrastructure_failure(result) is True
    assert infrastructure_failure(result, code_failure_context=CONTEXT) is False


@pytest.mark.parametrize("trace", [
    'Traceback:\n/usr/lib/python3/site-packages/package.py:2: in <module>\nE   ModuleNotFoundError: No module named external\n',
    '  File "/app/tests/test_subject.py", line 2, in <module>\nImportError: broken fixture\n',
    '  File "/app2/src/subject.py", line 2, in <module>\nImportError: unrelated repo\n',
    '  File "/app/src/../src/subject.py", line 2, in <module>\nImportError: noncanonical path\n',
    '  File "/outside/subject.py", line 2, in <module>\nImportError: unrelated path\n',
    'ImportError: source named /app/src/subject.py, but no traceback frame\n',
    '  File "/app/src/subject.py", line 2, in <module>\nAssertionError: unrelated failure\nERROR collecting tests\n',
])
def test_unrelated_missing_dependency_paths_and_non_frames_stay_conservative(trace):
    assert infrastructure_failure(failure(trace), code_failure_context=CONTEXT) is True


@pytest.mark.parametrize("changes", [
    {"timed_out": True}, {"error": {"code": "workspace_changed_during_check"}},
    {"exit_code": 127}, {"exit_code": 137}, {"exit_code": -9},
    {"exit_code": 3}, {"exit_code": 5}, {"truncated": True},
    {"stderr": "MemoryError: allocation failed"}, {"stderr": "out of memory"},
    {"stderr": "PermissionError: denied"}, {"stderr": "ConnectionError: offline"},
    {"stderr": "command not found"}, {"stderr": "no such file or directory"},
    {"name": "check_other"}, {"argv": ["untrusted", "command"]}, {"argv": None},
    {"verification_kind": "model_claim"},
])
def test_hard_errors_untrusted_checks_and_truncation_cannot_be_overridden(changes):
    assert infrastructure_failure(failure(**changes), code_failure_context=CONTEXT) is True


def test_result_flags_or_dicts_cannot_supply_internal_provenance():
    result = failure(code_induced_import=True)
    assert infrastructure_failure(result) is True
    assert infrastructure_failure(result, code_failure_context={"modified_sources": ["src/subject.py"]}) is True


@pytest.fixture
def runtime(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "subject.py").write_text("def value():\n    return 0\n")
    class NoModel:
        model = "offline-no-inference"
        context_size = 8192
        max_output_tokens = 1024
    base = HarnessConfig.preset("C")
    config = replace(base, workflow_mode="evidence", action_protocol="schema", review_mode="off",
        persist=False, checkpoint_enabled=False, compaction_enabled=False,
        context=replace(base.context, enabled=False), evidence_policy="rules", model_routing="off",
        routing_max_failed_repairs=1,
        evidence_test_commands=((sys.executable, "-c", "import subject; assert subject.value() == 1"),))
    agent = Agent(NoModel(), config=config, registry=ToolRegistry(Workspace(root, ToolConfig(allow_tests=True))),
                  runs_dir=tmp_path / "runs", max_steps=10, deadline_seconds=30)
    value = EvidenceWorkflow(agent)
    value.requirements = TaskRequirements.fix()
    value._setup("Fix the synthetic subject fixture.")
    value.definitions = value._definitions()
    value.enabled = {item["function"]["name"] for item in value.definitions}
    value.state = AgentState(messages=[{"role": "system", "content": EVIDENCE_SYSTEM_PROMPT},
                                      {"role": "user", "content": value.task}])
    value.session.bind(value.state)
    yield value
    if value.session.store:
        value.session.store.release()


def check(runtime):
    return runtime._perform_action(runtime.state, {"name": "run_check", "arguments": {"name": "check_1"}},
                                   time.monotonic() + 10)


def edit(runtime, old, new):
    target = runtime.workspace.root / "subject.py"
    source = runtime.evidence.observe_source("subject.py", origin="direct")
    runtime.direct = [source.id]
    runtime._prepare(runtime.state, runtime.definitions, time.monotonic() + 10)
    return runtime._perform_action(runtime.state, {"name": "apply_edits", "arguments": {"edits": [{
        "path": "subject.py", "old_text": old, "new_text": new}]}}, time.monotonic() + 10)


@pytest.mark.parametrize("baseline_passes", [False, True])
def test_real_baseline_and_accepted_patch_provenance_make_actual_import_failure_repairable(runtime, baseline_passes):
    target = runtime.workspace.root / "subject.py"
    if baseline_passes:
        target.write_text("def value():\n    return 1\n")
    baseline = check(runtime)
    assert baseline["success"] is baseline_passes and not infrastructure_failure(baseline)
    before = target.read_text()
    assert edit(runtime, before, "from nonexistent_provenance_fixture import thing\n" + before)["success"]
    result = check(runtime)
    assert "ModuleNotFoundError" in result["stderr"]
    assert infrastructure_failure(result) is True
    context = runtime._code_failure_context("check_1", result, time.monotonic() + 10)
    assert isinstance(context, CodeFailureContext) and context.modified_sources == ("subject.py",)
    assert runtime.evidence.get(context.baseline_evidence_id, fresh=False).revision == 0
    assert runtime._check_infrastructure_failure("check_1", result, time.monotonic() + 10) is False
    progress = RoutingProgress()
    assert progress.observe("run_check", {"name": "check_1"}, result, runtime.revision, runtime.config,
                            code_failure_context=context) == "failed_repairs"


def test_baseline_archive_is_used_even_after_rolling_test_memory_is_cleared(runtime):
    assert not infrastructure_failure(check(runtime))
    before = (runtime.workspace.root / "subject.py").read_text()
    assert edit(runtime, before, "from nonexistent_provenance_fixture import thing\n" + before)["success"]
    runtime.session.memory.test_results.clear()
    result = check(runtime)
    assert runtime._check_infrastructure_failure("check_1", result, time.monotonic() + 10) is False


def test_missing_or_already_broken_initial_environment_never_gains_provenance(runtime):
    target = runtime.workspace.root / "subject.py"
    target.write_text("from missing_at_baseline import thing\n")
    assert infrastructure_failure(check(runtime))
    before = target.read_text()
    assert edit(runtime, before, before + "# unrelated edit\n")["success"]
    result = check(runtime)
    assert runtime._check_infrastructure_failure("check_1", result, time.monotonic() + 10) is True


def test_current_snapshot_must_still_match_and_command_must_be_trusted(runtime):
    check(runtime)
    target = runtime.workspace.root / "subject.py"
    before = target.read_text()
    edit(runtime, before, "from nonexistent_provenance_fixture import thing\n" + before)
    result = check(runtime)
    assert runtime._code_failure_context("check_1", result, time.monotonic() + 10)
    assert runtime._code_failure_context("untrusted_name", result, time.monotonic() + 10) is None
    target.write_text(target.read_text() + "# changed after check\n")
    assert runtime._code_failure_context("check_1", result, time.monotonic() + 10) is None


def test_actual_reverted_source_does_not_count_as_currently_modified(runtime):
    check(runtime)
    target = runtime.workspace.root / "subject.py"
    original = target.read_text()
    broken = "from nonexistent_provenance_fixture import thing\n" + original
    edit(runtime, original, broken)
    imported_failure = check(runtime)
    edit(runtime, broken, original)
    check(runtime)
    assert runtime._code_failure_context("check_1", imported_failure, time.monotonic() + 10) is None


def test_14b_check_loop_keeps_remaining_call_after_proven_code_induced_import(runtime):
    check(runtime)
    target = runtime.workspace.root / "subject.py"
    before = target.read_text()
    edit(runtime, before, "from nonexistent_provenance_fixture import thing\n" + before)
    runtime.routing = {"repair": {"status": "active", "pending_checks": ["check_1"], "calls_used": 1}}
    assert runtime._run_repair_checks(time.monotonic() + 10) is False
    assert runtime.routing["repair"]["status"] == "active"
    assert runtime.routing["repair"]["pending_checks"] == []
    assert runtime.state.stop_reason != "repair_environment_failure"


def test_healthy_baseline_from_a_different_command_is_not_reused(runtime):
    check(runtime)
    target = runtime.workspace.root / "subject.py"
    before = target.read_text()
    edit(runtime, before, "from nonexistent_provenance_fixture import thing\n" + before)
    runtime.tools.commands["check_1"] = (sys.executable, "-c", "import subject")
    result = check(runtime)
    assert infrastructure_failure(result)
    assert runtime._code_failure_context("check_1", result, time.monotonic() + 10) is None


def test_source_race_during_provenance_validation_is_conservative(runtime, monkeypatch):
    check(runtime)
    target = runtime.workspace.root / "subject.py"
    before = target.read_text()
    edit(runtime, before, "from nonexistent_provenance_fixture import thing\n" + before)
    result = check(runtime)
    original = runtime.tools._bytes
    def raced(path):
        payload = original(path)
        if path == target:
            target.write_text(target.read_text() + "# concurrent change\n")
        return payload
    monkeypatch.setattr(runtime.tools, "_bytes", raced)
    assert runtime._code_failure_context("check_1", result, time.monotonic() + 10) is None


def test_external_edits_without_an_accepted_model_patch_do_not_supply_provenance(runtime):
    check(runtime)
    target = runtime.workspace.root / "subject.py"
    target.write_text("from nonexistent_provenance_fixture import thing\n" + target.read_text())
    result = check(runtime)
    assert runtime.revision == 0 and infrastructure_failure(result)
    assert runtime._code_failure_context("check_1", result, time.monotonic() + 10) is None


def test_pytest_name_error_collection_is_code_only_with_verified_context():
    result = failure('src/subject.py:4: in Subject\nE   NameError: name partial_structure is not defined\n')
    assert infrastructure_failure(result)
    assert not infrastructure_failure(result, code_failure_context=CONTEXT)
    assert infrastructure_failure(failure('tests/test_subject.py:4: in test_import\nE   NameError: unmodified fixture\n'),
                                  code_failure_context=CONTEXT)


def test_actual_pytest_name_error_collection_has_baseline_and_changed_source_provenance(runtime):
    root = runtime.workspace.root
    (root / "test_subject.py").write_text("import subject\ndef test_value():\n    assert subject.value() == 1\n")
    runtime.tools.commands["check_1"] = (sys.executable, "-m", "pytest", "-q", "test_subject.py")
    baseline = check(runtime)
    assert baseline["exit_code"] == 1 and not infrastructure_failure(baseline)
    before = (root / "subject.py").read_text()
    after = "class Subject:\n    unresolved = staticmethod(undefined_name)\n" + before
    assert edit(runtime, before, after)["success"]
    result = check(runtime)
    assert result["exit_code"] == 2 and "NameError" in result["stdout"]
    assert infrastructure_failure(result)
    assert not runtime._check_infrastructure_failure("check_1", result, time.monotonic() + 10)


def test_base_auto_check_propagates_provenance_to_failed_repair_routing(runtime, monkeypatch):
    from patchrook_agent.evidence_verification import run_base_checks
    check(runtime)
    target = runtime.workspace.root / "subject.py"
    before = target.read_text()
    edit(runtime, before, "from nonexistent_provenance_fixture import thing\n" + before)
    runtime.config = replace(runtime.config, evidence_auto_check_edits=True)
    runtime.routing = {"active_model": "qwen3:8b"}
    reasons = []
    monkeypatch.setattr(runtime, "_routing_signal", lambda reason, deadline: reasons.append(reason) or False)
    monkeypatch.setattr(runtime, "_maybe_start_repair", lambda deadline: False)
    assert run_base_checks(runtime, time.monotonic() + 10) is False
    assert reasons == ["failed_repairs"]
    assert runtime.state.stop_reason != "base_check_environment_failure"


def test_malformed_check_names_and_internal_context_fail_conservatively(runtime):
    assert runtime._code_failure_context([], failure(), time.monotonic() + 10) is None
    malformed = replace(CONTEXT, baseline_evidence_id=None)
    assert infrastructure_failure(failure(), code_failure_context=malformed)
    malformed = replace(CONTEXT, modified_sources=(["unhashable"],))
    assert infrastructure_failure(failure(), code_failure_context=malformed)


def test_modified_ancestor_cannot_hide_an_unrelated_dependency_failure():
    trace = ('Traceback (most recent call last):\n'
             '  File "/app/src/subject.py", line 2, in <module>\n'
             '    import external_dependency\n'
             '  File "/usr/lib/site-packages/external_dependency.py", line 4, in <module>\n'
             '    import missing_transitive_dependency\n'
             'ModuleNotFoundError: No module named missing_transitive_dependency\n')
    assert infrastructure_failure(failure(trace), code_failure_context=CONTEXT)
