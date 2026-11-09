"""Lean completion uses observed pytest scopes and bounded new evidence."""

import json

import pytest

from patchrook_agent.lean_progress import LeanProgress
from patchrook_agent.tools import Workspace


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "tests").mkdir()
    for path in ("tests/test_a.py", "tests/test_b.py", "module.py"):
        (tmp_path / path).write_text("# Source is not read by the evidence helper\n")
    return Workspace(tmp_path)


def result(status="passed", *, stdout="", duration=0.01):
    return {"success": status == "passed", "status": status,
            "exit_code": None if status == "timed_out" else 0 if status == "passed" else 1,
            "stdout": stdout, "stderr": "", "duration_seconds": duration}


def run(progress, path="tests/test_a.py", status="passed", **kwargs):
    return progress.observe("run_tests", {"path": path}, result(status, **kwargs))


def change(progress):
    return progress.observe("edit_file", {"path": "module.py"},
                            {"success": True, "path": "module.py", "modified": True})


def read(progress, *, first=1, source="value = 1", name="read_file", **kwargs):
    record = {"success": True, "path": "module.py", "start_line": first, "end_line": first,
              "content": f"{first}: {source}", "truncated": False, **kwargs}
    return progress.observe(name, {"path": "module.py"} if name == "read_file" else {"files": [{"path": "module.py"}]},
                            record if name == "read_file" else {"success": True, "files": [record]})


def test_unrelated_green_test_does_not_hide_a_failure(workspace):
    progress = LeanProgress(workspace)
    run(progress, "tests/test_a.py", "failed")
    run(progress, "tests/test_b.py")
    assert progress.verification_status == "failed"
    assert progress.missing() == ["resolve observed failed tests for tests/test_a.py"]
    assert "tests/test_a.py" in progress.summary()
    run(progress, "tests/test_a.py")
    assert progress.verification_status == "passed" and not progress.missing()


@pytest.mark.parametrize("passing", ["tests/test_a.py::TestClass", "tests/test_a.py", "tests", "."])
def test_same_or_broader_selection_resolves_failed_node(workspace, passing):
    progress = LeanProgress(workspace)
    run(progress, "tests/test_a.py::TestClass::test_a[case]", "failed")
    run(progress, passing)
    assert progress.verification_status == "passed" and not progress.failed


@pytest.mark.parametrize("passing", ["tests/test_a.py::test_alpha", "tests/test_a.py::test_a[other]", "tests/test_b.py"])
def test_similar_or_sibling_selection_cannot_clear_failed_node(workspace, passing):
    progress = LeanProgress(workspace)
    run(progress, "tests/test_a.py::test_a[case]", "failed")
    run(progress, passing)
    assert progress.verification_status == "failed"


def test_function_selection_covers_its_parameter_cases(workspace):
    progress = LeanProgress(workspace)
    run(progress, "tests/test_a.py::test_a[case]", "failed")
    run(progress, "tests/test_a.py::test_a")
    assert progress.verification_status == "passed"


def test_subset_pass_cannot_hide_failed_file_or_directory(workspace):
    progress = LeanProgress(workspace)
    run(progress, "tests", "failed")
    run(progress, "tests/test_a.py")
    assert progress.verification_status == "failed"
    run(progress, "tests")
    assert progress.verification_status == "passed"


def test_mutation_invalidates_passes_but_retains_failure_scope(workspace):
    progress = LeanProgress(workspace)
    run(progress, "tests/test_a.py", "failed")
    run(progress, "tests/test_b.py")
    assert change(progress) is None
    assert progress.generation == 1 and not progress.passed and progress.failed
    run(progress, "tests/test_b.py")
    assert progress.verification_status == "failed"
    run(progress, "tests")
    assert progress.verification_status == "passed"
    change(progress)
    assert progress.verification_status == "not_run"
    assert progress.missing() == ["observe passing tests after the last change"]


def test_later_failed_subset_invalidates_old_broader_pass(workspace):
    progress = LeanProgress(workspace)
    run(progress, ".")
    run(progress, "tests/test_a.py::test_a", "failed")
    assert not progress.passed and progress.verification_status == "failed"
    run(progress, "tests/test_b.py")
    assert progress.verification_status == "failed"


def test_required_targets_must_all_be_covered_after_last_change(workspace):
    progress = LeanProgress(workspace, ["tests/test_a.py", "tests/test_b.py::test_b"])
    run(progress, "tests/test_a.py")
    assert progress.verification_status == "not_run"
    assert progress.missing() == ["observe passing tests after the last change for tests/test_b.py::test_b"]
    run(progress, "tests/test_b.py")
    assert not progress.missing()
    change(progress)
    assert len(progress.missing()) == 2
    run(progress, "tests")
    assert not progress.missing()


def test_read_only_completion_does_not_force_edits_or_tests(workspace):
    progress = LeanProgress(workspace)
    read(progress)
    assert progress.missing(require_tests=False) == []
    assert progress.generation == 0
    run(progress, status="failed")
    assert progress.missing(require_tests=False) == ["resolve observed failed tests for tests/test_a.py"]


def test_observed_timeout_remains_until_scope_passes(workspace):
    progress = LeanProgress(workspace)
    run(progress, status="timed_out")
    assert progress.verification_status == "timed_out"
    run(progress, "tests/test_b.py")
    assert progress.verification_status == "timed_out"
    run(progress, ".")
    assert progress.verification_status == "passed"


@pytest.mark.parametrize("path", ["../outside.py", "-q", "tests::-q", "tests::test_a", "tests/test_a.py::", "missing.py"])
def test_invalid_or_unconfined_test_scope_is_never_evidence(workspace, path):
    progress = LeanProgress(workspace)
    run(progress, path)
    assert progress.verification_status == "not_run" and not progress.passed
    with pytest.raises(ValueError):
        LeanProgress(workspace, [path])


def test_aliased_and_absolute_selections_are_canonical(workspace):
    progress = LeanProgress(workspace)
    run(progress, "./tests/../tests/test_a.py", "failed")
    run(progress, str(workspace.root / "tests/test_a.py"))
    assert progress.verification_status == "passed"
    assert list(progress.passed) == ["tests/test_a.py"]


def test_forged_or_setup_test_results_do_not_credit_passing(workspace):
    progress = LeanProgress(workspace)
    for record in ({"success": True, "status": "passed", "exit_code": True},
                   {"success": False, "status": "passed", "exit_code": 0},
                   {"success": False, "error": {"code": "permission_denied"}},
                   {"success": True, "status": "passed"}):
        progress.observe("run_tests", {"path": "tests"}, record)
    assert progress.verification_status == "not_run"


def test_repeated_reads_get_one_recovery_then_bounded_stop(workspace):
    progress = LeanProgress(workspace)
    assert read(progress) is None
    assert read(progress, duration_seconds=3, artifact="new-id") is None
    assert read(progress, duration_seconds=8, artifact="another-id") == "recover"
    progress.acknowledge_recovery()
    assert read(progress) is None
    assert read(progress) == "stop"
    assert progress.stall_count == 4 and "no new evidence" in progress.recovery_hint()


def test_new_line_or_changed_source_resets_stall(workspace):
    progress = LeanProgress(workspace)
    read(progress)
    read(progress)
    assert read(progress) == "recover"
    progress.acknowledge_recovery()
    assert read(progress, first=2) is None and progress.stall_count == 0
    read(progress, first=2)
    assert read(progress, first=2, source="value = 2") is None and progress.stall_count == 0
    assert not progress.recovery_delivered


def test_partial_source_and_hidden_artifacts_do_not_reset_stall(workspace):
    progress = LeanProgress(workspace)
    assert read(progress, partial_line=True, truncated=True) is None
    assert progress.stall_count == 1
    assert read(progress, partial_line=True, truncated=True, artifact={"full_source": "different"}) == "recover"
    progress.observe("read_artifact", {}, {"success": True, "content": "source"})
    assert progress.stall_count == 3


@pytest.mark.parametrize("name", ["read_files", "read_code"])
def test_batch_reads_count_only_new_visible_lines(workspace, name):
    progress = LeanProgress(workspace)
    read(progress, name=name)
    assert read(progress, name=name) is None
    assert read(progress, name=name) == "recover"
    assert read(progress, name=name, first=5) is None


def test_equivalent_search_and_listing_results_ignore_incidental_metadata(workspace):
    progress = LeanProgress(workspace)
    for index in range(3):
        action = progress.observe("search_code", {"pattern": "value", "max_results": index + 1},
                                  {"success": True, "matches": [{"path": "module.py", "line_number": 1, "text": "value = 1"}],
                                   "duration_seconds": index, "artifact_id": str(index)})
    assert action == "recover"
    assert progress.observe("search_code", {}, {"success": True, "matches": [
        {"path": "module.py", "line_number": 2, "text": "value = 2"}]}) is None
    listing = {"success": True, "entries": ["tests/", "module.py"]}
    progress.observe("list_files", {}, listing)
    progress.observe("list_files", {}, {**listing, "duration_seconds": 9})
    assert progress.observe("list_files", {}, listing) == "recover"


def test_read_and_search_share_source_identity(workspace):
    progress = LeanProgress(workspace)
    read(progress)
    matched = {"success": True, "matches": [{"path": "module.py", "line_number": 1, "text": "value = 1"}]}
    assert progress.observe("search_code", {"pattern": "value"}, matched) is None
    assert read(progress) == "recover"
    assert progress.observe("search_code", {"pattern": "value"}, {"success": True, "matches": [
        {"path": "module.py", "line_number": 2, "text": "value = 1"}]}) is None


def test_repeated_test_failures_ignore_pytest_timing_but_changed_outcomes_reset(workspace):
    progress = LeanProgress(workspace)
    run(progress, status="failed", stdout="1 failed in 0.01s")
    run(progress, status="failed", stdout="1 failed in 3.25s")
    assert run(progress, status="failed", stdout="1 failed in 0.44s") == "recover"
    assert run(progress, status="failed", stdout="2 failed in 0.44s") is None
    assert progress.stall_count == 0
    assert run(progress) is None and progress.verification_status == "passed"


def test_only_actual_mutation_resets_stall_and_invalidates_passes(workspace):
    progress = LeanProgress(workspace)
    run(progress)
    read(progress)
    progress.observe("edit_file", {}, {"success": True, "modified": False})
    assert progress.verification_status == "passed" and progress.stall_count == 1
    change(progress)
    assert progress.verification_status == "not_run" and progress.stall_count == 0
    # A post-edit read is fresh evidence of the current source generation.
    assert read(progress) is None


def test_serialization_preserves_debt_and_recovery_delivery_on_resume(workspace):
    progress = LeanProgress(workspace, ["tests/test_a.py"])
    run(progress, status="failed")
    read(progress)
    read(progress)
    assert read(progress) == "recover"
    progress.acknowledge_recovery()
    stored = json.loads(json.dumps(progress.to_dict()))
    restored = LeanProgress.from_dict(stored, workspace, ["tests/test_a.py"])
    assert restored.to_dict() == stored and restored.action is None
    assert restored.verification_status == "failed"
    assert read(restored) is None
    assert read(restored) == "stop"
    assert LeanProgress.from_dict(restored.to_dict(), workspace).action == "stop"
    restored.invalidate()
    assert restored.failed and restored.stall_count == 0 and restored.generation == 1


def test_pending_recovery_survives_resume_without_replay(workspace):
    progress = LeanProgress(workspace)
    read(progress)
    read(progress)
    read(progress)
    restored = LeanProgress.from_dict(progress.to_dict(), workspace)
    assert restored.action == "recover"
    restored.acknowledge_recovery()
    assert restored.action is None


@pytest.mark.parametrize("change", [
    {"schema_version": True}, {"generation": True}, {"generation": -1}, {"stall_count": 500},
    {"recovery_delivered": 1}, {"seen": ["bad"]}, {"passed": [{"path": "../outside", "nodes": [], "directory": False}]},
    {"test_outcomes": {"../outside": "a" * 64}}, {"unknown": 1},
    {"failed": [{"selection": {"path": "tests", "nodes": [], "directory": True}, "status": []}]},
])
def test_corrupt_progress_state_is_rejected(workspace, change):
    progress = LeanProgress(workspace)
    with pytest.raises(ValueError):
        LeanProgress.from_dict({**progress.to_dict(), **change}, workspace)


def test_resume_rejects_changed_required_test_scope(workspace):
    progress = LeanProgress(workspace, ["tests/test_a.py"])
    with pytest.raises(ValueError, match="Required tests differ"):
        LeanProgress.from_dict(progress.to_dict(), workspace, ["tests/test_b.py"])
