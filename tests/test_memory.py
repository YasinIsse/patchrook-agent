import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from patchrook_agent.memory import (
    SessionStore, WorkingMemory, compare_snapshots, render_progress, snapshot_files,
)
from patchrook_agent.tools.base import ToolConfig, Workspace


def memory(session_id="session-1"):
    return WorkingMemory(session_id=session_id, objective="Fix addition; preserve multiplication.")


def test_working_memory_records_observed_facts_not_claimed_success():
    state = memory()
    state.observe_tool("read_file", {"path": "maths.py"}, {
        "success": False, "error": {"code": "not_found", "message": "File is missing"},
    }, 1)
    assert state.verification == "not_run" and not state.completed_steps
    assert state.unresolved_errors[-1]["error"]["message"] == "File is missing"
    state.observe_tool("read_file", {"path": "maths.py"}, {
        "success": True, "path": "maths.py", "content": "1: return a - b",
    }, 2)
    assert state.relevant_files == ["maths.py"] and not state.unresolved_errors
    assert state.last_completed_step == 2


def test_discovered_and_modified_paths_are_deduplicated():
    state = memory()
    state.observe_tool("list_files", {}, {"success": True, "entries": ["src/", "src/math.py"]}, 1)
    state.observe_tool("search_code", {}, {"success": True, "matches": [{"path": "src/math.py"}]}, 1)
    for step in (2, 3):
        state.observe_tool("edit_file", {"path": "src/math.py"}, {
            "success": True, "path": "src/math.py", "modified": True,
        }, step)
    assert state.relevant_files == ["src/math.py"]
    assert state.modified_files == ["src/math.py"]
    assert state.verification == "not_run"
    assert "Verify" in state.active_task


def test_noop_edits_do_not_claim_mutation_or_invalidate_verification():
    state = memory()
    state.verification = "passed"
    state.observe_tool("edit_file", {"path": "math.py"}, {
        "success": True, "path": "math.py", "modified": False,
    }, 1)
    assert state.verification == "passed" and state.modified_files == []


def test_test_failures_survive_edits_until_observed_passing_result():
    state = memory()
    failure = "AssertionError: expected 5, got -1\n  maths.py:2"
    state.observe_tool("run_tests", {"path": "tests/test_math.py"}, {
        "success": False, "status": "failed", "exit_code": 1, "stdout": failure, "stderr": "",
    }, 1)
    state.observe_tool("edit_file", {"path": "maths.py"}, {
        "success": True, "path": "maths.py", "modified": True,
    }, 2)
    assert state.verification == "not_run"
    assert json.loads(state.context_summary())["unresolved_errors"][0]["stdout"] == failure
    state.observe_tool("run_tests", {"path": "tests/test_math.py"}, {
        "success": True, "status": "passed", "exit_code": 0, "stdout": "1 passed", "stderr": "",
    }, 3)
    assert state.verification == "passed" and not state.unresolved_errors
    assert len(state.test_results) == 2


def test_unrelated_success_does_not_clear_unresolved_error():
    state = memory()
    state.observe_tool("edit_file", {"path": "a.py"}, {
        "success": False, "error": {"code": "no_match", "message": "Exact old_text missing"},
    }, 1)
    state.observe_tool("read_file", {"path": "b.py"}, {"success": True, "path": "b.py"}, 2)
    assert state.unresolved_errors[0]["path"] == "a.py"


def test_passing_selected_tests_resolves_no_collection_attempt_but_not_other_test_failure():
    state = memory()
    state.observe_tool("run_tests", {"path": "tests/test_other.py"}, {
        "success": False, "status": "failed", "exit_code": 1,
        "stdout": "AssertionError: unrelated failing test", "stderr": "",
    }, 1)
    state.observe_tool("run_tests", {"path": "src/math.py"}, {
        "success": False, "status": "failed", "exit_code": 5,
        "stdout": "no tests ran", "stderr": "",
    }, 2)
    assert len(state.unresolved_errors) == 2
    state.observe_tool("run_tests", {"path": "tests/test_math.py"}, {
        "success": True, "status": "passed", "exit_code": 0,
        "stdout": "1 passed", "stderr": "",
    }, 3)
    assert state.verification == "passed"
    assert len(state.unresolved_errors) == 1
    assert state.unresolved_errors[0]["path"] == "tests/test_other.py"
    assert state.unresolved_errors[0]["exit_code"] == 1
    assert "no tests ran" not in render_progress(state)
    assert "unrelated failing test" in render_progress(state)


def test_memory_schema_roundtrip_and_strict_validation():
    state = memory()
    assert WorkingMemory.from_dict(state.to_dict()) == state
    for changes in ({"schema_version": 2}, {"last_completed_step": "2"}, {"unexpected": True}):
        with pytest.raises(ValidationError):
            WorkingMemory.from_dict({**state.to_dict(), **changes})


def test_progress_is_readable_projection_of_observed_state():
    state = memory()
    state.observe_tool("run_tests", {"path": "tests/test_math.py"}, {
        "success": False, "status": "failed", "exit_code": 1,
        "stdout": "FAILED test_add_numbers", "stderr": "",
    }, 4)
    rendered = render_progress(state)
    assert state.objective in rendered
    assert "Status: failed" in rendered and "FAILED test_add_numbers" in rendered
    assert "Last completed agent step: 4" in rendered
    assert "## Next Steps" in rendered


def test_private_checkpoint_roundtrip_and_session_reopen(tmp_path):
    store = SessionStore(tmp_path / "runs")
    state = memory(store.session_id)
    payload = {"state": {"steps": 2}, "active_messages": [{"role": "user", "content": state.objective}],
               "pending_operation": {"name": "edit_file", "arguments": {"path": "maths.py"}}}
    store.write_checkpoint(payload, state)
    reopened = SessionStore(tmp_path / "runs", store.session_id)
    loaded = reopened.load_checkpoint()
    assert loaded["state"]["steps"] == 2
    assert loaded["pending_operation"] == payload["pending_operation"]
    assert loaded["memory"] == state.to_dict()
    assert json.loads((store.directory / "state.json").read_text()) == state.to_dict()
    assert (store.directory / "TASKS.md").read_text() == render_progress(state)
    assert store.directory.stat().st_mode & 0o777 == 0o700
    assert (store.directory / "checkpoint.json").stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("session_id", ["../escape", "/tmp/escape", "", "a/b", "a" * 100])
def test_session_id_traversal_rejected(tmp_path, session_id):
    with pytest.raises(ValueError):
        SessionStore(tmp_path, session_id)


def test_session_symlink_escape_rejected(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "escape").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="symlink"):
        SessionStore(runs, "escape")


def test_corrupt_or_future_checkpoint_fails_validation(tmp_path):
    store = SessionStore(tmp_path)
    store.write_checkpoint({}, memory(store.session_id))
    path = store.directory / "checkpoint.json"
    original = path.read_text()
    for corrupted in ("not JSON", json.dumps({"schema_version": 2}),
                      original.replace(store.session_id, "different-session")):
        path.write_text(corrupted)
        with pytest.raises(ValueError):
            store.load_checkpoint()


def test_interruption_during_atomic_checkpoint_keeps_previous_checkpoint(tmp_path, monkeypatch):
    store = SessionStore(tmp_path)
    state = memory(store.session_id)
    store.write_checkpoint({"state": {"steps": 1}}, state)
    replace = os.replace

    def interrupt(source, destination):
        if Path(destination).name == "checkpoint.json":
            raise OSError("simulated interrupted commit")
        return replace(source, destination)

    monkeypatch.setattr(os, "replace", interrupt)
    with pytest.raises(OSError):
        store.write_checkpoint({"state": {"steps": 2}}, state)
    assert store.load_checkpoint()["state"]["steps"] == 1
    assert not list(store.directory.glob(".checkpoint-*"))


def test_snapshots_detect_external_edits_missing_files_and_symlink_changes(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    workspace = Workspace(root)
    (root / "math.py").write_text("return a - b")
    snapshots = snapshot_files(workspace, ["math.py", "new.py"])
    assert compare_snapshots(workspace, snapshots) == []
    (root / "math.py").write_text("return a + b")
    (root / "new.py").write_text("created externally")
    assert compare_snapshots(workspace, snapshots) == ["math.py", "new.py"]
    external = tmp_path / "secret"
    external.write_text("private")
    (root / "math.py").unlink()
    (root / "math.py").symlink_to(external)
    assert "math.py" in compare_snapshots(workspace, snapshots)


def test_oversized_files_are_not_assumed_unchanged(tmp_path):
    (tmp_path / "large").write_bytes(b"x" * 100)
    workspace = Workspace(tmp_path, ToolConfig(max_file_bytes=10))
    snapshots = snapshot_files(workspace, ["large"])
    assert snapshots["large"]["kind"] == "unverified_large"
    assert compare_snapshots(workspace, snapshots) == ["large"]


def test_skill_failures_resolve_by_name_without_hiding_another_skill():
    memory = WorkingMemory(session_id="skill-scope", objective="Consult relevant skills")
    failure = {"success": False, "error": {"code": "skill_not_loaded", "message": "Load instructions first"}}
    memory.observe_tool("read_skill", {"name": "debugging", "reference": "references/example.md"}, failure, 1)
    memory.observe_tool("read_skill", {"name": "investigation", "reference": "references/example.md"}, failure, 2)
    assert {error["skill"] for error in memory.unresolved_errors} == {"debugging", "investigation"}
    memory.observe_tool("read_skill", {"name": "debugging"}, {
        "success": True, "name": "debugging", "path": "debugging/SKILL.md",
    }, 3)
    assert [error["skill"] for error in memory.unresolved_errors] == ["investigation"]
    assert memory.loaded_skills == ["debugging"]
