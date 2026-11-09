import json

import pytest

from patchrook_agent.context.budget import estimate_tokens
from patchrook_agent.memory import SessionStore, WorkingMemory
from patchrook_agent.memory.projection import MEMORY_PREFIX


def projection_data(projection):
    assert projection.content.startswith(MEMORY_PREFIX)
    return json.loads(projection.content[len(MEMORY_PREFIX):])


def memory():
    return WorkingMemory(session_id="projection-test", objective="Fix addition; preserve multiplication.")


def test_projection_budget_preserves_current_facts_and_complete_disk_memory(tmp_path):
    state = memory()
    store = SessionStore(tmp_path)
    state.session_id = store.session_id
    state.relevant_files = [f"package/module_{index}.py" for index in range(2000)]
    state.completed_steps = [f"Read package/module_{index}.py." for index in range(500)]
    state.modified_files = ["package/module_1.py"]
    state.active_task = "Verify the updated files"
    state.remaining_steps = ["Run tests/test_module_1.py", "Report observed results"]
    state.decisions = ["Preserve the multiplication API"]
    original = state.to_dict()

    projection = state.project(max_tokens=512)
    data = projection_data(projection)
    assert not projection.over_budget and projection.estimated_tokens <= 512
    assert projection.estimated_tokens == estimate_tokens([
        {"role": "system", "content": projection.content},
    ], [])
    assert data["active_task"] == state.active_task
    assert data["modified_files"] == state.modified_files
    assert data["remaining"] == state.remaining_steps
    assert data["verification"] == "not_run"
    assert data["decisions"] == state.decisions
    assert state.objective not in projection.content
    assert "Original user request" in data["objective"]
    assert data["details"]["artifact"] == "memory"
    assert data["details"]["optional"] is True
    assert projection.omitted["relevant_files"] >= 1980
    assert projection.omitted["completed_steps"] >= 492
    assert state.to_dict() == original

    store.write_checkpoint({}, state)
    assert json.loads((store.directory / "state.json").read_text()) == original


def test_projection_preserves_exact_unresolved_failure_once():
    state = memory()
    failure = "AssertionError: expected 'a\\nb', got 'c'\n  tests/test_math.py:27"
    state.observe_tool("run_tests", {"path": "tests/test_math.py"}, {
        "success": False, "status": "failed", "exit_code": 1,
        "stdout": failure, "stderr": "RuntimeWarning: exact evidence",
    }, 3)
    projection = state.project(max_tokens=512)
    data = projection_data(projection)
    assert not projection.over_budget
    assert data["unresolved_errors"][0]["stdout"] == failure
    assert data["unresolved_errors"][0]["stderr"] == "RuntimeWarning: exact evidence"
    assert data["latest_test"]["status"] == "failed"
    assert data["latest_test"]["evidence"] == "unresolved_errors[0]"
    assert "stdout" not in data["latest_test"]


def test_identical_active_error_is_referenced_without_mutating_raw_evidence():
    state = memory()
    result = {"success": False, "status": "failed", "exit_code": 1,
              "stdout": "AssertionError: preserve this exact failure\n  test_math.py:8", "stderr": ""}
    state.observe_tool("run_tests", {"path": "tests/test_math.py"}, result, 4)
    message = {"role": "tool", "tool_name": "run_tests", "tool_call_id": "call-failed",
               "content": json.dumps(result)}
    before = state.to_dict()
    projection = state.project(max_tokens=512, active_messages=[message])
    data = projection_data(projection)
    error = data["unresolved_errors"][0]
    assert "stdout" not in error and "stderr" not in error
    assert error["exit_code"] == 1 and error["status"] == "failed"
    assert error["evidence"]["tool_call_id"] == "call-failed"
    assert data["latest_test"]["evidence"]["tool_call_id"] == "call-failed"
    assert result["stdout"] not in projection.content
    assert json.loads(message["content"]) == result
    assert state.to_dict() == before


@pytest.mark.parametrize("replacement", [
    {"success": True, "error": {"code": "missing", "message": "Exact failure"}},
    {"success": False, "error": {"code": "missing", "message": "Another failure"}},
    {"success": False},
])
def test_nonidentical_or_successful_observations_cannot_replace_exact_errors(replacement):
    state = memory()
    result = {"success": False, "error": {"code": "missing", "message": "Exact failure"}}
    state.observe_tool("read_file", {"path": "src/math.py"}, result, 2)
    projection = state.project(max_tokens=512, active_messages=[
        {"role": "tool", "tool_name": "read_file", "content": json.dumps(replacement)},
        {"role": "tool", "tool_name": "run_tests", "content": json.dumps(result)},
        {"role": "tool", "tool_name": "read_file", "content": "not JSON"},
    ])
    assert projection_data(projection)["unresolved_errors"][0]["error"] == result["error"]


def test_essential_overflow_is_reported_without_losing_error_or_constraints():
    state = memory()
    evidence = "AssertionError: unique exact line\n" * 300
    state.observe_tool("run_tests", {"path": "tests/test_math.py"}, {
        "success": False, "status": "failed", "exit_code": 1,
        "stdout": evidence, "stderr": "",
    }, 1)
    state.blockers = ["Must preserve the public API"]
    before = state.to_dict()
    projection = state.project(max_tokens=512)
    data = projection_data(projection)
    assert projection.over_budget and projection.estimated_tokens > projection.budget_tokens
    assert "memory_projection_tokens" in projection.reason
    assert data["unresolved_errors"][0]["stdout"] == evidence
    assert data["blockers"] == state.blockers
    assert state.to_dict() == before


def test_exact_latest_failure_is_retained_even_without_an_unresolved_error_record():
    state = memory()
    state.test_results = [{"path": "tests/test_math.py", "status": "failed", "exit_code": 1,
                           "step": 7, "stdout": "exact test failure", "stderr": "exact traceback"}]
    state.verification = "failed"
    data = projection_data(state.project(max_tokens=512))
    assert data["latest_test"]["stdout"] == "exact test failure"
    assert data["latest_test"]["stderr"] == "exact traceback"
    assert data["verification"] == "failed"


def test_optional_history_is_bounded_but_huge_optional_entry_cannot_starve_later_facts():
    state = memory()
    state.current_plan = ["x" * 10000, "Run focused tests"]
    state.relevant_files = ["src/math.py"]
    state.completed_steps = [f"Read file {index}" for index in range(30)]
    state.test_results = [{"status": "passed", "exit_code": 0, "step": index,
                           "stdout": "old output" * 100} for index in range(4)]
    state.verification = "passed"
    projection = state.project(max_tokens=400)
    data = projection_data(projection)
    assert not projection.over_budget and projection.estimated_tokens <= 400
    assert data["plan"] == ["Run focused tests"]
    assert data["relevant_files"] == ["src/math.py"]
    assert projection.omitted["current_plan"] == 1
    assert projection.omitted["completed_steps"] >= 22
    assert projection.omitted["test_results"] == 3
    assert "old output" not in projection.content


@pytest.mark.parametrize("budget", [0, -1, 1.5, True, "512"])
def test_projection_budget_requires_positive_integer(budget):
    with pytest.raises(ValueError, match="positive integer"):
        memory().project(max_tokens=budget)
