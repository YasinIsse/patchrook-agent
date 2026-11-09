import json

import pytest
from pydantic import ValidationError

from patchrook_agent.memory import SessionStore, WorkingMemory, render_progress
from patchrook_agent.task_state import TaskState, normalized_test_outcome


def failed(label="rounding", seconds="0.02", *, selection="."):
    return {"success": False, "status": "failed", "exit_code": 1,
            "stdout": f"E   assert {label}\nFAILED tests/test_checkout.py::test_total\n1 failed, 2 passed in {seconds}s",
            "stderr": "", "duration_seconds": float(seconds), "path": selection,
            "test_counts": {"failed": 1, "passed": 2}}


def passed(selection="."):
    return {"success": True, "status": "passed", "exit_code": 0, "path": selection,
            "stdout": "3 passed in 0.03s", "stderr": "", "test_counts": {"passed": 3}}


def edit(state, old="ceil", new="ceil - 1", *, path="promotions.py", step=1):
    state.observe_action("edit_file", {"path": path, "old_text": old, "new_text": new},
                         {"success": True, "modified": True, "path": path}, step=step)


def test_legacy_memory_defaults_and_task_state_checkpoint_roundtrip(tmp_path):
    memory = WorkingMemory(session_id="task-ledger", objective="Fix returns and rounding.")
    legacy = memory.to_dict()
    legacy.pop("task_state")
    assert WorkingMemory.from_dict(legacy).task_state == TaskState()
    memory.task_state.active_goal = "Fix percentage rounding"
    edit(memory.task_state)
    memory.task_state.observe_action("run_tests", {}, failed(), step=2, source_digest="A")
    store = SessionStore(tmp_path)
    memory.session_id = store.session_id
    store.write_checkpoint({}, memory)
    restored = WorkingMemory.from_dict(store.load_checkpoint()["memory"])
    assert restored.task_state == memory.task_state
    assert len(restored.task_state.attempts) == 1
    assert memory.verification == "not_run"  # ledger cannot assert WorkingMemory verification


def test_protocol_retry_accounting_is_durable_and_separate_from_strategic_rounds(tmp_path):
    memory = WorkingMemory(session_id="protocol-retry", objective="Fix the observed calculation.")
    memory.task_state.phase = "propose"
    memory.task_state.episodes = 1
    memory.task_state.rounds = 1
    memory.task_state.protocol_retries = 1
    store = SessionStore(tmp_path)
    memory.session_id = store.session_id
    store.write_checkpoint({}, memory)
    restored = WorkingMemory.from_dict(store.load_checkpoint()["memory"])
    assert restored.task_state.protocol_retries == 1
    assert restored.task_state.rounds == 1 and restored.task_state.episodes == 1
    assert restored.task_state.brief()["protocol_retries"] == 1
    restored.task_state.clear_pending("Workspace changed outside the harness")
    assert restored.task_state.protocol_retries == 0 and restored.task_state.rounds == 0
    assert "protocol_retries" not in restored.task_state.brief()


def test_legacy_task_state_without_protocol_retry_counter_defaults_to_zero():
    saved = TaskState(phase="propose", rounds=1, episodes=1).model_dump()
    saved.pop("protocol_retries")
    restored = TaskState.model_validate(saved)
    assert restored.protocol_retries == 0 and restored.rounds == 1
    assert "protocol_retries" not in restored.brief()


@pytest.mark.parametrize("value", [-1, 2, True, "1", 1.5])
def test_protocol_retry_counter_is_strictly_bounded_on_restore_and_assignment(value):
    with pytest.raises(ValidationError):
        TaskState.model_validate({**TaskState().model_dump(), "protocol_retries": value})
    state = TaskState()
    with pytest.raises(ValidationError):
        state.protocol_retries = value
    assert state.protocol_retries == 0


@pytest.mark.parametrize("changes", [
    {"phase": "completed"}, {"rounds": 3}, {"rounds": True}, {"episodes": 9},
    {"evidence_reads": 2}, {"evidence_reads": "1"}, {"proposal_revision": -1},
    {"history_omitted": "0"}, {"proposals": [{}, {}, {}, {}]}, {"selected": []},
    {"selected": {"path": []}}, {"proposals": [{"approach": 42}]},
    {"proposal_evidence": {"bad": {"not JSON"}}}, {"attempts": {"goal": True}},
    {"attempts": {str(index): 1 for index in range(17)}},
    {"attempt_selections": {"goal": {".": -1}}},
    {"pending_changes": [{"path": "a.py", "step": 1}]},
    {"failed_states": [{"source_digest": "claimed"}]},
    {"history": [{"event": "action", "step": True}]},
    {"history": [{"event": "action", "step": 1}] * 97},
    {"unexpected": "value"},
])
def test_task_state_strict_restore_rejects_malformed_or_oversized_data(changes):
    with pytest.raises(ValidationError):
        TaskState.model_validate({**TaskState().model_dump(), **changes})


def test_all_completed_repeats_are_recorded_but_bounded_and_noops_are_not_edits():
    state = TaskState()
    for step in range(100):
        state.observe_action("edit_file", {"path": "checkout.py", "old_text": "same", "new_text": "same"},
                             {"success": False, "error": {"code": "no_change"}}, step=step)
    assert len(state.history) == 96 and state.history_omitted == 4
    assert [item["step"] for item in state.history[:2]] == [4, 5]
    assert all(item["error"] == "no_change" for item in state.history)
    assert not state.pending_changes and state.mutation == 0 and not state.trigger
    assert len(state.brief()["recent_actions"]) == 6


def test_baseline_failure_and_identical_reruns_do_not_count_repair_attempts():
    state = TaskState(active_goal="Fix rounding")
    state.observe_action("run_tests", {}, failed(seconds="0.02"), step=0, source_digest="A")
    state.observe_action("run_tests", {}, failed(seconds="1.37"), step=1, source_digest="A")
    assert not state.attempts and not state.trigger
    assert len(state.failed_states) == 1
    assert len(state.history) == 2
    assert state.failed_states[0]["changes"] == []


def test_two_failed_actual_attempts_same_goal_require_review():
    state = TaskState(active_goal="Fix rounding")
    edit(state)
    state.observe_action("run_tests", {}, failed("6601 == 6600"), step=2, source_digest="A")
    assert not state.trigger and list(state.attempts.values()) == [1]
    edit(state, old="ceil - 1", new="ceil - 2", step=3)
    state.observe_action("run_tests", {}, failed("6602 == 6600"), step=4, source_digest="B")
    assert state.trigger == "failed_attempts" and list(state.attempts.values()) == [2]
    assert state.pending_changes == []


def test_failed_patch_cycle_is_detected_across_goals_and_retains_both_witnesses():
    state = TaskState(active_goal="Initial rounding theory")
    state.observe_action("run_tests", {}, failed("4951 == 4952"), step=0, source_digest="original")
    edit(state, new="ceil - 1")
    state.observe_action("run_tests", {}, failed("6601 == 6600"), step=2, source_digest="minus-one")
    state.active_goal = "Try original operation again"
    edit(state, old="ceil - 1", new="ceil", step=3)
    state.observe_action("run_tests", {}, failed("4951 == 4952", seconds="9.1"),
                         step=4, source_digest="original")
    assert state.trigger == "repeated_failed_state"
    assert len(state.failed_states) == 2
    brief = json.dumps(state.brief(), ensure_ascii=False)
    assert "6601 == 6600" in brief and "4951 == 4952" in brief
    assert "ceil - 1" in brief and '"ceil"' in brief


def test_state_without_a_measured_source_hash_does_not_claim_cycle():
    state = TaskState(active_goal="Fix")
    state.observe_action("run_tests", {}, failed(), step=0)
    edit(state)
    state.observe_action("run_tests", {}, failed(), step=2)
    assert not state.trigger
    assert list(state.attempts.values()) == [1]


def test_goal_paraphrase_does_not_reset_attempts_for_the_same_changed_path():
    state = TaskState(active_goal="Try rounding idea one")
    edit(state)
    state.observe_action("run_tests", {}, failed("first"), step=2, source_digest="A")
    assert state.active_goal == "promotions.py"
    state.active_goal = "A completely different name for the same edit"
    edit(state, new="other formula", step=3)
    state.observe_action("run_tests", {}, failed("second"), step=4, source_digest="B")
    assert state.trigger == "failed_attempts" and list(state.attempts.values()) == [2]
    assert state.failed_states[-1]["stable_goal"] == "promotions.py"
    assert state.failed_states[-1]["paths"] == ["promotions.py"]


def test_multiple_edits_before_one_test_are_one_attempt_per_goal_path():
    state = TaskState(active_goal="Fix rounding")
    edit(state)
    edit(state, old="ceil - 1", new="ceil - 2", step=2)
    edit(state, path="adjustments.py", step=3)
    state.observe_action("run_tests", {}, failed(), step=4, source_digest="A")
    assert list(state.attempts.values()) == [1, 1]
    assert len(state.failed_states[-1]["changes"]) == 3
    assert not state.trigger


def test_timing_noise_is_normalized_but_failure_text_and_amounts_are_not():
    first = failed(seconds="0.02")
    second = {**failed(seconds="13.47"), "prompt_eval_seconds": 990, "duration_seconds": 1000}
    assert normalized_test_outcome(first) == normalized_test_outcome(second)
    assert normalized_test_outcome(first) != normalized_test_outcome(failed("new failure"))
    first["stderr"] = '/tmp/patchrook-worktree-a12b/tests/test_checkout.py:10'
    second["stderr"] = '/tmp/patchrook-worktree-c34d/tests/test_checkout.py:10'
    assert normalized_test_outcome(first) == normalized_test_outcome(second)


def test_full_suite_pass_clears_failure_debt_but_preserves_action_audit():
    state = TaskState(active_goal="Fix rounding")
    edit(state)
    state.observe_action("run_tests", {}, failed(), step=2, source_digest="A")
    edit(state, old="ceil - 1", new="floor", step=3)
    state.observe_action("run_tests", {}, passed(), step=4, source_digest="B")
    assert not state.trigger and not state.failed_states and not state.attempts
    assert not state.pending_changes and len(state.history) == 4
    assert state.history[-1]["outcome"] == "passed"
    state.observe_action("run_tests", {}, failed(), step=5, source_digest="A")
    assert not state.trigger  # a prior resolved episode cannot manufacture a new cycle


def test_specific_test_pass_cannot_clear_other_selection_debt():
    state = TaskState(active_goal="Fix shared helper")
    edit(state)
    state.observe_action("run_tests", {"path": "tests/test_a.py"}, failed(selection="tests/test_a.py"),
                         step=2, source_digest="A")
    edit(state, new="different")
    state.observe_action("run_tests", {"path": "tests/test_b.py"}, failed(selection="tests/test_b.py"),
                         step=4, source_digest="B")
    assert state.trigger == "failed_attempts"
    state.observe_action("run_tests", {}, passed("tests/test_a.py"), step=5, source_digest="B")
    assert [item["selection"] for item in state.failed_states] == ["tests/test_b.py"]
    assert list(state.attempts.values()) == [1]
    assert not state.trigger


def test_unrelated_passing_selection_cannot_erase_an_observed_failed_cycle():
    state = TaskState(active_goal="Fix rounding")
    state.observe_action("run_tests", {}, failed(selection="tests/test_a.py"), step=0, source_digest="A")
    edit(state)
    state.observe_action("run_tests", {}, failed(selection="tests/test_a.py"), step=2, source_digest="A")
    assert state.trigger == "repeated_failed_state" and list(state.attempts.values()) == [1]
    state.observe_action("run_tests", {}, passed("tests/test_b.py"), step=3, source_digest="A")
    assert state.trigger == "repeated_failed_state"
    state.observe_action("run_tests", {}, passed("tests/test_a.py"), step=4, source_digest="B")
    assert not state.trigger and not state.failed_states


def test_external_change_invalidates_pending_selection_and_cycle_comparisons():
    state = TaskState(active_goal="Fix", phase="selected", selected={"id": "p1"},
                      proposals=[{"id": "p1"}], proposal_evidence={"digest": "original"}, rounds=1)
    state.observe_action("run_tests", {}, failed(), step=0, source_digest="A")
    edit(state)
    state.clear_pending("Workspace source changed outside the harness")
    assert state.pending_changes == [] and state.selected is None and state.proposals == []
    assert state.phase == "direct" and state.rounds == 0 and state.proposal_evidence == {}
    assert state.comparison_epoch == 1 and state.proposal_revision == 1
    assert state.history[-1]["event"] == "invalidated"
    edit(state, step=3)
    state.observe_action("run_tests", {}, failed(), step=4, source_digest="A")
    assert not state.trigger and list(state.attempts.values()) == [1]


def test_brief_is_bounded_preserves_two_distinct_failures_and_never_mutates_full_state():
    state = TaskState(active_goal="A" * 512, review_reason="B" * 512, phase="selected",
                      selected={"id": "proposal-a", "goal": "G" * 512})
    for index in range(4):
        edit(state, old="x" * 320, new="y" * 320, path=f"{index}.py", step=2 * index)
        state.observe_action("run_tests", {}, failed(f"variant-{index}"),
                             step=2 * index + 1, source_digest=f"source-{index}")
    original = state.model_dump()
    for budget in (512, 900, 1800):
        brief = state.brief(max_chars=budget)
        assert len(json.dumps(brief, ensure_ascii=False, separators=(",", ":"), sort_keys=True)) <= budget
        assert len(brief["failed_witnesses"]) == 2
        assert [item["source"] for item in brief["failed_witnesses"]] == ["source-2", "source-3"]
    assert state.model_dump() == original


def test_large_write_is_explicit_excerpt_not_fabricated_exact_current_source():
    state = TaskState()
    state.observe_action("write_file", {"path": "big.py", "content": "a" * 3000},
                         {"success": True, "modified": True}, step=1)
    change = state.pending_changes[-1]
    assert "accepted_delta" not in change
    assert change["accepted_delta_excerpt"]["content"] == {
        "head": "a" * 80, "tail": "a" * 80, "characters": 3000}
    assert "current_source" not in json.dumps(state.model_dump())


def test_failed_selected_implementation_is_linked_to_observed_outcome():
    selected = {"id": "p2", "kind": "implement", "path": "promotions.py",
                "approach": "Subtract a cent", "expected": "All totals pass", "status": "executed"}
    state = TaskState(phase="selected", active_goal="Fix rounding", selected=selected)
    edit(state)
    state.observe_action("run_tests", {}, failed(), step=2, source_digest="A")
    assert state.failed_states[-1]["proposal"] == {key: value for key, value in selected.items() if key != "status"}
    assert state.failed_states[-1]["outcome"]["status"] == "failed"
    assert state.failed_states[-1]["source_digest"] == "A"


def test_failed_test_rerun_after_restore_retains_failed_proposal_and_edit_witness():
    selected = {"id": "subtract_one", "kind": "implement_change", "path": "promotions.py",
                "approach": "Subtract a cent from the rounded discount", "expected": "Every total passes",
                "status": "executed"}
    state = TaskState(phase="selected", active_goal="promotions.py", selected=selected)
    edit(state)
    state.observe_action("run_tests", {}, failed("6601 == 6600"), step=2, source_digest="A")
    previous = state.failed_states[-1]
    restored = TaskState.model_validate(state.model_dump())
    restored.selected["status"] = "observed_failed"
    restored.phase = "propose"
    restored.observe_action("run_tests", {}, failed("6601 == 6600", seconds="4.73"),
                            step=3, source_digest="A")
    assert len(restored.failed_states) == 1
    retained = restored.failed_states[-1]
    for key in ("proposal", "changes", "attempt_keys", "stable_goal", "paths"):
        assert retained[key] == previous[key]
    assert retained["step"] == 3 and list(restored.attempts.values()) == [1]
    assert not restored.trigger and restored.pending_changes == []
    view = json.dumps(restored.brief())
    assert "subtract_one" in view and "ceil - 1" in view and "6601 == 6600" in view


def test_new_failed_edit_does_not_inherit_a_previous_proposal_association():
    selected = {"id": "first", "kind": "implement_change", "path": "promotions.py",
                "approach": "Subtract a cent", "expected": "Totals pass", "status": "executed"}
    state = TaskState(phase="selected", active_goal="promotions.py", selected=selected)
    edit(state)
    state.observe_action("run_tests", {}, failed(), step=2, source_digest="A")
    state.phase = "direct"
    state.selected["status"] = "observed_failed"
    edit(state, old="ceil - 1", new="ceil", step=3)
    state.observe_action("run_tests", {}, failed(), step=4, source_digest="A")
    assert "proposal" not in state.failed_states[-1]
    assert state.failed_states[-1]["changes"][-1]["accepted_delta"]["new_text"] == "ceil"


def test_task_ledger_appears_in_progress_and_budgeted_memory_projection():
    memory = WorkingMemory(session_id="ledger-view", objective="Fix without changing tax policy")
    memory.task_state.active_goal = "Fix percentage rounding"
    memory.task_state.observe_action("read_file", {"path": "promotions.py", "start_line": 38, "end_line": 45},
                                     {"success": True, "path": "promotions.py"}, step=1)
    memory.task_state.record_event("selected", step=1, proposal_id="p1", reason="Observed fractional rounding failure")
    summary = json.loads(memory.context_summary())
    assert summary["task_state"]["active_goal"] == "Fix percentage rounding"
    projected = memory.project(max_tokens=512)
    assert not projected.over_budget
    assert "task_state" in projected.content
    rendered = render_progress(memory)
    assert "## Recent Actions" in rendered and "promotions.py" in rendered and "selected" in rendered


def test_optional_recent_actions_shrink_before_the_memory_projection_overflows():
    memory = WorkingMemory(session_id="ledger-budget", objective="Preserve source constraints")
    memory.task_state.active_goal = "Investigate " + "a" * 200
    for step in range(6):
        memory.task_state.record_event("selected", step=step, proposal_id=f"p{step}",
                                      reason="Observed need " + "b" * 180)
    projection = memory.project(max_tokens=400)
    assert not projection.over_budget
    assert projection.estimated_tokens <= 400
    assert memory.task_state.active_goal in json.dumps(memory.task_state.model_dump())
    assert len(memory.task_state.history) == 6
