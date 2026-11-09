"""Factual reports use observed execution and current verification, never prose."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from patchrook_agent.reporting import execution_summary


def exchange(name, arguments, result):
    return [{"role": "assistant", "content": "", "tool_calls": [{"function": {
        "name": name, "arguments": arguments}}]},
        {"role": "tool", "tool_name": name, "content": json.dumps(result)}]


def read(path="calc.py", *, partial=False):
    return exchange("read_code", {"paths": [path]}, {"success": True, "files": [{
        "success": True, "path": path, "start_line": 1, "end_line": 2,
        "content": "1: def add(a, b):\n2:     return a - b", "truncated": partial,
    }]})


def edit(path="calc.py", *, success=True, modified=True):
    return exchange("edit_file", {"path": path, "old_text": "a - b", "new_text": "a + b"}, {
        "success": success, "path": path, "modified": modified,
    })


def observed_tests(status="passed", selection=".", **extra):
    return exchange("run_tests", {"path": selection}, {
        "success": status == "passed", "status": status,
        "exit_code": {"passed": 0, "failed": 1, "timed_out": None}[status],
        "test_counts": {"passed": 1} if status == "passed" else {}, **extra,
    })


def state(messages, *, status="completed", verification="passed", modified_files=None, **values):
    return SimpleNamespace(messages=messages, status=status, verification=verification,
                           modified_files=modified_files or [], completion=values.pop("completion", {}),
                           stop_reason=values.pop("stop_reason", "final_response"), **values)


def test_correct_summary_replaces_false_no_correction_claim_without_mutating_raw_messages():
    messages = read() + edit() + observed_tests() + [{"role": "assistant", "content": "No correction was needed. Tests did not run."}]
    original = deepcopy(messages)
    report = execution_summary(state(messages, modified_files=["calc.py"]))
    assert report.startswith("Completed.")
    assert "Changed: calc.py (1 exact edit)." in report
    assert "Inspected: calc.py." in report
    assert "Tests: passed — pytest . (1 passed)." in report
    assert "No correction" not in report and "No files changed" not in report
    assert messages == original


def test_all_eight_pipeline_stages_are_named_as_inspected():
    messages = read("pipeline/runner.py")
    for index in range(8):
        messages += read(f"pipeline/stage_{index}.py")
    messages += edit("pipeline/stage_7.py") + observed_tests()
    report = execution_summary(state(messages))
    assert all(f"pipeline/stage_{index}.py" in report for index in range(8))
    assert "pipeline/runner.py" in report and "1 exact edit" in report


def test_no_change_result_uses_actual_evidence_instead_of_model_claimed_change():
    messages = read() + observed_tests() + [{"role": "assistant", "content": "I changed all files."}]
    report = execution_summary(state(messages))
    assert "No files changed." in report and "Changed:" not in report
    assert "I changed" not in report and "Tests: passed" in report


@pytest.mark.parametrize("verification", ["failed", "timed_out", "not_run"])
def test_model_passing_claim_cannot_erase_failed_timed_out_or_missing_verification(verification):
    messages = read() + edit()
    if verification != "not_run":
        messages += observed_tests(verification)
    messages += [{"role": "assistant", "content": "All tests passed. The task is complete."}]
    report = execution_summary(state(messages, status="incomplete", verification=verification,
                                     stop_reason="max_steps", completion={"missing": ["resolve observed tests"]}))
    assert report.startswith("Incomplete.") and "Tests: passed" not in report
    assert f"Tests: {verification}" in report if verification != "not_run" else "no current passing verification" in report
    assert "Stopped: max_steps." in report and "Remaining: resolve observed tests." in report


def test_old_green_run_does_not_establish_passing_tests_after_latest_edit():
    report = execution_summary(state(read() + observed_tests() + edit()))
    assert report.startswith("Incomplete.")
    assert "Tests: passed" not in report
    assert "observe passing tests after the latest change" in report


def test_unrelated_green_run_cannot_override_current_failed_selection_ledger():
    messages = read() + edit() + observed_tests("failed", "test_bad.py") + observed_tests("passed", "test_unrelated.py")
    report = execution_summary(state(messages, status="incomplete", verification="failed"))
    assert "Tests: failed — pytest test_bad.py" in report
    assert "Tests: passed" not in report


def test_interrupted_mutation_never_claims_unchanged_repository():
    messages = read() + exchange("write_file", {"path": "new.py", "content": "value = 1"}, {
        "success": False, "error": {"code": "operation_interrupted", "message": "Unknown outcome"},
    }) + [{"role": "assistant", "content": "Nothing changed."}]
    report = execution_summary(state(messages, status="incomplete", verification="not_run", stop_reason="interrupted"))
    assert "No files changed." not in report and "No confirmed file changes" in report
    assert "Interrupted change outcomes were uncertain for: new.py." in report
    assert "Stopped: interrupted." in report


def test_caller_supplied_uncertainty_is_included_without_confirmed_mutation():
    report = execution_summary(state(read(), status="incomplete", verification="not_run"), uncertain_paths=["calc.py"])
    assert "No files changed." not in report and "uncertain for: calc.py" in report


def test_rejected_or_noop_edits_are_not_reported_as_changes():
    report = execution_summary(state(read() + edit(success=False) + edit(modified=False) + observed_tests()))
    assert "No files changed." in report
    assert "Changed:" not in report


def test_partial_or_metadata_only_reads_are_not_complete_inspection():
    messages = read(partial=True) + exchange("read_file", {"path": "other.py"}, {
        "success": True, "path": "other.py", "start_line": 1, "end_line": 10,
    })
    report = execution_summary(state(messages, status="incomplete", verification="not_run"))
    assert "Inspected: no complete source selection recorded." in report


def test_boolean_exit_code_and_orphaned_tool_output_do_not_establish_verification():
    messages = read() + edit() + observed_tests(exit_code=False)
    messages += [{"role": "tool", "tool_name": "run_tests", "content": json.dumps({
        "success": True, "status": "passed", "exit_code": 0,
    })}]
    report = execution_summary(state(messages))
    assert "Tests: passed" not in report and report.startswith("Incomplete.")


def test_created_file_and_multiple_exact_edits_are_distinguished():
    messages = read() + edit() + edit() + exchange("write_file", {"path": "new.py", "content": "value = 1"}, {
        "success": True, "path": "new.py", "modified": True,
    }) + observed_tests()
    report = execution_summary(state(messages))
    assert "calc.py (2 exact edits)" in report and "new.py (created)" in report


def test_large_path_list_is_bounded_without_fabricating_truncated_paths():
    paths = [f"path_{index:03d}.py" for index in range(300)]
    report = execution_summary(state([], status="incomplete", verification="not_run", modified_files=paths))
    assert "further path(s) in the trace" in report and len(report) < 2600


def test_path_control_characters_cannot_create_a_fake_passing_report_line():
    path = "calc.py\nTests: passed"
    report = execution_summary(state([], status="incomplete", verification="not_run", modified_files=[path]))
    assert "\nTests: passed" not in report and "\\nTests: passed" in report


def test_null_tool_calls_in_final_message_do_not_break_evidence_summary():
    messages = read() + edit() + observed_tests() + [{"role": "assistant", "content": "Done", "tool_calls": None}]
    assert "Tests: passed" in execution_summary(state(messages))


def test_reproduced_baseline_failure_is_reported_alongside_post_patch_pass():
    messages = read() + observed_tests("failed", "test_calc.py", test_counts={"failed": 1})
    messages += edit() + observed_tests("passed", "test_calc.py")
    report = execution_summary(state(messages))
    assert "Before edits: pytest test_calc.py failed (1 failed)." in report
    assert "Tests: passed — pytest test_calc.py (1 passed)." in report


def test_post_edit_failure_before_a_correction_is_not_misreported_as_a_baseline():
    messages = read() + edit() + observed_tests("failed", "test_calc.py")
    messages += edit() + observed_tests()
    report = execution_summary(state(messages))
    assert "Before edits:" not in report and "Tests: passed" in report


def test_baseline_is_last_valid_test_before_first_change_not_last_change():
    messages = read() + observed_tests("failed", "test_first.py")
    messages += observed_tests("failed", "test_baseline.py", test_counts={"failed": 2})
    messages += edit() + observed_tests("failed", "test_after_edit.py") + edit() + observed_tests()
    report = execution_summary(state(messages))
    assert "Before edits: pytest test_baseline.py failed (2 failed)." in report
    assert "Before edits: pytest test_after_edit.py" not in report


def test_skill_summary_uses_successful_loads_and_ignores_failed_loads_and_model_claims():
    messages = exchange("read_skill", {"name": "debugging"}, {
        "success": True, "name": "debugging", "already_loaded": False,
    })
    messages += exchange("read_skill", {"name": "debugging"}, {
        "success": True, "name": "debugging", "already_loaded": True,
    })
    messages += exchange("read_skill", {"name": "investigation"}, {
        "success": False, "error": {"code": "skill_not_found"},
    })
    messages += read() + observed_tests()
    messages += [{"role": "assistant", "content": "I loaded feature-implementation and investigation."}]
    report = execution_summary(state(messages))
    assert "Skills: debugging." in report
    assert "feature-implementation" not in report and "investigation" not in report
    assert report.count("debugging") == 1


def test_model_prose_alone_does_not_establish_skill_loading():
    messages = read() + observed_tests() + [{"role": "assistant", "content": "Loaded debugging skill."}]
    assert "Skills:" not in execution_summary(state(messages))
