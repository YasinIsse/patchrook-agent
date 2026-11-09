"""Only byte-identical check output already present can become a reference."""

from copy import deepcopy
import json
import time

from patchrook_agent.evidence_feedback import repair_action_correction
from test_evidence_repair_optional import runtime


def failure():
    return {"success": False, "kind": "command_check", "verification_kind": "configured_command", "name": "check_1",
        "argv": ["python", "public.py"], "status": "exited", "exit_code": 1, "timed_out": False,
        "stdout": '{"failed":18,"representative_failure":{"data":{"count":"3"}}}', "stderr": "",
        "duration_seconds": .2, "truncated": False}


def correction(result):
    return "Observed action failure: " + json.dumps(result)


def test_exact_check_output_becomes_reference_without_mutating_any_original_data():
    result = failure()
    observed = [{key: result[key] for key in ("name", "exit_code", "stdout", "stderr")}]
    before = deepcopy((result, observed))
    original = correction(result)
    shown = repair_action_correction(original, observed, {"check_1": ("python", "public.py")})
    assert "failed_checks" in shown and result["stdout"] not in shown
    assert (result, observed) == before and correction(result) == original


def test_noncheck_and_nonidentical_or_incomplete_receipts_are_preserved_exactly():
    result = failure()
    checks = [{key: result[key] for key in ("name", "exit_code", "stdout", "stderr")}]
    commands = {"check_1": ("python", "public.py")}
    for changed in ({"stdout": "DIFFERENT ASSERTION"}, {"stderr": "unique failure"}, {"argv": ["python", "other.py"]},
                    {"timed_out": True}, {"truncated": True}, {"diagnostic": "unique information"},
                    {"error": {"code": "actual_error", "message": "preserve this"}}):
        value = {**result, **changed}
        text = correction(value)
        assert repair_action_correction(text, checks, commands) == text
    for text in ("Copy exact fresh source before editing.", "Model response done_reason=length: no action.",
                 "Observed action failure: not JSON", correction({"error": {"code": "no_match", "message": "preserve"}})):
        assert repair_action_correction(text, checks, commands) == text


def test_repair_packet_keeps_exact_failure_and_internal_correction_but_renders_output_once(tmp_path):
    value = runtime(tmp_path, failure="EXACT_FAILED_ASSERTION")
    result = failure()
    result.update(argv=list(value.tools.commands["check_1"]), stdout="EXACT_FAILED_ASSERTION")
    value.correction = correction(result)
    original = value.correction
    messages = value._prepare_repair(value.state, value._repair_definitions(), time.monotonic() + 10, record=False)
    prefix = "Observed repair state (data):\n"
    shown = json.loads(messages[-1]["content"][len(prefix):])
    assert shown["failed_checks"][0]["stdout"] == "EXACT_FAILED_ASSERTION"
    assert "EXACT_FAILED_ASSERTION" not in shown["action_correction"]
    assert value.correction == original and value.routing["repair"]["calls_used"] == value.attempts == 0
    assert not value.tools._presented
