"""Pure routing progress and checkpoint validation; no inference or tool calls."""

from copy import deepcopy
from dataclasses import asdict, replace

import pytest

from patchrook_agent.config import HarnessConfig
from patchrook_agent.model_routing import RoutingProgress, restore_routing
from patchrook_agent.models.decision import DecisionResult


SMALL, LARGE = "qwen3:8b", "qwen3:14b"


@pytest.fixture(autouse=True)
def no_model_calls(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Routing state validation attempted a model or network call")

    monkeypatch.setattr("httpx.Client", forbidden)
    monkeypatch.setattr("patchrook_agent.models.ollama.OllamaClient.chat", forbidden)


@pytest.fixture
def config():
    return replace(HarnessConfig(), workflow_mode="evidence", decision_model="on", model_routing="tev",
                   evidence_test_commands=(("python", "-m", "pytest"),))


def checkpoint(*, switched=False, decided=True):
    return {"schema_version": 2, "initial_model": SMALL,
            "active_model": LARGE if switched else SMALL, "escalation_model": LARGE,
            "initial_decided": decided,
            "initial_route": {"label": "small", "called": False,
                              "fallback_reason": "interrupted_routing"} if decided else None,
            "pending_switch": None,
            "switches": [{"from_model": SMALL, "to_model": LARGE, "reason": "failed_repairs",
                          "step": 4, "management_seconds": .1}] if switched else [],
            "model_calls": {}, "model_usage": {}, "progress": RoutingProgress().to_dict(),
            "repair": {"status": "active" if switched else "idle", "reason": "failed_repairs" if switched else "",
                       "calls_used": 0, "call_inflight": False, "pending_checks": [],
                       "last_revision": 0, "source_ranges": []}}


def restore(value, config, *, steps=8, attempts=9):
    return restore_routing(value, config, value["active_model"], steps, attempts)


@pytest.mark.parametrize("switched,decided", [(False, False), (False, True), (True, True)])
def test_valid_routing_restore_is_detached_and_preserves_switch_identity(config, switched, decided):
    value = checkpoint(switched=switched, decided=decided)
    original = deepcopy(value)
    route, progress = restore(value, config)
    assert value == original and "progress" not in route
    assert isinstance(progress, RoutingProgress)
    assert route["active_model"] == (LARGE if switched else SMALL)
    route["switches"].clear()
    progress.observations.append("0" * 64)
    assert value == original


def test_pending_switch_restores_without_executing_or_charging_it(config):
    value = checkpoint()
    value["pending_switch"] = {"target": LARGE, "reason": "invalid_responses"}
    value["repair"].update(status="requested", reason="invalid_responses")
    route, _ = restore(value, config)
    assert route["pending_switch"] == value["pending_switch"]
    assert route["active_model"] == SMALL and route["switches"] == []


def test_model_usage_distinguishes_unknown_measurements_and_measured_zero(config):
    value = checkpoint()
    value["model_calls"] = {SMALL: 2}
    value["model_usage"] = {SMALL: {"prompt_tokens": None, "completion_tokens": 0,
                                    "inference_seconds": 1.0, "load_seconds": None,
                                    "prompt_token_calls": 0, "completion_token_calls": 1,
                                    "load_calls": 0}}
    route, _ = restore(value, config)
    assert route["model_usage"][SMALL]["prompt_tokens"] is None
    assert route["model_usage"][SMALL]["completion_tokens"] == 0
    value["model_usage"][SMALL]["prompt_token_calls"] = 1
    with pytest.raises(ValueError, match="coverage"):
        restore(value, config)


def test_real_decision_timeout_fallback_with_unknown_scores_can_resume(config):
    value = checkpoint()
    result = DecisionResult(("small", "large", "uncertain"), {}, fallback_reason="timeout", called=True)
    value["initial_route"] = {**asdict(result), "label": "small"}
    route, _ = restore(value, config)
    assert route["initial_route"]["fallback_reason"] == "timeout"
    assert route["initial_route"]["scores"] == {}
    value["initial_route"]["label"] = "large"
    with pytest.raises(ValueError, match="base coder"):
        restore(value, config)


def test_invalid_response_audit_is_bounded_but_does_not_authorize_large_model(config):
    value = checkpoint()
    value["initial_route"] = {"label": "small", "called": True, "fallback_reason": "invalid_routing_response",
                              "ranking": ["bad", "bad"], "scores": {"bad": .5}}
    route, _ = restore(value, config)
    assert route["active_model"] == SMALL
    value["initial_route"]["ranking"] = ["x"] * 13
    with pytest.raises(ValueError, match="ranking"):
        restore(value, config)


@pytest.mark.parametrize("key,value", [
    ("schema_version", True), ("schema_version", 1), ("schema_version", 3),
    ("initial_model", LARGE), ("initial_model", "remote:cloud"),
    ("initial_model", "bad\nname"), ("active_model", "other:8b"),
    ("escalation_model", "other:14b"), ("initial_decided", 1),
    ("initial_route", None), ("initial_route", {"label": "invalid", "called": False}),
    ("initial_route", {"label": "small", "called": 1}),
    ("initial_route", {"label": "small", "called": False, "elapsed_seconds": float("nan")}),
    ("initial_route", {"label": "small", "called": False, "scores": {"other": .5}}),
    ("initial_route", {"label": "small", "called": False, "ranking": ["small"]}),
    ("initial_route", {"label": "small", "called": False, "unknown": True}),
    ("pending_switch", {"target": SMALL, "reason": "failed_repairs"}),
    ("pending_switch", {"target": LARGE, "reason": "resume_any_model"}),
    ("switches", [{"from_model": SMALL, "to_model": LARGE, "reason": "failed_repairs",
                   "step": 100, "management_seconds": 0}]),
    ("model_calls", {SMALL: True}), ("model_calls", {SMALL: -1}),
    ("model_calls", {SMALL: 10}), ("model_calls", {"unowned-model": 1}),
    ("model_usage", {SMALL: {"prompt_tokens": -1, "completion_tokens": 0,
                             "inference_seconds": 0, "load_seconds": 0,
                             "prompt_token_calls": 1, "completion_token_calls": 1, "load_calls": 1}}),
    ("model_usage", {SMALL: {"prompt_tokens": 0, "completion_tokens": 0,
                             "inference_seconds": float("inf"), "load_seconds": 0,
                             "prompt_token_calls": 1, "completion_token_calls": 1, "load_calls": 1}}),
    ("model_usage", {SMALL: {"prompt_tokens": 0, "completion_tokens": 0,
                             "inference_seconds": 0, "load_seconds": -1,
                             "prompt_token_calls": 1, "completion_token_calls": 1, "load_calls": 1}}),
    ("progress", {"invalid_responses": 9, "repeated_actions": 0, "observations": [], "failed_repairs": []}),
])
def test_malformed_checkpoint_rejected_without_model_or_network(config, key, value):
    data = checkpoint()
    data[key] = value
    with pytest.raises(ValueError):
        restore(data, config)


def test_extra_fields_and_model_mismatch_rejected(config):
    value = checkpoint()
    with pytest.raises(ValueError):
        restore(value | {"extra": True}, config)
    with pytest.raises(ValueError):
        restore_routing(value, config, LARGE, 8, 9)
    with pytest.raises(ValueError):
        restore_routing(value, config, SMALL, True, 9)


def test_switch_history_must_be_forward_only_and_match_active_model(config):
    value = checkpoint(switched=True)
    value["switches"][0]["from_model"] = LARGE
    with pytest.raises(ValueError):
        restore(value, config)
    value = checkpoint(switched=True)
    value["switches"].append(deepcopy(value["switches"][0]))
    with pytest.raises(ValueError):
        restore(value, config)
    value = checkpoint(switched=True)
    value["pending_switch"] = {"target": LARGE, "reason": "failed_repairs"}
    with pytest.raises(ValueError):
        restore(value, config)
    value = checkpoint(switched=True)
    value["switches"] = []
    with pytest.raises(ValueError):
        restore(value, config)


def completed_repair(*, returned=False):
    value = checkpoint(switched=True)
    value["repair"].update(status="verified", calls_used=1, last_revision=1,
                           source_ranges=[{"path": "calc.py", "start_line": 1, "end_line": 8}])
    value["model_calls"] = {SMALL: 3, LARGE: 1}
    if returned:
        value["active_model"] = SMALL
        value["switches"].append({"from_model": LARGE, "to_model": SMALL, "reason": "repair_verified",
                                   "step": 5, "management_seconds": .1})
    return value


@pytest.mark.parametrize("returned", [False, True])
def test_verified_repair_can_restore_before_or_after_return_to_base(config, returned):
    value = completed_repair(returned=returned)
    if not returned:
        value["pending_switch"] = {"target": SMALL, "reason": "repair_verified"}
    route, _ = restore(value, config)
    assert route["repair"]["status"] == "verified"
    assert route["active_model"] == (SMALL if returned else LARGE)


def test_one_repair_episode_cannot_return_before_verification_or_reenter(config):
    value = completed_repair(returned=True)
    value["repair"]["status"] = "active"
    with pytest.raises(ValueError, match="verified"):
        restore(value, config)
    value = completed_repair(returned=True)
    value["pending_switch"] = {"target": LARGE, "reason": "failed_repairs"}
    with pytest.raises(ValueError, match="pending"):
        restore(value, config)


def test_consumed_unknown_generations_stay_charged_across_resume(config):
    value = checkpoint(switched=True)
    value["repair"].update(calls_used=2, call_inflight=False)
    value["switches"][0]["step"] = 1
    value["model_calls"] = {SMALL: 1}
    route, _ = restore(value, config, steps=1, attempts=3)
    assert route["repair"]["calls_used"] == 2
    with pytest.raises(ValueError, match="episode"):
        restore(value, config, steps=1, attempts=1)
    with pytest.raises(ValueError, match="episode"):
        restore(value, replace(config, routing_max_repair_calls=1))


def test_pending_verification_restores_only_trusted_named_checks(config):
    value = checkpoint(switched=True)
    value["repair"].update(calls_used=1, pending_checks=["check_1"], last_revision=1)
    route, _ = restore(value, config)
    assert route["repair"]["pending_checks"] == ["check_1"]
    for checks in (["check_2"], ["check_1", "check_1"], "check_1"):
        value["repair"]["pending_checks"] = checks
        with pytest.raises(ValueError, match="checks"):
            restore(value, config)


@pytest.mark.parametrize("field,bad", [
    ("calls_used", 3), ("calls_used", True), ("call_inflight", 1),
    ("status", "exploring"), ("last_revision", -1),
    ("source_ranges", [{"path": "../outside.py", "start_line": 1, "end_line": 2}]),
    ("source_ranges", [{"path": "/outside.py", "start_line": 1, "end_line": 2}]),
    ("source_ranges", [{"path": "calc.py", "start_line": 2, "end_line": 1}]),
    ("source_ranges", [{"path": "calc.py", "start_line": True, "end_line": 2}]),
])
def test_malformed_repair_packet_state_rejected_without_execution(config, field, bad):
    value = checkpoint(switched=True)
    value["repair"][field] = bad
    with pytest.raises(ValueError):
        restore(value, config)


def test_repair_call_and_pending_checks_phases_cannot_overlap(config):
    value = checkpoint(switched=True)
    value["repair"].update(calls_used=1, call_inflight=True)
    assert restore(value, config)[0]["repair"]["call_inflight"] is True
    value["repair"]["pending_checks"] = ["check_1"]
    with pytest.raises(ValueError, match="generation"):
        restore(value, config)
    value["repair"].update(status="exhausted", call_inflight=False)
    with pytest.raises(ValueError, match="checks|Terminal"):
        restore(value, config)


def test_native_multiple_reads_per_step_remain_valid_checkpoint_progress(config):
    value = checkpoint()
    value["progress"]["repeated_actions"] = 2
    restore(value, config, steps=1, attempts=1)


def failed_check(**options):
    return {"success": False, "exit_code": 1, "stdout": "FAILED test_total.py::test_total\n",
            "verification_kind": "configured_command", **options}


def test_failed_baseline_and_duplicate_check_do_not_count_as_two_repairs(config):
    progress = RoutingProgress()
    args = {"name": "check_1"}
    assert progress.observe("run_check", args, failed_check(), 0, config) is None
    assert progress.failed_repairs == []
    assert progress.observe("run_check", args, failed_check(), 1, config) is None
    assert progress.observe("run_check", args, failed_check(), 1, config) is None
    assert len(progress.failed_repairs) == 1
    assert progress.observe("run_check", args, failed_check(), 2, config) == "failed_repairs"


@pytest.mark.parametrize("failure", [
    {"exit_code": 2, "stdout": "ERROR collecting tests"},
    {"exit_code": 1, "stderr": "ModuleNotFoundError: No module named dependency"},
    {"exit_code": 1, "timed_out": True}, {"exit_code": 137},
    {"error": {"code": "test_permission_denied"}},
])
def test_environment_failures_never_trigger_repair_escalation(config, failure):
    progress = RoutingProgress()
    for revision in range(1, 8):
        assert progress.observe("run_check", {"name": "check_1"}, failed_check(**failure), revision, config) is None
    assert progress.failed_repairs == [] and progress.repeated_actions == 0


def test_repeated_reads_ignore_volatile_ids_but_new_source_is_progress(config):
    progress = RoutingProgress()
    args = {"path": "calc.py", "question": "First wording"}
    source = {"success": True, "content": "return a - b", "source_sha256": "a" * 64}
    assert progress.observe("read_file", args, source | {"evidence_id": "1"}, 0, config) is None
    for index in range(1, 4):
        result = progress.observe("read_file", args | {"question": "Same need"},
                                  source | {"evidence_id": str(index), "duration_seconds": index}, 0, config)
    assert result == "repeated_actions"
    assert progress.observe("read_file", args, source | {"content": "return a + b"}, 1, config) is None
    assert progress.repeated_actions == 0


def test_search_page_sizes_and_default_offsets_do_not_create_new_evidence(config):
    progress = RoutingProgress()
    matches = [{"path": "calc.py", "line_number": number, "text": f"value_{number}"}
               for number in range(1, 4)]
    assert progress.observe("search_code", {"pattern": "value", "path": ".", "max_results": 10},
                            {"success": True, "matches": matches}, 0, config) is None
    for args, observed in [({"max_results": 2}, matches[:2]),
                           ({"max_results": 10, "offset": 0}, matches),
                           ({"max_results": 3, "question": "Different wording"}, matches)]:
        reason = progress.observe("search_code", {"pattern": "value", "path": ".", **args},
                                  {"success": True, "matches": observed}, 0, config)
    assert reason == "repeated_actions"
    assert progress.observe("search_code", {"pattern": "new"}, {"success": True, "matches": [
        {"path": "other.py", "line_number": 1, "text": "new"}]}, 0, config) is None


def test_overlapping_reads_only_reset_progress_when_they_add_source(config):
    progress = RoutingProgress()
    def read(first, lines):
        return progress.observe("read_file", {"path": "calc.py", "start_line": first},
            {"success": True, "path": "calc.py", "start_line": first,
             "source_sha256": "a" * 64, "content": "\n".join(f"{index}: line{index}" for index in lines)}, 0, config)
    read(1, range(1, 6))
    read(2, range(2, 5))
    assert progress.repeated_actions == 1
    read(4, range(4, 7))
    assert progress.repeated_actions == 0
    read(1, range(1, 7))
    assert progress.repeated_actions == 1


def test_invalid_response_threshold_and_successful_mutation_reset(config):
    progress = RoutingProgress()
    assert progress.invalid(config) is None
    assert progress.invalid(config) == "invalid_responses"
    assert progress.observe("apply_edits", {}, {"success": True, "modified_files": ["calc.py"]}, 1, config) is None
    assert progress.invalid_responses == progress.repeated_actions == 0


@pytest.mark.parametrize("field,value", [("observations", ["not-a-hash"]),
                                       ("failed_repairs", [["check_1", 0, "0" * 64]]),
                                       ("repeated_actions", True)])
def test_malformed_progress_rejected(field, value):
    data = RoutingProgress().to_dict()
    data[field] = value
    with pytest.raises(ValueError):
        RoutingProgress.from_dict(data)
