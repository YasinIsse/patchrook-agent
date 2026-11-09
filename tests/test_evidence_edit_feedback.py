"""Rejected edits retain bounded proposal facts and use existing repair gates."""

import hashlib
import json

import pytest

from patchrook_agent.config import HarnessConfig
from patchrook_agent.evidence_workflow import _rejected_edit_summary
from patchrook_agent.model_routing import RoutingProgress
from test_model_routing_workflow import Coder, LARGE, SMALL, action, check, edit, read, route, setup, text, tool_results


@pytest.fixture(autouse=True)
def no_model_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Feedback tests must not contact a model")
    monkeypatch.setattr("httpx.Client", forbidden)


def repeated_proposal():
    return action("apply_edits", edits=[{"path": "calc.py", "old_text": "def add(a, b):",
        "new_text": "def add(a, b):\n    # Implementation here " + str(index)} for index in range(5)])


def test_rejected_summary_groups_exact_anchors_without_replaying_placeholder_code():
    proposal = json.loads(repeated_proposal()["content"])["arguments"]
    summary = _rejected_edit_summary(proposal)
    assert summary["edit_count"] == 5 and summary["edit_authorization"] is False
    assert "old_text" not in summary["anchors"][0]
    assert summary["anchors"][0]["old_text_sha256"] == hashlib.sha256(b"def add(a, b):").hexdigest()
    assert summary["anchors"][0]["repeat_count"] == 5
    assert len(summary["anchors"][0]["replacement_chars"]) == 5
    assert "Implementation here" not in json.dumps(summary)


def test_long_anchor_has_only_full_length_hash_and_summary_keeps_a_fixed_bound():
    summary = _rejected_edit_summary({"edits": [{"path": "calc.py", "old_text": "x" * 2000, "new_text": "y"}]})
    assert "old_text" not in summary["anchors"][0]
    assert summary["anchors"][0]["old_text_chars"] == 2000
    assert summary["anchors"][0]["old_text_sha256"] == hashlib.sha256(("x" * 2000).encode()).hexdigest()
    huge = _rejected_edit_summary({"edits": [{"path": "p" * 4000, "old_text": "x", "new_text": "y"}]})
    assert huge["truncated"] and not huge["anchors"]
    assert len(json.dumps(huge, separators=(",", ":"))) <= 900


def test_first_rejection_reaches_next_request_and_never_calls_14b(tmp_path, monkeypatch):
    small = Coder(SMALL, repeated_proposal(), read())
    large = Coder(LARGE)
    agent, loads, unloads = setup(tmp_path, monkeypatch, small, large, max_steps=2)
    state = agent.run("Fix addition.")
    failure = tool_results(state, "apply_edits")[0]
    assert failure["error"]["code"] == "source_not_presented"
    assert failure["rejected_edit"]["anchors"][0]["repeat_count"] == 5
    prompt = text(small.requests[1])
    assert '"repeat_count":5' in prompt and '"old_text":' not in prompt
    assert "read_file only for missing source" in prompt and "Implementation here" not in prompt
    assert not large.requests and not loads and not unloads
    assert route(state)["repair"]["status"] == "idle"
    assert not state.modified_files


def test_two_invalid_edits_request_one_gated_verified_repair(tmp_path, monkeypatch):
    small = Coder(SMALL, read(), check(), repeated_proposal(), repeated_proposal(), action("finish"))
    large = Coder(LARGE, edit())
    agent, loads, unloads = setup(tmp_path, monkeypatch, small, large)
    state = agent.run("Fix addition and preserve comments.")
    assert state.status == "completed" and state.verification == "passed", state.final_response
    failures = [result for result in tool_results(state, "apply_edits") if not result["success"]]
    assert [result["error"]["code"] for result in failures] == ["overlapping_edits", "overlapping_edits"]
    assert len(large.requests) == 1 and route(state)["repair"]["calls_used"] == 1
    assert route(state)["repair"]["reason"] == "invalid_responses"
    assert loads == [LARGE, SMALL] and unloads == [SMALL, LARGE]
    packet = text(large.requests[0])
    assert "recent_rejections" in packet and '"repeat_count":5' in packet
    assert "AssertionError" in packet and "return a - b" in packet


def test_two_invalid_edits_without_failed_check_cannot_load_14b(tmp_path, monkeypatch):
    small = Coder(SMALL, read(), repeated_proposal(), repeated_proposal(), action("finish"))
    large = Coder(LARGE)
    agent, loads, unloads = setup(tmp_path, monkeypatch, small, large, max_steps=4)
    state = agent.run("Fix addition.")
    assert route(state)["repair"]["status"] == "requested"
    assert route(state)["repair"]["calls_used"] == 0
    assert not large.requests and not loads and not unloads
    assert not state.modified_files and state.verification == "not_run"


def test_anchor_in_rejection_feedback_never_grants_source_authority(tmp_path, monkeypatch):
    small = Coder(SMALL, repeated_proposal(), repeated_proposal())
    large = Coder(LARGE)
    agent, loads, unloads = setup(tmp_path, monkeypatch, small, large, max_steps=2)
    state = agent.run("Fix addition.")
    failures = tool_results(state, "apply_edits")
    assert [failure["error"]["code"] for failure in failures] == ["source_not_presented", "source_not_presented"]
    assert '"old_text":"def add(a, b):"' not in text(small.requests[1])
    assert '"old_text_sha256":' in text(small.requests[1])
    assert not state.modified_files and not large.requests and not loads


@pytest.mark.parametrize("code", ["source_not_presented", "overlapping_edits", "invalid_arguments", "ambiguous_match",
                                 "no_match", "duplicate_edit", "no_change"])
def test_consecutive_guard_rejections_reuse_existing_invalid_counter_and_schema(code):
    config = HarnessConfig()
    progress = RoutingProgress()
    result = {"success": False, "error": {"code": code}}
    assert progress.observe("apply_edits", {}, result, 0, config) is None
    assert progress.observe("apply_edits", {}, result, 0, config) == "invalid_responses"
    assert RoutingProgress.from_dict(progress.to_dict()).to_dict() == progress.to_dict()


@pytest.mark.parametrize("code", ["stale_source", "io_error", "deadline_exceeded", "permission_denied", "rollback_failed"])
def test_races_and_infrastructure_do_not_consume_invalid_proposal_allowance(code):
    config = HarnessConfig()
    progress = RoutingProgress()
    result = {"success": False, "error": {"code": code}}
    for _ in range(2):
        assert progress.observe("apply_edits", {}, result, 0, config) is None
    assert progress.invalid_responses == 0


def test_successful_read_breaks_invalid_proposal_streak():
    config = HarnessConfig()
    progress = RoutingProgress()
    bad = {"success": False, "error": {"code": "invalid_arguments"}}
    assert progress.observe("apply_edits", {}, bad, 0, config) is None
    assert progress.observe("read_file", {"path": "calc.py"}, {"success": True, "path": "calc.py",
        "start_line": 1, "source_sha256": "a" * 64, "content": "1: VALUE = 1"}, 0, config) is None
    assert progress.observe("apply_edits", {}, bad, 0, config) is None
