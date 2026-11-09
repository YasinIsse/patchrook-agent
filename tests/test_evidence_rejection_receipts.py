"""Rejected candidate code stays absent while exact diagnostics reach repairs."""

from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import pytest

from patchrook_agent.evidence_feedback import recent_rejection_receipts, rejected_edit_summary, safe_receipt
from test_model_routing_workflow import Coder, LARGE, SMALL, action, edit, route, setup


def feedback(messages):
    marker = "Observed repair state (data):\n"
    return json.loads(next(row["content"][len(marker):] for row in messages if row["content"].startswith(marker)))


def test_mid_string_preview_is_absent_and_full_lengths_hashes_errors_are_bounded():
    old = 'class Result:\n    """' + "public contract " * 30 + 'complete closing text"""'
    replacement = 'def candidate():\n    """THIS_REJECTED_BODY_MUST_NOT_REPLAY'
    summary = rejected_edit_summary({"edits": [{"path": "subject.py", "old_text": old, "new_text": replacement}]},
        error={"code": "python_syntax_error", "message": "Python syntax regression at line405: unterminated string."},
        step=3, model=LARGE, revision=0, inference_attempt=3)
    encoded = json.dumps(summary)
    assert old[:160] not in encoded and replacement not in encoded
    assert "old_text" not in summary["anchors"][0] and "new_text" not in summary["anchors"][0]
    assert summary["anchors"][0]["old_text_chars"] == len(old)
    assert summary["anchors"][0]["old_text_sha256"] == hashlib.sha256(old.encode()).hexdigest()
    assert summary["anchors"][0]["replacement_chars"] == [len(replacement)]
    assert summary["error"]["code"] == "python_syntax_error" and "line405" in summary["error"]["message"]
    assert summary["applied"] is False and summary["revision"] == 0
    assert len(encoded) <= 900


@pytest.mark.parametrize("payload", ["x" * 4000, "😀" * 4000])
def test_error_and_path_bounds_hold_even_with_unicode(payload):
    summary = rejected_edit_summary({"edits": [{"path": payload, "old_text": "CODE", "new_text": "OTHER_CODE"}]},
        error={"code": "python_syntax_error", "message": payload})
    assert len(json.dumps(summary, separators=(",", ":"))) <= 900
    assert summary["error"]["code"] == "python_syntax_error"
    assert "OTHER_CODE" not in json.dumps(summary)


def test_legacy_summary_is_redacted_and_its_actual_outer_error_is_restored():
    state = SimpleNamespace(messages=[{"role": "assistant", "content": "discarded"},
        {"role": "tool", "tool_name": "apply_edits", "content": json.dumps({
            "success": False, "error": {"code": "python_syntax_error", "message": "unterminated string at405"},
            "rejected_edit": {"edit_count": 1, "anchors": [{"path": "subject.py", "old_text": '"""CUT_AT160',
                "old_text_chars": 250, "old_text_truncated": True, "repeat_count": 1, "replacement_chars": [33]}]}})}],
        recovery_notes=[])
    receipt = recent_rejection_receipts(state, {"initial_model": SMALL})[0]
    assert "CUT_AT160" not in json.dumps(receipt)
    assert receipt["error"] == {"code": "python_syntax_error", "message": "unterminated string at405"}
    assert receipt["step"] == 1 and receipt["model"] == SMALL


def test_syntax_then_length_handoff_and_second_repair_receive_latest_diagnostics(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    source = 'def add(a, b):\n    """' + "public information " * 14 + 'END_OF_FULL_ANCHOR"""\n    return a - b\n'
    (root / "calc.py").write_text(source)
    first_bad = action("apply_edits", edits=[{"path": "calc.py", "old_text": source,
        "new_text": 'def add(a, b):\n    """FIRST_REJECTED_CODE'}])
    truncated = {"role": "assistant", "content": '{"name":"apply_edits","arguments":{"edits":['}
    class LengthCoder(Coder):
        def chat(self, *args, **kwargs):
            response = super().chat(*args, **kwargs)
            if response.message == truncated:
                response.done_reason = "length"
            return response
    small = LengthCoder(SMALL, first_bad, truncated)
    captured = []
    def repair_one(messages):
        packet = feedback(messages)
        receipts = packet["recent_rejections"]
        assert [item["error"]["code"] for item in receipts] == ["python_syntax_error", "model_response_length"]
        assert [item["step"] for item in receipts] == [1, 2]
        assert all(item["model"] == SMALL and item["applied"] is False for item in receipts)
        assert receipts[-1]["done_reason"] == "length"
        assert (root / "calc.py").read_text() == source
        captured.append(packet)
        return action("apply_edits", edits=[{"path": "calc.py", "old_text": source,
            "new_text": 'def add(a, b):\n    """SECOND_REJECTED_CODE'}])
    def repair_two(messages):
        packet = feedback(messages)
        receipts = packet["recent_rejections"]
        assert [item["error"]["code"] for item in receipts] == ["model_response_length", "python_syntax_error"]
        assert [item["step"] for item in receipts] == [2, 3]
        assert receipts[-1]["model"] == LARGE and "unterminated" in receipts[-1]["error"]["message"]
        assert "line 2" in receipts[-1]["error"]["message"]
        assert (root / "calc.py").read_text() == source
        fresh = "\n".join(row["content"] for row in messages if row["content"].startswith("Requested source (data) "))
        assert "END_OF_FULL_ANCHOR\"\"\"" in fresh and "return a - b" in fresh
        assert "FIRST_REJECTED_CODE" not in json.dumps(messages) and "SECOND_REJECTED_CODE" not in json.dumps(messages)
        captured.append(packet)
        return edit()
    large = Coder(LARGE, repair_one, repair_two)
    agent, _, _ = setup(tmp_path, monkeypatch, small, large, persist=True,
        config_overrides={"evidence_initial_checks": True, "evidence_seed_symbols": True})
    agent.registry.workspace.config = replace(agent.registry.workspace.config, validate_python_edits=True)
    state = agent.run("Fix `add` and preserve its public contract.")
    assert state.status == "completed" and state.verification == "passed"
    assert route(state)["repair"]["calls_used"] == 2 and len(large.requests) == 2
    assert (root / "calc.py").read_text() == source.replace("a - b", "a + b")
    checkpoint = json.loads((agent._session.store.directory / "checkpoint.json").read_text())
    restored = SimpleNamespace(**{key: checkpoint["state"][key] for key in ("messages", "recovery_notes")})
    receipts = recent_rejection_receipts(restored, checkpoint["execution"]["routing"])
    assert [row["error"]["code"] for row in receipts] == ["model_response_length", "python_syntax_error"]
    assert len(captured) == 2
