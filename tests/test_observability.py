import json
from pathlib import Path

import pytest

from patchrook_agent.observability import EventLogger, ExecutionMetrics, redact_data


def test_events_append_without_rewriting_existing_history(tmp_path):
    logger = EventLogger(tmp_path, "session-1")
    logger.emit("model_request", step=1, model="qwen3:14b", messages=[{"role": "user", "content": "Inspect"}])
    first = logger.path.read_bytes()
    logger.emit("model_response", step=1, prompt_tokens=123, completion_tokens=4)
    assert logger.path.read_bytes().startswith(first)
    events = [json.loads(line) for line in logger.path.read_text().splitlines()]
    assert [event["event"] for event in events] == ["model_request", "model_response"]
    assert all(event["session_id"] == "session-1" and event["step"] == 1 for event in events)
    assert logger.path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("fragment", [b'{"event":"unfinished', b'{"content":"\xf0\x9f'])
def test_event_after_interrupted_write_keeps_tail_bytes_and_starts_a_parseable_line(tmp_path, fragment):
    logger = EventLogger(tmp_path, "session")
    logger.emit("before_interruption", step=1)
    complete_prefix = logger.path.read_bytes()
    with logger.path.open("ab") as stream:
        stream.write(fragment)
    logger.emit("after_interruption", step=2, message="Recovered")
    saved = logger.path.read_bytes()
    assert saved.startswith(complete_prefix + fragment + b"\n")
    lines = saved.splitlines()
    assert json.loads(lines[0])["event"] == "before_interruption"
    assert lines[1] == fragment
    recovered = json.loads(lines[2])
    assert recovered["event"] == "after_interruption" and recovered["step"] == 2
    assert logger.path.stat().st_mode & 0o777 == 0o600


def test_complete_event_without_final_newline_is_preserved_on_append(tmp_path):
    logger = EventLogger(tmp_path, "session")
    previous = b'{"event":"previous"}'
    logger.path.write_bytes(previous)
    logger.emit("next")
    saved = logger.path.read_bytes()
    assert saved.startswith(previous + b"\n")
    assert [json.loads(line)["event"] for line in saved.splitlines()] == ["previous", "next"]


def test_redaction_handles_nested_json_content_credentials_and_preserves_metrics(tmp_path):
    logger = EventLogger(tmp_path, "session")
    original = {
        "api_key": "private-api-key", "prompt_tokens": 40,
        "content": json.dumps({"password": "hunter2", "stdout": "Bearer abc123XYZ"}),
        "stdout": "api_key='secret-value' password=more-secret\nhttp://user:password@example.com",
        "private": "-----BEGIN PRIVATE KEY-----\nsecret-body\n-----END PRIVATE KEY-----",
    }
    logger.emit("tool_result", result=original)
    saved = logger.path.read_text()
    for secret in ("private-api-key", "hunter2", "abc123XYZ", "secret-value", "more-secret", "user:password", "secret-body"):
        assert secret not in saved
    assert json.loads(saved)["result"]["prompt_tokens"] == 40
    assert original["api_key"] == "private-api-key"


def test_known_key_redaction():
    assert "sk-abcdefghijklmnop" not in redact_data("value: sk-abcdefghijklmnop")
    assert "ghp_abcdefghijklmnop" not in redact_data("value: ghp_abcdefghijklmnop")


def test_content_logging_can_be_disabled_without_losing_outcome(tmp_path):
    logger = EventLogger(tmp_path, "session", log_content=False)
    logger.emit("tool_result", tool="run_tests", arguments={"path": "secret.py"}, result={
        "success": False, "exit_code": 1, "stdout": "sensitive output", "stderr": "sensitive error",
    })
    event = json.loads(logger.path.read_text())
    assert event["result"]["success"] is False and event["result"]["exit_code"] == 1
    assert "sensitive output" not in logger.path.read_text()
    assert "secret.py" not in logger.path.read_text()


def test_logging_can_be_disabled_or_redaction_explicitly_disabled(tmp_path):
    logger = EventLogger(tmp_path, "session", enabled=False)
    logger.emit("ignored", password="private")
    assert not logger.path.exists()
    logger = EventLogger(tmp_path, "session", redact=False)
    logger.emit("example", password="explicitly unredacted")
    assert "explicitly unredacted" in logger.path.read_text()


def test_event_file_symlink_is_not_followed(tmp_path):
    outside = tmp_path / "private"
    outside.write_text("unchanged")
    (tmp_path / "events.jsonl").symlink_to(outside)
    logger = EventLogger(tmp_path, "session")
    with pytest.raises(OSError):
        logger.emit("example")
    assert outside.read_text() == "unchanged"


def test_metrics_distinguish_estimated_context_and_actual_usage_and_resume_accounting():
    metrics = ExecutionMetrics()
    metrics.record_context(1, 1000, 0.25)
    metrics.record_model(1, 700, 20, 3.0)
    metrics.record_context(2, 2000, 0.5)
    metrics.record_model(2, 1400, 40, 5.0)
    metrics.record_tool("read_file", True, 0.01)
    metrics.record_tool("edit_file", False, 0.02)
    metrics.record_tool("edit_file", True, 0.03, retry=True)
    metrics.compactions.append({"tokens_before": 2000, "tokens_after": 1000})
    metrics.checkpoint_count = 4
    metrics.resume_attempts = 1
    metrics.resume_success = 1
    metrics.wall_seconds = 9
    restored = ExecutionMetrics.from_dict(metrics.to_dict())
    summary = restored.summary()
    assert summary["inference_calls"] == 2
    assert summary["actual_input_tokens"] == 2100 and summary["actual_output_tokens"] == 60
    assert summary["inference_seconds"] == 8
    assert summary["average_context_utilization"] == 0.375
    assert summary["peak_context_utilization"] == 0.5
    assert summary["compactions"][0]["reduction_percent"] == 50
    assert summary["tools_by_name"]["edit_file"]["retries"] == 1
    assert summary["checkpoint_count"] == 4 and summary["resume_success"] == 1
    assert restored.per_step[0]["estimated_context_tokens"] == 1000
    assert restored.per_step[0]["actual_input_tokens"] == 700


def test_empty_metrics_have_finite_zero_summary():
    summary = ExecutionMetrics().summary()
    assert summary["average_context_utilization"] == 0
    assert summary["peak_context_utilization"] == 0
    assert summary["actual_input_tokens"] == 0


def test_unknown_model_usage_survives_checkpoint_without_becoming_reported_zero():
    metrics = ExecutionMetrics()
    metrics.record_context(1, 1000, 0.25)
    metrics.per_step[0].update(response_received=False, actual_input_tokens=None,
                               actual_output_tokens=None, request_wall_seconds=0.5)
    restored = ExecutionMetrics.from_dict(metrics.to_dict())
    assert restored.per_step[0]["actual_input_tokens"] is None
    assert restored.summary()["actual_input_tokens"] == 0  # completed-response total only
    restored.per_step[0]["response_received"] = True
    restored.record_model(1, 700, 20, 2.0)
    assert restored.summary()["actual_input_tokens"] == 700
    assert restored.summary()["actual_output_tokens"] == 20
    # Additional measured model work on a step remains additive.
    restored.record_model(1, 100, 5, 0.5)
    assert restored.summary()["actual_input_tokens"] == 800
    assert restored.summary()["inference_calls"] == 2


@pytest.mark.parametrize("record", [
    {"step": 1, "actual_input_tokens": None},
    {"step": 1, "response_received": True, "actual_input_tokens": None},
    {"step": 1, "response_received": False, "estimated_context_tokens": None},
    {"step": 1, "response_received": "false", "actual_input_tokens": None},
])
def test_unknown_tokens_require_an_explicit_unreceived_response(record):
    with pytest.raises(ValueError):
        ExecutionMetrics.from_dict({"per_step": [record]})


@pytest.mark.parametrize("data", [{"unknown": 1}, {"resume_attempts": -1}, {"inference_calls": "2"},
                                       {"context_utilizations": [float("nan")]}, {"per_step": [{}]}])
def test_corrupt_checkpoint_metrics_are_rejected(data):
    with pytest.raises(ValueError):
        ExecutionMetrics.from_dict(data)
