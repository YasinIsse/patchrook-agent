"""Optional Ollama timing evidence stays measurable, nullable, and offline."""

from copy import deepcopy
import json
from pathlib import Path

import httpx
import pytest

from patchrook_agent.agent import Agent
from patchrook_agent.config import HarnessConfig
from patchrook_agent.models.ollama import ModelResponse, OllamaClient
from patchrook_agent.observability.metrics import ExecutionMetrics


HTTPX_CLIENT = httpx.Client


@pytest.fixture(autouse=True)
def prohibit_live_http(monkeypatch):
    def unexpected_http(*args, **kwargs):
        raise AssertionError("Inference timing tests must not contact a live model")

    monkeypatch.setattr(httpx, "Client", unexpected_http)


def mock_response(monkeypatch, telemetry, *, omitted_fields=()):
    payload = {"message": {"role": "assistant", "content": "Inspected."},
               "prompt_eval_count": 100, "eval_count": 20, **telemetry}
    for key in omitted_fields:
        payload.pop(key, None)
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=json.dumps(payload)))
    monkeypatch.setattr(httpx, "Client", lambda **options: HTTPX_CLIENT(transport=transport, **options))
    return OllamaClient().chat([{"role": "user", "content": "Inspect."}])


@pytest.mark.parametrize("cached", [None, 0, 70])
def test_response_cache_count_preserves_absent_zero_and_measured_values(monkeypatch, cached):
    telemetry = {} if cached is None else {"prompt_eval_cached_count": cached}
    response = mock_response(monkeypatch, telemetry)
    assert response.prompt_eval_cached_count == cached
    assert response.prompt_duration_ns is None
    assert response.completion_duration_ns is None and response.load_duration_ns is None


def test_explicit_zero_component_durations_remain_known(monkeypatch):
    response = mock_response(monkeypatch, {"prompt_eval_duration": 0, "eval_duration": 0,
                                          "load_duration": 0, "prompt_eval_cached_count": 0})
    assert response.prompt_duration_ns == response.completion_duration_ns == response.load_duration_ns == 0
    assert response.prompt_eval_cached_count == 0


@pytest.mark.parametrize("invalid", [-1, True, 1.5, "123", float("nan"), float("inf")])
def test_malformed_optional_telemetry_does_not_discard_the_assistant_response(monkeypatch, invalid):
    response = mock_response(monkeypatch, {"prompt_eval_duration": invalid, "eval_duration": invalid,
                                          "load_duration": invalid, "prompt_eval_cached_count": invalid})
    assert response.message["content"] == "Inspected."
    assert response.prompt_duration_ns is None and response.completion_duration_ns is None
    assert response.load_duration_ns is None and response.prompt_eval_cached_count is None


def test_impossible_cache_count_and_missing_prompt_count_are_unknown(monkeypatch):
    response = mock_response(monkeypatch, {"prompt_eval_cached_count": 101})
    assert response.prompt_eval_cached_count is None
    response = mock_response(monkeypatch, {"prompt_eval_count": None, "prompt_eval_cached_count": 0})
    assert response.prompt_eval_cached_count is None


def test_model_response_positional_done_reason_remains_compatible():
    original = ModelResponse({"role": "assistant", "content": "Done."}, 100, 20, 0, 0, 0, 0, "length")
    assert original.done_reason == "length" and original.prompt_eval_cached_count is None
    extended = ModelResponse(original.message, 100, 20, 0, 0, 0, 0, "stop", 0)
    assert extended.done_reason == "stop" and extended.prompt_eval_cached_count == 0


def test_metrics_rates_use_matching_known_cohorts_and_preserve_partial_coverage():
    metrics = ExecutionMetrics()
    metrics.record_model(1, 100, 20, 4, prompt_eval_duration_ns=2_000_000_000,
                         eval_duration_ns=1_000_000_000, load_duration_ns=500_000_000,
                         prompt_eval_cached_count=60)
    metrics.record_model(2, 200, 30, 4, prompt_eval_duration_ns=4_000_000_000,
                         eval_duration_ns=0, load_duration_ns=0, prompt_eval_cached_count=0)
    metrics.record_model(3, 300, 15, 8, prompt_eval_duration_ns=5_000_000_000,
                         eval_duration_ns=3_000_000_000)
    metrics.record_model(4, 100, 10, 1)
    summary = metrics.summary()
    assert summary["prompt_eval_seconds"] == 11
    assert summary["eval_seconds"] == 4 and summary["load_seconds"] == 0.5
    assert summary["prompt_eval_cached_count"] == 60
    assert summary["uncached_input_tokens"] == 240
    assert summary["prefill_tokens_per_second"] == 240 / 6
    assert summary["generation_tokens_per_second"] == 35 / 4
    assert summary["timing_coverage"] == {
        "completed_calls": 4, "recorded_calls": 4, "prompt_eval_calls": 3, "eval_calls": 3,
        "prompt_token_calls": 4, "completion_token_calls": 4,
        "load_calls": 2, "cache_calls": 2, "uncached_input_calls": 2,
        "prefill_throughput_calls": 2, "generation_throughput_calls": 2,
    }
    assert metrics.per_step[1]["prompt_eval_cached_count"] == 0
    assert metrics.per_step[1]["eval_seconds"] == 0
    assert metrics.per_step[1]["generation_tokens_per_second"] is None
    assert metrics.per_step[2]["prefill_tokens_per_second"] is None
    assert ExecutionMetrics.from_dict(metrics.to_dict()).summary() == summary


def test_multiple_model_responses_on_one_step_keep_rates_and_coverage_additive():
    metrics = ExecutionMetrics()
    metrics.record_model(1, 100, 20, 3, prompt_eval_duration_ns=2_000_000_000,
                         eval_duration_ns=1_000_000_000, prompt_eval_cached_count=60)
    metrics.record_model(1, 200, 30, 4)
    row = metrics.per_step[0]
    assert row["actual_input_tokens"] == 300 and row["actual_output_tokens"] == 50
    assert row["prompt_eval_seconds"] == 2 and row["prompt_eval_cached_count"] == 60
    assert row["prefill_tokens_per_second"] == 20
    assert row["generation_tokens_per_second"] == 20
    assert row["timing_coverage"]["recorded_calls"] == 2
    assert row["timing_coverage"]["prompt_eval_calls"] == 1


def test_legacy_checkpoint_has_unknown_components_and_can_accumulate_new_evidence():
    legacy = {"inference_calls": 1, "per_step": [
        {"step": 1, "actual_input_tokens": 500, "actual_output_tokens": 20, "inference_seconds": 10}]}
    unchanged = deepcopy(legacy)
    metrics = ExecutionMetrics.from_dict(legacy)
    assert legacy == unchanged
    summary = metrics.summary()
    assert summary["prompt_eval_seconds"] is None and summary["eval_seconds"] is None
    assert summary["load_seconds"] is None and summary["prompt_eval_cached_count"] is None
    assert summary["prefill_tokens_per_second"] is None
    assert summary["timing_coverage"]["completed_calls"] == 1
    assert summary["timing_coverage"]["recorded_calls"] == 0
    assert metrics.per_step[0]["prompt_eval_seconds"] is None
    metrics.record_model(2, 100, 20, 3, prompt_eval_duration_ns=2_000_000_000,
                         eval_duration_ns=1_000_000_000, prompt_eval_cached_count=0)
    summary = metrics.summary()
    assert summary["actual_input_tokens"] == 600
    assert summary["prompt_eval_seconds"] == 2 and summary["prompt_eval_cached_count"] == 0
    assert summary["prefill_tokens_per_second"] == 50
    assert summary["timing_coverage"]["completed_calls"] == 2
    assert summary["timing_coverage"]["cache_calls"] == 1


def test_zero_duration_never_produces_an_infinite_or_invented_rate():
    metrics = ExecutionMetrics()
    metrics.record_model(1, 100, 20, 0, prompt_eval_duration_ns=0, eval_duration_ns=0,
                         load_duration_ns=0, prompt_eval_cached_count=0)
    summary = metrics.summary()
    assert summary["prompt_eval_seconds"] == summary["eval_seconds"] == summary["load_seconds"] == 0
    assert summary["prefill_tokens_per_second"] is None
    assert summary["generation_tokens_per_second"] is None
    assert summary["timing_coverage"]["prompt_eval_calls"] == 1


@pytest.mark.parametrize("change", [
    {"prompt_eval_seconds": -1}, {"eval_seconds": True}, {"load_seconds": float("nan")},
    {"prompt_eval_cached_count": 101}, {"uncached_input_tokens": 99},
    {"completion_tokens": -1},
])
def test_corrupt_persisted_timing_measurements_are_rejected(change):
    metrics = ExecutionMetrics()
    metrics.record_model(1, 100, 20, 3, prompt_eval_duration_ns=2_000_000_000,
                         eval_duration_ns=1_000_000_000, prompt_eval_cached_count=0)
    saved = metrics.to_dict()
    saved["per_step"][0]["inference_measurements"][0].update(change)
    with pytest.raises(ValueError):
        ExecutionMetrics.from_dict(saved)


def test_timing_cannot_claim_more_recorded_responses_than_completed_calls():
    metrics = ExecutionMetrics()
    metrics.record_model(1, 100, 20, 3)
    saved = metrics.to_dict()
    saved["inference_calls"] = 0
    with pytest.raises(ValueError, match="coverage exceeds"):
        ExecutionMetrics.from_dict(saved)


class ResponseModel:
    model = "timing-unit-fixture"
    context_size = 8192
    max_output_tokens = 512

    def __init__(self, response):
        self.response = response

    def chat(self, **options):
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response


def test_agent_records_native_nanosecond_events_and_component_seconds(tmp_path):
    response = ModelResponse({"role": "assistant", "content": "Inspected."}, 100, 20,
                             2_000_000_000, 1_000_000_000, 0, 3_000_000_000,
                             prompt_eval_cached_count=0)
    state = Agent(ResponseModel(response), workspace=tmp_path, config=HarnessConfig.preset("C"),
                  runs_dir=tmp_path / "runs").run("Inspect without changing files.")
    events = [json.loads(line) for line in (Path(state.session_dir) / "events.jsonl").read_text().splitlines()]
    event = next(item for item in events if item["event"] == "model_response")
    assert event["prompt_eval_duration"] == 2_000_000_000
    assert event["eval_duration"] == 1_000_000_000 and event["load_duration"] == 0
    assert event["component_duration_unit"] == "nanoseconds"
    assert event["prompt_eval_cached_count"] == 0
    assert state.metrics["prompt_eval_seconds"] == 2
    assert state.metrics["eval_seconds"] == 1 and state.metrics["load_seconds"] == 0
    assert state.metrics["prefill_tokens_per_second"] == 50
    assert state.metrics["generation_tokens_per_second"] == 20
    assert state.metrics["timing_coverage"]["completed_calls"] == 1


@pytest.mark.parametrize("error,reason", [(httpx.ReadTimeout("slow"), "model_timeout"),
                                         (KeyboardInterrupt(), "interrupted")])
def test_failed_or_interrupted_request_does_not_invent_component_measurements(tmp_path, error, reason):
    state = Agent(ResponseModel(error), workspace=tmp_path, config=HarnessConfig.preset("C"),
                  runs_dir=tmp_path / "runs").run("Inspect without changing files.")
    assert state.stop_reason == reason
    assert state.metrics["completed_inference_calls"] == 0
    assert state.metrics["inference_calls"] == 1
    assert state.metrics["prompt_eval_seconds"] is None
    assert state.metrics["prompt_eval_cached_count"] is None
    assert state.metrics["timing_coverage"]["completed_calls"] == 0
    assert state.metrics["timing_coverage"]["recorded_calls"] == 0


@pytest.mark.parametrize("invalid_count", [None, True, -1, "20", 1.5])
def test_missing_or_invalid_counts_do_not_produce_rates_from_legacy_zero_defaults(monkeypatch, invalid_count):
    response = mock_response(monkeypatch, {
        "prompt_eval_count": invalid_count, "eval_count": invalid_count,
        "prompt_eval_duration": 2_000_000_000, "eval_duration": 1_000_000_000,
        "prompt_eval_cached_count": 0,
    })
    assert response.prompt_tokens == response.completion_tokens == 0
    assert not response.prompt_tokens_reported and not response.completion_tokens_reported
    assert response.prompt_eval_cached_count is None
    metrics = ExecutionMetrics()
    metrics.record_model(1, response.prompt_tokens, response.completion_tokens, 3,
                         prompt_eval_duration_ns=response.prompt_duration_ns,
                         eval_duration_ns=response.completion_duration_ns,
                         prompt_eval_cached_count=response.prompt_eval_cached_count,
                         prompt_tokens_reported=response.prompt_tokens_reported,
                         completion_tokens_reported=response.completion_tokens_reported)
    summary = metrics.summary()
    assert summary["prompt_eval_seconds"] == 2 and summary["eval_seconds"] == 1
    assert summary["prefill_tokens_per_second"] is None
    assert summary["generation_tokens_per_second"] is None
    assert summary["timing_coverage"]["prompt_token_calls"] == 0
    assert summary["timing_coverage"]["completion_token_calls"] == 0


def test_explicit_zero_counts_with_measured_duration_have_measured_zero_rates(monkeypatch):
    response = mock_response(monkeypatch, {"prompt_eval_count": 0, "eval_count": 0,
                                          "prompt_eval_duration": 2_000_000_000,
                                          "eval_duration": 1_000_000_000,
                                          "prompt_eval_cached_count": 0})
    assert response.prompt_tokens_reported and response.completion_tokens_reported
    metrics = ExecutionMetrics()
    metrics.record_model(1, 0, 0, 3, prompt_eval_duration_ns=response.prompt_duration_ns,
                         eval_duration_ns=response.completion_duration_ns,
                         prompt_eval_cached_count=response.prompt_eval_cached_count,
                         prompt_tokens_reported=response.prompt_tokens_reported,
                         completion_tokens_reported=response.completion_tokens_reported)
    summary = metrics.summary()
    assert summary["prefill_tokens_per_second"] == 0
    assert summary["generation_tokens_per_second"] == 0
    assert summary["prompt_eval_cached_count"] == summary["uncached_input_tokens"] == 0


@pytest.mark.parametrize("omitted", ["prompt_eval_count", "eval_count"])
def test_actual_omitted_count_only_disables_its_corresponding_rate(monkeypatch, omitted):
    response = mock_response(monkeypatch, {"prompt_eval_duration": 2_000_000_000,
                                          "eval_duration": 1_000_000_000,
                                          "prompt_eval_cached_count": 0}, omitted_fields=(omitted,))
    metrics = ExecutionMetrics()
    metrics.record_model(1, response.prompt_tokens, response.completion_tokens, 3,
                         prompt_eval_duration_ns=response.prompt_duration_ns,
                         eval_duration_ns=response.completion_duration_ns,
                         prompt_eval_cached_count=response.prompt_eval_cached_count,
                         prompt_tokens_reported=response.prompt_tokens_reported,
                         completion_tokens_reported=response.completion_tokens_reported)
    summary = metrics.summary()
    assert summary["prefill_tokens_per_second"] == (None if omitted == "prompt_eval_count" else 50)
    assert summary["generation_tokens_per_second"] == (None if omitted == "eval_count" else 20)
    assert response.prompt_tokens_reported is (omitted != "prompt_eval_count")
    assert response.completion_tokens_reported is (omitted != "eval_count")


def test_agent_event_marks_unreported_counts_and_rates_unknown(tmp_path, monkeypatch):
    response = mock_response(monkeypatch, {"prompt_eval_duration": 2_000_000_000,
                                          "eval_duration": 1_000_000_000,
                                          "prompt_eval_cached_count": 0,
                                          "total_duration": 3_000_000_000},
                             omitted_fields=("prompt_eval_count", "eval_count"))
    state = Agent(ResponseModel(response), workspace=tmp_path, config=HarnessConfig.preset("C"),
                  runs_dir=tmp_path / "runs").run("Inspect without changing files.")
    events = [json.loads(line) for line in (Path(state.session_dir) / "events.jsonl").read_text().splitlines()]
    event = next(item for item in events if item["event"] == "model_response")
    assert event["prompt_tokens_reported"] is False and event["completion_tokens_reported"] is False
    assert event["prompt_eval_cached_count"] is None
    assert state.metrics["prefill_tokens_per_second"] is None
    assert state.metrics["generation_tokens_per_second"] is None
