"""Decision configuration, reporting and metrics without local inference."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from patchrook_agent import cli
from patchrook_agent.agent import AgentState
from patchrook_agent.config import HarnessConfig
from patchrook_agent.observability.metrics import ExecutionMetrics
from patchrook_agent.reporting import execution_summary


def test_decision_defaults_preserve_legacy_config_and_roundtrip():
    original = replace(HarnessConfig.preset("C"), workflow_mode="coding")
    assert original.decision_model == "off"
    saved = original.to_dict()
    for name in ("decision_model", "decision_timeout_seconds", "decision_memory_policy"):
        saved.pop(name)
    assert HarnessConfig.from_dict(saved) == original
    enabled = replace(original, decision_model="on", decision_timeout_seconds=12,
                      decision_memory_policy="resident")
    assert HarnessConfig.from_dict(enabled.to_dict()) == enabled
    assert replace(enabled, decision_model="off").context == original.context
    assert {name: getattr(enabled, name) for name in ("temperature", "top_p", "top_k", "min_p")} == {
        name: getattr(original, name) for name in ("temperature", "top_p", "top_k", "min_p")}


@pytest.mark.parametrize("options", [
    {"decision_model": True}, {"decision_model": []}, {"decision_model": "tev1:4b"},
    {"decision_timeout_seconds": 0}, {"decision_timeout_seconds": float("nan")},
    {"decision_timeout_seconds": True}, {"decision_memory_policy": None},
    {"decision_memory_policy": []}, {"decision_memory_policy": "automatic"},
    {"decision_model": "on", "workflow_mode": "standard"},
])
def test_invalid_decision_config_is_rejected(options):
    with pytest.raises(ValueError):
        HarnessConfig(workflow_mode=options.pop("workflow_mode", "coding"), **options)


@pytest.mark.parametrize("mode", ["tev", "off"])
def test_cli_routing_toggle_keeps_coder_settings_and_trusted_checks(monkeypatch, tmp_path, mode):
    captured = {}

    class Agent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self, *args, **kwargs):
            return AgentState(status="completed", verification="passed", final_response="Done")

    monkeypatch.setattr(cli, "Agent", Agent)
    monkeypatch.setattr(cli, "_validate_models", lambda *args: None)
    assert cli.main(["Fix the parser", "--workspace", str(tmp_path), "--model-routing", mode,
                     "--decision-timeout", "11", "--check", "true"]) == 0
    config = captured["config"]
    assert config.decision_model == ("on" if mode == "tev" else "off")
    assert config.decision_timeout_seconds == 11 and config.decision_memory_policy == "serial"
    assert captured["model"].model == "qwen3:8b"
    assert captured["model"].context_size == 8192
    assert captured["model"].sampling_settings == {"temperature": .7, "top_p": .8, "top_k": 20, "min_p": 0}
    assert captured["registry"].workspace.config.allow_tests is True


@pytest.mark.parametrize("options", [["--decision-timeout", "10"], ["--model-routing", "off"]])
def test_decision_resume_overrides_rejected_before_checkpoint_or_model(monkeypatch, options):
    monkeypatch.setattr(cli, "Agent", lambda **kwargs: pytest.fail("Must not initialize model"))
    with pytest.raises(SystemExit) as error:
        cli.main(["--resume", "existing", *options])
    assert error.value.code == 2


def test_metrics_keep_decision_costs_separate_and_roundtrip():
    metrics = ExecutionMetrics(decision_enabled=True)
    metrics.record_decision(elapsed_seconds=1.2, management_seconds=.3, input_tokens=100)
    metrics.record_decision(elapsed_seconds=.01, cache_hit=True)
    metrics.record_decision(elapsed_seconds=2.0, fallback_reason="timeout")
    summary = metrics.summary()
    assert summary["decision_enabled"] is True
    assert summary["decision_calls"] == 2 and summary["decision_cache_hits"] == 1
    assert summary["decision_failures"] == summary["decision_fallbacks"] == 1
    assert summary["decision_seconds"] == pytest.approx(3.21)
    assert summary["decision_management_seconds"] == .3 and summary["decision_input_tokens"] == 100
    assert summary["inference_calls"] == summary["actual_input_tokens"] == 0
    assert ExecutionMetrics.from_dict(metrics.to_dict()) == metrics
    legacy = {name: value for name, value in metrics.to_dict().items() if not name.startswith("decision_")}
    assert ExecutionMetrics.from_dict(legacy).decision_enabled is False


def test_preflight_failure_is_fallback_without_a_model_call_or_known_load_time():
    metrics = ExecutionMetrics(decision_enabled=True)
    metrics.record_decision(elapsed_seconds=.3, management_seconds=.3,
                            fallback_reason="unload_failed", called=False)
    assert metrics.decision_calls == 0
    assert metrics.decision_fallbacks == metrics.decision_failures == 1
    assert metrics.decision_load_seconds is None and metrics.decision_load_measurements == 0
    assert ExecutionMetrics.from_dict(metrics.to_dict()) == metrics


def test_known_load_measurements_keep_coverage_and_request_duration_separate():
    metrics = ExecutionMetrics(decision_enabled=True)
    metrics.record_decision(elapsed_seconds=1.5, inference_seconds=1.2,
                            management_seconds=.3, load_seconds=.8)
    assert metrics.decision_seconds == 1.5 and metrics.decision_inference_seconds == 1.2
    assert metrics.decision_load_seconds == .8 and metrics.decision_load_measurements == 1
    assert ExecutionMetrics.from_dict(metrics.to_dict()) == metrics


@pytest.mark.parametrize("field,value", [("decision_enabled", 1), ("decision_calls", True),
                                        ("decision_failures", -1), ("decision_seconds", float("inf")),
                                        ("decision_management_seconds", -1), ("decision_input_tokens", 1.5)])
def test_bad_checkpoint_decision_metrics_rejected(field, value):
    with pytest.raises(ValueError):
        ExecutionMetrics.from_dict({field: value})


@pytest.mark.parametrize("enabled", [False, True])
def test_reports_record_decision_mode_from_metrics(enabled):
    state = SimpleNamespace(status="incomplete", messages=[], modified_files=[], completion={},
                            metrics={"decision_enabled": enabled})
    summary = execution_summary(state)
    assert "Decision assistance: " + ("on (tev1:4b)." if enabled else "off.") in summary
    assert summary.startswith("Incomplete.")
    assert "Tests: no current passing verification" in summary


