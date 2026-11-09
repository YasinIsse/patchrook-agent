"""Routing configuration and CLI boundaries without model inference."""

from dataclasses import replace
import json

import pytest

from patchrook_agent import cli
from patchrook_agent.agent import AgentState
from patchrook_agent.config import HarnessConfig


def routing_config(**options):
    values = {"workflow_mode": "evidence", "action_protocol": "schema",
              "decision_model": "on", "model_routing": "tev", "review_mode": "off"}
    return HarnessConfig(**(values | options))


def test_routing_config_roundtrip_and_legacy_defaults():
    original = HarnessConfig()
    saved = original.to_dict()
    for name in ("model_routing", "escalation_model", "routing_max_failed_repairs",
                 "routing_max_invalid_responses", "routing_max_repeated_actions", "routing_max_repair_calls"):
        saved.pop(name)
    assert HarnessConfig.from_dict(saved) == original
    config = routing_config()
    assert HarnessConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config
    assert config.escalation_model == "qwen3:14b"
    assert (config.routing_max_failed_repairs, config.routing_max_invalid_responses,
            config.routing_max_repeated_actions) == (2, 2, 3)
    assert config.routing_max_repair_calls == 2
    assert replace(config, routing_max_repair_calls=1).routing_max_repair_calls == 1


@pytest.mark.parametrize("policy", ["rules", "window"])
def test_routing_allows_tev_without_semantic_evidence_admission(policy):
    config = routing_config(evidence_policy=policy)
    assert config.model_routing == "tev" and config.decision_model == "on"
    with pytest.raises(ValueError, match="decision_model"):
        replace(config, model_routing="off")
    assert replace(config, model_routing="off", decision_model="off").evidence_policy == policy


@pytest.mark.parametrize("options", [
    {"model_routing": True}, {"model_routing": []}, {"model_routing": "auto"},
    {"workflow_mode": "coding"}, {"decision_model": "off"},
    {"evidence_policy": "decision"}, {"decision_memory_policy": "resident"},
    {"escalation_model": ""}, {"escalation_model": " qwen3:14b"},
    {"escalation_model": "qwen3:14b:cloud"}, {"escalation_model": []},
    {"escalation_model": "model\x00name"},
    {"routing_max_failed_repairs": 0}, {"routing_max_failed_repairs": 9},
    {"routing_max_invalid_responses": True}, {"routing_max_invalid_responses": 9},
    {"routing_max_repeated_actions": 1}, {"routing_max_repeated_actions": 17},
    {"routing_max_repair_calls": 0}, {"routing_max_repair_calls": 3},
    {"routing_max_repair_calls": True}, {"routing_max_repair_calls": 1.5},
])
def test_invalid_routing_settings_rejected(options):
    with pytest.raises(ValueError):
        routing_config(**options)


def capture_agent(monkeypatch):
    captured = {}

    class Agent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self, *args, **kwargs):
            return AgentState(status="completed", verification="passed", final_response="Done")

    monkeypatch.setattr(cli, "Agent", Agent)
    monkeypatch.setattr(cli, "_validate_models", lambda *args: None)
    return captured


def test_cli_defaults_enable_tev_and_preserve_model_context(tmp_path, monkeypatch):
    captured = capture_agent(monkeypatch)
    assert cli.main(["Fix the parser", "--workspace", str(tmp_path), "--check", "true"]) == 0
    config = captured["config"]
    assert config.model_routing == "tev" and config.decision_model == "on"
    assert config.escalation_model == "qwen3:14b" and config.decision_memory_policy == "serial"
    assert config.evidence_context_tokens == config.context.input_limit == 6963
    assert captured["model"].model == "qwen3:8b"
    assert captured["model"].context_size == config.context.context_size == 8192
    assert captured["model"].max_output_tokens == 1024
    assert captured["registry"].workspace.config.allow_tests is True
    assert captured["registry"].workspace.config.validate_python_edits is True


def test_cli_explicit_baseline_disables_routing_and_tev(tmp_path, monkeypatch):
    captured = capture_agent(monkeypatch)
    assert cli.main(["Fix the parser", "--workspace", str(tmp_path), "--model-routing", "off", "--check", "true"]) == 0
    config = captured["config"]
    assert config.model_routing == "off" and config.decision_model == "off"


@pytest.mark.parametrize("options", [
    ["--model-routing", "off"], ["--model-routing", "tev"],
    ["--escalation-model", "qwen3:14b"],
])
def test_routing_resume_overrides_rejected_before_loading_checkpoint(monkeypatch, options):
    monkeypatch.setattr(cli, "Agent", lambda **kwargs: pytest.fail("Must not construct agent"))
    with pytest.raises(SystemExit) as exc:
        cli.main(["--resume", "saved", *options])
    assert exc.value.code == 2


@pytest.mark.parametrize("options", [
    ["--model-routing", "auto"],
    ["--escalation-model", ""],
    ["--decision-timeout", "0"],
])
def test_invalid_cli_routing_settings_fail_without_inference(monkeypatch, options):
    monkeypatch.setattr(cli, "Agent", lambda **kwargs: pytest.fail("Must not construct agent"))
    with pytest.raises(SystemExit) as exc:
        cli.main(["Fix the parser", *options])
    assert exc.value.code == 2
