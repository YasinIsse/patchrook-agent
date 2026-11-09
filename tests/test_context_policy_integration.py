"""Request preparation and persisted context policy through the public agent."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import httpx
import pytest

from patchrook_agent import cli
from patchrook_agent.agent import Agent, AgentState
from patchrook_agent.config import HarnessConfig
from patchrook_agent.context import ContextConfig
from patchrook_agent.memory.projection import MEMORY_PREFIX
from patchrook_agent.memory.state import WorkingMemory
from patchrook_agent.models import ollama as ollama_module
from patchrook_agent.models.ollama import ModelResponse, OllamaClient
from patchrook_agent import session as session_module
from patchrook_agent.session import Session
from patchrook_agent.tools import ToolRegistry, Workspace


def final():
    return {"role": "assistant", "content": "Inspected the observed repository."}


def call(name, **arguments):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": name, "arguments": arguments}}]}


def exchange(name, arguments, result):
    return [call(name, **arguments), {
        "role": "tool", "tool_name": name, "content": json.dumps(result)}]


class SequenceModel:
    model = "deterministic-context-policy-fixture"

    def __init__(self, *messages, context_size=8192, max_output_tokens=512):
        self.context_size = context_size
        self.max_output_tokens = max_output_tokens
        self.messages = iter(messages)
        self.requests = []

    def chat(self, **options):
        self.requests.append(deepcopy(options))
        return ModelResponse(next(self.messages), 11, 7, 0, 0, 0, 10_000_000)


def ephemeral(variant="C", **options):
    return replace(HarnessConfig.preset(variant), persist=False, checkpoint_enabled=False, **options)


def test_actual_session_attempts_and_reductions_are_distinct(tmp_path):
    config = ephemeral("B", context=ContextConfig(context_size=8192, trigger=0.4, keep_recent_turns=0))
    session = Session(config, Workspace(tmp_path), SequenceModel(), tmp_path / "runs", "Preserve public APIs.")
    messages = [{"role": "system", "content": "Inspect first."},
                {"role": "user", "content": "Preserve public APIs. " * 450}]
    state = AgentState(messages=deepcopy(messages), active_messages=deepcopy(messages))
    first = session.prepare(state, [])
    assert first.attempted and first.safe and not first.compacted
    state.steps = 1
    second = session.prepare(state, [])
    assert not second.attempted and second.safe
    initial = session.metrics.summary()
    assert initial["compaction_attempt_count"] == 1
    assert initial["compaction_count"] == 0
    assert initial["compaction_attempts"][0]["tokens_before"] == initial["compaction_attempts"][0]["tokens_after"]
    assert initial["compaction_attempts"][0]["outcome_reason"] == "headroom_unavailable"
    state.active_messages += exchange("read_file", {"path": "historical.py"}, {
        "success": True, "path": "historical.py", "content": "old_source = 1\n" * 200})
    for index in range(3):
        state.active_messages += exchange("read_file", {"path": f"current{index}.py"}, {
            "success": True, "path": f"current{index}.py", "content": "1: return a - b"})
    state.steps = 2
    third = session.prepare(state, [])
    assert third.safe and third.compacted and third.attempted
    measured = session.metrics.summary()
    assert measured["compaction_attempt_count"] == 2
    assert measured["compaction_count"] == 1
    assert measured["compaction_attempts"][-1]["reduced"] is True
    assert measured["context_preparation_seconds"] >= measured["compaction_seconds"]
    assert measured["compaction_seconds"] == first.compaction_seconds + third.compaction_seconds


def test_preparation_time_includes_memory_projection_without_charging_compaction(tmp_path, monkeypatch):
    ticks = [100.0]
    monkeypatch.setattr(session_module.time, "monotonic", lambda: ticks[0])
    project = WorkingMemory.project

    def measured_project(self, **options):
        ticks[0] += 0.25
        return project(self, **options)

    monkeypatch.setattr(WorkingMemory, "project", measured_project)
    session = Session(ephemeral(), Workspace(tmp_path), SequenceModel(), tmp_path / "runs", "Inspect addition.")
    messages = [{"role": "system", "content": "Stable instructions."},
                {"role": "user", "content": "Inspect addition."}]
    state = AgentState(messages=deepcopy(messages), active_messages=deepcopy(messages))
    prepared = session.prepare(state, [])
    assert prepared.safe and not prepared.attempted
    assert session.metrics.context_preparation_seconds == 0.25
    assert session.metrics.compaction_seconds == prepared.compaction_seconds == 0
    assert prepared.messages[-1]["content"].startswith(MEMORY_PREFIX)
    assert sum(message["content"].startswith(MEMORY_PREFIX) for message in prepared.messages) == 1
    assert state.active_messages == messages
    assert session.metrics.per_step[0]["memory_projection_tokens"] <= 512


def test_projection_overflow_stops_agent_before_inference_with_actionable_message(tmp_path):
    model = SequenceModel(final())
    agent = Agent(model, workspace=tmp_path, config=ephemeral(memory_projection_tokens=64),
                  runs_dir=tmp_path / "runs")
    state = agent.run("Inspect addition without changing unrelated behavior.")
    assert state.status == "incomplete" and state.stop_reason == "memory_projection_limit"
    assert "memory_projection_tokens" in state.final_response == state.context_limit_message
    assert state.metrics["memory_projection_budget"] == 64
    assert state.metrics["inference_calls"] == 0 and not model.requests
    assert state.metrics["compaction_seconds"] == 0


def test_effective_output_reserve_tracks_model_prediction_limit(tmp_path):
    model = SequenceModel(final(), context_size=4096, max_output_tokens=1024)
    with pytest.raises(ValueError, match="trigger must precede"):
        Session(ephemeral("B", context=ContextConfig(context_size=4096)), Workspace(tmp_path),
                model, tmp_path / "runs", "Inspect.")
    config = ephemeral("B", context=ContextConfig(context_size=4096, trigger=0.6))
    state = Agent(model, workspace=tmp_path, config=config, runs_dir=tmp_path / "runs").run("Inspect.")
    assert state.status == "completed"
    assert state.metrics["reserved_output_tokens"] == 1024
    assert state.metrics["effective_input_limit"] == 4096 - 1024 - 128
    assert state.metrics["context_window_tokens"] == 4096


@pytest.mark.parametrize("legacy", [False, True])
def test_current_and_legacy_checkpoints_keep_original_configuration_on_resume(tmp_path, legacy):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    config = replace(HarnessConfig.preset("C"), context=ContextConfig(
        context_size=8192, trigger=0.6, target=0.35))
    first = Agent(SequenceModel(call("read_file", path="calc.py")), workspace=repo,
                  config=config, runs_dir=tmp_path / "runs", pause_after_step=1)
    paused = first.run("Inspect addition; preserve the original public API.")
    assert paused.stop_reason == "paused"
    path = Path(paused.session_dir) / "checkpoint.json"
    checkpoint = json.loads(path.read_text())
    if legacy:
        checkpoint["config"].pop("memory_projection_tokens")
        for saved_config in (checkpoint["config"]["context"], checkpoint["context"]["config"]):
            saved_config.pop("headroom_tokens")
            saved_config.pop("estimation_margin_tokens")
        for key in ("compaction_attempts", "last_reclaimable", "last_request_fingerprint"):
            checkpoint["context"].pop(key)
        for key in ("compaction_attempts", "context_preparation_seconds"):
            checkpoint["metrics"].pop(key)
        checkpoint["state"].pop("context_limit_message")
        path.write_text(json.dumps(checkpoint))
    resumed_model = SequenceModel(final())
    resumed = Agent(resumed_model, workspace=repo, runs_dir=tmp_path / "runs")
    state = resumed.resume(paused.session_id)
    assert state.status == "completed"
    assert state.memory["objective"] == "Inspect addition; preserve the original public API."
    assert state.metrics["context_window_tokens"] == 8192
    assert state.metrics["memory_projection_budget"] == 512
    assert resumed._session.context.config.trigger == 0.6
    assert resumed._session.context.config.target == 0.35
    assert resumed._session.context.config.estimation_margin_tokens == 128
    assert resumed._session.context.config.headroom_tokens == 512
    assert len(resumed_model.requests) == 1


def test_private_ineffective_attempt_cooldown_persists_through_agent_resume(tmp_path):
    config = replace(HarnessConfig.preset("C"), context=ContextConfig(context_size=8192, trigger=0.4))
    first = Agent(SequenceModel(call("list_files", path=".")), workspace=tmp_path,
                  config=config, runs_dir=tmp_path / "runs", pause_after_step=1)
    paused = first.run("Preserve public APIs. " * 350)
    assert paused.stop_reason == "paused"
    assert paused.metrics["compaction_attempt_count"] == 1
    checkpoint = json.loads((Path(paused.session_dir) / "checkpoint.json").read_text())
    assert checkpoint["context"]["last_reclaimable"] == {}
    assert checkpoint["context"]["last_request_fingerprint"]
    model = SequenceModel(final())
    resumed = Agent(model, workspace=tmp_path, runs_dir=tmp_path / "runs")
    state = resumed.resume(paused.session_id)
    assert state.status == "completed"
    assert state.metrics["compaction_attempt_count"] == 1
    assert state.metrics["compaction_count"] == 0
    assert "last_request_fingerprint" not in json.dumps(model.requests)
    assert "last_reclaimable" not in json.dumps(model.requests)


@pytest.mark.parametrize("size", [8192, 16384])
def test_real_client_payload_harness_budget_and_report_use_same_selected_window(tmp_path, monkeypatch, size):
    captured = []

    def respond(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"message": final(), "prompt_eval_count": 17, "eval_count": 5})

    original_client = httpx.Client
    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(ollama_module.httpx, "Client",
                        lambda **options: original_client(transport=transport, **options))
    model = OllamaClient(context_size=size)
    agent = Agent(model, workspace=tmp_path, config=ephemeral(), runs_dir=tmp_path / "runs")
    state = agent.run("Inspect addition.")
    assert state.status == "completed"
    assert captured[0]["options"]["num_ctx"] == agent._session.context.config.context_size == size
    assert state.metrics["context_window_tokens"] == size
    assert state.metrics["effective_input_limit"] == ContextConfig(context_size=size).input_limit
    assert captured[0]["options"]["num_predict"] == state.metrics["reserved_output_tokens"] == 512
    assert captured[0]["think"] is False


@pytest.mark.parametrize("explicit_size", [None, 8192, 16384])
def test_cli_selected_context_matches_actual_client_and_harness(tmp_path, monkeypatch, explicit_size):
    captured = {}

    class CaptureAgent:
        def __init__(self, **options):
            captured.update(options)

        def run(self, objective, *, requirements=None, tools=None):
            return AgentState(status="completed", final_response="Inspected.")

    monkeypatch.setattr(cli, "Agent", CaptureAgent)
    monkeypatch.setattr(cli, "_validate_models", lambda *args: None)
    workspace = tmp_path / "repository"
    workspace.mkdir()
    args = ["Inspect", "--workspace", str(workspace), "--task-mode", "inspect", "--runs-dir", str(tmp_path / "runs")]
    if explicit_size:
        args += ["--context-size", str(explicit_size)]
    assert cli.main(args) == 0
    expected = explicit_size or 8192
    assert isinstance(captured["model"], OllamaClient)
    assert captured["model"].context_size == captured["config"].context.context_size == expected
    assert captured["config"].context.trigger == 0.8
    assert captured["config"].context.target is None


def test_client_and_cli_reject_32k_before_running_an_agent(monkeypatch):
    with pytest.raises(ValueError, match="at most 16384"):
        OllamaClient(context_size=32768)
    invoked = []
    monkeypatch.setattr(cli, "Agent", lambda **options: invoked.append(options))
    with pytest.raises(SystemExit) as error:
        cli.main(["Inspect", "--context-size", "32768"])
    assert error.value.code == 2 and invoked == []
