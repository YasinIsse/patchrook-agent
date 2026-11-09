"""Supported CLI defaults, fail-fast setup, verification and resume contracts."""

from dataclasses import replace
import json
from pathlib import Path
import shlex
from types import SimpleNamespace
import sys

import httpx
import pytest

from patchrook_agent import cli, defaults
from patchrook_agent.agent import AgentState
from patchrook_agent.completion import TaskRequirements
from patchrook_agent.evidence_workflow import EvidenceWorkflow
from patchrook_agent.models.ollama import OllamaClient
from patchrook_agent.tools import ToolConfig, ToolRegistry, Workspace


CHECK = shlex.join([sys.executable, "-c", "assert True"])
VALIDATE_MODELS = cli._validate_models


@pytest.fixture(autouse=True)
def models(monkeypatch):
    """No CLI unit test can accidentally generate or load a model."""
    calls = []
    monkeypatch.setattr(cli, "_validate_models", lambda url, names: calls.append((url, names)))
    return calls


def fake_agent(monkeypatch, *, result=None, after_run=None):
    calls = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            calls["constructor"] = kwargs
            self.__dict__.update(kwargs)

        def run(self, task, *, requirements=None, tools=None):
            calls.update(task=task, requirements=requirements, tools=tools)
            if after_run:
                after_run(self)
            return result or AgentState(status="completed", verification="passed", stop_reason="verified_finish")

        def resume(self, session_id, *, allow_tests=False):
            calls["resume"] = (session_id, allow_tests)
            return result or AgentState(status="completed", verification="passed", stop_reason="verified_finish")

    monkeypatch.setattr(cli, "Agent", FakeAgent)
    return calls


def run_args(tmp_path, *options):
    return ["Fix the bug", "--workspace", str(tmp_path), "--check", CHECK, *options]


def test_plain_command_uses_verified_routed_profile(tmp_path, monkeypatch, models, capsys):
    calls = fake_agent(monkeypatch)
    monkeypatch.chdir(tmp_path)
    check = (sys.executable, "-m", "pytest", "-q")
    monkeypatch.setattr(cli, "_default_check", lambda root: check)
    assert cli.main(["Fix the parser"]) == 0
    options = calls["constructor"]
    config, model = options["config"], options["model"]
    assert calls["task"] == "Fix the parser"
    assert calls["requirements"] == TaskRequirements.fix()
    assert config.workflow_mode == "evidence" and config.action_protocol == "schema"
    assert config.evidence_policy == "rules" and config.model_routing == "tev"
    assert config.decision_model == "on" and config.escalation_model == "qwen3:14b"
    assert config.decision_memory_policy == "serial" and config.decision_timeout_seconds == 90
    assert config.context.context_size == 8192 and config.evidence_context_tokens == 6963
    assert config.evidence_optional_tokens == 1536 and not config.compaction_enabled
    assert config.evidence_initial_checks and config.evidence_auto_check_edits and config.evidence_seed_symbols
    assert config.evidence_recover_python_function_anchors and config.evidence_refresh_modified_python
    assert config.routing_max_failed_repairs == 1 and config.routing_max_repair_calls == 2
    assert config.evidence_test_commands == (check,)
    assert config.routing_generation_profiles.base.think is False
    assert config.routing_generation_profiles.repair.think is False
    assert config.routing_generation_profiles.base.temperature == .7
    assert config.routing_generation_profiles.repair.temperature == .6
    assert config.routing_generation_profiles.repair.top_p == .95
    assert model.model == "qwen3:8b" and model.max_output_tokens == 1024
    assert model.context_size == 8192 and model.request_timeout_seconds == 3600
    assert options["think"] is False and options["max_steps"] == 12
    assert options["deadline_seconds"] == 3600
    tool_config = options["registry"].workspace.config
    assert tool_config.allow_tests and tool_config.validate_python_edits
    assert tool_config.test_timeout_seconds == 120 and tool_config.max_output_chars == 12000
    assert models == [("http://localhost:11434", ["qwen3:8b", "tev1:4b", "qwen3:14b"])]
    text = capsys.readouterr().out
    assert "Running in" in text and "checks: passed" in text
    assert "context_records" not in text and "inference_calls" not in text


def test_model_budget_sampling_and_thinking_overrides_reach_both_phases(tmp_path, monkeypatch, models):
    calls = fake_agent(monkeypatch)
    assert cli.main(run_args(tmp_path, "--model", "coder:9b", "--escalation-model", "coder:15b",
        "--base-url", "http://server:11434", "--context-size", "4096", "--max-output-tokens", "768",
        "--max-steps", "20", "--deadline", "5400", "--test-timeout", "80", "--decision-timeout", "120",
        "--think", "--temperature", ".4", "--top-p", ".9", "--top-k", "12", "--min-p", ".05")) == 0
    options = calls["constructor"]
    config, model = options["config"], options["model"]
    assert config.evidence_context_tokens == config.context.input_limit
    assert config.context.context_size == model.context_size == 4096
    assert options["think"] is True and options["max_steps"] == 20 and options["deadline_seconds"] == 5400
    assert model.request_timeout_seconds == 5400 and model.max_output_tokens == 768
    assert model.model == "coder:9b" and config.escalation_model == "coder:15b"
    assert config.routing_generation_profiles.base.think and config.routing_generation_profiles.repair.think
    assert config.routing_generation_profiles.repair.max_output_tokens == 768
    assert model.sampling_settings == {"temperature": .4, "top_p": .9, "top_k": 12, "min_p": .05}
    config.routing_generation_profiles.base.validate_settings(options["model"].__dict__, True)
    assert options["registry"].workspace.config.test_timeout_seconds == 80
    assert config.decision_timeout_seconds == 120
    assert models == [("http://server:11434", ["coder:9b", "tev1:4b", "coder:15b"])]


def test_single_model_override_disables_tev_and_repair_profile(tmp_path, monkeypatch, models):
    calls = fake_agent(monkeypatch)
    assert cli.main(run_args(tmp_path, "--model", "qwen3:14b", "--model-routing", "off")) == 0
    config = calls["constructor"]["config"]
    assert config.model_routing == "off" and config.decision_model == "off"
    assert config.routing_generation_profiles is None
    assert models == [("http://localhost:11434", ["qwen3:14b"])]


def test_explicit_checks_are_argv_and_preserve_quoted_arguments(tmp_path, monkeypatch):
    calls = fake_agent(monkeypatch)
    command = shlex.join([sys.executable, "-c", "print('literal $HOME')", "argument with spaces"])
    assert cli.main(["Fix", "--workspace", str(tmp_path), "--check", command, "--check", CHECK]) == 0
    assert calls["constructor"]["config"].evidence_test_commands == (
        (sys.executable, "-c", "print('literal $HOME')", "argument with spaces"),
        (sys.executable, "-c", "assert True"))


@pytest.mark.parametrize("mode", ["fix", "code"])
def test_completed_without_passing_checks_does_not_exit_successfully(tmp_path, monkeypatch, mode):
    fake_agent(monkeypatch, result=AgentState(status="completed", verification="not_run"))
    assert cli.main(run_args(tmp_path, "--task-mode", mode)) == 1


def test_inspect_is_read_only_and_requires_no_check_environment(tmp_path, monkeypatch, models):
    calls = fake_agent(monkeypatch, result=AgentState(status="completed", verification="not_run"))
    monkeypatch.setattr(cli, "_checks", lambda *_: pytest.fail("No checks in inspect mode"))
    assert cli.main(["Inspect parser", "--workspace", str(tmp_path), "--task-mode", "inspect"]) == 0
    options = calls["constructor"]
    assert calls["requirements"] == TaskRequirements.inspect()
    assert options["config"].model_routing == "off"
    assert not options["registry"].workspace.config.allow_tests
    assert {item["function"]["name"] for item in calls["tools"]} == {
        "list_files", "read_file", "search_code", "read_evidence", "finish"}
    assert models == [("http://localhost:11434", ["qwen3:8b"])]


def test_json_output_is_small_and_omits_trace_content(tmp_path, monkeypatch, capsys):
    state = AgentState(status="completed", verification="passed", stop_reason="verified_finish",
        final_response="private source", messages=[{"content": "private prompt"}],
        metrics={"inference_calls": 3, "decision_calls": 1, "context_records": ["private evidence"],
                 "model_routing": {"model_calls": {"qwen3:8b": 2, "qwen3:14b": 1}}})
    fake_agent(monkeypatch, result=state)
    assert cli.main(run_args(tmp_path, "--json")) == 0
    text = capsys.readouterr().out
    result = json.loads(text)
    assert result["success"] is True and result["coder_calls"] == 3
    assert result["model_calls"] == {"qwen3:8b": 2, "qwen3:14b": 1}
    assert "private" not in text and "Running" not in text


def test_default_check_prefers_target_venv_without_using_harness_python(tmp_path, monkeypatch):
    executable = tmp_path / ".venv/bin/python"
    executable.parent.mkdir(parents=True)
    executable.symlink_to(sys.executable)
    observed = []
    monkeypatch.setattr(cli.subprocess, "run", lambda argv, **kw: (
        observed.append((argv, kw)), SimpleNamespace(returncode=0))[1])
    assert cli._default_check(tmp_path) == (str(executable), "-m", "pytest", "-q")
    assert observed[0][0][0] == str(executable) and observed[0][1]["cwd"] == tmp_path


def test_default_check_uses_activated_path_python_without_local_venv(tmp_path, monkeypatch):
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/activated/python" if name == "python3" else None)
    monkeypatch.setattr(cli.subprocess, "run", lambda *_args, **_kw: SimpleNamespace(returncode=0))
    assert cli._default_check(tmp_path) == ("/activated/python", "-m", "pytest", "-q")


@pytest.mark.parametrize("reason", ["missing_python", "missing_pytest", "python_timeout"])
def test_bad_default_check_environment_fails_before_model_metadata(tmp_path, monkeypatch, models, reason):
    calls = fake_agent(monkeypatch)
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/rg" if name == "rg" else (
        None if reason == "missing_python" else "/python"))
    def probe(*args, **kwargs):
        if reason == "python_timeout":
            raise cli.subprocess.TimeoutExpired(args[0], 10)
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(cli.subprocess, "run", probe)
    with pytest.raises(SystemExit) as error:
        cli.main(["Fix", "--workspace", str(tmp_path)])
    assert error.value.code == 2 and not calls and not models


def test_missing_ripgrep_fails_before_agent_or_models(tmp_path, monkeypatch, models):
    calls = fake_agent(monkeypatch)
    monkeypatch.setattr(cli.shutil, "which", lambda _name: None)
    with pytest.raises(SystemExit) as error:
        cli.main(run_args(tmp_path))
    assert error.value.code == 2 and not calls and not models


@pytest.mark.parametrize("check", ["", "   ", "no-such-command-for-patchrook", "'/unterminated"])
def test_bad_explicit_check_fails_before_models(tmp_path, monkeypatch, models, check):
    calls = fake_agent(monkeypatch)
    with pytest.raises(SystemExit) as error:
        cli.main(["Fix", "--workspace", str(tmp_path), "--check", check])
    assert error.value.code == 2 and not calls and not models


@pytest.mark.parametrize("flags", [
    ["--max-steps", "0"], ["--deadline", "nan"], ["--test-timeout", "0"],
    ["--decision-timeout", "0"], ["--max-output-tokens", "100"],
    ["--temperature", "-1"], ["--top-p", "2"], ["--top-k", "-1"],
    ["--task-mode", "inspect", "--model-routing", "tev"],
    ["--model", "qwen3:14b"],
])
def test_invalid_options_fail_before_inference(tmp_path, monkeypatch, flags):
    calls = fake_agent(monkeypatch)
    with pytest.raises(SystemExit) as error:
        cli.main(run_args(tmp_path, *flags))
    assert error.value.code == 2 and not calls


def test_artifacts_cannot_be_created_inside_target_repository(tmp_path, monkeypatch, models):
    calls = fake_agent(monkeypatch)
    with pytest.raises(SystemExit) as error:
        cli.main(run_args(tmp_path, "--runs-dir", str(tmp_path / "runs")))
    assert error.value.code == 2 and not calls and not models


def test_quoted_user_runs_path_is_normalized_before_agent_creation(tmp_path, monkeypatch):
    calls = fake_agent(monkeypatch)
    assert cli.main(run_args(tmp_path, "--runs-dir", "~/patchrook-test-artifacts")) == 0
    assert calls["constructor"]["runs_dir"] == (Path.home() / "patchrook-test-artifacts").resolve()


def test_ollama_readiness_requires_only_installed_model_metadata(monkeypatch):
    requests = []
    def get(url, **options):
        requests.append((url, options))
        return httpx.Response(200, request=httpx.Request("GET", url), json={"models": [
            {"name": name} for name in ("qwen3:8b", "tev1:4b", "qwen3:14b")]})
    monkeypatch.setattr(cli.httpx, "get", get)
    VALIDATE_MODELS("http://localhost:11434/", ["qwen3:8b", "tev1:4b", "qwen3:14b"])
    assert requests == [("http://localhost:11434/api/tags", {"timeout": 10})]


def test_ollama_missing_model_has_actionable_error(monkeypatch):
    monkeypatch.setattr(cli.httpx, "get", lambda url, **_: httpx.Response(200,
        request=httpx.Request("GET", url), json={"models": [{"name": "qwen3:8b"}]}))
    with pytest.raises(ValueError, match="tev1:4b.*ollama pull"):
        VALIDATE_MODELS("http://localhost:11434", ["qwen3:8b", "tev1:4b"])


def test_ollama_unavailable_and_cloud_models_fail_before_generation(monkeypatch):
    def unavailable(*_args, **_kwargs):
        raise httpx.ConnectError("refused")
    monkeypatch.setattr(cli.httpx, "get", unavailable)
    with pytest.raises(ValueError, match="Start Ollama"):
        VALIDATE_MODELS("http://localhost:11434", ["qwen3:8b"])
    with pytest.raises(ValueError, match="local Ollama"):
        VALIDATE_MODELS("http://localhost:11434", ["model:cloud"])


def checkpoint(settings=None):
    return {"config": {"model_routing": "tev", "escalation_model": "qwen3:14b"},
        "execution": {"routing": {"initial_model": "qwen3:8b"},
            "task_requirements": TaskRequirements.fix().to_dict(), "model_settings": settings or {
                "model": "qwen3:14b", "base_url": "http://saved:11434", "context_size": 8192,
                "max_output_tokens": 768, "request_timeout_seconds": 2200,
                "temperature": .6, "top_p": .95, "top_k": 20, "min_p": 0.}}}


def test_resume_restores_active_model_profile_and_authorizes_saved_checks(tmp_path, monkeypatch, models):
    calls = fake_agent(monkeypatch)
    class Store:
        def __init__(self, runs_dir, session_id):
            assert runs_dir == tmp_path and session_id == "saved"
        def load_checkpoint(self):
            return checkpoint()
    monkeypatch.setattr(cli, "SessionStore", Store)
    assert cli.main(["--resume", "saved", "--runs-dir", str(tmp_path)]) == 0
    model = calls["constructor"]["model"]
    assert model.model == "qwen3:14b" and model.max_output_tokens == 768
    assert model.base_url == "http://saved:11434" and model.request_timeout_seconds == 2200
    assert model.sampling_settings == {"temperature": .6, "top_p": .95, "top_k": 20, "min_p": 0.}
    assert calls["resume"] == ("saved", True) and "requirements" not in calls
    assert set(models[0][1]) == {"qwen3:8b", "qwen3:14b", "tev1:4b"}


@pytest.mark.parametrize("flags", [
    ["--model", "qwen3:8b"], ["--check", CHECK], ["--think"], ["--no-think"],
    ["--max-steps", "12"], ["--deadline", "3600"], ["--base-url", "http://localhost:11434"],
    ["--model-routing", "off"], ["--task-mode", "code"], ["--temperature", ".7"],
])
def test_resume_override_is_rejected_before_checkpoint_access(monkeypatch, flags):
    monkeypatch.setattr(cli, "SessionStore", lambda *_: pytest.fail("No checkpoint access"))
    with pytest.raises(SystemExit) as error:
        cli.main(["--resume", "saved", *flags])
    assert error.value.code == 2


@pytest.mark.parametrize("flags", [[], ["Task", "--resume", "saved"],
    ["Task", "--variant", "A"], ["Task", "--workflow", "coding"], ["Task", "--allow-tests"]])
def test_invalid_or_removed_cli_options_fail_before_agent(monkeypatch, flags):
    calls = fake_agent(monkeypatch)
    with pytest.raises(SystemExit) as error:
        cli.main(flags)
    assert error.value.code == 2 and not calls


@pytest.mark.parametrize("fails", [False, True])
def test_cleanup_releases_only_active_repair_and_preserves_result(tmp_path, monkeypatch, fails):
    unloads = []
    def unload(runtime, name, deadline):
        unloads.append((name, deadline))
        if fails:
            raise OSError("unavailable")
    monkeypatch.setattr(EvidenceWorkflow, "_unload_model", unload)
    emitted = []
    agent = SimpleNamespace(config=defaults.coding_config(), model=OllamaClient(model="qwen3:14b"),
        registry=ToolRegistry(Workspace(tmp_path)), _session=SimpleNamespace(emit=lambda *a, **kw: emitted.append((a, kw))))
    receipt = cli._release_repair_model(agent)
    assert unloads[0][0] == "qwen3:14b"
    assert receipt["status"] == ("failed" if fails else "released_or_absent")
    assert emitted[0][1]["model_generations"] == 0 and receipt["management_seconds"] >= 0


@pytest.mark.parametrize("routing,model", [("tev", "qwen3:8b"), ("off", "qwen3:14b")])
def test_cleanup_leaves_base_or_explicit_single_model_cached(monkeypatch, routing, model):
    monkeypatch.setattr(EvidenceWorkflow, "_unload_model", lambda *_: pytest.fail("Must leave cached"))
    agent = SimpleNamespace(config=defaults.coding_config(model_routing=routing), model=OllamaClient(model=model))
    assert cli._release_repair_model(agent)["status"] == "skipped"


def test_cli_failure_still_cleans_active_14b_without_masking_exit(tmp_path, monkeypatch):
    unloads = []
    def unload(runtime, name, deadline):
        unloads.append(name)
        raise OSError("cleanup failure")
    monkeypatch.setattr(EvidenceWorkflow, "_unload_model", unload)
    fake_agent(monkeypatch, result=AgentState(status="incomplete", verification="failed", stop_reason="repair_call_limit"),
        after_run=lambda agent: setattr(agent, "model", OllamaClient(model="qwen3:14b")))
    assert cli.main(run_args(tmp_path)) == 1
    assert unloads == ["qwen3:14b"]
