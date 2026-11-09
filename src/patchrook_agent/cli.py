"""Run a local coding task with the supported 8B / Tev / 14B profile."""

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time

import httpx

from patchrook_agent.agent import Agent
from patchrook_agent.completion import TaskRequirements
from patchrook_agent import defaults
from patchrook_agent.memory.checkpoint import SessionStore
from patchrook_agent.models.ollama import OllamaClient
from patchrook_agent.session import DEFAULT_RUNS_DIR
from patchrook_agent.tools.base import ToolConfig, ToolError, Workspace
from patchrook_agent.tools.registry import ToolRegistry


READ_ONLY_TOOLS = tuple({"function": {"name": name}} for name in
    ("list_files", "read_file", "search_code", "read_evidence", "finish"))


def _parser():
    parser = argparse.ArgumentParser(description="Fix a repository using local 8B, Tev routing and bounded 14B repair.")
    parser.add_argument("task", nargs="?", help="Task to complete in the current directory")
    parser.add_argument("--workspace", type=Path, help="Target repository (default: current directory)")
    parser.add_argument("--check", action="append", metavar="COMMAND",
        help="Trusted verification command; repeat for multiple checks (default: Python pytest -q). No implicit shell.")
    parser.add_argument("--task-mode", choices=("fix", "code", "inspect"),
        help="fix requires a change and passing checks (default); code allows already-correct code; inspect is read-only")
    models = parser.add_argument_group("model overrides")
    models.add_argument("--model", help="Base coder (default: qwen3:8b)")
    models.add_argument("--escalation-model", help="Repair coder (default: qwen3:14b)")
    models.add_argument("--model-routing", choices=("tev", "off"), help="Routing policy (default: tev)")
    models.add_argument("--base-url", help="Ollama address (default: http://localhost:11434)")
    models.add_argument("--think", action=argparse.BooleanOptionalAction, default=None,
                        help="Enable reasoning for both coders (default: off)")
    models.add_argument("--context-size", type=int, choices=(4096, 8192, 16384), help="Shared context tokens (default: 8192)")
    models.add_argument("--max-output-tokens", type=int, help="Coder output limit (default: 1024)")
    limits = parser.add_argument_group("budget overrides")
    limits.add_argument("--max-steps", type=int, help="Total coder-call limit (default: 12)")
    limits.add_argument("--deadline", type=float, metavar="SECONDS", help="Task deadline (default: 3600)")
    limits.add_argument("--test-timeout", type=float, metavar="SECONDS", help="Limit per check (default: 120)")
    limits.add_argument("--decision-timeout", type=float, metavar="SECONDS", help="Tev decision budget (default: 90)")
    sampling = parser.add_argument_group("sampling overrides")
    for flag, kind, value in (("temperature", float, ".7"), ("top-p", float, ".8"),
                              ("top-k", int, "20"), ("min-p", float, "0")):
        sampling.add_argument("--" + flag, type=kind, help=f"Base coder {flag} (default: {value})")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR,
                        help="Session directory outside the target repository (default: ~/.local/state/patchrook-agent/runs)")
    parser.add_argument("--resume", metavar="SESSION_ID", help="Resume a session with its saved settings and checks")
    parser.add_argument("--json", action="store_true", help="Print a compact JSON result instead of human-readable output")
    return parser


def _value(args, name, default):
    value = getattr(args, name)
    return default if value is None else value


def _default_check(root):
    candidates = (root / ".venv/bin/python", root / ".venv/Scripts/python.exe")
    python = next((str(p) for p in candidates if p.is_file() and os.access(p, os.X_OK)), None)
    python = python or shutil.which("python3") or shutil.which("python")
    if not python:
        raise ValueError("No Python interpreter found for pytest. Activate the project environment or supply --check COMMAND.")
    try:
        ready = subprocess.run([python, "-c",
            "import importlib.util,sys; sys.exit(importlib.util.find_spec('pytest') is None)"],
            cwd=root, capture_output=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ValueError("Could not validate the project's Python. Activate its environment or supply --check COMMAND.") from error
    if ready.returncode:
        raise ValueError(f"pytest is missing from {python}. Install the project's test dependencies or supply --check COMMAND.")
    return (python, "-m", "pytest", "-q")


def _checks(root, values):
    if values is None:
        commands = (_default_check(root),)
    else:
        commands = tuple(tuple(shlex.split(value)) for value in values)
    if not commands or any(not command for command in commands):
        raise ValueError("Each --check must contain an executable command.")
    for command in commands:
        name = command[0]
        if "/" in name or "\\" in name:
            executable = Path(name) if Path(name).is_absolute() else root / name
            available = executable.is_file() and os.access(executable, os.X_OK)
        else:
            available = shutil.which(name) is not None
        if not available:
            raise ValueError(f"Check executable not found: {name}. Activate the project environment or correct --check.")
    return commands


def _validate_models(base_url, names):
    for name in names:
        if not isinstance(name, str) or not name.strip() or name != name.strip() or ":cloud" in name or "\0" in name:
            raise ValueError("Models must be explicit local Ollama model names.")
    try:
        response = httpx.get(base_url.rstrip("/") + "/api/tags", timeout=10)
        response.raise_for_status()
        installed = {row.get("name", row.get("model")) for row in response.json().get("models", [])}
    except (httpx.HTTPError, ValueError, TypeError, AttributeError) as error:
        raise ValueError(f"Cannot read Ollama models at {base_url}. Start Ollama and check --base-url.") from error
    missing = sorted(set(names) - installed)
    if missing:
        raise ValueError("Missing Ollama models: " + ", ".join(missing) +
                         ". Install each with ollama pull MODEL.")


def _release_repair_model(agent):
    """Release this run's active larger coder without changing its outcome."""
    config, model = getattr(agent, "config", None), getattr(agent, "model", None)
    if (config is None or config.model_routing != "tev" or not isinstance(model, OllamaClient)
            or model.model != config.escalation_model):
        return {"status": "skipped", "management_seconds": 0.0}
    started = time.monotonic()
    receipt = {"model": model.model, "status": "released_or_absent"}
    try:
        from patchrook_agent.evidence_workflow import EvidenceWorkflow
        EvidenceWorkflow(agent)._unload_model(model.model, started + 10)
    except Exception as error:
        receipt.update(status="failed", error_type=type(error).__name__)
    receipt["management_seconds"] = time.monotonic() - started
    session = getattr(agent, "_session", None)
    if session is not None:
        try:
            session.emit("trial_model_cleanup", model_generations=0, **receipt)
        except Exception:
            pass
    return receipt


def _summary(state):
    metrics = state.metrics or {}
    routing = metrics.get("model_routing") or {}
    return {"status": state.status, "stop_reason": state.stop_reason, "checks": state.verification,
        "modified_files": state.modified_files, "session_id": state.session_id,
        "session_dir": state.session_dir, "coder_calls": metrics.get("inference_calls"),
        "router_calls": metrics.get("decision_calls"), "model_calls": routing.get("model_calls", {})}


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if bool(args.task) == bool(args.resume):
        parser.error("Provide a task or --resume SESSION_ID.")
    saved_options = ("task_mode", "check", "model", "escalation_model", "model_routing", "base_url", "think",
        "context_size", "max_output_tokens", "max_steps", "deadline", "test_timeout", "decision_timeout",
        "temperature", "top_p", "top_k", "min_p")
    if args.resume and any(getattr(args, name) is not None for name in saved_options):
        parser.error("Resume uses the saved models, budgets and checks; omit run-setting overrides.")
    agent = None
    requires_checks = True
    try:
        try:
            args.runs_dir = args.runs_dir.expanduser().resolve()
        except RuntimeError as error:
            raise ValueError("--runs-dir could not be resolved.") from error
        if not shutil.which("rg"):
            raise ValueError("ripgrep (rg) is missing. Install it before running the harness.")
        if args.resume:
            checkpoint = SessionStore(args.runs_dir, args.resume).load_checkpoint()
            settings = checkpoint["execution"]["model_settings"]
            requires_checks = checkpoint["execution"].get("task_requirements", {}).get("require_tests", True)
            config = checkpoint.get("config", {})
            names = [settings["model"]]
            if config.get("model_routing") == "tev":
                names += [defaults.ROUTER_MODEL, config.get("escalation_model", defaults.REPAIR_MODEL),
                          checkpoint["execution"]["routing"]["initial_model"]]
            _validate_models(settings["base_url"], names)
            model = OllamaClient(**{key: settings[key] for key in
                ("model", "base_url", "context_size", "max_output_tokens")},
                request_timeout_seconds=settings.get("request_timeout_seconds", 1800.),
                temperature=settings.get("temperature", 0.), top_p=settings.get("top_p"),
                top_k=settings.get("top_k"), min_p=settings.get("min_p"))
            agent = Agent(model=model, workspace=args.workspace, runs_dir=args.runs_dir)
            if not args.json:
                print(f"Resuming {args.resume} with its saved settings and checks.", flush=True)
            state = agent.resume(args.resume, allow_tests=True)
        else:
            mode = args.task_mode or "fix"
            inspect_only = mode == "inspect"
            requires_checks = not inspect_only
            if inspect_only and (args.check is not None or args.model_routing == "tev"):
                raise ValueError("Inspect mode is read-only and does not use checks or model routing.")
            root = Workspace(args.workspace or Path.cwd()).root
            if args.runs_dir.expanduser().resolve().is_relative_to(root):
                raise ValueError("--runs-dir must be outside the target repository.")
            commands = () if inspect_only else _checks(root, args.check)
            routing = _value(args, "model_routing", "off" if inspect_only else "tev")
            model_name = _value(args, "model", defaults.BASE_MODEL)
            escalation = _value(args, "escalation_model", defaults.REPAIR_MODEL)
            if routing == "tev" and model_name == escalation:
                raise ValueError("Choose a different repair model or use --model-routing off for a single coder.")
            config = defaults.coding_config(checks=commands,
                context_size=_value(args, "context_size", defaults.CONTEXT_SIZE),
                max_output_tokens=_value(args, "max_output_tokens", defaults.OUTPUT_TOKENS),
                model_routing=routing, escalation_model=escalation, think=bool(args.think), inspect_only=inspect_only,
                decision_timeout=_value(args, "decision_timeout", defaults.DECISION_TIMEOUT_SECONDS),
                **{name: _value(args, name, default) for name, default in
                   (("temperature", .7), ("top_p", .8), ("top_k", 20), ("min_p", 0.))})
            max_steps = _value(args, "max_steps", defaults.MAX_STEPS)
            if max_steps < 1:
                raise ValueError("--max-steps must be positive.")
            deadline = _value(args, "deadline", defaults.DEADLINE_SECONDS)
            tools = ToolConfig(allow_tests=not inspect_only, validate_python_edits=True,
                test_timeout_seconds=_value(args, "test_timeout", defaults.TEST_TIMEOUT_SECONDS), max_output_chars=12000)
            model = OllamaClient(model=model_name, base_url=_value(args, "base_url", defaults.BASE_URL),
                context_size=config.context.context_size, max_output_tokens=_value(args, "max_output_tokens", defaults.OUTPUT_TOKENS),
                request_timeout_seconds=deadline, **{name: getattr(config, name) for name in
                    ("temperature", "top_p", "top_k", "min_p")})
            names = [model_name] + ([defaults.ROUTER_MODEL, escalation] if routing == "tev" else [])
            _validate_models(model.base_url, names)
            agent = Agent(model=model, registry=ToolRegistry(Workspace(root, tools)), config=config,
                max_steps=max_steps, think=bool(args.think),
                deadline_seconds=deadline, runs_dir=args.runs_dir)
            if not args.json:
                print(f"Running in {root} with {model_name}" +
                      (f", {defaults.ROUTER_MODEL} routing and {escalation} repair." if routing == "tev" else "."), flush=True)
                if commands:
                    print("Checks: " + "; ".join(shlex.join(command) for command in commands), flush=True)
            requirements = {"code": TaskRequirements.code, "fix": TaskRequirements.fix,
                            "inspect": TaskRequirements.inspect}[mode]()
            if inspect_only:
                state = agent.run(args.task, tools=list(READ_ONLY_TOOLS), requirements=requirements)
            else:
                state = agent.run(args.task, requirements=requirements)
    except (OSError, ValueError, ToolError, KeyError, TypeError, httpx.HTTPError) as error:
        parser.error(str(error))
    finally:
        if agent is not None:
            _release_repair_model(agent)
    success = state.status == "completed" and (
        state.verification == "passed" if requires_checks else state.verification not in {"failed", "timed_out"})
    summary = _summary(state)
    summary["success"] = success
    if args.json:
        print(json.dumps(summary))
    else:
        print(f"Harness: {state.status}; stop: {state.stop_reason}; checks: {state.verification}")
        if state.modified_files:
            print("Changed: " + ", ".join(state.modified_files))
        if state.status != "completed" and state.final_response:
            print(state.final_response)
        if state.session_id:
            print(f"Session: {state.session_id}\nArtifacts: {state.session_dir}")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
