"""Sequential model -> validated tool -> observation execution loop."""

from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import httpx
from pydantic import TypeAdapter

from patchrook_agent.actions import (
    action_instructions, action_schema, lean_action_instructions, lean_action_schema, parse_action,
)
from patchrook_agent.completion import TaskRequirements, inspection_observed
from patchrook_agent.coding import (
    CODING_SYSTEM_PROMPT, LEGACY_CODING_SYSTEM_PROMPT, discover_source_paths, requires_baseline,
)
from patchrook_agent.config import HarnessConfig
from patchrook_agent.decision_assistance import DecisionAssistance
from patchrook_agent.context.compaction import ContextManager
from patchrook_agent.context.lean import LEAN_SYSTEM_PROMPT, WORKING_SET_PREFIX
from patchrook_agent.inspection import InspectionLedger, canonical_path, visible_lines
from patchrook_agent.lean_progress import LeanProgress, _material_source
from patchrook_agent.read_recovery import corrective_read_action
from patchrook_agent.repair import build_repair_plan
from patchrook_agent.repair_actions import action_for_choice
from patchrook_agent.review_workflow import ReviewWorkflow, REVIEW_TOOLS
from patchrook_agent.proposal_generation import generation_schema
from patchrook_agent.reporting import execution_summary
from patchrook_agent.skills.requests import explicit_skill_names, skill_access_requested
from patchrook_agent.memory.checkpoint import SessionStore, compare_snapshots
from patchrook_agent.memory.state import WorkingMemory
from patchrook_agent.models.ollama import OllamaClient
from patchrook_agent.observability.metrics import ExecutionMetrics
from patchrook_agent.progress import (
    SearchProgress, empty_search_guidance, pending_local_imports, pending_read_selections, source_read_boundaries,
)
from patchrook_agent.session import DEFAULT_RUNS_DIR, Session
from patchrook_agent.tools import ToolConfig, ToolError, ToolRegistry, Workspace


PRE_BATCH_GUIDANCE_SYSTEM_PROMPT = """You are PatchRook Agent, a local autonomous coding assistant.
Complete software engineering tasks accurately and efficiently.
Available tools: list_files, read_file, read_files, search_code, write_file, edit_file, run_tests.
The harness validates and executes actions; use structured tool calls only.
1. Inspect relevant code before editing.
2. Read known file paths directly. Prefer read_files for multiple known files, using narrow line ranges.
3. Prefer targeted exact edits over rewriting entire files.
4. Search only to locate unknown code or line numbers. Never repeat an unchanged search that found no matches.
5. Never claim a tool succeeded without observing its result.
6. Run relevant tests after modifications when possible.
7. If tests fail, investigate and attempt a correction.
8. Never claim tests passed unless an observed test result passed.
9. Report incomplete work and remaining failures.
10. Complete requested actions with tools before finishing. Report changes and observed verification, never just a plan to act.
Prioritize correctness and few inference rounds. Batch independent reads; reuse evidence already available.
"""


SYSTEM_PROMPT = PRE_BATCH_GUIDANCE_SYSTEM_PROMPT.replace(
    "2. Read known file paths directly. Prefer read_files for multiple known files, using narrow line ranges.",
    "2. Read known paths directly. When 2-10 files are known, use one read_files call instead of sequential read_file calls. "
    "Use search match line numbers to read narrow function ranges, expanding only when needed.",
) + 'Batch example: read_files arguments {"files":[{"path":"a.py","start_line":10,"end_line":15},{"path":"b.py","start_line":20,"end_line":25}]}\n'


LEGACY_SYSTEM_PROMPT = """You are PatchRook Agent, a local autonomous coding assistant.
Complete software engineering tasks accurately and efficiently.
Available tools: list_files, read_file, search_code, write_file, edit_file, run_tests.
The harness validates and executes actions; use structured tool calls only.
1. Inspect relevant code before editing.
2. Search narrowly and read relevant line ranges to save context.
3. Prefer targeted exact edits over rewriting entire files.
4. Use tools only when necessary. Multiple requested calls run in order.
5. Never claim a tool succeeded without observing its result.
6. Run relevant tests after modifications when possible.
7. If tests fail, investigate and attempt a correction.
8. Never claim tests passed unless an observed test result passed.
9. Report incomplete work and remaining failures.
10. Finish with a concise summary of changes and verification.
Prioritize correctness over unnecessary tool calls.
"""


def _stable_observation(value: Any) -> Any:
    """Ignore transport identities and timings when detecting actual progress."""
    if isinstance(value, dict):
        return {key: _stable_observation(item) for key, item in value.items()
                if key not in {"id", "index", "duration_seconds", "tool_call_id", "artifact_id"}}
    if isinstance(value, list):
        return [_stable_observation(item) for item in value]
    return value


@dataclass
class AgentState:
    messages: list[dict[str, Any]] = field(default_factory=list)
    steps: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_inference_seconds: float = 0.0
    total_tool_seconds: float = 0.0
    tool_calls: int = 0
    tool_failures: int = 0
    status: str = "incomplete"
    stop_reason: str = "max_steps"
    final_response: str = ""
    modified_files: list[str] = field(default_factory=list)
    verification: str = "not_run"
    session_id: str = ""
    session_dir: str = ""
    active_messages: list[dict[str, Any]] = field(default_factory=list)
    memory: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    recovery_notes: list[str] = field(default_factory=list)
    context_limit_message: str = ""
    completion: dict[str, Any] = field(default_factory=dict)


class Agent:
    def __init__(
        self,
        model: OllamaClient | None = None,
        max_steps: int = 10,
        think: bool = False,
        *,
        workspace: str | Path | None = None,
        registry: ToolRegistry | None = None,
        deadline_seconds: float = 900.0,
        max_tool_failures: int = 3,
        config: HarnessConfig | None = None,
        runs_dir: str | Path | None = None,
        pause_after_step: int | None = None,
        decision_client=None,
    ):
        if max_steps < 1 or max_tool_failures < 1:
            raise ValueError("Step and failure limits must be positive")
        if isinstance(deadline_seconds, bool) or not math.isfinite(deadline_seconds) or deadline_seconds <= 0:
            raise ValueError("Task deadline must be positive and finite")
        if workspace is not None and registry is not None:
            raise ValueError("Supply workspace or registry, not both")
        self.config = config or HarnessConfig.preset("A")
        self.model = model or OllamaClient(**{key: getattr(self.config, key)
                                             for key in ("temperature", "top_p", "top_k", "min_p")})
        self.max_steps = max_steps
        self.think = think
        self.registry = registry or ToolRegistry(Workspace(workspace or Path.cwd()))
        self.deadline_seconds = deadline_seconds
        self.max_tool_failures = max_tool_failures
        if isinstance(self.model, OllamaClient):
            self.config = replace(self.config, **{key: getattr(self.model, key)
                                                  for key in ("temperature", "top_p", "top_k", "min_p")})
        selected_window = getattr(self.model, "context_size", self.config.context.context_size)
        reserve = max(512, self.config.context.reserved_output_tokens,
                      getattr(self.model, "max_output_tokens", 512))
        self.config = replace(self.config, context=replace(self.config.context,
                                                          context_size=selected_window,
                                                          reserved_output_tokens=reserve,
                                                          enabled=(False if self.config.workflow_mode == "evidence"
                                                                   else self.config.context.enabled)))
        self.runs_dir = Path(runs_dir).expanduser() if runs_dir is not None else DEFAULT_RUNS_DIR
        if pause_after_step is not None and (type(pause_after_step) is not int or pause_after_step < 1):
            raise ValueError("pause_after_step must be a positive step number")
        self.pause_after_step = pause_after_step
        self._explicit_workspace = workspace is not None or registry is not None
        self._decision_client = decision_client

    def _model_settings(self) -> dict[str, Any]:
        settings = {key: getattr(self.model, key) for key in (
            "model", "base_url", "context_size", "max_output_tokens", "request_timeout_seconds",
        ) if hasattr(self.model, key)}
        if isinstance(self.model, OllamaClient):
            settings.update(self.model.sampling_settings, stream=False, truncate=False, shift=False)
        return settings

    def _execution(self) -> dict[str, Any]:
        return {"max_steps": self.max_steps, "deadline_seconds": self.deadline_seconds,
                "max_tool_failures": self.max_tool_failures, "think": self.think,
                "model_settings": self._model_settings(), "tool_config": asdict(self.registry.workspace.config),
                "enabled_tools": sorted(self._enabled), "pending_operation": self._pending,
                "inflight_model": self._inflight_model, "inference_attempts": self._inference_attempts,
                "unmeasured_inference_calls": self._unmeasured_inferences,
                "failures_without_progress": self._failures,
                "successful_observations": sorted(self._successful), "tool_attempts": self._tool_attempts,
                "revalidate_paths": sorted(self._revalidate_paths),
                "verification_required": self._verification_required,
                "task_requirements": self._requirements.to_dict() if self._requirements is not None else None,
                "action_protocol_version": 1,
                "completion_recoveries": self._completion_recoveries,
                "search_progress": self._search_progress.to_dict(),
                "search_feedback": self._search_feedback,
                **({"decision": self._decision.to_dict()} if self._decision is not None else {}),
                **({"lean_progress": self._lean_progress.to_dict(),
                    "lean_context": self._session.lean.to_dict(),
                    "lean_recovery_hint": self._lean_recovery_hint,
                    "lean_read_recoveries": self._lean_read_recoveries,
                    "lean_invalid_calls": self._lean_invalid_calls}
                   if self.config.workflow_mode in {"lean", "coding"} else {}),
                **({"coding": {"gathered": self._coding_gathered,
                                "source_queue": list(self._coding_queue),
                                "refresh_paths": sorted(self._coding_refresh_paths),
                                "test_pending": self._coding_test_pending,
                                "skills_prepared": self._coding_skills_prepared,
                                "baseline_complete": self._coding_baseline_complete,
                                "repair": deepcopy(self._repair_state)}}
                   if self.config.workflow_mode == "coding" else {})}

    def _append(self, state: AgentState, message: dict[str, Any]) -> None:
        state.messages.append(message)
        state.active_messages.append(message)

    def _setup(self, state: AgentState, objective: str, *, store: SessionStore | None = None) -> None:
        self._session = Session(self.config, self.registry.workspace, self.model,
                                self.runs_dir, objective, store=store)
        self._session.baseline_reserve = getattr(self.model, "max_output_tokens", 512)
        self.registry.lean_mode = self.config.workflow_mode in {"lean", "coding"}
        self.registry.code_context = self.config.workflow_mode != "coding"
        if self.registry.lean_mode:
            self.registry.enable_lean_tools()
        self.registry.attach_skills(self._session.skills)
        self.registry.attach_artifacts(self._session.artifacts)
        self.registry.capture_enabled = self.config.persist and self.config.logging_enabled and self.config.log_content
        self._pending = None
        self._inflight_model = False
        self._inference_attempts = 0
        self._unmeasured_inferences = 0
        self._failures = 0
        self._successful: set[str] = set()
        self._tool_attempts: dict[str, int] = {}
        self._revalidate_paths: set[str] = set()
        self._verification_required = False
        self._requirements: TaskRequirements | None = None
        self._completion_recoveries = 0
        self._search_progress = SearchProgress()
        self._search_feedback: dict | None = None
        self._lean_progress = (LeanProgress(self.registry.workspace,
                                            material_source_progress=self.config.workflow_mode == "coding")
                               if self.registry.lean_mode else None)
        self._lean_recovery_hint = ""
        self._lean_read_recoveries = 0
        self._lean_invalid_calls: dict[str, str] = {}
        self._coding_gathered = False
        self._coding_skills_prepared = False
        self._coding_baseline_complete = False
        self._coding_queue: list[str] = []
        self._coding_refresh_paths: set[str] = set()
        self._coding_test_pending = False
        self._repair_plan = None
        self._repair_model_messages = []
        self._repair_state = {"fingerprint": "", "episode": "", "selected_id": "", "narrow": False,
                              "evidence_requested": False, "requested_ranges": [],
                              "action_id": "", "action_status": "idle"}
        self._session.repair_packet = None
        self._session.completion_review = False
        self._session.review_source_ranges = []
        self._review = ReviewWorkflow(self)
        # Off creates no decision client, does no discovery beyond the original
        # workflow, and makes no decision-service or model-management requests.
        self._decision = (DecisionAssistance(
            self.config, self.model, self.registry.workspace, client=self._decision_client,
            record=lambda purpose, result: self._record_decision(state, purpose, result),
            emit=lambda event, **data: self._session.emit(event, state.steps, **data))
            if self.config.decision_model == "on" else None)
        self._session.metrics.decision_enabled = self._decision is not None
        self._session.decision = self._decision
        self._session.bind(state)

    def _record_decision(self, state, purpose, result):
        self._session.metrics.record_decision(
            elapsed_seconds=result.elapsed_seconds, management_seconds=result.management_seconds,
            input_tokens=result.input_tokens, fallback_reason=result.fallback_reason,
            cache_hit=result.cache_hit, called=result.called,
            inference_seconds=result.inference_seconds, load_seconds=result.load_seconds)
        self._session.emit("decision_result", state.steps, purpose=purpose,
                           model="tev1:4b", **asdict(result))
        self._session.refresh(state)

    def _checkpoint(self, state: AgentState) -> None:
        self._session.checkpoint(state, self._execution())

    @staticmethod
    def _call_parts(call: Any) -> tuple[str, Any]:
        function = call.get("function", {}) if isinstance(call, dict) else {}
        function = function if isinstance(function, dict) else {}
        name = function.get("name", "unknown")
        return name if isinstance(name, str) else "unknown", function.get("arguments", {})

    @staticmethod
    def _argument_path(arguments: Any) -> str | None:
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:
                return None
        path = arguments.get("path") if isinstance(arguments, dict) else None
        return path.split("::", 1)[0] if isinstance(path, str) else None

    def _canonical_path(self, arguments: Any) -> str | None:
        path = self._argument_path(arguments)
        if path is not None:
            try:
                return self.registry.workspace.relative(self.registry.workspace.resolve(path))
            except ToolError:
                return path
        return None

    def _fits_context(self, state: AgentState, tools: list[dict[str, Any]]) -> bool:
        # An estimate for code/JSON, not a model tokenizer. The Ollama client
        # also requests truncate=False and shift=False as the server-side guard.
        # Never discard history when the configured budget is exhausted.
        payload = json.dumps({"messages": state.messages, "tools": tools},
                             ensure_ascii=False, separators=(",", ":"))
        estimated_tokens = math.ceil(len(payload.encode("utf-8")) / 3) + 32 * len(state.messages)
        reserve = getattr(self.model, "max_output_tokens", 512)
        return estimated_tokens + reserve < getattr(self.model, "context_size", 8192)

    @staticmethod
    def _repair_arguments(arguments: Any) -> Any:
        if isinstance(arguments, str):
            try:
                return json.loads(arguments)
            except ValueError:
                pass
        return arguments

    def _invalid_call_key(self, name: str, arguments: Any) -> str:
        return json.dumps([name, self._repair_arguments(arguments)], sort_keys=True, ensure_ascii=False)

    def _invalid_target_now_exists(self, name: str, arguments: Any, code: str | None) -> bool:
        if name != "run_tests" or code != "not_found":
            return False
        arguments = self._repair_arguments(arguments)
        path = arguments.get("path") if isinstance(arguments, dict) else None
        if not isinstance(path, str):
            return False
        try:
            return self.registry.workspace.resolve(path.split("::", 1)[0]).exists()
        except (OSError, ValueError, ToolError):
            return False

    def _tool_repair_instruction(self, name: str, arguments: Any, result: dict) -> str:
        code = result.get("error", {}).get("code")
        if name == "run_tests" and code in {"not_found", "invalid_arguments", "invalid_path"}:
            hint = ("The test request was rejected before pytest ran. Do not repeat the rejected request. "
                    "Call run_tests with empty arguments {} for repository-wide pytest discovery.")
            if "list_files" in self._enabled:
                hint += " To select a narrower target, first use list_files to discover an existing test path."
            return hint
        if name == "read_skill" and code in {"invalid_skill", "skill_not_found", "invalid_arguments"}:
            names = result.get("valid_names")
            arguments = self._repair_arguments(arguments)
            selected = arguments.get("name") if isinstance(arguments, dict) else None
            if isinstance(names, list) and names and all(isinstance(item, str) for item in names) and selected not in names:
                return ("The skill name was rejected. Choose a name from these available trusted skills: "
                        + json.dumps(names, ensure_ascii=False) + ". Load one only if it is needed for the task.")
        return ""

    def _execute(self, state: AgentState, call: Any, enabled: set[str],
                 deadline: float, *, rejection: str = "", controller: bool = False) -> dict[str, Any]:
        name, arguments = self._call_parts(call)
        signature = json.dumps([name, arguments], sort_keys=True, ensure_ascii=False)
        attempts = self._tool_attempts.get(signature, 0)
        self._tool_attempts[signature] = attempts + 1
        state.tool_calls += 1
        self._pending = call
        self._checkpoint(state)
        self._session.emit("tool_request", state.steps, name=name, arguments=arguments)
        started = time.monotonic()
        self.registry.last_execution = None
        path = self._canonical_path(arguments)
        invalid_key = self._invalid_call_key(name, arguments) if self.registry.lean_mode else None
        prior_invalid = self._lean_invalid_calls.get(invalid_key)
        if name not in enabled:
            result = self.registry.error("unknown_tool", "Tool is not enabled for this task")
        elif rejection:
            result = self.registry.error("inspection_required", rejection)
        elif (self.config.workflow_mode == "coding" and name == "read_artifact"
              and (not isinstance(self._repair_arguments(arguments), dict)
                   or self._repair_arguments(arguments).get("artifact_id") not in self._coding_artifact_ids(state))):
            result = self.registry.error("invalid_artifact", "Use an available artifact_id returned by a tool in this session.")
        elif prior_invalid and not self._invalid_target_now_exists(name, arguments, prior_invalid):
            result = self.registry.error("repeated_invalid_call",
                                         f"The identical {name} request was already rejected ({prior_invalid}). "
                                         "Use the corrected options from the preceding recovery instruction.")
            result["executed"] = False
        elif name in {"edit_file", "write_file"} and path in self._revalidate_paths:
            result = self.registry.error("workspace_changed", "Read this file again before modifying it after resume")
        else:
            result = self.registry.dispatch(name, arguments,
                                            remaining_seconds=deadline - started)
        duration = time.monotonic() - started
        state.total_tool_seconds += duration
        result = self._session.expand_observation(name, result, self.registry.last_execution, step=state.steps)
        if not result.get("success"):
            state.tool_failures += 1
        elif self.registry.lean_mode:
            self._lean_invalid_calls.pop(invalid_key, None)
        if name in {"write_file", "edit_file"} and result.get("success") and result.get("modified"):
            self._lean_invalid_calls.clear()
            if result["path"] not in state.modified_files:
                state.modified_files.append(result["path"])
            state.verification = "not_run"
            if self.config.workflow_mode == "coding":
                self._coding_test_pending = True
                self._coding_refresh_paths.add(result["path"])
                if self._decision is not None:
                    self._decision.invalidate()
        if self.registry.lean_mode and name in enabled:
            if name == "edit_file" and result.get("error", {}).get("code") == "no_change":
                self._lean_recovery_hint = (
                    "The edit changed nothing. Do not repeat identical old_text and new_text. "
                    "Use the observed source and unresolved test failure to propose a different correction; "
                    "inspect related code only if a specific detail is missing.")
                self._session.emit("no_change_recovery", state.steps, name=name, guidance=self._lean_recovery_hint)
            repair = self._tool_repair_instruction(name, arguments, result)
            if repair:
                self._lean_invalid_calls[invalid_key] = result["error"]["code"]
                self._lean_recovery_hint = repair
                self._session.emit("tool_argument_recovery", state.steps, name=name, guidance=repair)
        if name == "run_tests" and result.get("status") in {"passed", "failed", "timed_out"}:
            state.verification = result["status"]
        reads = ([result] if name == "read_file" else result.get("files", []) if name in {"read_files", "read_code"} else [])
        fresh_inspection = InspectionLedger()
        if self._revalidate_paths:
            fresh_inspection.observe(name, arguments, result, workspace=self.registry.workspace)
        for observed in reads:
            if not isinstance(observed, dict) or not observed.get("success"):
                continue
            read_path = self._canonical_path({"path": observed.get("path")})
            evidence = fresh_inspection.files.get(read_path)
            if read_path in self._revalidate_paths and evidence is not None and evidence.inspected:
                self._revalidate_paths.remove(read_path)
            if not self._revalidate_paths:
                self._session.memory.blockers = [note for note in self._session.memory.blockers
                                                 if not note.startswith("Relevant files changed since checkpoint:")]
        self._session.observe_tool(state, name, arguments, result, duration, retry=attempts > 0,
                                   available=self.registry.last_execution)
        if self._lean_progress is not None:
            previous_stall = self._lean_progress.stall_count
            self._lean_progress.observe(name, arguments, result)
            if controller and name in {"read_code", "read_file"}:
                # Bounded controller gathering can encounter empty modules; it
                # is not a repeated unproductive model decision.
                self._lean_progress.stall_count = min(previous_stall, self._lean_progress.stall_count)
            state.verification = self._session.memory.verification = self._lean_progress.verification_status
        # Instructions are pinned exactly once on the next context preparation;
        # the tool observation only acknowledges that the document was loaded.
        if name == "read_skill" and result.get("success"):
            result = {key: value for key, value in result.items() if key != "content"}
        observation = {"role": "tool", "tool_name": name,
                       "content": json.dumps(result, ensure_ascii=False, separators=(",", ":"))}
        if isinstance(call, dict) and isinstance(call.get("id"), str):
            observation["tool_call_id"] = call["id"]
        self._append(state, observation)
        if (self.config.workflow_mode == "coding" and not controller and name == "read_file"
                and (self._session.repair_packet is not None or self._session.completion_review)
                and result.get("success")):
            self._remember_repair_read(state, result)
        if self.config.workflow_mode == "coding":
            self._review.observed(state, name, result)
        self._search_progress.observe(name, arguments, result)
        if self._search_progress.action is None:
            self._search_feedback = None
        self._pending = None
        if self._decision is not None:
            if not result.get("success"):
                self._decision.failure(self._session.memory.objective, name, result,
                                       self._lean_progress.generation, deadline,
                                       allow_call=self.config.workflow_mode != "coding")
            elif name == "run_tests":
                self._decision.recovery_hint = ""
        self._checkpoint(state)
        return result

    def run(self, task: str, tools: list[dict[str, Any]] | None = None, *,
            requirements: TaskRequirements | None = None) -> AgentState:
        if self.config.workflow_mode == "evidence":
            from patchrook_agent.evidence_workflow import EvidenceWorkflow
            return EvidenceWorkflow(self).run(task, tools=tools, requirements=requirements)
        if (self.config.action_protocol == "schema" or self.config.workflow_mode in {"lean", "coding"}) and requirements is None:
            requirements = TaskRequirements.code()
        if requirements is not None and not isinstance(requirements, TaskRequirements):
            raise ValueError("requirements must be TaskRequirements or None")
        prompt = (CODING_SYSTEM_PROMPT if self.config.workflow_mode == "coding" else
                  LEAN_SYSTEM_PROMPT if self.config.workflow_mode == "lean" else SYSTEM_PROMPT)
        state = AgentState(messages=[{"role": "system", "content": prompt},
                                     {"role": "user", "content": task}])
        state.active_messages = list(state.messages)
        self._setup(state, task)
        self._requirements = requirements
        definitions = self.registry.definitions()
        if tools is not None:
            # Preserve the earlier optional tools argument as a subset selector;
            # the registry's trusted schemas remain authoritative.
            requested = {item.get("function", {}).get("name") for item in tools}
            definitions = [item for item in definitions if item["function"]["name"] in requested]
        enabled = {item["function"]["name"] for item in definitions}
        self._enabled = enabled
        if "read_files" not in enabled and self.config.workflow_mode not in {"lean", "coding"}:
            # New restricted runs use the already trusted pre-batch policy.
            # Resume preserves its original raw and active instructions.
            state.messages[0]["content"] = LEGACY_SYSTEM_PROMPT
        self.registry.guidance_tools = None if enabled == set(self.registry.tools) else enabled
        definitions = self.registry.definitions(enabled_tools=enabled)
        if requirements is not None and requirements.instructions() and self.config.workflow_mode not in {"lean", "coding"}:
            self._append(state, {"role": "system", "content": requirements.instructions()})
        self._session.emit("session_start", objective=task, workspace=str(self.registry.workspace.root),
                           config=self.config.to_dict(), model_settings=self._model_settings(), think=self.think,
                           task_requirements=requirements.to_dict() if requirements is not None else None)
        self._checkpoint(state)
        return self._continue(state, definitions)

    def _completion_missing(self, state: AgentState) -> list[str]:
        if self._requirements is None:
            return []
        evidence = {"inspected": inspection_observed(state.messages),
                    "changed": bool(state.modified_files), "verification": state.verification}
        missing = self._requirements.missing(**evidence)
        if self._lean_progress is not None:
            needed = self._requirements.require_tests or bool(state.modified_files) or self._verification_required
            for item in self._lean_progress.missing(require_tests=needed):
                if item not in missing:
                    missing.append(item)
        if self.config.workflow_mode == "coding":
            for name in explicit_skill_names(self._session.memory.objective):
                if name not in self._session.skills.loaded:
                    missing.append("load requested trusted skill: " + name)
            if requires_baseline(self._session.memory.objective) and not self._coding_baseline_complete:
                missing.append("run requested baseline tests before editing")
        state.completion = {"requirements": self._requirements.to_dict(),
                            "evidence": evidence, "satisfied": not missing, "missing": missing}
        if self._lean_progress is not None:
            state.completion["progress"] = self._lean_progress.summary()
            state.completion["outstanding_tests"] = list(self._lean_progress.failed)
        return missing

    def _lean_progress_recovery(self, state: AgentState) -> str:
        """Choose the next missing action without prescribing a code change."""
        prefix = "Recent calls added no new evidence. Do not repeat unchanged reads or searches. "
        requirements = self._requirements or TaskRequirements()
        if self._revalidate_paths or (requirements.require_inspection and not inspection_observed(state.messages)):
            reader = next((name for name in ("read_code", "read_files", "read_file") if name in self._enabled), None)
            next_action = (f"Next action: inspect relevant source with {reader}." if reader else
                           "Source inspection is still required; explain which needed reading tool is unavailable.")
        elif self._lean_progress.failed:
            scopes = ", ".join(scope[:120] for scope in list(self._lean_progress.failed)[:3])
            if self.config.workflow_mode == "coding":
                next_action = ("Next action: use the observed implementation and exact failed tests for " + scopes
                               + " to identify a targeted correction and submit edit_file. "
                                 "Read only a specific missing implementation range that prevents the correction; "
                                 "comments, signatures and unchanged rereads do not advance this task. "
                                 "The harness will rerun tests after an actual edit.")
            else:
                next_action = ("Next action: investigate the observed test failure for " + scopes
                               + ", correct its cause, then rerun the relevant tests.")
        elif requirements.require_change and not state.modified_files:
            editor = next((name for name in ("edit_file", "write_file") if name in self._enabled), None)
            next_action = (f"Next action: use the observed source to identify and apply the requested change with {editor}. "
                           "Read more only if a specific missing detail prevents that decision." if editor else
                           "A requested source change remains; explain which needed editing tool is unavailable.")
        elif (requirements.require_tests or state.modified_files or self._verification_required) and state.verification != "passed":
            next_action = ("Next action: run the relevant tests with run_tests and inspect their result."
                           if "run_tests" in self._enabled else
                           "Verification is still required; explain which needed test tool is unavailable.")
        else:
            next_action = "Finish with the work and verification actually observed if the requested task is complete."
        return prefix + next_action

    def _search_gate(self, state: AgentState) -> bool:
        """Apply durable search decisions before another inference, including resume."""
        if self._lean_progress is not None:
            action = self._lean_progress.action
            if action == "stop":
                state.stop_reason = "stalled_progress"
                state.final_response = "Incomplete: repeated actions produced no new useful evidence after recovery guidance."
                self._session.emit("progress_stalled", state.steps, progress=self._lean_progress.to_dict())
                return False
            if action == "recover":
                if (self.config.workflow_mode == "coding" and self._repair_plan is not None
                        and not self._repair_state["narrow"]):
                    # Change the evidence presented once; another warning or
                    # decision-model swap would leave the same read loop intact.
                    self._repair_state["narrow"] = True
                    self._set_repair_packet(state)
                self._lean_recovery_hint = self._lean_recovery_hint or self._lean_progress_recovery(state)
                self._lean_progress.acknowledge_recovery()
                self._session.emit("progress_recovery", state.steps, guidance=self._lean_recovery_hint)
                self._checkpoint(state)
            return True
        progress = self._search_progress.to_dict()
        if self._search_progress.action == "stop":
            state.stop_reason = "stalled_search"
            state.final_response = "Incomplete: repeated the same empty search after recovery guidance."
            self._session.emit("search_stalled", state.steps, progress=progress)
            return False
        if self._search_progress.action == "recover" and self._search_feedback != progress:
            self._search_feedback = progress
            note = empty_search_guidance(self._session.memory.objective, enabled_tools=self._enabled)
            self._append(state, {"role": "system", "content": note})
            self._session.emit("search_recovery", state.steps, progress=progress, guidance=note)
            self._checkpoint(state)
        return True

    def _schema_completion(self, state: AgentState) -> tuple[bool, list[str]]:
        missing = self._completion_missing(state)
        if self._revalidate_paths:
            missing = [*missing, "reinspect workspace changes before completion"]
        if (state.modified_files or self._verification_required) and state.verification != "passed":
            requirement = "observe passing tests after the last change"
            if requirement not in missing:
                missing.append(requirement)
        if state.verification in {"failed", "timed_out"} and not missing:
            missing.append("resolve the observed test failure")
        return not missing, missing

    def _read_guidance(self, state: AgentState) -> str:
        if not self._enabled & {"read_file", "read_files", "read_code"}:
            return ""
        if self.config.workflow_mode == "lean" and "read_code" in self._enabled:
            correction = corrective_read_action(state.messages, self.registry.workspace)
            if correction is not None:
                return (" Correct the read arguments with: "
                        + json.dumps(correction, ensure_ascii=False, separators=(",", ":")))
        boundaries = (source_read_boundaries(state.messages, self.registry.workspace,
                                             external=self._session.lean.stale_before)
                      if self.config.workflow_mode == "lean" else None)
        selections = pending_read_selections(state.messages, self.registry.workspace, limit=8,
                                              source_boundaries=boundaries)
        paths = {selection["path"] for selection in selections}
        for path in pending_local_imports(state.messages, self.registry.workspace,
                                          task=self._session.memory.objective, source_boundaries=boundaries):
            if path not in paths:
                selections.append({"path": path})
                paths.add(path)
        bounded = []
        for selection in selections[:8]:
            candidate = json.dumps(bounded + [selection], ensure_ascii=False, separators=(",", ":"))
            if len(candidate) > 900:
                break
            bounded.append(selection)
        if not bounded:
            return ""
        if self.config.workflow_mode == "lean" and "read_code" in self._enabled:
            # Keep automatic reads separate from exact continuation ranges.
            ranged = next((item for item in bounded if "start_line" in item or "end_line" in item), None)
            if ranged is not None and "read_file" in self._enabled:
                suggested = {"action": "tool", "tool": "read_file", "arguments": ranged}
            else:
                suggested = {"action": "tool", "tool": "read_code",
                             "arguments": {"paths": [item["path"] for item in bounded]}}
            return (" Suggested next read (guidance, not inspection): "
                    + json.dumps(suggested, ensure_ascii=False, separators=(",", ":")))
        reader = ("Batch independent selections with read_code." if "read_code" in self._enabled else
                  "Batch independent selections with read_files." if "read_files" in self._enabled else
                  "Use read_file for one selection.")
        return (" Suggested next reads from partial/omitted selections or unread local files: "
                + json.dumps(bounded, ensure_ascii=False, separators=(",", ":"))
                + ". Read known ranges directly; these hints do not count as inspection. " + reader)

    def _continue(self, state: AgentState, definitions: list[dict[str, Any]]) -> AgentState:
        deadline = time.monotonic() + max(0.0, self.deadline_seconds - self._session.wall_base)
        try:
            self._loop(state, definitions, deadline)
        except KeyboardInterrupt:
            state.status, state.stop_reason = "incomplete", "interrupted"
            self._session.emit("interrupted", state.steps, pending_operation=self._pending,
                               inflight_model=self._inflight_model)
        finally:
            self._completion_missing(state)
            self._report_observed_outcome(state)
            self._session.finish(state, self._execution())
        return state

    def _report_observed_outcome(self, state: AgentState) -> None:
        if self.config.workflow_mode == "coding":
            state.final_response = execution_summary(state, uncertain_paths=self._revalidate_paths)
            # Inspection questions need their semantic answer in addition to
            # factual execution status. Coding/fix reports never reuse model
            # claims about changes or tests.
            if (state.status == "completed" and self._requirements is not None
                    and not self._requirements.require_change and not self._requirements.require_tests
                    and not state.modified_files and not self._revalidate_paths):
                last = next((message for message in reversed(state.messages)
                             if message.get("role") == "assistant"), {})
                answer = last.get("content")
                if not last.get("tool_calls") and isinstance(answer, str) and answer.strip():
                    state.final_response += "\n\n" + answer.strip()
            self._session.emit("execution_summary", state.steps, summary=state.final_response,
                               source="observed_execution")

    def _coding_call(self, state: AgentState, name: str, arguments: dict,
                     deadline: float) -> dict:
        """Record controller actions through the same validated execution path."""
        call = {"type": "function", "function": {"name": name, "arguments": arguments}}
        self._session.emit("coding_controller", state.steps, name=name, arguments=arguments)
        self._append(state, {"role": "assistant", "content": "", "tool_calls": [call]})
        return self._execute(state, call, self._enabled, deadline, controller=True)

    def _coding_prepare(self, state: AgentState, deadline: float) -> bool:
        """Gather source once, refresh stale files, and verify each patch batch."""
        if not self._coding_skills_prepared:
            for name in explicit_skill_names(self._session.memory.objective):
                if name in self._session.skills.loaded:
                    continue
                if time.monotonic() >= deadline:
                    state.stop_reason = "deadline_exceeded"
                    return False
                if "read_skill" not in self._enabled:
                    state.stop_reason = "requested_skill_unavailable"
                    return False
                result = self._coding_call(state, "read_skill", {"name": name}, deadline)
                if not result.get("success"):
                    state.stop_reason = "requested_skill_unavailable"
                    return False
            self._coding_skills_prepared = True
            self._checkpoint(state)
        if not self._coding_gathered:
            if not self._coding_queue:
                self._coding_queue = discover_source_paths(self.registry.workspace, self._session.memory.objective)
                if (self._decision is not None and self.config.review_mode == "off"
                        and not requires_baseline(self._session.memory.objective)):
                    self._coding_queue = self._decision.prioritize(
                        self._session.memory.objective, self._coding_queue, deadline)
                self._checkpoint(state)
            # A small batch leaves room for exact numbered bodies under the
            # existing observation cap. Omitted files are retried individually.
            while self._coding_queue:
                if time.monotonic() >= deadline:
                    state.stop_reason = "deadline_exceeded"
                    return False
                paths = self._coding_queue[:3]
                if "read_code" not in self._enabled:
                    state.stop_reason = "inspection_unavailable"
                    state.final_response = "Incomplete: the coding workflow requires read_code for source gathering."
                    return False
                result = self._coding_call(state, "read_code", {"paths": paths}, deadline)
                returned = {item.get("path") for item in result.get("files", []) if isinstance(item, dict)}
                # All returned records (including explicit errors) are observed;
                # the model may request precise continuations if content clips.
                consumed = [path for path in paths if path in returned]
                if not consumed:
                    if len(paths) > 1:
                        result = self._coding_call(state, "read_code", {"paths": paths[:1]}, deadline)
                    consumed = paths[:1]
                self._coding_queue = [path for path in self._coding_queue if path not in consumed]
                self._checkpoint(state)
            self._coding_gathered = True
            self._checkpoint(state)
        if requires_baseline(self._session.memory.objective) and not self._coding_baseline_complete:
            if state.modified_files or self._verification_required:
                state.stop_reason = "baseline_not_observed_before_edit"
                return False
            if not self.registry.workspace.config.allow_tests or "run_tests" not in self._enabled:
                state.stop_reason = "verification_not_authorized"
                return False
            result = self._coding_call(state, "run_tests", {"path": "."}, deadline)
            if result.get("status") not in {"passed", "failed", "timed_out"}:
                state.stop_reason = "verification_error"
                return False
            self._coding_baseline_complete = True
            self._coding_test_pending = False
            self._checkpoint(state)
        for path in sorted(self._revalidate_paths):
            if time.monotonic() >= deadline:
                state.stop_reason = "deadline_exceeded"
                return False
            result = self._coding_call(state, "read_code", {"paths": [path]}, deadline)
            if path in self._revalidate_paths:
                state.stop_reason = "workspace_changed"
                state.final_response = f"Incomplete: changed source {path} could not be reinspected."
                return False
        for path in sorted(self._coding_refresh_paths):
            if time.monotonic() >= deadline:
                state.stop_reason = "deadline_exceeded"
                return False
            # A failed test must be paired with actual post-patch source so a
            # correction can use exact current text on the following turn.
            self._coding_call(state, "read_code", {"paths": [path]}, deadline)
            self._coding_refresh_paths.discard(path)
            self._checkpoint(state)
        if self._coding_test_pending:
            if not self.registry.workspace.config.allow_tests or "run_tests" not in self._enabled:
                state.stop_reason = "verification_not_authorized"
                state.final_response = "Incomplete: the coding workflow requires authorized pytest execution."
                return False
            result = self._coding_call(state, "run_tests", {"path": "."}, deadline)
            self._coding_test_pending = False
            self._checkpoint(state)
            if result.get("status") not in {"passed", "failed", "timed_out"}:
                state.stop_reason = "verification_error"
                state.final_response = "Incomplete: pytest could not run. " + result.get("error", {}).get("message", "")
                return False
        self._coding_repair(state, deadline)
        return True

    def _set_repair_packet(self, state):
        packet = self._repair_plan.packet(self._repair_state["selected_id"],
                                          narrow=self._repair_state["narrow"])
        if self._repair_state["evidence_requested"]:
            item = next(item for item in self._repair_plan.items if item.id == packet["id"])
            span = getattr(item, "evidence_range", None)
            if span:
                packet["source_ranges"].append({"path": item.path, "start_line": span[0],
                                                  "end_line": min(span[1], span[0] + 79)})
        self._session.repair_packet = packet
        item = next(item for item in self._repair_plan.items if item.id == packet["id"])
        packet["selected_path"] = item.path
        packet["selected_symbol"] = item.symbol
        packet["requested_ranges"] = deepcopy(self._repair_state["requested_ranges"])
        self._session.emit("repair_packet", state.steps, phase="repair", **packet)

    def _remember_repair_read(self, state, result):
        """Keep a few specifically requested missing ranges across later reads.

        Canonical controller inspection can already contain these lines even
        when the compact request omitted them. Remember selectors only; current
        observed text and actual prepared visibility still govern every edit.
        Broad rereads and header-only extensions cannot expand this focus.
        """
        path = canonical_path(result.get("path"))
        lines = visible_lines(result)
        if path is None or not lines or len(lines) > 24 or max(lines) - min(lines) >= 24:
            return
        represented = set(self._coding_visible_lines(path, self._repair_model_messages))
        represented.update(number for selection in self._repair_state["requested_ranges"]
                       if selection["path"] == path
                       for number in range(selection["start_line"], selection["end_line"] + 1))
        novel = {number: text for number, text in lines.items() if number not in represented}
        if not novel or not _material_source(path, novel) or len(self._repair_state["requested_ranges"]) >= 4:
            return
        selector = {"path": path, "start_line": min(lines), "end_line": max(lines)}
        self._repair_state["requested_ranges"].append(selector)
        self._session.emit("repair_evidence_requested", state.steps, **selector)

    def _coding_repair(self, state, deadline):
        """Compile observed failure work after verification, before coder use."""
        self._session.completion_review = bool(
            state.modified_files and state.verification == "passed" and not self._lean_progress.failed
            and not self._revalidate_paths and not self._coding_test_pending and not self._coding_refresh_paths)
        self._session.review_source_ranges = (
            deepcopy(self._repair_state["requested_ranges"]) if self._session.completion_review else [])
        if (not self._lean_progress.failed or self._revalidate_paths
                or self._requirements is not None and not self._requirements.require_change):
            self._repair_plan = None
            self._session.repair_packet = None
            return
        plan = build_repair_plan(state.messages, self._session.memory.objective,
                                 revision=self._lean_progress.generation,
                                 stale_before=self._session.lean.stale_before)
        if plan is None or not plan.items:
            self._repair_plan = None
            self._session.repair_packet = None
            return
        self._repair_plan = plan
        episode = hashlib.sha256(json.dumps([
            self._lean_progress.generation,
            {scope: self._lean_progress._test_outcomes.get(scope)
             for scope in self._lean_progress.failed}], sort_keys=True).encode()).hexdigest()
        if self._repair_state["fingerprint"] != plan.fingerprint:
            same_episode = self._repair_state["episode"] == episode
            if same_episode:
                # Extra comments, signatures, or overlapping source must not
                # undo the single recovery or pay for another routing decision.
                self._repair_state["fingerprint"] = plan.fingerprint
            else:
                choice = (self._decision.select_repair(self._session.memory.objective, plan, deadline,
                                                       allow_call=state.steps == 0,
                                                       allow_inspection="read_file" in self._enabled)
                          if self._decision is not None and self.config.review_mode == "off" else plan.items[0].id)
                action = action_for_choice(plan, choice, allow_inspection="read_file" in self._enabled)
                selected = next(item for item in plan.items if item.id == action.item_id)
                self._repair_state = {"fingerprint": plan.fingerprint, "episode": episode,
                                      "selected_id": selected.id, "narrow": False,
                                      "evidence_requested": False, "requested_ranges": [],
                                      "action_id": action.id, "action_status": "pending"}
        self._repair_state["episode"] = episode
        if not any(item.id == self._repair_state["selected_id"] for item in plan.items):
            self._repair_state["selected_id"] = plan.items[0].id
        if self._repair_state["action_status"] == "pending":
            self._dispatch_repair_action(state, plan, deadline)
        self._set_repair_packet(state)
        self._checkpoint(state)

    def _dispatch_repair_action(self, state, plan, deadline):
        """Execute a closed read/prepare action without a Qwen navigation turn."""
        action = action_for_choice(plan, self._repair_state["action_id"],
                                   allow_inspection="read_file" in self._enabled)
        self._repair_state["action_id"] = action.id
        self._repair_state["selected_id"] = action.item_id
        self._checkpoint(state)
        details = {"action_id": action.id, "kind": action.kind, "item_id": action.item_id,
                   "path": action.path, "start_line": action.start_line, "end_line": action.end_line,
                   "fingerprint": plan.fingerprint}
        self._session.emit("decision_action", state.steps, **details)
        result = None
        if action.kind in {"inspect_helper", "inspect_config"}:
            # Paths and ranges originate in observed, bounded work items, not
            # generated arguments. The usual registry validates them again.
            if "read_file" in self._enabled and time.monotonic() < deadline:
                result = self._coding_call(state, "read_file", {
                    "path": action.path, "start_line": action.start_line, "end_line": action.end_line}, deadline)
                lines = visible_lines(result)
                if result.get("success") and lines and max(lines) - min(lines) < 24:
                    self._repair_state["requested_ranges"] = [{
                        "path": action.path, "start_line": min(lines), "end_line": max(lines)}]
            else:
                result = {"success": False, "error": {"code": "action_unavailable",
                          "message": "The selected evidence read is unavailable within the task's tools or deadline."}}
        self._repair_state["action_status"] = "applied"
        self._session.emit("decision_action_result", state.steps, **details,
                           success=result.get("success") if result is not None else True,
                           operation="read_file" if result is not None else "prepare_patch",
                           error=result.get("error") if result is not None else None)
        self._checkpoint(state)

    def _coding_edit_error(self, state: AgentState, call: dict, model_messages: list[dict]) -> str:
        """Require patch text to have been visible before this model response."""
        name, arguments = self._call_parts(call)
        arguments = self._repair_arguments(arguments)
        if name not in {"edit_file", "write_file"} or not isinstance(arguments, dict):
            return ""
        path = self._canonical_path(arguments)
        if not path:
            return ""  # The registry reports malformed arguments precisely.
        if name == "write_file":
            try:
                exists = self.registry.workspace.resolve(path).exists()
            except ToolError:
                return ""
            return "Use edit_file with observed exact source to modify an existing file." if exists else ""
        old = arguments.get("old_text")
        if not isinstance(old, str) or not old:
            return ""
        lines = self._coding_visible_lines(path, model_messages)
        block, previous = [], None
        for number, source in sorted(lines.items()):
            if previous is not None and number != previous + 1:
                if old in "\n".join(block) + "\n":
                    return ""
                block = []
            block.append(source)
            previous = number
        if old in "\n".join(block) + "\n":
            return ""
        return "Read the specific missing source with read_code or read_file before proposing this exact edit."

    def _coding_visible_lines(self, path, model_messages):
        lines = {}
        # Only credit source in the request Qwen actually received. Controller
        # reads omitted by the context budget are still available on disk, but
        # cannot authorize a guessed patch from unseen text.
        for message in model_messages:
            content = message.get("content", "")
            if message.get("role") == "user" and content.startswith(WORKING_SET_PREFIX):
                active = False
                for line in content.splitlines():
                    if line.startswith("Observed source "):
                        active = line.startswith(f"Observed source {path} lines ")
                        continue
                    if active:
                        number, separator, source = line.partition(": ")
                        if separator and number.isdigit():
                            lines[int(number)] = source
                        else:
                            active = False
            if message.get("role") != "tool":
                continue
            try:
                result = json.loads(content)
            except (ValueError, TypeError):
                continue
            if not isinstance(result, dict):
                continue
            records = ([result] if message.get("tool_name") == "read_file" else
                       result.get("files", []) if message.get("tool_name") in {"read_code", "read_files"} else [])
            for record in records:
                if isinstance(record, dict) and record.get("success") and self._canonical_path(record) == path:
                    lines.update(visible_lines(record))
        return lines

    def _coding_artifact_ids(self, state: AgentState) -> tuple[str, ...]:
        if self._session.artifacts is None:
            return ()
        observed = []
        for message in state.messages:
            if message.get("role") != "tool":
                continue
            try:
                result = json.loads(message.get("content", ""))
            except (ValueError, TypeError):
                continue
            if isinstance(result, dict) and isinstance(result.get("artifact_id"), str):
                observed.append(result["artifact_id"])
        return self._session.artifacts.available_ids(observed)

    def _coding_definitions(self, state: AgentState, definitions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        available = {"edit_file", "write_file", "read_code", "read_file"}
        if self._session.repair_packet is not None or self._session.completion_review:
            # The controller owns file discovery. A repair episode needs exact
            # replacements and a precise source-range escape; order stays fixed.
            available.discard("read_code")
        artifact_ids = self._coding_artifact_ids(state)
        if artifact_ids:
            available.add("read_artifact")
        if skill_access_requested(self._session.memory.objective):
            available.add("read_skill")
        selected = []
        for definition in definitions:
            if definition["function"]["name"] in available:
                # Phase-specific enums must not mutate the registry definitions
                # reused on subsequent turns or after checkpoint recovery.
                item = deepcopy(definition)
                name = item["function"]["name"]
                if name == "read_code":
                    item["function"]["description"] = "Request missing source files; use read_file for a specific missing range."
                elif name == "write_file":
                    item["function"]["description"] = "Create a new UTF-8 file. Use edit_file for existing files."
                    item["function"]["parameters"]["properties"].pop("overwrite", None)
                if name == "read_skill":
                    item["function"]["parameters"]["properties"]["name"]["enum"] = [
                        skill["name"] for skill in self._session.skills.index()]
                elif name == "read_artifact":
                    item["function"]["description"] = "Read omitted evidence using an available returned artifact_id."
                    item["function"]["parameters"]["properties"]["artifact_id"]["enum"] = list(artifact_ids)
                selected.append(item)
        return selected

    def _loop(self, state: AgentState, definitions: list[dict[str, Any]], deadline: float) -> None:
        coding_mode = self.config.workflow_mode == "coding"
        base_definitions = definitions
        for _ in range(state.steps, self.max_steps):
            if coding_mode and not self._coding_prepare(state, deadline):
                break
            if coding_mode and not self._review.before_model(state, deadline):
                break
            if coding_mode:
                definitions = self._coding_definitions(state, base_definitions)
                definitions = self._review.definitions(state, definitions)
            model_enabled = {item["function"]["name"] for item in definitions}
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                state.stop_reason = "deadline_exceeded"
                break
            if not self._search_gate(state):
                break
            proposal_mode = coding_mode and self._review.proposing
            schema_mode = self.config.action_protocol == "schema" and not proposal_mode
            lean_mode = self.config.workflow_mode in {"lean", "coding"}
            allow_complete, missing = self._schema_completion(state) if schema_mode or lean_mode else (False, [])
            schema_builder = lean_action_schema if lean_mode else action_schema
            response_schema = (generation_schema() if proposal_mode else
                               schema_builder(definitions, allow_complete=allow_complete) if schema_mode else None)
            if schema_mode or lean_mode:
                if coding_mode:
                    inspection_only = (self._requirements is not None and not self._requirements.require_change
                                       and not self._requirements.require_tests)
                    if inspection_only:
                        guidance = "Answer the user's inspection question using observed source; request a specific missing range only if needed."
                    else:
                        guidance = ("If the requested work is complete, reply DONE; otherwise submit remaining edits or request specific missing source."
                                    if allow_complete else "Next: patch the observed source, or read a specific missing detail. Missing: "
                                    + "; ".join(missing) + ".")
                    if self._session.repair_packet is not None:
                        guidance = ("Repair phase: address the selected repair work item using the exact observed failure and source. "
                                    "Submit a small targeted edit_file replacement. Use read_file only for a specific missing range "
                                    "that prevents the correction. The harness verifies every actual change. "
                                    "All outstanding failures and preservation constraints still apply.")
                    elif self._session.completion_review:
                        guidance = ("Completion review: compare the original task with the observed accepted changes and "
                                    "current verification. If the requested work is satisfied, reply DONE. Otherwise submit "
                                    "the remaining targeted correction or read a specific missing range. Passing tests alone "
                                    "do not establish that every requested requirement is complete.")
                    review_guidance = self._review.guidance()
                    if self._review.proposing:
                        guidance = review_guidance
                    elif review_guidance:
                        guidance += "\n" + review_guidance
                else:
                    guidance = (lean_action_instructions(definitions, allow_complete, missing, schema_mode=schema_mode)
                                if lean_mode else action_instructions(definitions, allow_complete, missing))
                    guidance += self._read_guidance(state)
                if lean_mode and self._lean_recovery_hint:
                    guidance += "\nRecovery instruction: " + self._lean_recovery_hint
                if self._decision is not None:
                    if self._decision.architecture:
                        guidance += "\nStatic repository relationships (data):\n" + self._decision.architecture
                    if self._decision.recovery_hint:
                        guidance += "\nSuggested investigation (advice; exact observations govern): " + self._decision.recovery_hint
                if coding_mode:
                    prepared, definitions = self._review.prepare(state, definitions, guidance)
                    model_enabled = {item["function"]["name"] for item in definitions}
                else:
                    prepared = self._session.prepare(
                        state, definitions, response_schema=response_schema,
                        protocol_instructions=guidance)
            else:
                prepared = self._session.prepare(state, definitions)
            if not prepared.safe:
                state.stop_reason = (prepared.reason if prepared.reason in {"invalid_tool_history", "memory_projection_limit"}
                                     else "context_limit")
                state.context_limit_message = prepared.limit_message
                state.final_response = prepared.limit_message
                break
            if coding_mode:
                self._repair_model_messages = prepared.messages
            attempt_row = None
            request_started = None
            parsed_action = None
            try:
                options = {"messages": prepared.messages,
                           "tools": None if schema_mode or proposal_mode else definitions, "think": self.think}
                if schema_mode or proposal_mode:
                    options["response_format"] = response_schema
                if isinstance(self.model, OllamaClient):
                    options["timeout_seconds"] = max(0.001, deadline - time.monotonic())
                self._inflight_model = True
                self._inference_attempts += 1
                self._checkpoint(state)
                self._session.emit("model_request", state.steps + 1, messages=prepared.messages,
                                   tools=[] if schema_mode else definitions,
                                   **({"review_phase": self._session.memory.task_state.phase,
                                       "review_state": self._session.memory.task_state.brief(),
                                       "proposal_evidence": self._review.evidence(prepared.messages)
                                       if self._review.proposing else {},
                                       **({"proposal_transport": "constrained_json",
                                           "proposal_target": self._review.target()} if proposal_mode else {})}
                                      if coding_mode else {}),
                                   **({"repair_packet": self._session.repair_packet}
                                      if self._session.repair_packet is not None else {}),
                                   **({"format": response_schema,
                                       "action_protocol": "proposal_json" if proposal_mode else "schema"}
                                      if schema_mode or proposal_mode else {}),
                                   settings=self._model_settings(), think=self.think,
                                   estimated_tokens=prepared.estimated_tokens, utilization=prepared.utilization)
                request_started = time.monotonic()
                attempt_row = next(row for row in self._session.metrics.per_step if row["step"] == state.steps + 1)
                attempt_row.update(response_received=False, actual_input_tokens=None, actual_output_tokens=None)
                response = self.model.chat(**options)
                self._inflight_model = False
                if lean_mode:
                    self._session.record_lean_usage(response.prompt_tokens,
                                                    prompt_tokens_reported=getattr(response, "prompt_tokens_reported", True))
                    self._lean_recovery_hint = ""
                attempt_row.update(response_received=True, request_wall_seconds=time.monotonic() - request_started)
                state.steps += 1
                state.total_prompt_tokens += response.prompt_tokens
                state.total_completion_tokens += response.completion_tokens
                state.total_inference_seconds += response.total_duration_ns / 1e9
                self._session.metrics.record_model(state.steps, response.prompt_tokens,
                                                   response.completion_tokens, response.total_duration_ns / 1e9,
                                                   prompt_eval_duration_ns=getattr(response, "prompt_duration_ns", None),
                                                   eval_duration_ns=getattr(response, "completion_duration_ns", None),
                                                   load_duration_ns=getattr(response, "load_duration_ns", None),
                                                   prompt_eval_cached_count=getattr(response, "prompt_eval_cached_count", None),
                                                   prompt_tokens_reported=getattr(response, "prompt_tokens_reported", True),
                                                   completion_tokens_reported=getattr(response, "completion_tokens_reported", True))
                message = response.message
                self._session.emit("model_response", state.steps, message=message,
                                   prompt_tokens=response.prompt_tokens, completion_tokens=response.completion_tokens,
                                   duration_seconds=response.total_duration_ns / 1e9,
                                   prompt_eval_duration=getattr(response, "prompt_duration_ns", None),
                                   eval_duration=getattr(response, "completion_duration_ns", None),
                                   load_duration=getattr(response, "load_duration_ns", None),
                                   component_duration_unit="nanoseconds",
                                   prompt_eval_cached_count=getattr(response, "prompt_eval_cached_count", None),
                                   prompt_tokens_reported=getattr(response, "prompt_tokens_reported", True),
                                   completion_tokens_reported=getattr(response, "completion_tokens_reported", True),
                                   done_reason=getattr(response, "done_reason", "stop"))
                if not isinstance(message, dict) or message.get("role") != "assistant":
                    raise ValueError("Model did not return an assistant message")
                if schema_mode and getattr(response, "done_reason", "stop") != "length":
                    try:
                        parsed_action = parse_action(message, definitions, allow_complete=allow_complete)
                    except ValueError as exc:
                        self._session.emit("action_rejected", state.steps, error=str(exc), message=message)
                        state.stop_reason = "invalid_schema_action"
                        state.final_response = "Incomplete: model response did not satisfy the action protocol."
                        break
                    self._session.emit("action_parsed", state.steps, action=parsed_action, completion_available=allow_complete)
                    if parsed_action["kind"] == "tool":
                        message = {"role": "assistant", "content": "", "tool_calls": [parsed_action["call"]]}
                    else:
                        content = parsed_action.get("summary", parsed_action.get("reason", ""))
                        message = {"role": "assistant", "content": content,
                                   "_schema_action": json.loads(response.message["content"])}
                if proposal_mode:
                    # Preserve the server response as data. Unrequested native
                    # calls cannot enter canonical executable tool history.
                    self._append(state, {"role": "assistant",
                                         "content": message.get("content", "") if isinstance(message.get("content", ""), str) else "",
                                         "_proposal_response": deepcopy(message)})
                else:
                    self._append(state, message)
                self._checkpoint(state)
            except httpx.TimeoutException:
                self._inflight_model = False
                if attempt_row is not None and request_started is not None:
                    attempt_row["request_wall_seconds"] = time.monotonic() - request_started
                self._session.emit("model_error", state.steps + 1, error="timeout")
                state.stop_reason = "deadline_exceeded" if time.monotonic() >= deadline else "model_timeout"
                break
            except httpx.HTTPStatusError as exc:
                self._inflight_model = False
                if attempt_row is not None and request_started is not None:
                    attempt_row["request_wall_seconds"] = time.monotonic() - request_started
                detail = exc.response.text.lower()
                if "context" in detail and any(word in detail for word in ("length", "limit", "exceed")):
                    state.stop_reason = "context_limit"
                    state.context_limit_message = (
                        "Ollama rejected the request at its context limit. The estimate is approximate; "
                        "use narrower source ranges, fewer loaded references, or a smaller optional memory view. "
                        "The server was instructed not to truncate history.")
                    state.final_response = state.context_limit_message
                else:
                    state.status, state.stop_reason = "error", "model_error: HTTPStatusError"
                self._session.emit("model_error", state.steps + 1, error=state.stop_reason)
                break
            except Exception as exc:
                self._inflight_model = False
                if attempt_row is not None and request_started is not None:
                    attempt_row["request_wall_seconds"] = time.monotonic() - request_started
                state.status = "error"
                state.stop_reason = f"model_error: {type(exc).__name__}"
                self._session.emit("model_error", state.steps + 1, error=state.stop_reason)
                break
            if time.monotonic() >= deadline:
                state.stop_reason = "deadline_exceeded"
                break
            if getattr(response, "done_reason", "stop") == "length":
                state.stop_reason = "response_limit"
                break
            if proposal_mode:
                self._review.handle_generation(state, message, prepared.messages, deadline)
                if self._session.memory.task_state.phase == "stopped":
                    break
                if self.pause_after_step is not None and state.steps >= self.pause_after_step:
                    state.stop_reason = "paused"
                    break
                continue
            if parsed_action is not None and parsed_action["kind"] == "blocked":
                retry = (corrective_read_action(state.messages, self.registry.workspace)
                         if lean_mode and self._lean_read_recoveries == 0 and "read_code" in self._enabled else None)
                if retry is not None:
                    self._lean_read_recoveries += 1
                    self._lean_recovery_hint = ("The read failed because of its arguments. Retry this corrected call: "
                                                + json.dumps(retry, ensure_ascii=False, separators=(",", ":")))
                    self._session.emit("read_argument_recovery", state.steps, guidance=self._lean_recovery_hint)
                    self._checkpoint(state)
                    continue
                state.stop_reason = "model_blocked"
                state.final_response = "Incomplete: " + parsed_action["reason"]
                break
            calls = message.get("tool_calls") or []
            if not isinstance(calls, list) or len(calls) > 16:
                state.status, state.stop_reason = "error", "invalid_tool_calls"
                break
            if not calls:
                if coding_mode and self._review.proposing:
                    retry = self._review.reject_protocol(
                        state, "Empty or prose response: call propose_next_steps with two concrete alternatives and evidence_refs.")
                    self._append(state, {"role": "system", "content":
                        "Review phase: submit two concrete alternatives with propose_next_steps; work is not complete."})
                    self._checkpoint(state)
                    if not retry:
                        break
                    continue
                state.final_response = message.get("content") or ""
                if not isinstance(state.final_response, str) or not state.final_response.strip():
                    state.final_response = ""
                    state.stop_reason = "empty_response"
                    break
                if self._revalidate_paths:
                    state.final_response = "Workspace changes require inspection before completion."
                    state.stop_reason = "workspace_changed"
                    break
                missing = self._completion_missing(state)
                # Verification has an existing approved fallback below. Missing
                # inspection or an explicitly required patch needs model action.
                unmet_work = [item for item in missing if item != "observe passing tests after the last change"]
                if unmet_work:
                    if self._completion_recoveries == 0:
                        self._completion_recoveries += 1
                        note = "Task incomplete: " + "; ".join(missing) + ". Use structured tools to complete the work before finishing."
                        self._append(state, {"role": "system", "content": note})
                        self._session.emit("completion_recovery", state.steps, missing=missing)
                        state.final_response = ""
                        self._checkpoint(state)
                        continue
                    state.stop_reason = "completion_requirements_unmet"
                    state.final_response = "Incomplete: " + "; ".join(missing) + "."
                    break
                tests_required = self._requirements is not None and self._requirements.require_tests
                if (state.modified_files or self._verification_required or tests_required) and state.verification == "not_run":
                    if not self.registry.workspace.config.allow_tests or "run_tests" not in self._enabled:
                        state.final_response = "Tests were not run because test execution is not authorized or run_tests is unavailable."
                        state.stop_reason = "verification_not_authorized"
                        break
                    # A final answer cannot skip verification after a mutation.
                    # The harness runs the approved default suite and lets the
                    # model inspect its result before issuing a final report.
                    calls = [{"type": "function", "function": {
                        "name": "run_tests", "arguments": {"path": "."}}}]
                    state.final_response = ""
                    self._append(state, {"role": "assistant", "content": "Verify changes before finishing.",
                                         "tool_calls": calls})
                else:
                    if state.verification in {"failed", "timed_out"}:
                        state.stop_reason = "tests_" + state.verification
                    else:
                        # Required observed work is complete. Independent task
                        # correctness remains a separate acceptance decision.
                        state.status, state.stop_reason = "completed", "final_response"
                    break
            stop = False
            if coding_mode and self._review.enabled and (self._review.proposing or any(
                    self._call_parts(call)[0] in REVIEW_TOOLS for call in calls)):
                if (len(calls) == 1 and self._call_parts(calls[0])[0] in REVIEW_TOOLS
                        and self._call_parts(calls[0])[0] in model_enabled):
                    self._review.handle(state, calls[0], prepared.messages, deadline)
                else:
                    # A proposal batch must never smuggle a mutation alongside
                    # its unapproved alternatives. Acknowledge every call.
                    for call in calls:
                        name, _ = self._call_parts(call)
                        result = {"success": False, "error": {"code": "review_required",
                                  "message": "Submit one propose_next_steps call; no operation in this batch was executed."}}
                        observation = {"role": "tool", "tool_name": name, "content": json.dumps(result)}
                        if isinstance(call, dict) and isinstance(call.get("id"), str):
                            observation["tool_call_id"] = call["id"]
                        self._append(state, observation)
                    self._review.reject_protocol(
                        state, "Mixed or unapproved tool batch: submit only one propose_next_steps call; no operation was executed.")
                    self._checkpoint(state)
                if self._session.memory.task_state.phase == "stopped":
                    break
                if self.pause_after_step is not None and state.steps >= self.pause_after_step:
                    state.stop_reason = "paused"
                    break
                continue
            # Evaluate inspection against the context available to the model,
            # before any same-response read can manufacture edit permission.
            rejections = [(self._coding_edit_error(state, call, prepared.messages) or self._review.edit_error(call))
                          if coding_mode else "" for call in calls]
            for call, rejection in zip(calls, rejections):
                execution_enabled = model_enabled if coding_mode else self._enabled
                # The no-change completion fallback is a controller test call.
                if coding_mode and calls and not message.get("tool_calls"):
                    execution_enabled = self._enabled
                result = self._execute(state, call, execution_enabled, deadline, rejection=rejection)
                fingerprint = json.dumps(_stable_observation([call, result]), sort_keys=True, ensure_ascii=False)
                if not result.get("success"):
                    self._failures += 1
                elif fingerprint not in self._successful:
                    self._successful.add(fingerprint)
                    self._failures = 0
                if result.get("error", {}).get("code") == "repeated_invalid_call":
                    state.stop_reason, stop = "repeated_tool_failures", True
                    state.final_response = "Incomplete: " + result["error"]["message"]
                    break
                if time.monotonic() >= deadline:
                    state.stop_reason, stop = "deadline_exceeded", True
                    break
                if self._failures >= self.max_tool_failures:
                    state.stop_reason, stop = "repeated_tool_failures", True
                    break
            if stop:
                break
            if coding_mode and self._coding_test_pending and not self._coding_prepare(state, deadline):
                break
            if not self._search_gate(state):
                break
            if self.pause_after_step is not None and state.steps >= self.pause_after_step:
                state.stop_reason = "paused"
                break

    def resume(self, session_id: str, *, allow_tests: bool = False) -> AgentState:
        """Continue known state; interrupted mutations are never replayed."""
        try:
            if type(allow_tests) is not bool:
                raise ValueError("allow_tests must be an explicit boolean")
            saved = SessionStore(self.runs_dir, session_id).load_checkpoint()
            if saved.get("config", {}).get("workflow_mode") == "evidence":
                from patchrook_agent.evidence_workflow import EvidenceWorkflow
                return EvidenceWorkflow.resume(self, session_id, allow_tests=allow_tests)
            return self._resume(session_id, allow_tests=allow_tests)
        except (KeyError, TypeError, IndexError) as exc:
            raise ValueError("Invalid or incomplete checkpoint data") from exc
        finally:
            session = getattr(self, "_session", None)
            if session and session.store and hasattr(session.store, "release"):
                session.store.release()

    def _resume(self, session_id: str, *, allow_tests: bool = False) -> AgentState:
        resume_started = time.monotonic()
        store = SessionStore(self.runs_dir, session_id)
        saved = store.load_checkpoint()
        execution = saved.get("execution")
        if not isinstance(execution, dict):
            raise ValueError("Checkpoint has no valid execution settings")
        config = HarnessConfig.from_dict(saved["config"])
        if config.action_protocol == "schema" and execution.get("action_protocol_version") != 1:
            raise ValueError("Unsupported checkpoint action protocol version")
        if not config.checkpoint_enabled:
            raise ValueError("Checkpoint resume is disabled for this configuration")
        root = Workspace(saved["workspace"]).root
        if str(root) != saved["workspace"]:
            raise ValueError("Checkpoint workspace path no longer resolves to its original location")
        if self._explicit_workspace and root != self.registry.workspace.root:
            raise ValueError("Checkpoint workspace differs from the configured workspace")
        saved_model_settings = execution.get("model_settings")
        if saved_model_settings != self._model_settings():
            raise ValueError("Model settings differ from the checkpoint; use the original model and context")
        state = TypeAdapter(AgentState).validate_json(json.dumps(saved["state"]), strict=True)
        trusted_systems = [{"role": "system", "content": prompt}
                           for prompt in (SYSTEM_PROMPT, PRE_BATCH_GUIDANCE_SYSTEM_PROMPT, LEGACY_SYSTEM_PROMPT,
                                          LEAN_SYSTEM_PROMPT, CODING_SYSTEM_PROMPT, LEGACY_CODING_SYSTEM_PROMPT)]
        if (state.session_id != session_id or len(state.messages) < 2
                or state.messages[0] not in trusted_systems):
            raise ValueError("Invalid agent checkpoint state")
        memory = WorkingMemory.from_dict(saved["memory"])
        if state.messages[1] != {"role": "user", "content": memory.objective}:
            raise ValueError("Checkpoint objective does not match original conversation")
        if (len(state.active_messages) < 2 or state.active_messages[0] != state.messages[0]
                or state.active_messages[1] != state.messages[1]):
            raise ValueError("Active checkpoint context does not preserve the original objective and instructions")
        self.max_steps = execution["max_steps"]
        self.deadline_seconds = execution["deadline_seconds"]
        self.max_tool_failures = execution["max_tool_failures"]
        if type(self.max_steps) is not int or self.max_steps < 1 or type(self.max_tool_failures) is not int or self.max_tool_failures < 1:
            raise ValueError("Invalid execution limits in checkpoint")
        if (isinstance(self.deadline_seconds, bool) or not isinstance(self.deadline_seconds, (float, int))
                or not 0 < self.deadline_seconds < float("inf")):
            raise ValueError("Invalid task deadline in checkpoint")
        self.think = execution["think"]
        if type(self.think) is not bool:
            raise ValueError("Invalid reasoning setting in checkpoint")
        tool_settings = dict(execution["tool_config"])
        tool_settings["allow_tests"] = bool(allow_tests)
        self.registry = ToolRegistry(Workspace(root, ToolConfig(**tool_settings)))
        self.config = config
        self._setup(state, memory.objective, store=store)
        session = self._session
        session.memory = memory
        memory.restore_inspection(state.messages, self.registry.workspace)
        session.metrics = ExecutionMetrics.from_dict(saved["metrics"])
        session.metrics.decision_enabled = self._decision is not None
        if self._decision is not None and "decision" in execution:
            self._decision.restore(execution["decision"])
        # A checkpoint cannot contain the duration of its own final disk write.
        # The subsequent durable event supplies that observed timing on resume.
        if session.logger and session.logger.path.is_file():
            with session.logger.path.open(encoding="utf-8", errors="replace") as events:
                for line in events:
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if (event.get("event") == "checkpoint" and event.get("session_id") == session_id
                            and event.get("checkpoint_count") == session.metrics.checkpoint_count):
                        for key in ("checkpoint_seconds", "wall_seconds"):
                            value = event.get(key)
                            if isinstance(value, (int, float)) and 0 <= value < float("inf"):
                                setattr(session.metrics, key, max(getattr(session.metrics, key), value))
        session.metrics.resume_attempts += 1
        session.emit("resume_attempt", state.steps)
        restored_context = ContextManager.from_dict(saved["context"])
        if restored_context.config != session.context.config:
            raise ValueError("Checkpoint context settings do not match the saved harness and model settings")
        session.context = restored_context
        session.skills.restore(saved["skills"])
        session.snapshots = saved["file_snapshots"]
        changed = compare_snapshots(self.registry.workspace, session.snapshots)
        self._revalidate_paths = set(execution.get("revalidate_paths", [])) | set(changed)
        self._inference_attempts = execution.get("inference_attempts", state.steps)
        self._unmeasured_inferences = execution.get("unmeasured_inference_calls", 0)
        if execution.get("inflight_model"):
            self._unmeasured_inferences += 1
            state.recovery_notes.append("An interrupted model request has unknown token usage and unfinished duration; observed totals are lower bounds.")
        self._failures = execution.get("failures_without_progress", 0)
        self._successful = set(execution.get("successful_observations", []))
        self._tool_attempts = execution.get("tool_attempts", {})
        self._verification_required = execution.get("verification_required", False)
        if type(self._verification_required) is not bool:
            raise ValueError("Invalid checkpoint verification requirement")
        if self.config.workflow_mode in {"lean", "coding"}:
            self._lean_progress = LeanProgress.from_dict(execution["lean_progress"], self.registry.workspace)
            self._session.lean.restore(execution["lean_context"])
            self._lean_recovery_hint = execution.get("lean_recovery_hint", "")
            if not isinstance(self._lean_recovery_hint, str):
                raise ValueError("Invalid lean recovery hint")
            self._lean_read_recoveries = execution.get("lean_read_recoveries", 0)
            if type(self._lean_read_recoveries) is not int or not 0 <= self._lean_read_recoveries <= 1:
                raise ValueError("Invalid lean read recovery count")
            self._lean_invalid_calls = execution.get("lean_invalid_calls", {})
            if (not isinstance(self._lean_invalid_calls, dict)
                    or any(not isinstance(key, str) or value not in {"not_found", "invalid_arguments", "invalid_path", "invalid_skill", "skill_not_found"}
                           for key, value in self._lean_invalid_calls.items())):
                raise ValueError("Invalid lean tool repair history")
        if self.config.workflow_mode == "coding":
            coding = execution.get("coding")
            required_keys = {"gathered", "source_queue", "refresh_paths", "test_pending"}
            if (not isinstance(coding, dict) or not required_keys <= set(coding)
                    or set(coding) - required_keys - {"skills_prepared", "baseline_complete", "repair"}
                    or type(coding["gathered"]) is not bool or type(coding["test_pending"]) is not bool
                    or not isinstance(coding["source_queue"], list) or len(coding["source_queue"]) > 24
                    or any(not isinstance(path, str) for path in coding["source_queue"])
                    or not isinstance(coding["refresh_paths"], list) or len(coding["refresh_paths"]) > 16
                    or any(not isinstance(path, str) for path in coding["refresh_paths"])
                    or type(coding.get("skills_prepared", False)) is not bool
                    or type(coding.get("baseline_complete", False)) is not bool):
                raise ValueError("Invalid coding workflow checkpoint")
            self._coding_gathered = coding["gathered"]
            self._coding_queue = list(coding["source_queue"])
            self._coding_refresh_paths = set(coding["refresh_paths"])
            self._coding_test_pending = coding["test_pending"]
            self._coding_skills_prepared = coding.get("skills_prepared", False)
            self._coding_baseline_complete = coding.get("baseline_complete", False)
            repair = coding.get("repair", self._repair_state)
            if isinstance(repair, dict) and "episode" not in repair:
                repair = {**repair, "episode": ""}
            if isinstance(repair, dict) and "requested_ranges" not in repair:
                repair = {**repair, "requested_ranges": []}
            if isinstance(repair, dict):
                repair = {"action_id": "", "action_status": "idle", **repair}
            if (not isinstance(repair, dict) or set(repair) != set(self._repair_state)
                    or not isinstance(repair["fingerprint"], str)
                    or repair["fingerprint"] and (len(repair["fingerprint"]) != 64
                        or any(char not in "0123456789abcdef" for char in repair["fingerprint"]))
                    or not isinstance(repair["selected_id"], str) or len(repair["selected_id"]) > 256
                    or not isinstance(repair["episode"], str)
                    or repair["episode"] and (len(repair["episode"]) != 64
                        or any(char not in "0123456789abcdef" for char in repair["episode"]))
                    or type(repair["narrow"]) is not bool or type(repair["evidence_requested"]) is not bool):
                raise ValueError("Invalid repair workflow checkpoint")
            if (not isinstance(repair["action_id"], str) or len(repair["action_id"]) > 48
                    or not isinstance(repair["action_status"], str)
                    or repair["action_status"] not in {"idle", "pending", "applied"}):
                raise ValueError("Invalid repair action checkpoint")
            requested = repair["requested_ranges"]
            if (not isinstance(requested, list) or len(requested) > 4
                    or any(not isinstance(item, dict) or set(item) != {"path", "start_line", "end_line"}
                           or not isinstance(item["path"], str) or canonical_path(item["path"]) != item["path"]
                           or Path(item["path"]).is_absolute() or item["path"].startswith("../")
                           or type(item["start_line"]) is not int or type(item["end_line"]) is not int
                           or not 1 <= item["start_line"] <= item["end_line"] <= item["start_line"] + 23
                           for item in requested)):
                raise ValueError("Invalid requested repair evidence checkpoint")
            # Packet source is always rebuilt from canonical observations.
            self._repair_state = deepcopy(repair)
        requirements = execution.get("task_requirements")
        self._requirements = TaskRequirements.from_dict(requirements) if requirements is not None else None
        self._completion_recoveries = execution.get("completion_recoveries", 0)
        if type(self._completion_recoveries) is not int or not 0 <= self._completion_recoveries <= 1:
            raise ValueError("Invalid checkpoint completion recovery count")
        self._search_progress = SearchProgress.from_dict(execution["search_progress"]) if "search_progress" in execution else SearchProgress()
        self._search_feedback = execution.get("search_feedback")
        if self._search_feedback is not None:
            SearchProgress.from_dict(self._search_feedback)
        self._enabled = set(execution["enabled_tools"])
        known = {item["function"]["name"] for item in self.registry.definitions()}
        if not self._enabled <= known:
            raise ValueError("Checkpoint requests unavailable tools")
        self.registry.guidance_tools = None if self._enabled == set(self.registry.tools) else self._enabled
        lean_invalidated = self._repair_interrupted(state, execution.get("pending_operation"))
        if changed:
            if self._decision is not None:
                self._decision.invalidate()
                self._decision.source_priority = []
            self._lean_invalid_calls.clear()
            self._search_progress.reset()
            self._search_feedback = None
            note = "Relevant files changed since checkpoint: " + ", ".join(changed)
            state.recovery_notes.append(note)
            session.memory.blockers.append(note)
            state.verification = session.memory.verification = "not_run"
            if self.config.workflow_mode == "coding":
                self._coding_test_pending = True
                session.memory.task_state.clear_pending("Workspace changed since the saved decision")
            if self._lean_progress is not None:
                if not lean_invalidated:
                    self._lean_progress.invalidate()
                session.lean.invalidate_paths(changed, history_length=len(state.messages))
            self._append(state, {"role": "system", "content": note + ". Prior snippets may be stale; read changed files before editing or finishing."})
        session.metrics.resume_success += 1
        if self.config.workflow_mode == "coding":
            self._review.validate_restored()
        session.wall_base = session.metrics.wall_seconds + time.monotonic() - resume_started
        session.started = time.monotonic()
        session.bind(state)
        session.emit("resumed", state.steps, changed_files=changed,
                     unmeasured_inference_calls=self._unmeasured_inferences)
        if state.status == "completed" and not changed:
            self._completion_missing(state)
            self._report_observed_outcome(state)
            session.finish(state, self._execution())
            return state
        state.status, state.stop_reason, state.final_response = "incomplete", "max_steps", ""
        self._checkpoint(state)
        definitions = self.registry.definitions(enabled_tools=self._enabled)
        return self._continue(state, definitions)

    def _repair_interrupted(self, state: AgentState, pending: Any) -> bool:
        # Only an unfinished final assistant exchange can have missing results.
        lean_invalidated = False
        assistant_index = next((index for index in range(len(state.active_messages) - 1, -1, -1)
                                if state.active_messages[index].get("role") == "assistant"), None)
        if assistant_index is None:
            return False
        message = state.active_messages[assistant_index]
        calls = message.get("tool_calls") or []
        following = state.active_messages[assistant_index + 1:]
        completed = sum(item.get("role") == "tool" for item in following)
        if not isinstance(calls, list) or completed > len(calls):
            raise ValueError("Invalid tool relationships in checkpoint")
        for call in calls[completed:]:
            name, arguments = self._call_parts(call)
            unknown = pending is not None and call == pending
            result = self.registry.error("operation_interrupted" if unknown else "not_executed",
                                         "Operation outcome is unknown; inspect the workspace; do not replay automatically."
                                         if unknown else "This requested operation was not executed before interruption.")
            observation = {"role": "tool", "tool_name": name, "content": json.dumps(result)}
            if isinstance(call, dict) and isinstance(call.get("id"), str):
                observation["tool_call_id"] = call["id"]
            self._append(state, observation)
            self._session.memory.observe_tool(name, arguments, result, state.steps)
            self._session.emit("interrupted_tool_recovery", state.steps, name=name, arguments=arguments, result=result)
            if unknown:
                aggregate = self._session.metrics.tools_by_name.setdefault(name, {
                    "calls": 0, "successes": 0, "failures": 0, "retries": 0, "seconds": 0.0,
                })
                aggregate["calls"] += 1
                aggregate["unknown"] = aggregate.get("unknown", 0) + 1
                path = self._canonical_path(arguments)
                if path and name in {"write_file", "edit_file"}:
                    self._revalidate_paths.add(path)
                    self._verification_required = True
                    state.verification = self._session.memory.verification = "not_run"
                    if self.config.workflow_mode == "coding":
                        self._coding_test_pending = True
                        self._session.memory.task_state.clear_pending("An interrupted mutation needs fresh observation")
                if name == "run_tests":
                    state.verification = self._session.memory.verification = "not_run"
                    if self.config.workflow_mode == "coding":
                        self._coding_test_pending = True
                if self._lean_progress is not None and name in {"write_file", "edit_file", "run_tests"}:
                    if not lean_invalidated:
                        self._lean_progress.invalidate()
                        lean_invalidated = True
                    if path and name in {"write_file", "edit_file"}:
                        self._session.lean.invalidate_paths([path], history_length=len(state.messages))
                state.recovery_notes.append(f"Unknown outcome for interrupted {name}; no automatic replay.")
        return lean_invalidated
