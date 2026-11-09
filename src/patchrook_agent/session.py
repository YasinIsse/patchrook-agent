"""Session artifacts and measured context preparation, outside the agent loop."""

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import re
import time
from uuid import uuid4
from typing import Any

from patchrook_agent.config import HarnessConfig
from patchrook_agent.coding import CODING_SYSTEM_PROMPT
from patchrook_agent.context.budget import estimate_tokens
from patchrook_agent.context.compaction import ContextManager, PreparedContext
from patchrook_agent.context.lean import LeanContext
from patchrook_agent.memory.checkpoint import SessionStore, atomic_write, atomic_write_json, snapshot_files
from patchrook_agent.memory.progress import render_progress
from patchrook_agent.memory.state import WorkingMemory
from patchrook_agent.models.schema import schema_messages, schema_request_tokens
from patchrook_agent.observability.events import EventLogger, redact_data
from patchrook_agent.observability.artifacts import ArtifactStore
from patchrook_agent.observability.metrics import ExecutionMetrics
from patchrook_agent.skills.loader import SkillLoader
from patchrook_agent.tools.base import ToolError


DEFAULT_RUNS_DIR = Path.home() / ".local" / "state" / "patchrook-agent" / "runs"


class Session:
    def __init__(self, config: HarnessConfig, workspace, model, runs_dir: Path,
                 objective: str, *, store: SessionStore | None = None):
        self.config, self.workspace = config, workspace
        self.store = store or (SessionStore(runs_dir) if config.persist else None)
        self.session_id = self.store.session_id if self.store else uuid4().hex
        self.logger = (EventLogger(self.store.directory, self.session_id,
                                  enabled=config.logging_enabled, redact=config.redact_logs,
                                  log_content=config.log_content) if self.store else None)
        self.memory = WorkingMemory(session_id=self.session_id, objective=objective)
        self.metrics = ExecutionMetrics()
        self.metrics.decision_enabled = config.decision_model == "on"
        self.decision = None
        self.skills = SkillLoader(Path(config.trusted_skills_dir) if config.trusted_skills_dir else None,
                                  enabled=config.skills_enabled)
        self.metrics.skills_discovered = len(self.skills.index())
        self.artifacts = (ArtifactStore(self.store.directory,
                                       max_output_chars=max(256, workspace.config.max_output_chars - 160),
                                       redact=config.redact_logs)
                          if self.store and config.logging_enabled and config.log_content and config.variant != "A"
                          else None)
        context_size = getattr(model, "context_size", config.context.context_size)
        reserve = max(512, getattr(model, "max_output_tokens", 512), config.context.reserved_output_tokens)
        self.context = ContextManager(replace(config.context, context_size=context_size,
                                               reserved_output_tokens=reserve,
                                               enabled=config.compaction_enabled and config.context.enabled))
        self.lean = LeanContext(self.context.config, workspace=workspace)
        self.repair_packet = None
        self.completion_review = False
        self.review_source_ranges = []
        self.snapshots: dict[str, dict] = {}
        self.started = time.monotonic()
        self.wall_base = 0.0
        self.accounting: dict[str, int] = {}
        if self.store:
            self.store.acquire()

    def emit(self, event: str, step: int = 0, **data) -> None:
        if self.logger:
            self.logger.emit(event, step, **data)

    def bind(self, state) -> None:
        state.session_id = self.session_id
        state.session_dir = str(self.store.directory) if self.store else ""
        self.refresh(state)

    def refresh(self, state) -> None:
        self.metrics.wall_seconds = self.wall_base + time.monotonic() - self.started
        state.metrics = self.metrics.summary()
        state.metrics.update({name: getattr(state, name) for name in (
            "steps", "total_prompt_tokens", "total_completion_tokens", "total_inference_seconds",
            "total_tool_seconds", "tool_calls", "tool_failures",
        )})
        state.metrics.update(self.accounting)
        state.metrics.update(getattr(self, "evidence_metrics", {}))
        state.metrics["completed_inference_calls"] = self.metrics.inference_calls
        state.metrics["inference_calls"] = self.accounting.get("inference_attempts", self.metrics.inference_calls)
        state.metrics.update(context_window_tokens=self.context.config.context_size,
                             effective_input_limit=self.context.config.input_limit,
                             reserved_output_tokens=self.context.config.reserved_output_tokens,
                             memory_projection_budget=self.config.memory_projection_tokens)
        if getattr(self.config, "workflow_mode", "standard") in {"lean", "coding"}:
            state.metrics.update(context_estimator="lean_content", lean_correction_tokens=self.lean.correction_tokens,
                                 lean_observed_usage_calls=self.lean.observed_calls,
                                 lean_unmeasured_usage_calls=self.lean.unmeasured_calls)
        state.memory = self.memory.to_dict() if self.config.memory_enabled else {}

    def prepare(self, state, definitions: list[dict], *, response_schema: dict | None = None,
                protocol_instructions: str = "", system_prompt: str | None = None) -> PreparedContext:
        started = time.monotonic()
        projection = None
        schema_mode = response_schema is not None
        if system_prompt is not None and (self.config.workflow_mode != "coding" or not schema_mode
                                          or not isinstance(system_prompt, str) or not system_prompt.strip()):
            raise ValueError("A phase policy requires a nonempty coding schema prompt")
        if getattr(self.config, "workflow_mode", "standard") in {"lean", "coding"}:
            return self._prepare_lean(state, definitions, response_schema=response_schema,
                                      protocol_instructions=protocol_instructions, started=started,
                                      system_prompt=system_prompt)
        budget_definitions = [{"format": response_schema}] if schema_mode else definitions
        # Compaction and final dispatch must measure the same projected request.
        # Otherwise wire wrappers can cross the ceiling without a capacity retry.
        request_cost = ((lambda messages, _tools: schema_request_tokens(schema_messages(messages), response_schema))
                        if schema_mode else None)
        pins, tails = [], []
        if (self.config.compaction_enabled or self.config.memory_enabled or self.config.skills_enabled
                or self.config.variant != "A"):
            if self.skills.index_text():
                pins.append({"role": "system", "content": self.skills.index_text()})
            pins.extend(self.skills.loaded_messages())
            projection = (self.memory.project(max_tokens=self.config.memory_projection_tokens,
                                              active_messages=state.active_messages)
                          if self.config.memory_enabled else None)
            tails = [{"role": "system", "content": projection.content}] if projection else []
            if schema_mode and protocol_instructions:
                tails.append({"role": "system", "content": protocol_instructions})
            prepared = self.context.prepare(state.active_messages, budget_definitions,
                                             pinned_messages=pins, tail_messages=tails,
                                             **({"request_cost": request_cost} if schema_mode else {}))
            if projection:
                self.emit("memory_projection", state.steps + 1, estimated_tokens=projection.estimated_tokens,
                          budget_tokens=projection.budget_tokens, over_budget=projection.over_budget,
                          omitted=projection.omitted, reason=projection.reason)
                if projection.over_budget:
                    prepared.safe, prepared.reason = False, "memory_projection_limit"
                    prepared.limit_message = (projection.reason
                                              + " Reduce saved notes or raise HarnessConfig.memory_projection_tokens.")
        elif not schema_mode:
            # Instrument the unchanged Milestone 1 estimate and stopping rule.
            tokens = estimate_tokens(state.active_messages, definitions)
            reserve = getattr(self, "baseline_reserve", 512)
            size = self.context.config.context_size
            safe = tokens + reserve < size
            prepared = PreparedContext(state.active_messages, tokens, tokens / size, False,
                                       tokens, tokens, False, "disabled" if safe else "context_limit",
                                       safe, active_messages=state.active_messages)
        else:
            if protocol_instructions:
                tails.append({"role": "system", "content": protocol_instructions})
            # Schema mode always charges the full format and current protocol,
            # even when context compaction is deliberately disabled in A.
            prepared = self.context.prepare(state.active_messages, budget_definitions, tail_messages=tails,
                                             request_cost=request_cost)
        if schema_mode:
            try:
                wire_messages = schema_messages(prepared.messages)
                wire_tokens = schema_request_tokens(wire_messages, response_schema)
            except (ValueError, TypeError, RecursionError):
                prepared.safe, prepared.reason = False, "invalid_tool_history"
                prepared.limit_message = "Invalid schema action history; restore a complete canonical conversation from a checkpoint."
            else:
                prepared.messages = wire_messages
                prepared.estimated_tokens = wire_tokens
                prepared.utilization = wire_tokens / self.context.config.context_size
                prepared.tokens_after = wire_tokens
                prepared.target_reached = wire_tokens <= self.context.config.compaction_goal
                if not self.context.config.fits(wire_tokens):
                    if prepared.reason not in {"invalid_tool_history", "memory_projection_limit"}:
                        prepared.reason = "context_limit"
                        prepared.limit_message = (
                            f"Schema action request requires {wire_tokens} estimated input tokens; "
                            f"the effective ceiling is {self.context.config.input_limit}. "
                            "The full response schema and protocol are included. Use narrower source ranges "
                            "or reduce optional context; history will not be silently truncated.")
                    prepared.safe = False
        elapsed = time.monotonic() - started
        self.metrics.context_preparation_seconds += elapsed
        self.metrics.compaction_seconds += prepared.compaction_seconds
        self.metrics.record_context(state.steps + 1, prepared.estimated_tokens, prepared.utilization)
        row = self.metrics._step(state.steps + 1)
        row.update(context_preparation_seconds=elapsed, compaction_seconds=prepared.compaction_seconds,
                   compaction_attempted=prepared.attempted, context_reason=prepared.reason)
        if projection is not None:
            row.update(memory_projection_tokens=projection.estimated_tokens,
                       memory_projection_budget=projection.budget_tokens,
                       memory_projection_over_budget=projection.over_budget)
        if prepared.attempted:
            attempt = {"step": state.steps + 1, "tokens_before": prepared.tokens_before,
                       "tokens_after": prepared.tokens_after, "reduced": prepared.compacted,
                       "reason": prepared.attempt_reason, "outcome_reason": prepared.reason,
                       "seconds": prepared.compaction_seconds}
            self.metrics.compaction_attempts.append(attempt)
            self.emit("compaction_attempt", state.steps + 1,
                      **{key: value for key, value in attempt.items() if key != "step"})
        if prepared.compacted:
            record = {"step": state.steps + 1, "tokens_before": prepared.tokens_before,
                      "tokens_after": prepared.tokens_after, "target_reached": prepared.target_reached,
                      "reason": prepared.reason, "seconds": prepared.compaction_seconds}
            self.metrics.compactions.append(record)
            self.emit("compaction", state.steps + 1, **{key: value for key, value in record.items() if key != "step"})
        self.emit("context", state.steps + 1, estimated_tokens=prepared.estimated_tokens,
                  utilization=prepared.utilization, input_limit=self.context.config.input_limit,
                  context_window=self.context.config.context_size,
                  reserved_output_tokens=self.context.config.reserved_output_tokens,
                  estimation_margin_tokens=self.context.config.estimation_margin_tokens,
                  attempted=prepared.attempted, attempt_reason=prepared.attempt_reason,
                  target_reached=prepared.target_reached, reason=prepared.reason, safe=prepared.safe,
                  limit_message=prepared.limit_message)
        state.active_messages = prepared.active_messages
        return prepared

    def _prepare_lean(self, state, definitions: list[dict], *, response_schema: dict | None,
                      protocol_instructions: str, started: float, system_prompt: str | None = None) -> PreparedContext:
        # No generic plan, historical navigation summary or skill index is
        # injected into ordinary coding turns. Loaded trusted instructions are
        # retained whole; an explicitly requested discovery can show the index.
        skills = self.skills.loaded_messages()
        coding = self.config.workflow_mode == "coding"
        skill_tool_enabled = any(item.get("function", {}).get("name") == "read_skill" for item in definitions)
        if (not skills and (not coding or skill_tool_enabled)
                and re.search(r"\bskills?\b", self.memory.objective, re.IGNORECASE)):
            index = self.skills.index_text()
            if index:
                skills = [{"role": "system", "content": index}]
        completion = state.completion or {}
        if coding:
            summary = {}
            if state.verification != "not_run":
                summary["verification"] = state.verification
            if state.modified_files:
                summary["modified_files"] = list(state.modified_files)
            task_state = self.memory.task_state
            if task_state.history or task_state.phase != "direct":
                summary["task_state"] = task_state.brief()
        else:
            summary = {"modified_files": list(state.modified_files), "verification": state.verification}
        if not coding and completion.get("progress"):
            summary["current"] = completion["progress"]
        if not coding and completion.get("missing"):
            summary["missing"] = completion["missing"]
        if "outstanding_tests" in completion:
            summary["outstanding_tests"] = list(completion["outstanding_tests"])
        blockers = [note for note in self.memory.blockers if not note.startswith("Execution stopped:")]
        if blockers:
            summary["blockers"] = blockers
        # Hold a small focus while verification is unresolved. Paths come from
        # observed discovery/ranking, never a benchmark answer or hidden source.
        # The context builder uses exact, bounded excerpts and still protects
        # the latest native exchange and failed tests before optional source.
        focus = []
        if coding and completion.get("outstanding_tests"):
            priorities = (self.decision.source_priority if self.decision is not None
                          and self.decision.source_priority else list(self.memory.inspection.files))
            focus = list(dict.fromkeys(priorities))[:4]
        prepared = self.lean.prepare(state.messages, definitions=definitions,
                                     response_schema=response_schema,
                                     protocol_instructions=protocol_instructions,
                                     state_summary=summary, skills=skills,
                                     **({"system_prompt": system_prompt or CODING_SYSTEM_PROMPT, "compact": True,
                                         "source_paths": list(self.memory.inspection.files)[:24],
                                         **({"retained_source_paths": focus} if focus else {}),
                                         **({"source_priority": self.decision.source_priority}
                                            if self.decision is not None else {}),
                                         "incomplete_paths": [path for path, item in self.memory.inspection.files.items()
                                                              if item.pending_ranges or not item.inspected],
                                         **({"repair_packet": self.repair_packet}
                                            if self.repair_packet is not None else {}),
                                         **({"completion_review": True,
                                             "review_source_ranges": self.review_source_ranges}
                                            if self.completion_review else {})}
                                        if coding else {}))
        elapsed = time.monotonic() - started
        self.metrics.context_preparation_seconds += elapsed
        self.metrics.record_context(state.steps + 1, prepared.estimated_tokens, prepared.utilization)
        row = self.metrics._step(state.steps + 1)
        row.update(context_preparation_seconds=elapsed, compaction_seconds=0,
                   compaction_attempted=False, context_reason=prepared.reason,
                   context_estimator="lean_content", lean_correction_tokens=self.lean.correction_tokens)
        self.emit("context", state.steps + 1, estimated_tokens=prepared.estimated_tokens,
                  utilization=prepared.utilization, input_limit=self.context.config.input_limit,
                  context_window=self.context.config.context_size,
                  reserved_output_tokens=self.context.config.reserved_output_tokens,
                  estimation_margin_tokens=self.context.config.estimation_margin_tokens,
                  context_estimator="lean_content", lean_correction_tokens=self.lean.correction_tokens,
                  lean_observed_usage_calls=self.lean.observed_calls,
                  attempted=False, reason=prepared.reason, safe=prepared.safe,
                  limit_message=prepared.limit_message)
        # Keep canonical atomic exchanges for durable resume. The returned
        # messages are a distinct data-only view rebuilt from the raw log.
        state.active_messages = prepared.active_messages
        return prepared

    def record_lean_usage(self, prompt_tokens: int | None, *, prompt_tokens_reported: bool = True) -> None:
        if getattr(self.config, "workflow_mode", "standard") in {"lean", "coding"}:
            self.lean.record_usage(prompt_tokens, prompt_tokens_reported=prompt_tokens_reported)

    def expand_observation(self, name: str, result: dict, available: dict | None, *, step: int = 0) -> dict:
        if not self.artifacts or name == "read_artifact" or not result.get("truncated"):
            return result
        result = dict(result)
        try:
            result["artifact_id"] = self.artifacts.save(available or {"name": name, "result": result})
            self.emit("artifact_saved", step, artifact_id=result["artifact_id"], tool=name)
        except ToolError as exc:
            result["artifact_unavailable"] = exc.code
            self.emit("artifact_unavailable", step, tool=name, code=exc.code, message=exc.message)
        return result

    @staticmethod
    def _deduplicate_error_evidence(memory: dict, active: list[dict]) -> None:
        """Keep exact errors in their protected tool groups, rather than 3 copies.

        The disk memory remains full fidelity. Only its model projection uses a
        pointer when the identical debugging evidence is already in context.
        """
        results = []
        for message in active:
            if message.get("role") != "tool":
                continue
            try:
                result = json.loads(message.get("content", ""))
            except (TypeError, ValueError):
                continue
            if isinstance(result, dict) and result.get("success") is False:
                results.append((message.get("tool_name"), result))
        for error in memory.get("unresolved_errors", []):
            fields = [key for key in ("error", "status", "exit_code", "stdout", "stderr") if key in error]
            if fields and any(name == error.get("tool") and all(error[key] == result.get(key) for key in fields)
                              for name, result in results):
                code = error.get("error", {}).get("code")
                for key in fields:
                    error.pop(key)
                error.update(code=code, evidence="Exact failure preserved in the corresponding tool observation.")
        latest = memory.get("latest_test", {})
        if any(name == "run_tests" and result.get("status") == latest.get("status")
               and result.get("stdout", "") == latest.get("stdout", "")
               and result.get("stderr", "") == latest.get("stderr", "") for name, result in results):
            latest.pop("stdout", None)
            latest.pop("stderr", None)

    def observe_tool(self, state, name: str, arguments: Any, result: dict, duration: float,
                     *, retry: bool, available: dict | None = None) -> None:
        before = set(self.memory.relevant_files)
        self.memory.observe_tool(name, arguments, result, state.steps, self.workspace)
        self.metrics.record_tool(name, bool(result.get("success")), duration, retry=retry)
        row = next((row for row in self.metrics.per_step if row["step"] == state.steps), None)
        if row is not None:
            row.setdefault("tools", []).append({"name": name, "success": bool(result.get("success")),
                                                "seconds": duration, "retry": retry})
        if name == "read_skill" and result.get("success"):
            self.metrics.skills_loaded = len(self.skills.loaded)
            if not result.get("already_loaded") and not result.get("is_reference"):
                self.metrics.skills_activated += 1
        # Listing discovers names; it must not cause the harness to read every
        # listed file merely to create fingerprints. Snapshot inspected/matched
        # or modified files, plus explicit test selections.
        paths = set(self.memory.relevant_files) - before if name != "list_files" else set()
        if name in {"read_files", "read_code"}:
            paths.update(item["path"] for item in result.get("files", [])
                         if isinstance(item, dict) and item.get("success") is True
                         and isinstance(item.get("path"), str))
        if name in {"read_file", "edit_file", "write_file", "run_tests"}:
            args = arguments
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {}
            if isinstance(args, dict) and isinstance(args.get("path"), str):
                paths.add(args["path"].split("::", 1)[0])
            if result.get("path"):
                paths.add(result["path"])
        if paths and (self.config.checkpoint_enabled or self.config.workflow_mode == "coding"):
            canonical = set()
            for path in paths:
                try:
                    canonical.add(self.workspace.relative(self.workspace.resolve(path)))
                except (OSError, ValueError, ToolError):
                    # Invalid tool paths are already represented by their tool
                    # errors; they must not become trusted resume targets.
                    continue
            self.snapshots.update(snapshot_files(self.workspace, sorted(canonical)))
        if self.config.workflow_mode == "coding":
            # A content identity survives mutation counters and timing noise.
            # Snapshots contain hashes, never additional editable source.
            digest = hashlib.sha256(json.dumps([
                (path, value.get("kind"), value.get("sha256"))
                for path, value in sorted(self.snapshots.items()) if path != "."
            ], separators=(",", ":")).encode()).hexdigest() if self.snapshots else ""
            ledger = self.memory.task_state
            ledger.observe_action(name, arguments, result, step=state.steps, source_digest=digest)
            self.emit("task_state", state.steps, state=ledger.brief())
        self.emit("tool_result", state.steps, name=name, arguments=arguments,
                  result=(available or {}).get("result", result), observation=result,
                  available_output=(available or {}).get("available_output", {}),
                  duration_seconds=duration, retry=retry)

    def checkpoint(self, state, execution: dict) -> None:
        self.accounting = {key: execution.get(key, 0) for key in (
            "inference_attempts", "unmeasured_inference_calls",
        )}
        if not self.store or not self.config.checkpoint_enabled:
            if self.store and self.config.memory_enabled:
                atomic_write_json(self.store.directory / "state.json", self.memory.to_dict())
                atomic_write(self.store.directory / "TASKS.md", render_progress(self.memory))
            return
        started = time.monotonic()
        self.metrics.checkpoint_count += 1
        self.refresh(state)
        payload = {"workspace": str(self.workspace.root), "config": self.config.to_dict(),
                   "state": asdict(state), "metrics": self.metrics.to_dict(),
                   "context": self.context.to_dict(), "skills": self.skills.to_dict(),
                   "file_snapshots": self.snapshots, "execution": execution}
        self.store.write_checkpoint(payload, self.memory)
        self.metrics.checkpoint_seconds += time.monotonic() - started
        self.refresh(state)
        self.emit("checkpoint", state.steps, checkpoint_count=self.metrics.checkpoint_count,
                  checkpoint_seconds=self.metrics.checkpoint_seconds,
                  wall_seconds=self.metrics.wall_seconds,
                  pending_operation=execution.get("pending_operation"),
                  inflight_model=execution.get("inflight_model", False))

    def finish(self, state, execution: dict) -> None:
        try:
            self._finish(state, execution)
        finally:
            if self.store:
                self.store.release()

    def _finish(self, state, execution: dict) -> None:
        if state.status == "completed":
            self.memory.active_task = "Finished; see observed verification status"
            self.memory.remaining_steps = []
            self.memory.last_completed_step = state.steps
        elif state.stop_reason not in {"paused", "interrupted"}:
            note = f"Execution stopped: {state.stop_reason}."
            if note not in self.memory.blockers:
                self.memory.blockers.append(note)
        self.checkpoint(state, execution)
        self.refresh(state)
        self.emit("session_end", state.steps, status=state.status, stop_reason=state.stop_reason,
                  verification=state.verification, metrics=state.metrics)
        if self.store:
            report = {"schema_version": 1, "session_id": self.session_id,
                      "action_protocol": self.config.action_protocol,
                      "workflow_mode": getattr(self.config, "workflow_mode", "standard"),
                      "decision_model": self.config.decision_model,
                      "decision_timeout_seconds": self.config.decision_timeout_seconds,
                      "decision_memory_policy": self.config.decision_memory_policy,
                      "variant": self.config.variant, "status": state.status,
                      "stop_reason": state.stop_reason, "verification": state.verification,
                      "modified_files": state.modified_files, "metrics": state.metrics,
                      "per_step": self.metrics.per_step, "recovery_notes": state.recovery_notes,
                      "completion": state.completion}
            if self.config.log_content:
                report["final_response"] = state.final_response
            atomic_write_json(self.store.directory / "report.json",
                              redact_data(report) if self.config.redact_logs else report)
