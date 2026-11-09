"""Observed session facts, separate from raw model conversation history."""

import json
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from patchrook_agent.inspection import InspectionLedger, canonical_path
from patchrook_agent.task_state import TaskState

if TYPE_CHECKING:
    from .projection import MemoryProjection


class WorkingMemory(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, validate_assignment=True)

    schema_version: Literal[1] = 1
    session_id: str = Field(min_length=1)
    objective: str
    current_plan: list[str] = Field(default_factory=lambda: [
        "Inspect relevant code", "Complete requested work", "Verify changes when applicable",
    ])
    active_task: str = "Inspect relevant code"
    completed_steps: list[str] = Field(default_factory=list)
    remaining_steps: list[str] = Field(default_factory=lambda: [
        "Complete requested work", "Verify changes when applicable",
    ])
    relevant_files: list[str] = Field(default_factory=list)
    modified_files: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    test_results: list[dict[str, Any]] = Field(default_factory=list)
    verification: Literal["not_run", "passed", "failed", "timed_out"] = "not_run"
    unresolved_errors: list[dict[str, Any]] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    loaded_skills: list[str] = Field(default_factory=list)
    last_completed_step: int = Field(default=0, ge=0)
    inspection: InspectionLedger = Field(default_factory=InspectionLedger)
    task_state: TaskState = Field(default_factory=TaskState)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkingMemory":
        return cls.model_validate(data)

    def restore_inspection(self, messages: list[dict], workspace: Any = None) -> None:
        """Migrate legacy checkpoints from raw visible history; never invent coverage."""
        self.inspection = InspectionLedger.from_messages(messages, workspace)
        paths = set(self.relevant_files) | set(self.inspection.files)
        self.completed_steps = [note for note in self.completed_steps
                                if not any(note == f"Read {path}." or note.startswith(f"Read {path} lines ")
                                           for path in paths)]
        for path in self.inspection.files:
            self._inspection_progress(path, workspace)

    def _inspection_progress(self, path: str, workspace: Any = None) -> None:
        canonical = canonical_path(path, workspace)
        identities = {path, canonical} - {None}
        self.completed_steps = [note for note in self.completed_steps
                                if not any(note == f"Read {identity}." or note.startswith(f"Read {identity} lines ")
                                           for identity in identities)]
        item = self.inspection.files.get(canonical) if canonical is not None else None
        if item is None or not item.visible_ranges:
            return
        shown = item.visible_ranges[:4]
        ranges = ", ".join(f"{start}-{end}" for start, end in shown)
        if len(item.visible_ranges) > len(shown):
            ranges += f", +{len(item.visible_ranges) - len(shown)} ranges in inspection ledger"
        suffix = "; partial selection remains" if item.pending_ranges else ""
        self._remember(self.completed_steps, f"Read {canonical} lines {ranges}{suffix}.")

    @staticmethod
    def _remember(items: list[str], item: Any) -> None:
        if isinstance(item, str) and item and item not in items:
            items.append(item)

    def observe_tool(self, name: str, arguments: Any, result: dict[str, Any], step: int,
                     workspace: Any = None, *, _inspection_recorded: bool = False) -> None:
        """Update facts only after the harness observes a completed operation."""
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:
                arguments = {}
        arguments = arguments if isinstance(arguments, dict) else {}
        if not _inspection_recorded and name in {"read_file", "read_files", "read_code"}:
            self.inspection.observe(name, arguments, result, step=step, workspace=workspace)
        if name in {"read_files", "read_code"} and isinstance(result.get("files"), list):
            files = [item for item in result["files"] if isinstance(item, dict)]
            selections = arguments.get("files", [])
            selections = selections if isinstance(selections, list) else []
            for index, item in enumerate(files):
                selection = selections[index] if index < len(selections) and isinstance(selections[index], dict) else {}
                self.observe_tool("read_file", {**selection, "path": item.get("path", ".")}, item, step, workspace,
                                  _inspection_recorded=True)
            if result.get("success"):
                self.unresolved_errors = [error for error in self.unresolved_errors
                                          if error.get("tool") != name]
                return
            if not result.get("error") and any(item.get("success") is False for item in files):
                return
            if not result.get("error"):
                result = {**result, "error": {
                    "code": "batch_read_failed",
                    "message": "Some requested files could not be read; inspect the batch observation or artifact.",
                }}
        path = result.get("path") or arguments.get("path", ".")
        path = path if isinstance(path, str) else "."
        self.last_completed_step = max(step, self.last_completed_step)
        if not result.get("success"):
            failure = {"tool": name, "path": path, "step": step}
            if name in {"read_skill", "load_skill"} and isinstance(arguments.get("name"), str):
                failure["skill"] = arguments["name"]
            if "error" in result:
                failure["error"] = result["error"]
            else:
                failure.update({key: result[key] for key in (
                    "status", "exit_code", "stdout", "stderr",
                ) if key in result})
            # A repeated error replaces its earlier instance, not its exact text.
            self.unresolved_errors = [error for error in self.unresolved_errors
                                      if not (error.get("tool") == name
                                              and ((name in {"read_skill", "load_skill"}
                                                    and error.get("skill") == arguments.get("name"))
                                                   or (name not in {"read_skill", "load_skill"}
                                                       and error.get("path") == path)))]
            self.unresolved_errors.append(failure)
            self.active_task = "Investigate the observed tool failure"
        else:
            self.unresolved_errors = [error for error in self.unresolved_errors
                                      if not (error.get("tool") == name
                                              and ((name not in {"read_skill", "load_skill"}
                                                    and error.get("path") == path)
                                                   or (name in {"read_skill", "load_skill"}
                                                       and error.get("skill") == arguments.get("name"))
                                                   or (name == "run_tests" and path == ".")
                                                   or (name == "run_tests" and result.get("status") == "passed"
                                                       and result.get("exit_code") == 0
                                                       and error.get("exit_code") == 5)))]
            if name == "list_files":
                for entry in result.get("entries", []):
                    if isinstance(entry, str) and not entry.endswith("/"):
                        self._remember(self.relevant_files, entry)
                self._remember(self.completed_steps, f"Listed files under {path}.")
            elif name == "search_code":
                for match in result.get("matches", []):
                    if isinstance(match, dict):
                        self._remember(self.relevant_files, match.get("path"))
                self._remember(self.completed_steps, "Searched repository code.")
            elif name == "read_file":
                self._remember(self.relevant_files, path)
                self._inspection_progress(path, workspace)
            elif name in {"write_file", "edit_file"}:
                self._remember(self.relevant_files, path)
                if result.get("modified"):
                    self._remember(self.modified_files, path)
                    self.verification = "not_run"
                    self._remember(self.completed_steps, f"Modified {path} using {name}.")
                    self.active_task = "Verify the updated files"
                    self.remaining_steps = ["Run relevant tests", "Report observed results"]
            elif name in {"load_skill", "read_skill"}:
                self._remember(self.loaded_skills, result.get("name") or arguments.get("name"))
        if name == "run_tests" and result.get("status") in {"passed", "failed", "timed_out"}:
            observed = {"path": path, "step": step, **result}
            self.test_results.append(observed)
            self.verification = result["status"]
            if result.get("success") and result["status"] == "passed":
                self._remember(self.completed_steps, f"Tests passed for {path}.")
                self.active_task = "Report the observed changes and verification"
                self.remaining_steps = ["Report observed results"]
            else:
                self.active_task = "Investigate the failing tests"
                self.remaining_steps = ["Investigate observed test failures", "Run relevant tests again"]

    def context_summary(self) -> str:
        """A concise projection, preserving exact unresolved debugging evidence."""
        state: dict[str, Any] = {
            "active_task": self.active_task,
            "plan": self.current_plan,
            "completed": self.completed_steps[-8:],
            "remaining": self.remaining_steps,
            "relevant_files": self.relevant_files[:20],
            "modified_files": self.modified_files,
            "verification": self.verification,
        }
        if len(self.relevant_files) > 20:
            state["additional_relevant_files_on_disk"] = len(self.relevant_files) - 20
        if self.decisions:
            state["decisions"] = self.decisions
        if self.task_state.has_activity:
            state["task_state"] = self.task_state.brief()
        if self.blockers:
            state["blockers"] = self.blockers
        if self.loaded_skills:
            state["loaded_skills"] = self.loaded_skills
        if self.unresolved_errors:
            state["unresolved_errors"] = self.unresolved_errors
        if self.test_results:
            latest = self.test_results[-1]
            state["latest_test"] = {key: latest[key] for key in (
                "path", "status", "exit_code", "step",
            ) if key in latest}
            if latest.get("status") != "passed":
                state["latest_test"].update({key: latest[key] for key in (
                    "stdout", "stderr",
                ) if key in latest})
        return json.dumps(state, ensure_ascii=False, separators=(",", ":"))

    def project(self, *, max_tokens: int,
                active_messages: list[dict[str, Any]] | None = None) -> "MemoryProjection":
        """Build a bounded model view without changing complete observed memory."""
        from .projection import project_memory

        return project_memory(self, max_tokens=max_tokens, active_messages=active_messages)
