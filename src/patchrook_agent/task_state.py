"""Bounded task coordination facts derived from completed harness operations.

The ledger supplements the complete conversation and event log. A proposal is
an unverified plan, an accepted edit is an operation, and only a test observation
establishes a test outcome. None of these records grants source access.
"""

from copy import deepcopy
import hashlib
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


HISTORY_LIMIT = 96
FAILED_STATE_LIMIT = 16
PENDING_CHANGE_LIMIT = 8
ATTEMPT_LIMIT = 16
DELTA_LIMIT = 320


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True, allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _clip(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[:max(0, limit - 1)] + "…"


def _json_object(value: dict, *, limit: int) -> dict:
    """Reject malformed restored data, including non-JSON values under Any."""
    def visit(item: Any, depth: int = 0) -> None:
        if depth > 8:
            raise ValueError("task ledger JSON nesting exceeds eight levels")
        if item is None or type(item) in {bool, int, float}:
            return
        if isinstance(item, str):
            if len(item) > 8192:
                raise ValueError("task ledger string is too large")
            return
        if isinstance(item, list):
            if len(item) > 128:
                raise ValueError("task ledger array is too large")
            for child in item:
                visit(child, depth + 1)
            return
        if isinstance(item, dict):
            if len(item) > 64 or any(not isinstance(key, str) for key in item):
                raise ValueError("task ledger objects require bounded string keys")
            for child in item.values():
                visit(child, depth + 1)
            return
        raise ValueError("task ledger values must be JSON data")

    visit(value)
    if len(_json(value)) > limit:
        raise ValueError("task ledger record exceeds its size limit")
    return value


def normalized_test_outcome(result: dict[str, Any]) -> dict[str, Any]:
    """Remove measurement noise without hiding failing assertions or amounts."""
    outcome = {key: deepcopy(result[key]) for key in (
        "success", "status", "exit_code", "stdout", "stderr", "test_counts", "error",
    ) if key in result}
    for key in ("stdout", "stderr"):
        if isinstance(outcome.get(key), str):
            outcome[key] = re.sub(
                r"(?m)(\b\d+ (?:failed|passed|errors?|skipped|warnings?)(?:, \d+ [a-z]+)* in )\d+(?:\.\d+)?s\b",
                r"\1<elapsed>s", outcome[key])
            outcome[key] = re.sub(r"/tmp/patchrook-[^/\s\"']+/", "<workspace>/", outcome[key])
    return outcome


def _failure_witness(outcome: dict) -> str:
    text = "\n".join(str(outcome.get(key, "")) for key in ("stdout", "stderr"))
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    relevant = [line for line in lines if re.search(
        r"(?:^E\s|AssertionError|Error:|^FAILED\b|\bassert\b|\bfailed\b)", line)]
    return _clip("\n".join(relevant or lines[-4:]), 600)


def _change(arguments: dict, *, path: str, name: str, step: int, goal: str, mutation: int) -> dict:
    item: dict[str, Any] = {"path": path, "tool": name, "step": step, "goal": goal, "mutation": mutation}
    keys = ("old_text", "new_text") if name == "edit_file" else ("content",)
    delta = {key: arguments[key] for key in keys if isinstance(arguments.get(key), str)}
    if delta and all(len(text) <= DELTA_LIMIT for text in delta.values()):
        item["accepted_delta"] = delta
    elif delta:
        item["accepted_delta_excerpt"] = {
            key: {"head": text[:80], "tail": text[-80:], "characters": len(text)}
            for key, text in delta.items()}
    return item


def _string(item: dict, key: str, limit: int, *, required: bool = True) -> None:
    if key not in item and not required:
        return
    if not isinstance(item.get(key), str) or len(item[key]) > limit:
        raise ValueError(f"task ledger {key} must be a bounded string")


def _validate_change(item: dict) -> None:
    for key, limit in (("path", 1024), ("goal", 512), ("tool", 80)):
        _string(item, key, limit)
    if item["tool"] not in {"edit_file", "write_file"}:
        raise ValueError("pending changes require a completed mutation tool")
    for key in ("step", "mutation"):
        if type(item.get(key)) is not int or item[key] < 0:
            raise ValueError(f"pending change {key} must be a nonnegative integer")
    if "accepted_delta" in item:
        delta = item["accepted_delta"]
        if (not isinstance(delta, dict) or not delta or len(delta) > 2
                or any(key not in {"old_text", "new_text", "content"} for key in delta)
                or any(not isinstance(text, str) or len(text) > DELTA_LIMIT for text in delta.values())):
            raise ValueError("accepted delta must contain bounded exact observed arguments")
    if "accepted_delta_excerpt" in item and not isinstance(item["accepted_delta_excerpt"], dict):
        raise ValueError("accepted delta excerpt must be a JSON object")


def _validate_proposal_fields(item: dict) -> None:
    # The review protocol owns the complete proposal schema. These checks keep
    # restored partial metadata safe to inspect without treating it as approval.
    for key, bound in (("id", 32), ("kind", 32), ("path", 1024), ("approach", 240),
                       ("expected", 160), ("status", 32)):
        _string(item, key, bound, required=False)
    if "evidence_refs" in item:
        refs = item["evidence_refs"]
        if (not isinstance(refs, list) or len(refs) > 3
                or any(not isinstance(ref, str) or len(ref) > 32 for ref in refs)):
            raise ValueError("proposal evidence references must be bounded strings")


class TaskState(BaseModel):
    """Shared coordinator state; restore accepts legacy absence, never bad types."""

    model_config = ConfigDict(extra="forbid", strict=True, validate_assignment=True)

    phase: Literal["direct", "propose", "selected", "stopped"] = "direct"
    active_goal: str = Field(default="", max_length=512)
    review_reason: str = Field(default="", max_length=512)
    rounds: int = Field(default=0, ge=0, le=2)
    episodes: int = Field(default=0, ge=0, le=8)
    protocol_retries: int = Field(default=0, ge=0, le=1)
    evidence_reads: int = Field(default=0, ge=0, le=1)
    proposals: list[dict[str, Any]] = Field(default_factory=list, max_length=3)
    selected: dict[str, Any] | None = None
    proposal_evidence: dict[str, Any] = Field(default_factory=dict)
    proposal_revision: int = Field(default=0, ge=0)
    proposal_goal: str = Field(default="", max_length=180)
    history: list[dict[str, Any]] = Field(default_factory=list, max_length=HISTORY_LIMIT)
    history_omitted: int = Field(default=0, ge=0)
    failed_states: list[dict[str, Any]] = Field(default_factory=list, max_length=FAILED_STATE_LIMIT)
    attempts: dict[str, int] = Field(default_factory=dict)
    pending_changes: list[dict[str, Any]] = Field(default_factory=list, max_length=PENDING_CHANGE_LIMIT)
    # Mutation numbers distinguish a read/test replay from returning to a prior
    # failed workspace. An external workspace change starts a new comparison epoch.
    mutation: int = Field(default=0, ge=0)
    comparison_epoch: int = Field(default=0, ge=0)
    trigger_reason: Literal["", "repeated_failed_state", "failed_attempts"] = ""
    attempt_selections: dict[str, dict[str, int]] = Field(default_factory=dict)

    @field_validator("proposals", "history", "failed_states", "pending_changes")
    @classmethod
    def validate_records(cls, value: list[dict], info) -> list[dict]:
        limit = {"proposals": 4096, "history": 4096, "failed_states": 65536, "pending_changes": 4096}[info.field_name]
        for item in value:
            _json_object(item, limit=limit)
        if info.field_name in {"history", "failed_states", "pending_changes"}:
            for item in value:
                for key in ("step", "mutation", "epoch"):
                    if key in item and (type(item[key]) is not int or item[key] < 0):
                        raise ValueError(f"task ledger {key} must be a nonnegative integer")
        if info.field_name == "history":
            for item in value:
                _string(item, "event", 80)
                if type(item.get("step")) is not int or item["step"] < 0:
                    raise ValueError("task history requires an observed nonnegative step")
        if info.field_name == "proposals":
            for item in value:
                _validate_proposal_fields(item)
        if info.field_name == "pending_changes":
            for item in value:
                _validate_change(item)
        if info.field_name == "failed_states":
            for item in value:
                for key, bound in (("id", 64), ("source_digest", 128), ("outcome_digest", 64),
                                   ("selection", 1024), ("goal", 512), ("witness", 600)):
                    _string(item, key, bound)
                for key in ("mutation", "epoch", "step"):
                    if type(item.get(key)) is not int or item[key] < 0:
                        raise ValueError("failed state requires observed mutation, epoch and step")
                if not isinstance(item.get("outcome"), dict):
                    raise ValueError("failed state requires an observed outcome object")
                changes = item.get("changes")
                if not isinstance(changes, list) or len(changes) > PENDING_CHANGE_LIMIT:
                    raise ValueError("failed state changes must be a bounded list")
                for change in changes:
                    if not isinstance(change, dict):
                        raise ValueError("failed state changes must contain JSON objects")
                    _validate_change(change)
                keys = item.get("attempt_keys")
                if (not isinstance(keys, list) or len(keys) > PENDING_CHANGE_LIMIT
                        or any(not isinstance(key, str) or len(key) != 64 for key in keys)):
                    raise ValueError("failed state attempt keys must be bounded digests")
                _string(item, "stable_goal", 512)
                paths = item.get("paths")
                if (not isinstance(paths, list) or len(paths) > PENDING_CHANGE_LIMIT
                        or any(not isinstance(path, str) or len(path) > 1024 for path in paths)):
                    raise ValueError("failed state paths must be a bounded path inventory")
                if "proposal" in item:
                    if not isinstance(item["proposal"], dict):
                        raise ValueError("failed proposal must be a JSON object")
                    _validate_proposal_fields(item["proposal"])
                if "repeated" in item and type(item["repeated"]) is not bool:
                    raise ValueError("failed state repeat marker must be a boolean")
        return value

    @field_validator("selected", "proposal_evidence")
    @classmethod
    def validate_object(cls, value: dict | None, info) -> dict | None:
        if value is None:
            return None
        _json_object(value, limit=4096 if info.field_name == "selected" else 32768)
        if info.field_name == "selected":
            _validate_proposal_fields(value)
        return value

    @field_validator("attempts", "attempt_selections")
    @classmethod
    def validate_attempts(cls, value: dict, info) -> dict:
        if len(value) > ATTEMPT_LIMIT or any(len(key) > 128 for key in value):
            raise ValueError("task attempt inventory exceeds its bounded size")
        counters = list(value.values())
        if info.field_name == "attempt_selections":
            if any(len(scopes) > 16 or any(len(path) > 1024 for path in scopes) for scopes in counters):
                raise ValueError("task attempt selections exceed their bounded size")
            counters = [count for scopes in counters for count in scopes.values()]
        if any(type(count) is not int or count < 0 for count in counters):
            raise ValueError("task attempt counts must be nonnegative integers")
        return value

    @property
    def trigger(self) -> str:
        return self.trigger_reason

    @property
    def has_activity(self) -> bool:
        return bool(self.history or self.active_goal or self.selected or self.proposals or self.failed_states
                    or self.pending_changes or self.trigger or self.review_reason or self.phase != "direct")

    def record_event(self, kind: str, *, step: int = 0, **details: Any) -> None:
        """Record all brief events, including repeats; the raw log stays complete."""
        if not isinstance(kind, str) or not kind or len(kind) > 80:
            raise ValueError("task event kind must be a short nonempty string")
        if type(step) is not int or step < 0:
            raise ValueError("task event step must be a nonnegative integer")
        item = {"event": kind, "step": step, **deepcopy(details)}
        _json_object(item, limit=4096)
        history = [*self.history, item]
        self.history_omitted += max(0, len(history) - HISTORY_LIMIT)
        self.history = history[-HISTORY_LIMIT:]

    def observe_action(self, name: str, arguments: Any, result: dict[str, Any], *,
                       step: int, source_digest: str = "") -> None:
        """Only Session calls this after an observed, completed tool operation."""
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:
                arguments = {}
        arguments = arguments if isinstance(arguments, dict) else {}
        if not isinstance(result, dict):
            raise ValueError("a completed task action requires an observed result object")
        if not isinstance(source_digest, str) or len(source_digest) > 128:
            raise ValueError("source digest must be a bounded string")
        path = result.get("path", arguments.get("path", "."))
        path = _clip(path, 1024) if isinstance(path, str) else "."
        error = result.get("error")
        outcome = result.get("status") if name == "run_tests" else (
            "modified" if result.get("success") is True and result.get("modified") is True else
            "succeeded" if result.get("success") is True else "failed")
        record = {"tool": _clip(name, 80), "path": path, "outcome": _clip(str(outcome or "failed"), 80)}
        if isinstance(error, dict) and isinstance(error.get("code"), str):
            record["error"] = _clip(error["code"], 80)
        if name == "read_file":
            for key in ("start_line", "end_line"):
                value = result.get(key, arguments.get(key))
                if type(value) is int and value > 0:
                    record[key] = value
        if name in {"read_code", "read_files"}:
            files = result.get("files", [])
            if isinstance(files, list):
                paths = [item["path"] for item in files if isinstance(item, dict) and isinstance(item.get("path"), str)]
                record["paths"] = [_clip(item, 100) for item in paths[:8]]
        if name == "search_code" and isinstance(arguments.get("query"), str):
            record["query"] = _clip(arguments["query"], 160)
        self.record_event("action", step=step, **record)
        if (name in {"edit_file", "write_file"} and result.get("success") is True
                and result.get("modified") is True):
            self.mutation += 1
            self.pending_changes = [*self.pending_changes, _change(
                arguments, path=path, name=name, step=step, goal=self.active_goal, mutation=self.mutation,
            )][-PENDING_CHANGE_LIMIT:]
        if name == "run_tests" and result.get("status") in {"passed", "failed", "timed_out"}:
            self._observe_tests(path, result, step=step, source_digest=source_digest)

    def _observe_tests(self, path: str, result: dict, *, step: int, source_digest: str) -> None:
        passed = result.get("success") is True and result.get("status") == "passed" and result.get("exit_code") == 0
        if passed:
            full_suite = path in {".", "./"}
            self.failed_states = [] if full_suite else [
                item for item in self.failed_states if item.get("selection") != path]
            scopes = {} if full_suite else {
                key: {selection: count for selection, count in values.items() if selection != path}
                for key, values in self.attempt_selections.items()}
            scopes = {key: values for key, values in scopes.items() if values}
            self.attempt_selections = scopes
            self.attempts = {key: sum(values.values()) for key, values in scopes.items()}
            self.trigger_reason = ("repeated_failed_state" if any(
                item.get("repeated") is True for item in self.failed_states) else
                "failed_attempts" if any(count >= 2 for count in self.attempts.values()) else "")
            self.pending_changes = []
            return
        outcome = normalized_test_outcome(result)
        outcome_digest = _digest(outcome)
        identity = _digest({"source": source_digest, "outcome": outcome_digest, "selection": path,
                            "epoch": self.comparison_epoch})
        previous = next((item for item in reversed(self.failed_states) if item.get("id") == identity), None)
        # Source identity must be measured, not inferred from a claimed edit or
        # a missing hash. A baseline and a later unchanged failed state can cycle.
        repeated = bool(source_digest and self.pending_changes and previous is not None
                        and previous.get("mutation", 0) < self.mutation)
        if repeated:
            self.trigger_reason = "repeated_failed_state"
        keys: list[str] = []
        scopes = deepcopy(self.attempt_selections)
        paths = sorted({change["path"] for change in self.pending_changes})
        stable_goal = paths[0] if len(paths) == 1 else (
            "paths:" + "|".join(paths) if paths else self.active_goal)
        if len(stable_goal) > 512:
            stable_goal = "paths:" + _digest(paths)
        if len(paths) == 1 and self.phase == "direct":
            self.active_goal = stable_goal
        for change in self.pending_changes:
            # A new proposal label cannot reset the failed attempts for the
            # same edited file. Full-suite passes or external invalidation can.
            key = _digest({"path": change["path"]})
            if key in keys:
                continue
            keys.append(key)
            selections = scopes.setdefault(key, {})
            selections[path] = selections.get(path, 0) + 1
            if len(selections) > 16:
                selections.pop(next(iter(selections)))
        scopes = dict(list(scopes.items())[-ATTEMPT_LIMIT:])
        self.attempt_selections = scopes
        self.attempts = {key: sum(values.values()) for key, values in scopes.items()}
        if not self.trigger_reason and any(count >= 2 for count in self.attempts.values()):
            self.trigger_reason = "failed_attempts"
        item = {"id": identity, "source_digest": source_digest, "outcome_digest": outcome_digest,
                "selection": path, "goal": self.active_goal, "mutation": self.mutation,
                "stable_goal": stable_goal, "paths": paths,
                "repeated": repeated or bool(previous and previous.get("repeated") is True),
                "epoch": self.comparison_epoch, "step": step, "witness": _failure_witness(outcome),
                "outcome": {key: outcome[key] for key in ("status", "exit_code", "test_counts") if key in outcome},
                "changes": deepcopy(self.pending_changes), "attempt_keys": keys}
        if (self.pending_changes and self.selected and self.selected.get("kind") in {"implement", "implement_change"}
                and (self.selected.get("status") in {"selected", "executed"}
                     or ("status" not in self.selected and self.phase == "selected"))):
            item["proposal"] = {key: deepcopy(self.selected[key]) for key in (
                "id", "path", "approach", "expected", "kind",
            ) if key in self.selected}
        if not self.pending_changes and previous is not None:
            # A completed test can be durable before the controller clears its
            # pending-test flag. Repeating it after resume must retain the
            # observed edit/proposal witness, without charging another attempt.
            for key in ("changes", "attempt_keys", "proposal", "goal", "stable_goal", "paths"):
                if key in previous:
                    item[key] = deepcopy(previous[key])
        # A repeated read/test is still in history; it need not duplicate the
        # bounded distinct failure inventory or erase the earliest witness.
        states = [state for state in self.failed_states if state.get("id") != identity]
        self.failed_states = [*states, item][-FAILED_STATE_LIMIT:]
        self.pending_changes = []

    def clear_pending(self, reason: str) -> None:
        """Invalidate decisions and comparisons after an external workspace change."""
        if not isinstance(reason, str) or not reason:
            raise ValueError("pending invalidation requires a reason")
        self.record_event("invalidated", step=self.history[-1].get("step", 0) if self.history else 0,
                          reason=_clip(reason, 512))
        self.pending_changes = []
        self.proposals = []
        self.selected = None
        self.proposal_evidence = {}
        self.proposal_goal = ""
        self.proposal_revision += 1
        self.rounds = 0
        self.protocol_retries = 0
        self.evidence_reads = 0
        self.phase = "direct"
        self.review_reason = ""
        self.failed_states = []
        self.attempts = {}
        self.attempt_selections = {}
        self.trigger_reason = ""
        self.comparison_epoch += 1

    def brief(self, *, max_chars: int = 1800) -> dict[str, Any]:
        """Bounded model view; hypotheses and observed failed variants stay distinct."""
        if type(max_chars) is not int or max_chars < 512:
            raise ValueError("task state brief budget must be at least 512 characters")
        data: dict[str, Any] = {"phase": self.phase, "active_goal": _clip(self.active_goal, 160)}
        if self.rounds or self.episodes:
            data.update(rounds=self.rounds, episodes=self.episodes)
        if self.protocol_retries:
            data["protocol_retries"] = self.protocol_retries
        if self.review_reason:
            data["review_reason"] = _clip(self.review_reason, 120)
        if self.trigger:
            data["trigger"] = self.trigger
        if self.selected:
            data["selected"] = {key: _clip(str(value), 120) for key, value in self.selected.items()
                                if key in {"id", "proposal_id", "kind", "goal", "path", "target", "symbol", "status"}}
        recent = [{key: deepcopy(item[key]) for key in (
            "event", "step", "tool", "path", "paths", "query", "outcome", "proposal_id", "reason",
        ) if key in item} for item in self.history[-6:]]
        for item in recent:
            for key, value in list(item.items()):
                if isinstance(value, str):
                    item[key] = _clip(value, 100)
        if recent:
            data["recent_actions"] = recent
        # Keep both sides of an A/B failed patch cycle, even when ordinary
        # chronology must shrink. Full exact outputs remain in tool observations.
        witnesses = []
        for item in self.failed_states[-2:]:
            witness = {"source": item.get("source_digest", "")[:12], "step": item.get("step", 0),
                       "selection": _clip(item.get("selection", "."), 80),
                       "failure": _clip(item.get("witness", ""), 180)}
            changes = []
            for change in item.get("changes", [])[-2:]:
                entry = {"path": _clip(change.get("path", ""), 80)}
                if "accepted_delta" in change:
                    entry["accepted_delta"] = {key: value for key, value in change["accepted_delta"].items()
                                               if len(value) <= 96}
                elif "accepted_delta_excerpt" in change:
                    entry["delta"] = "Long accepted change; exact observation remains on disk."
                changes.append(entry)
            if changes:
                witness["changes"] = changes
            if isinstance(item.get("proposal"), dict):
                proposal = item["proposal"]
                witness["proposal"] = {key: _clip(str(proposal[key]), 96) for key in (
                    "id", "path", "approach",
                ) if key in proposal}
            witnesses.append(witness)
        if witnesses:
            data["failed_witnesses"] = witnesses
        while len(_json(data)) > max_chars and data.get("recent_actions"):
            data["recent_actions"].pop(0)
        if not data.get("recent_actions"):
            data.pop("recent_actions", None)
        if len(_json(data)) > max_chars:
            data.pop("selected", None)
        if len(_json(data)) > max_chars:
            for item in witnesses:
                item.pop("changes", None)
                item.pop("proposal", None)
                item["failure"] = _clip(item["failure"], 80)
                item["selection"] = _clip(item["selection"], 32)
            data["active_goal"] = _clip(data["active_goal"], 80)
            if "review_reason" in data:
                data["review_reason"] = _clip(data["review_reason"], 64)
        if len(_json(data)) > max_chars:
            data.pop("review_reason", None)
            data.pop("active_goal", None)
            for item in witnesses:
                item["failure"] = _clip(item["failure"], 32)
        return data
