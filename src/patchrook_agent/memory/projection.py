"""Deterministic, budgeted model views of complete observed working memory."""

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import json
from typing import TYPE_CHECKING, Any

from patchrook_agent.context.budget import estimate_tokens

if TYPE_CHECKING:
    from .state import WorkingMemory


MEMORY_PREFIX = "Working memory (observed):\n"
_FAILURE_FIELDS = ("error", "status", "exit_code", "stdout", "stderr")


@dataclass(frozen=True)
class MemoryProjection:
    content: str
    estimated_tokens: int
    budget_tokens: int
    over_budget: bool
    omitted: dict[str, int] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _render(data: dict[str, Any]) -> tuple[str, int]:
    content = MEMORY_PREFIX + json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return content, estimate_tokens([{"role": "system", "content": content}], [])


def _active_failures(messages: list[dict[str, Any]]) -> list[tuple[dict, dict]]:
    results = []
    for message in messages:
        if message.get("role") != "tool":
            continue
        try:
            result = json.loads(message.get("content", ""))
        except (TypeError, ValueError):
            continue
        if (isinstance(result, dict) and message.get("tool_name") == "read_files"
                and isinstance(result.get("files"), list)):
            results.extend(({**message, "tool_name": "read_file"}, item)
                           for item in result["files"]
                           if isinstance(item, dict) and item.get("success") is False)
        if isinstance(result, dict) and result.get("success") is False:
            results.append((message, result))
    return results


def _matching_observation(tool: str, evidence: dict[str, Any],
                          results: list[tuple[dict, dict]]) -> dict | None:
    fields = [key for key in _FAILURE_FIELDS if key in evidence]
    for message, result in results:
        if (fields and message.get("tool_name") == tool
                and all(key in result and evidence[key] == result[key] for key in fields)):
            return message
    return None


def _observation_reference(message: dict) -> dict[str, str]:
    reference = {"source": "active tool observation"}
    if isinstance(message.get("tool_call_id"), str):
        reference["tool_call_id"] = message["tool_call_id"]
    return reference


def project_memory(memory: "WorkingMemory", *, max_tokens: int,
                   active_messages: list[dict[str, Any]] | None = None) -> MemoryProjection:
    """Keep essential facts exact; spend remaining space on bounded recent facts.

    The original user message carries the full objective and constraints. Exact
    failures may point at identical active observations, which context handling
    must protect as tool-call/result groups. All omitted facts remain in the
    session's full memory artifact, never in an untracked lossy summary.
    """
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("memory projection max_tokens must be a positive integer")
    results = _active_failures(active_messages or [])
    data: dict[str, Any] = {
        "objective": "Original user request and its constraints.",
        "active_task": memory.active_task,
        "modified_files": list(memory.modified_files),
        "verification": memory.verification,
        "remaining": list(memory.remaining_steps),
    }
    for key in ("blockers", "decisions"):
        if getattr(memory, key):
            data[key] = list(getattr(memory, key))
    if memory.task_state.has_activity:
        # A bounded coordination view retains observed failed variants across
        # mutations; the full task ledger remains in the memory artifact.
        data["task_state"] = memory.task_state.brief()
    errors = deepcopy(memory.unresolved_errors)
    for error in errors:
        observation = _matching_observation(error.get("tool"), error, results)
        if observation is not None:
            details = error.pop("error", None)
            if isinstance(details, dict) and "code" in details:
                error["code"] = details["code"]
            error.pop("stdout", None)
            error.pop("stderr", None)
            error["evidence"] = _observation_reference(observation)
    if errors:
        data["unresolved_errors"] = errors
    if memory.test_results:
        latest = memory.test_results[-1]
        projected_test = {key: deepcopy(latest[key]) for key in (
            "path", "status", "exit_code", "step",
        ) if key in latest}
        if latest.get("status") != "passed":
            evidence = {key: latest[key] for key in _FAILURE_FIELDS if key in latest}
            observation = _matching_observation("run_tests", evidence, results)
            error_index = next((index for index, error in enumerate(memory.unresolved_errors)
                                if error.get("tool") == "run_tests"
                                and error.get("path") == latest.get("path")
                                and error.get("step") == latest.get("step")
                                and all(key in error and error[key] == value
                                        for key, value in evidence.items())), None)
            if observation is not None:
                projected_test["evidence"] = _observation_reference(observation)
            elif error_index is not None:
                projected_test["evidence"] = f"unresolved_errors[{error_index}]"
            else:
                projected_test.update({key: deepcopy(latest[key]) for key in (
                    "error", "stdout", "stderr",
                ) if key in latest})
        data["latest_test"] = projected_test

    # The fixed bounds limit both prompt growth and projection work even when a
    # session has a large file inventory. Optional entries must still fit the
    # explicit component budget, including the omission reference itself.
    optional = (
        ("plan", "current_plan", memory.current_plan[:8]),
        ("relevant_files", "relevant_files", memory.relevant_files[-20:]),
        ("completed", "completed_steps", memory.completed_steps[-8:]),
        ("loaded_skills", "loaded_skills", memory.loaded_skills[:16]),
    )
    totals = {source: len(getattr(memory, source)) for _, source, _ in optional}
    totals["test_results"] = max(0, len(memory.test_results) - 1)
    included = {source: 0 for _, source, _ in optional}

    def update_omissions() -> dict[str, int]:
        omitted = {source: total - included.get(source, 0)
                   for source, total in totals.items() if total > included.get(source, 0)}
        if omitted:
            data["details"] = {"artifact": "memory", "optional": True, "omitted": omitted}
        else:
            data.pop("details", None)
        return omitted

    omitted = update_omissions()
    content, tokens = _render(data)
    if tokens > max_tokens and "task_state" in data:
        # Recent coordination actions are optional before exact unresolved
        # failures. Shrink their view while retaining both failed variants.
        for characters in (1200, 800, 512):
            data["task_state"] = memory.task_state.brief(max_chars=characters)
            content, tokens = _render(data)
            if tokens <= max_tokens:
                break
    if tokens <= max_tokens:
        for target, source, entries in optional:
            selected: list[str] = []
            # Recent progress and paths are more useful if only part fits.
            candidates = reversed(entries) if target in {"relevant_files", "completed"} else entries
            for entry in candidates:
                selected.append(entry)
                data[target] = list(reversed(selected)) if target in {"relevant_files", "completed"} else list(selected)
                included[source] += 1
                candidate_omitted = update_omissions()
                candidate_content, candidate_tokens = _render(data)
                if candidate_tokens > max_tokens:
                    selected.pop()
                    included[source] -= 1
                    if selected:
                        data[target] = list(reversed(selected)) if target in {"relevant_files", "completed"} else list(selected)
                    else:
                        data.pop(target, None)
                    update_omissions()
                    continue
                omitted, content, tokens = candidate_omitted, candidate_content, candidate_tokens
    omitted = update_omissions()
    content, tokens = _render(data)
    over_budget = tokens > max_tokens
    reason = ("Essential observed memory exceeds its projection budget; increase memory_projection_tokens "
              "or resolve the protected failures before continuing." if over_budget else "")
    return MemoryProjection(content=content, estimated_tokens=tokens, budget_tokens=max_tokens,
                            over_budget=over_budget, omitted=omitted, reason=reason)
