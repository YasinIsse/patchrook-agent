"""Factual user summaries derived from execution evidence, without inference."""

from __future__ import annotations

from collections import Counter
import json
from typing import Any

from patchrook_agent.inspection import InspectionLedger, canonical_path


def _arguments(call: dict) -> dict:
    value = call.get("function", {}).get("arguments", {})
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return {}
    return value if isinstance(value, dict) else {}


def _observations(messages: list[dict]):
    """Pair actual observations with their requested arguments, never prose."""
    pending: list[dict] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant":
            calls = message.get("tool_calls")
            pending = [call for call in calls if isinstance(call, dict)] if isinstance(calls, list) else []
            continue
        if message.get("role") != "tool":
            continue
        name = message.get("tool_name")
        call_id = message.get("tool_call_id")
        call = next((item for item in pending if call_id is not None and item.get("id") == call_id), None)
        if call is None:
            call = next((item for item in pending if isinstance(item.get("function"), dict)
                         and item["function"].get("name") == name), None)
        if call is None or not isinstance(call.get("function"), dict) or call["function"].get("name") != name:
            continue
        pending.remove(call)
        try:
            result = json.loads(message.get("content", ""))
        except (ValueError, TypeError):
            continue
        if isinstance(result, dict):
            yield index, name, _arguments(call), result


def _identity(value: str) -> str:
    # A filename cannot introduce invented report lines through control text.
    return json.dumps(value, ensure_ascii=False) if any(char in value for char in "\n\r\t`") else value


def _paths(values, *, limit=24, budget=2000) -> str:
    ordered = list(dict.fromkeys(value for value in values if isinstance(value, str) and value))
    displayed = []
    for value in ordered[:limit]:
        rendered = _identity(value)
        if sum(len(item) + 2 for item in displayed) + len(rendered) > budget:
            break
        displayed.append(rendered)
    omitted = len(ordered) - len(displayed)
    text = ", ".join(displayed)
    if omitted:
        text += ("; " if text else "") + f"{omitted} further path(s) in the trace"
    return text


def _test_status(result: dict) -> str | None:
    status, code = result.get("status"), result.get("exit_code")
    if status == "passed" and result.get("success") is True and type(code) is int and code == 0:
        return status
    if status == "failed" and result.get("success") is False and type(code) is int and code != 0:
        return status
    if (status == "timed_out" and result.get("success") is False
            and (code is None or type(code) is int)):
        return status
    return None


def _test_counts(result: dict) -> str:
    counts = result.get("test_counts")
    if not isinstance(counts, dict):
        return ""
    labels = [f"{counts[name]} {name}" for name in (
        "passed", "failed", "error", "errors", "skipped", "xfailed", "xpassed", "deselected", "warning",
    ) if type(counts.get(name)) is int and counts[name] >= 0]
    return " (" + ", ".join(labels) + ")" if labels else ""


def execution_summary(state: Any, *, uncertain_paths=()) -> str:
    """Summarize observed changes, inspection and current verification.

    ``state`` is deliberately duck-typed to avoid a controller import cycle.
    Its status and verification ledger remain authoritative; old green tests
    and assistant claims cannot establish current passing verification.
    Canonical messages are never modified by this function.
    """
    messages = getattr(state, "messages", [])
    messages = messages if isinstance(messages, list) else []
    mutations: Counter[str] = Counter()
    created = set()
    uncertain = set(path for path in uncertain_paths if isinstance(path, str) and path)
    skills: list[str] = []
    tests: list[tuple[int, str, str, dict]] = []
    first_change = None
    last_change = -1
    for index, name, arguments, result in _observations(messages):
        if name in {"edit_file", "write_file"}:
            path = canonical_path(result.get("path") or arguments.get("path"))
            if result.get("success") is True and result.get("modified") is True and path:
                mutations[path] += 1
                if first_change is None:
                    first_change = index
                last_change = index
                if name == "write_file":
                    created.add(path)
            elif isinstance(result.get("error"), dict) and result["error"].get("code") == "operation_interrupted" and path:
                uncertain.add(path)
                if first_change is None:
                    first_change = index
                last_change = index
        if name == "read_skill" and result.get("success") is True:
            skill = result.get("name") or arguments.get("name")
            if isinstance(skill, str) and skill and skill not in skills:
                skills.append(skill)
        if name == "run_tests":
            status = _test_status(result)
            selection = arguments.get("path", ".")
            if status is not None and isinstance(selection, str):
                tests.append((index, status, selection, result))

    changed = list(mutations)
    # The controller's mutation ledger also covers compatible older journals
    # that have persisted modified_files but lack full tool payloads.
    for path in getattr(state, "modified_files", []):
        if isinstance(path, str) and path and path not in changed:
            changed.append(path)
    completion = getattr(state, "completion", {})
    completion = completion if isinstance(completion, dict) else {}
    requirements = completion.get("requirements", {})
    requirements = requirements if isinstance(requirements, dict) else {}
    missing = completion.get("missing", [])
    missing = [item for item in missing if isinstance(item, str)] if isinstance(missing, list) else []

    verification = getattr(state, "verification", "not_run")
    current_tests = [item for item in tests if item[0] > last_change]
    supporting_test = next((item for item in reversed(current_tests) if item[1] == verification), None)
    # A pass is reported only when both the current ledger and actual execution
    # agree. Outstanding failed selections can keep the ledger failed even
    # after a different, unrelated selection passes.
    if verification == "passed" and supporting_test is None:
        verification = "not_run"
    if verification not in {"passed", "failed", "timed_out", "not_run"}:
        verification = "not_run"

    tests_required = requirements.get("require_tests") is True or bool(changed) or bool(uncertain)
    if tests_required and verification != "passed" and not any("test" in item for item in missing):
        missing.append("observe passing tests after the latest change")
    status = getattr(state, "status", "incomplete")
    completed = status == "completed" and completion.get("satisfied") is not False and not missing
    lines = ["Completed." if completed else "Incomplete."]
    metrics = getattr(state, "metrics", {})
    if isinstance(metrics, dict) and type(metrics.get("decision_enabled")) is bool:
        lines.append("Decision assistance: " + ("on (tev1:4b)." if metrics["decision_enabled"] else "off."))

    if changed:
        details = []
        for path in changed:
            count = mutations[path]
            label = "created" if path in created else f"{count} exact edit{'s' if count != 1 else ''}" if count else "modified"
            details.append(f"{_identity(path)} ({label})")
        lines.append("Changed: " + _paths(details) + ".")
    elif uncertain:
        lines.append("No confirmed file changes were recorded.")
    else:
        lines.append("No files changed.")
    if uncertain:
        lines.append("Interrupted change outcomes were uncertain for: " + _paths(sorted(uncertain)) + ".")

    inspected = [path for path, evidence in InspectionLedger.from_messages(messages).files.items()
                 if evidence.inspected]
    lines.append("Inspected: " + (_paths(inspected) if inspected else "no complete source selection recorded") + ".")
    if skills:
        lines.append("Skills: " + _paths(skills) + ".")

    before_edits = next((item for item in reversed(tests)
                         if first_change is not None and item[0] < first_change), None)
    if before_edits is not None:
        _, status, selection, result = before_edits
        lines.append(f"Before edits: pytest {_identity(selection)} {status}{_test_counts(result)}.")

    if verification == "not_run":
        lines.append("Tests: no current passing verification recorded.")
    else:
        detail = ""
        if supporting_test is not None:
            _, _, selection, result = supporting_test
            detail = f" — pytest {_identity(selection)}"
            detail += _test_counts(result)
        lines.append(f"Tests: {verification}{detail}.")

    if not completed:
        reason = getattr(state, "stop_reason", "")
        if isinstance(reason, str) and reason:
            lines.append("Stopped: " + _identity(reason) + ".")
        if missing:
            bounded = [_identity(item) for item in missing[:6] if len(item) <= 500]
            if bounded:
                lines.append("Remaining: " + "; ".join(bounded) + ".")
    return "\n".join(lines)
