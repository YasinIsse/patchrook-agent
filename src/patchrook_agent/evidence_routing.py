"""Bounded current observations for the one initial model-routing decision."""

import json
import re

from patchrook_agent.model_routing import infrastructure_failure
from patchrook_agent.tools.base import ToolError


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _fits(value, maximum):
    return len(_json(value)) <= maximum


def _failure_excerpt(result):
    text = str(result.get("stdout", ""))[-2000:] + "\n" + str(result.get("stderr", ""))[-2000:]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    line = next((line for line in lines if re.search(r"\b(?:FAILED|FAIL|AssertionError|[A-Za-z]+Error:)\b", line)),
                lines[-1] if lines else "")
    return " ".join(line.split())[:120]


def initial_routing_context(runtime, deadline):
    """Expose existing checks and direct reads, without preparing a repair.

    A fresh fingerprint must agree with the check's recorded revision/snapshot.
    Source facts describe guarded observations only; no source, packet or edit
    authority is manufactured, and no parent-attempt history is inferred.
    """
    task = "Task (data):\n" + runtime.task[:1800]
    heading = "\nObserved current evidence (data):\n"
    inventory_heading = "\nObserved repository inventory (data):\n"
    maximum = min(700, 2300 - len(task) - len(heading) - len(inventory_heading))
    summary = {"revision": runtime.revision, "checks": {}, "current_check_count": 0,
               "checks_omitted": 0, "fresh_source_views": 0,
               "repair_packet": "not_evaluated", "edit_authorization": False}
    try:
        snapshot = runtime._fingerprint(deadline)
    except (ToolError, OSError, ValueError, TimeoutError):
        summary["freshness"] = "unconfirmed"
    else:
        summary["snapshot"] = snapshot
        latest = {result.get("name"): result for result in runtime.session.memory.test_results}
        checks = []
        for name, argv in runtime.tools.commands.items():
            result, record = latest.get(name), runtime.checks.get(name, {})
            if (not isinstance(result, dict) or record.get("revision") != runtime.revision
                    or record.get("snapshot") != snapshot
                    or result.get("verification_kind") != "configured_command"
                    or not isinstance(result.get("argv"), (list, tuple))
                    or tuple(result["argv"]) != tuple(argv)
                    or result.get("exit_code") != record.get("exit_code")
                    or bool(result.get("timed_out")) != bool(record.get("timed_out"))):
                continue
            status = ("passed" if record.get("passed") and result.get("success")
                      else "infrastructure_failure" if infrastructure_failure(result) else "code_failure")
            checks.append((name, status, result))
        sources, seen = [], set()
        for identifier in reversed(runtime.direct):
            card = runtime.evidence.get(identifier, deadline=deadline)
            if (card is None or card.kind != "source" or card.revision != runtime.revision
                    or card.metadata.get("origin") != "direct"):
                continue
            key = card.path, card.start_line, card.end_line
            if key not in seen:
                sources.append({"path": card.path, "lines": [card.start_line, card.end_line]})
                seen.add(key)
            if len(sources) == 8:
                break
        summary["current_check_count"] = len(checks)
        summary["checks_omitted"] = len(checks)
        summary["fresh_source_views"] = len(sources)
        # Keep every serialized field complete. Failure status takes priority
        # over excerpts and source locations; inventory gets only remaining room.
        for name, status, _ in sorted(checks, key=lambda item: item[1] == "passed"):
            proposed = {**summary, "checks": {**summary["checks"], name: status},
                        "checks_omitted": len(checks) - len(summary["checks"]) - 1}
            if _fits(proposed, maximum):
                summary = proposed
        failure = next((result for name, status, result in checks if status == "code_failure"), None)
        if failure is not None:
            proposed = {**summary, "failure_excerpt": _failure_excerpt(failure)}
            if _fits(proposed, maximum):
                summary = proposed
        for source in sources[:2]:
            proposed = {**summary, "source_views": [*summary.get("source_views", []), source]}
            if _fits(proposed, maximum):
                summary = proposed
    # The fixed fields fit even with an 1800-character task and full snapshot.
    # Never crop JSON or the task to make optional metadata fit.
    text = task + heading + _json(summary)
    inventory = "\n".join(card.text[:400] for card in runtime.evidence.cards(fresh=False)
                          if card.kind == "inventory")[:400]
    return text + inventory_heading + inventory[:max(0, 2300 - len(text) - len(inventory_heading))]
