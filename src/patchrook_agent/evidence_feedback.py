"""Bounded rejection metadata; proposed code never becomes replacement source."""

import hashlib
import json
import re


MODEL_FAILURE_PREFIX = "Model response rejection (data): "


def repair_action_correction(correction, failed_checks, commands):
    """Reference only an exact configured-check failure already in this packet."""
    prefix = "Observed action failure: "
    if not isinstance(correction, str) or not correction.startswith(prefix):
        return correction
    try:
        result = json.loads(correction[len(prefix):])
    except (TypeError, ValueError, RecursionError):
        return correction
    allowed = {"success", "kind", "verification_kind", "name", "argv", "status", "exit_code", "timed_out",
               "stdout", "stderr", "error", "duration_seconds", "truncated"}
    if (not isinstance(result, dict) or set(result) - allowed or result.get("success") is not False
            or result.get("kind") != "command_check" or result.get("verification_kind") != "configured_command"
            or result.get("timed_out") is not False or result.get("truncated") is not False
            or result.get("status") != "exited" or type(result.get("exit_code")) is not int
            or result["exit_code"] == 0 or not isinstance(result.get("name"), str) or result["name"] not in commands
            or result.get("argv") != list(commands[result["name"]])):
        return correction
    fields = ("name", "exit_code", "stdout", "stderr", "error")
    if any(isinstance(row, dict) and all(row.get(key) == result.get(key) for key in fields) for row in failed_checks):
        return f"Configured check {result['name']} failed; its exact output and error are in failed_checks."
    return correction


def public_case_increment(results, commands, checks, revision):
    """Reference one declared, currently failing public case; never copy its fixture."""
    for name, argv in commands.items():
        result, current = results.get(name), checks.get(name, {})
        if (not isinstance(result, dict) or current.get("revision") != revision
                or current.get("passed") is not False or result.get("success") is not False
                or result.get("verification_kind") != "configured_command" or result.get("name") != name
                or result.get("argv") != list(argv) or result.get("status") != "exited"
                or type(result.get("exit_code")) is not int or result["exit_code"] == 0
                or result.get("timed_out") is not False or result.get("truncated") is not False
                or result.get("error") or not isinstance(result.get("stdout"), str)):
            continue
        try:
            report = json.loads(result["stdout"])
        except (ValueError, RecursionError):
            continue
        if not isinstance(report, dict):
            continue
        representative, failures = report.get("representative_failure"), report.get("failures")
        if (not isinstance(representative, dict) or not isinstance(failures, list)
                or representative.get("edit_authorization") is not False
                or not all(isinstance(representative.get(key), dict) and representative[key]
                           for key in ("fixture", "call", "expected"))):
            continue
        case = representative.get("case")
        if not isinstance(case, str) or re.fullmatch(r"[A-Za-z][A-Za-z0-9_.:-]{0,95}", case) is None:
            continue
        counts = [report.get(key) for key in ("total", "passed", "failed")]
        if (not all(type(count) is int and count >= 0 for count in counts)
                or counts[2] == 0 or counts[0] != counts[1] + counts[2]):
            continue
        cases = []
        for group in failures:
            if (not isinstance(group, dict) or not isinstance(group.get("cases"), list)
                    or not all(isinstance(item, str) for item in group["cases"])):
                break
            cases.extend(group["cases"])
        else:
            if len(cases) == counts[2] and len(set(cases)) == len(cases) and case in cases:
                return {"check": name, "case": case, "instruction": (
                    "Make this currently failing public case pass with a working implementation in the next increment. "
                    "Use its representative_failure in failed_checks; preserve the full task and passing behavior. "
                    "All configured checks remain required for completion.")}
    return None


def patch_metadata(diff, paths):
    """Describe the exact internal patch without exposing obsolete source."""
    paths = list(dict.fromkeys(path for path in paths if isinstance(path, str)))
    raw = diff.encode("utf-8")
    result = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
              "modified_paths": paths[:4], "modified_path_count": len(paths),
              "paths_truncated": len(paths) > 4, "edit_authorization": False}
    while len(json.dumps(result, separators=(",", ":"))) > 900 and result["modified_paths"]:
        result["modified_paths"].pop()
        result["paths_truncated"] = True
    return result


def _bounded(value):
    def size():
        return len(json.dumps(value, separators=(",", ":")))
    while size() > 900 and value.get("anchors"):
        value["anchors"].pop()
        value["truncated"] = True
    error = value.get("error")
    while size() > 900 and isinstance(error, dict) and error.get("message"):
        error["message"] = error["message"][:-16]
        error["message_truncated"] = True
    return value


def safe_receipt(summary, error=None, **provenance):
    """Also strip executable-looking previews from older saved summaries."""
    fields = ("edit_count", "malformed_edits", "proposal_sha256", "truncated")
    result = {key: summary[key] for key in fields if key in summary}
    result.update(kind="rejected_candidate", applied=False, edit_authorization=False,
                  source_of_error="rejected_candidate_not_current_worktree")
    result.update({key: value for key, value in provenance.items()
                   if key in {"step", "model", "revision", "inference_attempt", "done_reason", "kind"}})
    result["anchors"] = []
    for row in summary.get("anchors", [])[:4]:
        if isinstance(row, dict):
            result["anchors"].append({key: row[key] for key in (
                "path", "kind", "old_text_chars", "old_text_sha256", "repeat_count", "replacement_chars") if key in row})
    if isinstance(error, dict):
        result["error"] = {key: error[key][:limit] for key, limit in (("code", 64), ("message", 240))
                           if isinstance(error.get(key), str)}
        if error.get("message_truncated") is True or isinstance(error.get("message"), str) and len(error["message"]) > 240:
            result["error"]["message_truncated"] = True
    return _bounded(result)


def rejected_edit_summary(arguments, *, error=None, **provenance):
    edits = arguments.get("edits")
    summary = {"edit_count": len(edits) if isinstance(edits, list) else None, "anchors": [], "truncated": False}
    if not isinstance(edits, list):
        summary["malformed_edits"] = True
    else:
        groups = {}
        for edit in edits[:16]:
            if not isinstance(edit, dict) or not isinstance(edit.get("path"), str):
                continue
            old = edit.get("old_text")
            old = old if isinstance(old, str) else None
            kind = ("replace" if set(edit) == {"path", "old_text", "new_text"}
                    else "create" if set(edit) == {"path", "content"} else "invalid")
            row = groups.setdefault((edit["path"], kind, old), {"path": edit["path"], "kind": kind,
                "old_text_chars": len(old) if old is not None else None,
                "old_text_sha256": hashlib.sha256(old.encode()).hexdigest() if old is not None else None,
                "repeat_count": 0, "replacement_chars": []})
            row["repeat_count"] += 1
            replacement = edit.get("new_text", edit.get("content"))
            row["replacement_chars"].append(len(replacement) if isinstance(replacement, str) else None)
        summary["anchors"] = list(groups.values())[:4]
        summary["truncated"] = len(groups) > 4 or len(edits) > 16
    try:
        raw = json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        summary["proposal_sha256"] = hashlib.sha256(raw).hexdigest()
    except (TypeError, ValueError, RecursionError):
        pass
    return safe_receipt(summary, error, **provenance)


def recent_rejection_receipts(state, routing=None):
    """Use persisted receipts, with chronological tool-error recovery for old runs."""
    found = []
    step = 0
    for message in state.messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant":
            step += 1
        if message.get("role") != "tool" or message.get("tool_name") != "apply_edits":
            continue
        try:
            result = json.loads(message["content"])
        except (KeyError, TypeError, ValueError):
            continue
        summary = result.get("rejected_edit") if isinstance(result, dict) else None
        if not isinstance(summary, dict):
            continue
        model = (routing or {}).get("initial_model")
        for switch in (routing or {}).get("switches", []):
            if type(switch.get("step")) is int and step > switch["step"]:
                model = switch.get("to_model")
        provenance = {key: summary[key] for key in ("step", "model", "revision", "inference_attempt") if key in summary}
        found.append(safe_receipt(summary, result.get("error"), **({"step": step, "model": model} | provenance)))
    for note in state.recovery_notes:
        if not isinstance(note, str) or not note.startswith(MODEL_FAILURE_PREFIX):
            continue
        try:
            receipt = json.loads(note[len(MODEL_FAILURE_PREFIX):])
        except (TypeError, ValueError):
            continue
        if isinstance(receipt, dict) and receipt.get("kind") == "invalid_model_response":
            found.append(safe_receipt({}, receipt.get("error"), **{key: receipt[key] for key in (
                "step", "model", "revision", "inference_attempt", "done_reason", "kind") if key in receipt}))
    found.sort(key=lambda row: (row.get("step") if type(row.get("step")) is int else -1,
                                row.get("inference_attempt") if type(row.get("inference_attempt")) is int else -1))
    return found[-2:]
