"""Offer one explicit read correction from observed argument errors."""

import json

from patchrook_agent.inspection import canonical_path
from patchrook_agent.tools.base import ToolError


_READERS = {"read_file", "read_files", "read_code"}
_ARGUMENT_ERRORS = {"invalid_arguments", "invalid_range", "invalid_selection", "symbol_not_found", "ambiguous_symbol"}
_CONCRETE_OBSTACLES = {"permission_denied", "outside_workspace", "not_found", "not_file", "io_error"}


def _object(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return {}
    return value if isinstance(value, dict) else {}


def _paths(arguments):
    result = []
    if isinstance(arguments.get("path"), str):
        result.append(arguments["path"])
    if isinstance(arguments.get("paths"), list):
        result.extend(path for path in arguments["paths"] if isinstance(path, str))
    if isinstance(arguments.get("files"), list):
        result.extend(item["path"] for item in arguments["files"]
                      if isinstance(item, dict) and isinstance(item.get("path"), str))
    return result


def corrective_read_action(messages: list[dict], workspace) -> dict | None:
    """Return path-only retry guidance; never execute or invent source evidence.

    Successful subsequent reads resolve the corresponding argument error.
    A newer unresolved concrete read obstacle takes precedence over any older
    argument mistake. Listings and navigation do not erase an unresolved error.
    """
    observations = []
    pending = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant" and isinstance(message.get("tool_calls"), list):
            pending = [call for call in message["tool_calls"] if isinstance(call, dict)]
        elif message.get("role") == "tool":
            name = message.get("tool_name")
            call = next((call for call in pending if call.get("id") == message.get("tool_call_id")
                         and message.get("tool_call_id") is not None), None)
            if call is None:
                call = next((call for call in pending if isinstance(call.get("function"), dict)
                             and call["function"].get("name") == name), {})
            if call in pending:
                pending.remove(call)
            function = call.get("function", {})
            arguments = _object(function.get("arguments", {})) if isinstance(function, dict) else {}
            observations.append((name, arguments, _object(message.get("content"))))
    resolved = set()
    for name, arguments, result in reversed(observations):
        if name not in _READERS:
            # A newer real filesystem/permission obstacle is not repaired by
            # retrying an older, otherwise recoverable source request.
            code = _object(result.get("error")).get("code")
            if isinstance(code, str) and code in _CONCRETE_OBSTACLES:
                return None
            continue
        records = result.get("files") if name != "read_file" else [result]
        records = records if isinstance(records, list) and records else [result]
        candidates = []
        recoverable = False
        for record in records:
            if not isinstance(record, dict):
                continue
            path = canonical_path(record.get("path"), workspace)
            if record.get("success") is True and path is not None:
                resolved.add(path)
                continue
            error = _object(record.get("error"))
            code = error.get("code")
            if not isinstance(code, str) or path in resolved:
                continue
            if code not in _ARGUMENT_ERRORS:
                return None
            paths = [record["path"]] if isinstance(record.get("path"), str) else _paths(arguments)
            if not paths:
                return None
            unresolved = [value for value in paths if canonical_path(value, workspace) not in resolved]
            if unresolved:
                recoverable = True
                candidates.extend(unresolved)
        if not recoverable:
            continue
        paths = []
        for candidate in candidates:
            path = canonical_path(candidate, workspace)
            if path is None:
                return None
            try:
                if not workspace.resolve(path).is_file():
                    return None
            except (OSError, ValueError, ToolError):
                return None
            if path not in paths:
                if len(json.dumps(paths + [path], ensure_ascii=False)) > 800:
                    break
                paths.append(path)
            if len(paths) == 10:
                break
        if paths:
            return {"action": "tool", "tool": "read_code", "arguments": {"paths": paths}}
    return None
