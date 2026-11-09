"""Pure schema-action helpers; tool argument execution stays in the registry."""

from copy import deepcopy
import json
from typing import Any


_MAX_ACTION_CHARS = 2_100_000
_MAX_INSTRUCTION_CHARS = 2500


def _definitions(definitions: list[dict]) -> dict[str, dict]:
    if not isinstance(definitions, list):
        raise ValueError("Action definitions must be a list")
    functions = {}
    for definition in definitions:
        if not isinstance(definition, dict) or definition.get("type") != "function":
            raise ValueError("Action definitions must contain function tools")
        function = definition.get("function")
        if not isinstance(function, dict):
            raise ValueError("Invalid action tool definition")
        name, parameters = function.get("name"), function.get("parameters")
        if not isinstance(name, str) or not name or name in functions:
            raise ValueError("Action tools must have unique nonempty names")
        if not isinstance(parameters, dict) or parameters.get("type") != "object":
            raise ValueError("Action tool parameters must be an object schema")
        functions[name] = function
    return functions


def _envelope(properties: dict[str, dict]) -> dict:
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def action_schema(definitions: list[dict], *, allow_complete: bool) -> dict:
    """Use the enabled tool schemas verbatim inside a single-action envelope."""
    if type(allow_complete) is not bool:
        raise ValueError("allow_complete must be a boolean")
    branches = []
    for name, function in _definitions(definitions).items():
        branch = _envelope({
            "action": {"type": "string", "enum": ["tool"]},
            "tool": {"type": "string", "enum": [name]},
            "arguments": deepcopy(function["parameters"]),
        })
        if isinstance(function.get("description"), str):
            branch["description"] = function["description"]
        branches.append(branch)
    if allow_complete:
        branches.append(_envelope({
            "action": {"type": "string", "enum": ["complete"]},
            "summary": {"type": "string", "minLength": 1},
        }))
    branches.append(_envelope({
        "action": {"type": "string", "enum": ["blocked"]},
        "reason": {"type": "string", "minLength": 1},
    }))
    return {"oneOf": branches}


def lean_action_schema(definitions: list[dict], *, allow_complete: bool) -> dict:
    """Constrain the action envelope; the registry validates tool arguments."""
    if type(allow_complete) is not bool:
        raise ValueError("allow_complete must be a boolean")
    names = list(_definitions(definitions))
    branches = [_envelope({
        "action": {"type": "string", "enum": ["tool"]},
        "tool": {"type": "string", "enum": names},
        "arguments": {"type": "object", "additionalProperties": True},
    })] if names else []
    if allow_complete:
        branches.append(_envelope({"action": {"type": "string", "enum": ["complete"]},
                                   "summary": {"type": "string", "minLength": 1}}))
    branches.append(_envelope({"action": {"type": "string", "enum": ["blocked"]},
                               "reason": {"type": "string", "minLength": 1}}))
    return {"oneOf": branches}


def _argument_contract(schema: dict, depth: int = 0) -> str:
    kind = schema.get("type", "value")
    if not isinstance(kind, str):
        return "value"
    if kind == "array":
        return "[" + _argument_contract(schema.get("items", {}), depth + 1) + "]"
    if kind == "object" and depth < 3:
        required = schema.get("required", [])
        return "{" + ",".join(name + ("" if name in required else "?") + ":" + _argument_contract(value, depth + 1)
                               for name, value in schema.get("properties", {}).items()) + "}"
    return {"integer": "int", "number": "number", "boolean": "bool"}.get(kind, str(kind))


def lean_action_instructions(definitions: list[dict], allow_complete: bool,
                             missing: list[str], *, schema_mode: bool = True) -> str:
    """Short contracts explain arguments once without copyable blocker text."""
    functions = _definitions(definitions)
    prefix = ('Return one JSON action: {"action":"tool","tool":"name","arguments":{...}}. '
              if schema_mode else "Use the enabled tools to complete the task. ")
    if schema_mode:
        prefix += ('Finish with {"action":"complete","summary":"changes and observed tests"}. '
                   if allow_complete else "Completion is unavailable. ")
    if missing:
        prefix += "Remaining: " + "; ".join(item[:100] for item in missing[:3]) + ". "
    if schema_mode:
        prefix += ("Use blocked only when tools cannot resolve a specific observed failure; "
                   "describe that failure and the remaining task in your own words. Correct argument errors and retry. ")
    if "read_code" in functions:
        prefix += "read_code takes paths only and selects useful code automatically. Reuse code already shown. "
        if "read_file" in functions:
            prefix += "Use read_file for a specific line range. "
    if "search_code" in functions:
        prefix += "search_code searches file CONTENT as literal text by default; mode=regex enables regex. "
    if "run_tests" in functions:
        prefix += 'Use run_tests with {} for repository-wide pytest discovery; set path only for a known test target. '
    contracts = []
    for name, function in functions.items():
        signature = _argument_contract(function["parameters"])
        description = " ".join(function.get("description", "").split())[:110]
        contracts.append(f"{name} {signature}: {description}")
    return prefix + "\n" + "\n".join(contracts)


def action_instructions(definitions: list[dict], allow_complete: bool, missing: list[str]) -> str:
    """Ground enabled tool semantics; the format schema still owns argument types."""
    if type(allow_complete) is not bool:
        raise ValueError("allow_complete must be a boolean")
    functions = _definitions(definitions)
    if not isinstance(missing, list) or any(not isinstance(item, str) for item in missing):
        raise ValueError("Missing completion evidence must be a list of strings")
    text = ('Return one schema-valid JSON object only. '
            'Tool: {"action":"tool","tool":"name","arguments":{...}}. ')
    if allow_complete:
        text += 'Finish: {"action":"complete","summary":"observed changes and verification"}. '
    else:
        text += "Completion is unavailable. "
    if missing:
        # Keep guidance bounded; the caller retains the full evidence ledger.
        bounded = "; ".join(item[:100] for item in missing[:4])[:160]
        text += "Missing: " + bounded + ". "
    routing = []
    if "read_file" in functions:
        routing.append("Read a task-named path directly with read_file.")
    if "read_files" in functions:
        routing.append("Batch established paths with read_files.")
    if "list_files" in functions:
        routing.append("Use list_files for filename discovery.")
    if "search_code" in functions:
        routing.append("search_code searches file CONTENT as literal text by default; mode=regex enables regex; empty matches do not prove a missing file.")
    if "read_file" in functions or "read_files" in functions:
        routing.append("For reads honor total_lines/truncated; follow next_start_line only when provided. At EOF inspect another relevant file.")
    text += " ".join(routing) + ("\n" if routing else "")
    text += "Enabled tool contracts (* required, ? optional; types/constraints are in the schema):\n"
    terminal = ('Use blocked only for a concrete obstacle after available read/list recovery: '
                '{"action":"blocked","reason":"obstacle and remaining work"}.')
    contracts = []
    for name, function in functions.items():
        parameters = function["parameters"]
        properties = parameters.get("properties", {})
        properties = properties if isinstance(properties, dict) else {}
        required = parameters.get("required", [])
        required = required if isinstance(required, list) else []
        arguments = ", ".join(argument + ("*" if argument in required else "?") for argument in properties)
        prefix = f"{name}({arguments}): "
        description = function.get("description", "")
        description = " ".join(description.split()) if isinstance(description, str) else ""
        if name == "search_code":
            description = "Search literal file contents by default; mode=regex enables regex. Empty matches are not a missing-file result."
            if "list_files" in functions:
                description += " Use list_files for filename discovery."
        contracts.append((prefix, description))
    available = _MAX_INSTRUCTION_CHARS - len(text) - len(terminal) - sum(len(prefix) + 1 for prefix, _ in contracts)
    if available < 0:
        raise ValueError("Enabled tool contracts exceed the action instruction budget")
    description_limit = min(160, available // max(1, len(contracts)))
    return text + "\n".join(prefix + description[:description_limit] for prefix, description in contracts) + "\n" + terminal


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON action field")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError("Invalid JSON constant: " + value)


def parse_action(message: dict, definitions: list[dict], *, allow_complete: bool) -> dict[str, Any]:
    """Normalize a strict content action; never turn arbitrary prose into a call.

    The registry remains authoritative for argument types, required fields,
    bounds and workspace permissions. This parser only validates the action
    envelope and enabled tool identity; it does not execute or repair anything.
    """
    if type(allow_complete) is not bool:
        raise ValueError("allow_complete must be a boolean")
    functions = _definitions(definitions)
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise ValueError("Schema action must be an assistant message")
    if "tool_calls" in message:
        raise ValueError("Native tool calls cannot be mixed with schema actions")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip() or len(content) > _MAX_ACTION_CHARS:
        raise ValueError("Schema action content must be a bounded JSON object")
    try:
        action = json.loads(content, object_pairs_hook=_object, parse_constant=_invalid_constant)
    except (ValueError, TypeError, RecursionError) as exc:
        raise ValueError("Invalid JSON action: " + str(exc)) from exc
    if not isinstance(action, dict):
        raise ValueError("Schema action must be a JSON object")
    kind = action.get("action")
    if not isinstance(kind, str):
        raise ValueError("Schema action name must be a string")
    if kind == "tool":
        if set(action) != {"action", "tool", "arguments"}:
            raise ValueError("Tool action requires only action, tool and arguments")
        name, arguments = action["tool"], action["arguments"]
        if not isinstance(name, str) or name not in functions:
            raise ValueError("Schema action requested a disabled or unknown tool")
        if not isinstance(arguments, dict):
            raise ValueError("Tool action arguments must be an object")
        return {"kind": "tool", "call": {"type": "function", "function": {
            "name": name, "arguments": deepcopy(arguments),
        }}}
    if kind in {"complete", "blocked"}:
        field = "summary" if kind == "complete" else "reason"
        if set(action) != {"action", field} or not isinstance(action[field], str) or not action[field].strip():
            raise ValueError(f"{kind} action requires a nonempty {field} and no extra fields")
        if kind == "complete" and not allow_complete:
            raise ValueError("Completion is unavailable until required evidence is observed")
        return {"kind": kind, field: action[field].strip()}
    raise ValueError("Unknown schema action")
