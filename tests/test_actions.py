"""A schema action is explicit, enabled, and never inferred from plan prose."""

from copy import deepcopy
import json

import pytest

from patchrook_agent.actions import action_instructions, action_schema, parse_action


READ = {"type": "function", "function": {
    "name": "read_file", "description": "Read numbered source lines.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"},
                                                          "start_line": {"type": "integer"}},
                   "required": ["path"], "additionalProperties": False},
}}
EDIT = {"type": "function", "function": {
    "name": "edit_file", "description": "Apply one exact replacement.",
    "parameters": {"type": "object", "properties": {name: {"type": "string"}
                                                          for name in ("path", "old_text", "new_text")},
                   "required": ["path", "old_text", "new_text"], "additionalProperties": False},
}}


def message(value):
    return {"role": "assistant", "content": json.dumps(value)}


def branch_actions(schema):
    return [branch["properties"]["action"]["enum"][0] for branch in schema["oneOf"]]


def test_required_work_excludes_completion_but_keeps_enabled_actions_and_blocking():
    schema = action_schema([READ], allow_complete=False)
    assert branch_actions(schema) == ["tool", "blocked"]
    branch = schema["oneOf"][0]
    assert branch["properties"]["tool"]["enum"] == ["read_file"]
    assert branch["properties"]["arguments"] == READ["function"]["parameters"]
    assert branch["additionalProperties"] is False
    assert set(branch["required"]) == {"action", "tool", "arguments"}


def test_completion_is_available_only_when_caller_enables_it():
    schema = action_schema([READ, EDIT], allow_complete=True)
    assert branch_actions(schema) == ["tool", "tool", "complete", "blocked"]
    normalized = parse_action(message({"action": "complete", "summary": " Read source; tests passed. "}),
                              [READ, EDIT], allow_complete=True)
    assert normalized == {"kind": "complete", "summary": "Read source; tests passed."}
    with pytest.raises(ValueError, match="unavailable"):
        parse_action(message({"action": "complete", "summary": "I will edit next."}), [READ], allow_complete=False)


def test_schema_and_normalized_arguments_do_not_mutate_or_alias_inputs():
    definitions = deepcopy([READ, EDIT])
    original = deepcopy(definitions)
    schema = action_schema(definitions, allow_complete=False)
    schema["oneOf"][0]["properties"]["arguments"]["properties"].clear()
    assert definitions == original
    action = {"action": "tool", "tool": "edit_file", "arguments": {"path": "calc.py", "old_text": "a-b", "new_text": "a+b"}}
    normalized = parse_action(message(action), definitions, allow_complete=False)
    assert normalized == {"kind": "tool", "call": {"type": "function", "function": {
        "name": "edit_file", "arguments": action["arguments"],
    }}}
    assert definitions == original


def test_blocked_outcome_remains_distinct_from_success_with_or_without_tools():
    for allowed in (False, True):
        parsed = parse_action(message({"action": "blocked", "reason": "The required module is missing."}),
                              [], allow_complete=allowed)
        assert parsed == {"kind": "blocked", "reason": "The required module is missing."}
    assert branch_actions(action_schema([], allow_complete=False)) == ["blocked"]


def test_disabled_tool_is_rejected_even_when_it_is_a_known_harness_function():
    with pytest.raises(ValueError, match="disabled"):
        parse_action(message({"action": "tool", "tool": "edit_file", "arguments": {}}), [READ], allow_complete=False)


@pytest.mark.parametrize("content", [
    "I will read calc.py next.", '```json\n{"action":"blocked","reason":"missing file"}\n```',
    '[{"action":"tool","tool":"read_file","arguments":{"path":"calc.py"}}]',
    '{"action":"blocked","reason":"missing file"} trailing prose',
    '{"action":"blocked","action":"complete","reason":"missing file"}',
    '{"action":"tool","tool":"read_file","arguments":{"path":"a.py","path":"b.py"}}',
    '{"action":"tool","tool":"read_file","arguments":{"start_line":NaN}}',
    '{"action":"tool","tool":"read_file","arguments":{"start_line":Infinity}}',
    "", "null", "true", "42",
])
def test_prose_invalid_json_ambiguous_objects_and_nonfinite_numbers_are_rejected(content):
    with pytest.raises(ValueError):
        parse_action({"role": "assistant", "content": content}, [READ], allow_complete=False)


@pytest.mark.parametrize("value", [
    {"action": "tool", "tool": "read_file", "arguments": {}, "summary": "I will read"},
    {"action": "tool", "tool": "read_file"},
    {"action": "tool", "tool": "read_file", "arguments": "{}"},
    {"action": "tool", "tool": [], "arguments": {}},
    {"action": "complete", "summary": "Done", "arguments": {}},
    {"action": "complete", "summary": " "},
    {"action": "complete", "summary": True},
    {"action": "blocked", "reason": " "},
    {"action": "blocked", "reason": "missing file", "tool": "read_file"},
    {"action": "finish", "summary": "Done"},
    {"action": [], "reason": "missing file"},
    {},
])
def test_action_envelope_is_strict_and_does_not_guess_the_intent(value):
    with pytest.raises(ValueError):
        parse_action(message(value), [READ], allow_complete=True)


@pytest.mark.parametrize("native", [[], None, [{"function": {"name": "read_file", "arguments": {"path": "calc.py"}}}]])
def test_native_tool_call_fields_cannot_bypass_schema_protocol(native):
    value = message({"action": "tool", "tool": "read_file", "arguments": {"path": "calc.py"}})
    value["tool_calls"] = native
    with pytest.raises(ValueError, match="Native tool calls"):
        parse_action(value, [READ], allow_complete=False)


def test_transport_metadata_is_allowed_but_nonassistant_actions_are_rejected():
    value = message({"action": "tool", "tool": "read_file", "arguments": {"path": "calc.py"}})
    value["thinking"] = ""
    assert parse_action(value, [READ], allow_complete=False)["kind"] == "tool"
    value["role"] = "user"
    with pytest.raises(ValueError, match="assistant"):
        parse_action(value, [READ], allow_complete=False)


def test_argument_validation_remains_the_registry_boundary():
    # Envelope parsing never coerces, fixes, or executes proposed arguments.
    value = message({"action": "tool", "tool": "read_file", "arguments": {"path": 4, "extra": "invalid"}})
    parsed = parse_action(value, [READ], allow_complete=False)
    assert parsed["call"]["function"]["arguments"] == {"path": 4, "extra": "invalid"}


def test_instructions_are_compact_enabled_only_and_do_not_repeat_argument_schemas():
    instructions = action_instructions([READ], False, ["inspect relevant source", "make the requested change", "observe passing tests"])
    assert "read_file" in instructions and "edit_file" not in instructions
    assert '"action":"complete"' not in instructions
    assert "Completion is unavailable" in instructions and "Missing:" in instructions
    assert "properties" not in instructions and len(instructions) <= 2500
    assert "read_file(path*, start_line?)" in instructions
    assert "Read numbered source lines." in instructions
    assert "At EOF inspect another relevant file" in instructions
    assert '"action":"complete"' in action_instructions([READ, EDIT], True, [])


def test_full_enabled_tool_guide_is_bounded_and_distinguishes_content_from_filename_search(tmp_path):
    from types import SimpleNamespace
    from patchrook_agent.tools import ToolRegistry, Workspace

    registry = ToolRegistry(Workspace(tmp_path))
    registry.attach_skills(SimpleNamespace(enabled=True, load=None))
    registry.attach_artifacts(SimpleNamespace(read=None))
    definitions = registry.definitions()
    assert len(definitions) == 9
    text = action_instructions(definitions, False, ["inspect source", "make the requested change", "observe passing tests"])
    assert len(text) <= 2500
    for definition in definitions:
        assert definition["function"]["name"] + "(" in text
    assert "search_code(pattern*, path?, max_results?, offset?, mode?)" in text
    assert "file CONTENT as literal text by default; mode=regex enables regex" in text
    assert "empty matches do not prove a missing file" in text
    assert "list_files for filename discovery" in text
    assert "concrete obstacle after available read/list recovery" in text


def test_read_only_subset_guide_does_not_advertise_missing_discovery_or_edit_tools():
    text = action_instructions([READ], False, ["inspect source"])
    assert "search_code" not in text and "list_files" not in text and "edit_file" not in text
    assert "Read a task-named path directly" in text


def test_search_only_subset_does_not_name_unavailable_path_discovery_tools():
    search = {"type": "function", "function": {
        "name": "search_code", "description": "Find paths and matching lines.",
        "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}}}
    text = action_instructions([search], False, ["inspect source"])
    assert "search_code(pattern*)" in text
    assert "list_files" not in text and "read_file" not in text
    assert "Find paths and matching lines" not in text


@pytest.mark.parametrize("definitions", [None, {}, [READ, READ], [{"type": "function", "function": {}}],
                                        [{"type": "function", "function": {"name": "read_file", "parameters": []}}]])
def test_invalid_enabled_definitions_do_not_produce_an_action_schema(definitions):
    with pytest.raises(ValueError):
        action_schema(definitions, allow_complete=False)
