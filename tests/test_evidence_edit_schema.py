"""Edit grammar matches the guarded tool contract; Draft validation is test-only."""

from copy import deepcopy
import hashlib
import json
import time

import httpx
import pytest

from patchrook_agent.evidence_tools import EvidenceTools
from patchrook_agent.evidence_workflow import _tool_instructions, action_schema
from patchrook_agent.models.ollama import OllamaClient
from patchrook_agent.tools import ToolConfig, Workspace


REAL_HTTP_CLIENT = httpx.Client
V4_MIXED = {
    "path": "src/cattrs/converters.py",
    "old_text": '    """Converts between structured and unstructured data."""\n    __slots__ = (',
    "new_text": '    """Converts between structured and unstructured data."""\n    __slots__ = (',
    "content": '    def partial_structure(self, data, target_type):\n'
               '        """Returns a PartialResult with structured and failed fields."""\n'
               '        # Implementation goes here.',
}


@pytest.fixture
def draft():
    # The optional benchmark environment already contains jsonschema. Keep it
    # out of the agent's runtime dependencies and skip only Draft cases when
    # that test dependency is absent from the invoking environment.
    return pytest.importorskip("jsonschema").Draft202012Validator


@pytest.fixture
def definition(tmp_path):
    return next(item for item in EvidenceTools(Workspace(tmp_path)).definitions()
                if item["function"]["name"] == "apply_edits")


@pytest.fixture
def validator(definition, draft):
    schema = definition["function"]["parameters"]
    draft.check_schema(schema)
    return draft(schema)


@pytest.mark.parametrize("item", [
    {"path": "subject.py", "old_text": "before", "new_text": "after"},
    {"path": "subject.py", "old_text": "delete this", "new_text": ""},
    {"path": "subject.py", "old_text": " ", "new_text": ""},
    {"path": "new.py", "content": "VALUE = 1\n"},
    {"path": "empty.py", "content": ""},
])
def test_draft_allows_exact_replacements_creations_and_empty_payloads(validator, item):
    validator.validate({"edits": [item]})


@pytest.mark.parametrize("item", [
    V4_MIXED,
    {"path": "subject.py"},
    {"path": "subject.py", "old_text": "before"},
    {"path": "subject.py", "new_text": "after"},
    {"old_text": "before", "new_text": "after"},
    {"content": "new"},
    {"path": "subject.py", "old_text": "", "new_text": "after"},
    {"path": "subject.py", "old_text": None, "new_text": "after"},
    {"path": "subject.py", "old_text": "before", "new_text": None},
    {"path": "subject.py", "content": None},
    {"path": "subject.py", "content": "new", "old_text": "before"},
    {"path": "subject.py", "old_text": "before", "new_text": "after", "extra": True},
    {"path": "subject.py", "content": "new", "extra": True},
])
def test_draft_rejects_actual_v4_mix_missing_fields_bad_types_and_extra_fields(validator, item):
    assert not validator.is_valid({"edits": [item]})


def test_draft_enforces_batch_limits_and_closed_outer_arguments(validator):
    valid = {"path": "subject.py", "content": ""}
    assert not validator.is_valid({"edits": []})
    assert validator.is_valid({"edits": [valid] * 16})
    assert not validator.is_valid({"edits": [valid] * 17})
    assert not validator.is_valid({"edits": [valid], "extra": True})


def test_compact_signature_shows_two_exclusive_required_shapes(definition):
    instructions = _tool_instructions([definition])
    assert "edits:array[{path,old_text,new_text}|{path,content}]" in instructions
    assert "old_text?" not in instructions and "new_text?" not in instructions and "content?" not in instructions
    assert "do not mix these shapes" in instructions


def test_compact_renderer_keeps_existing_single_object_and_scalar_array_shapes():
    definition = {"function": {"name": "batch", "description": "existing description", "parameters": {
        "type": "object", "properties": {"objects": {"type": "array", "items": {"type": "object",
        "required": ["path"], "properties": {"path": {"type": "string"}, "note": {"type": "string"}}}},
        "names": {"type": "array", "items": {"type": "string"}}}, "required": ["objects"]}}}
    instructions = _tool_instructions([definition])
    assert "objects:array[{path,note?}]" in instructions and "names?:array" in instructions


def test_valid_empty_replacement_and_creation_still_execute_atomically(tmp_path, validator):
    source = tmp_path / "subject.py"
    data = b"VALUE = 1\n"
    source.write_bytes(data)
    editor = EvidenceTools(Workspace(tmp_path, ToolConfig()))
    args = {"edits": [{"path": "subject.py", "old_text": data.decode(), "new_text": ""},
                      {"path": "new/empty.py", "content": ""}]}
    validator.validate(args)
    result = editor.execute("apply_edits", args, deadline=time.monotonic() + 10, visible_sources=[{
        "path": "subject.py", "start_line": 1, "end_line": 1, "content": data.decode(),
        "source_sha256": hashlib.sha256(data).hexdigest()}])
    assert result["success"] and source.read_bytes() == b""
    assert (tmp_path / "new/empty.py").is_file() and (tmp_path / "new/empty.py").read_bytes() == b""


def test_runtime_still_rejects_native_mixed_shape_without_writes(tmp_path, validator):
    assert not validator.is_valid({"edits": [V4_MIXED]})
    result = EvidenceTools(Workspace(tmp_path)).execute("apply_edits", {"edits": [V4_MIXED]},
                                                     deadline=time.monotonic() + 10)
    assert result["error"]["code"] == "invalid_arguments"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("protocol", ["schema", "native"])
def test_nested_edit_oneof_survives_real_http_serialization(definition, draft, monkeypatch, protocol):
    valid = {"name": "apply_edits", "arguments": {"edits": [{"path": "new.py", "content": ""}]}}
    mixed = {"name": "apply_edits", "arguments": {"edits": [V4_MIXED]}}
    original = deepcopy(definition)
    observed = []
    def server(request):
        payload = json.loads(request.content)
        observed.append(payload)
        if protocol == "schema":
            assert "tools" not in payload
            schema = payload["format"]
            draft.check_schema(schema)
            gate = draft(schema)
            gate.validate(valid)
            assert not gate.is_valid(mixed)
            message = {"role": "assistant", "content": json.dumps(valid)}
        else:
            assert "format" not in payload
            assert payload["tools"] == [definition]
            schema = payload["tools"][0]["function"]["parameters"]
            draft.check_schema(schema)
            gate = draft(schema)
            gate.validate(valid["arguments"])
            assert not gate.is_valid(mixed["arguments"])
            message = {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": valid["name"], "arguments": valid["arguments"]}}]}
        assert payload["think"] is False and payload["truncate"] is False
        return httpx.Response(200, json={"message": message, "done_reason": "stop",
            "prompt_eval_count": 100, "eval_count": 10})
    monkeypatch.setattr(httpx, "Client", lambda **options:
        REAL_HTTP_CLIENT(transport=httpx.MockTransport(server), **options))
    options = {"response_format": action_schema([definition])} if protocol == "schema" else {"tools": [definition]}
    response = OllamaClient(base_url="http://mock.invalid:11434").chat(
        [{"role": "system", "content": "Use the tool grammar."}, {"role": "user", "content": "Create an empty file."}],
        **options)
    assert len(observed) == 1 and response.completion_tokens == 10
    assert definition == original
