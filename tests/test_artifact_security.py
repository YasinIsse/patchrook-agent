"""Expandable evidence is exact, redacted, bounded, and session confined."""

import json
import os
from pathlib import Path

import pytest

from patchrook_agent.agent import Agent
from patchrook_agent.config import HarnessConfig
from patchrook_agent.models.ollama import ModelResponse
from patchrook_agent.observability.artifacts import ArtifactStore, MAX_ARTIFACT_BYTES
from patchrook_agent.observability import artifacts as artifact_module
from patchrook_agent.tools import ToolConfig, ToolError, ToolRegistry, Workspace


def store_at(path, **kwargs):
    path.mkdir()
    return ArtifactStore(path, **kwargs)


def reassemble(read, artifact_id, *, max_chars=137, limit=2000):
    pieces = []
    offset = 0
    for _ in range(2000):
        page = read(artifact_id=artifact_id, offset=offset, max_chars=max_chars)
        assert page["success"]
        assert page["artifact_id"] == artifact_id and page["offset"] == offset
        assert len(json.dumps(page, ensure_ascii=False, separators=(",", ":"))) <= limit
        pieces.append(page["content"])
        next_offset = page["next_offset"]
        if next_offset is None:
            assert not page["truncated"]
            return "".join(pieces)
        assert page["truncated"]
        assert next_offset == offset + len(page["content"]) and next_offset > offset
        offset = next_offset
    pytest.fail("artifact pagination did not finish")


def call(name, **arguments):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": name, "arguments": arguments}},
    ]}


class ExpandModel:
    context_size = 16384
    max_output_tokens = 512

    def __init__(self):
        self.calls = 0

    def chat(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            message = call("read_file", path="large.py")
        elif self.calls == 2:
            observation = next(json.loads(item["content"]) for item in reversed(kwargs["messages"])
                               if item.get("role") == "tool" and item.get("tool_name") == "read_file")
            message = call("read_artifact", artifact_id=observation["artifact_id"], max_chars=333)
        else:
            message = {"role": "assistant", "content": "Inspected the selected source and expanded its evidence."}
        return ModelResponse(message, 11, 5, 0, 0, 0, 0)


def test_managed_agent_preserves_full_source_in_expandable_artifact(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    lines = [f"value_{number} = 'λ café \\\"quoted\\\" {'x' * 40}'" for number in range(120)]
    (repo / "large.py").write_text("\n".join(lines) + "\n")
    registry = ToolRegistry(Workspace(repo, ToolConfig(max_output_chars=512)))
    state = Agent(ExpandModel(), registry=registry, config=HarnessConfig.preset("C"),
                  runs_dir=tmp_path / "runs").run("Inspect large.py and retrieve its omitted evidence.")
    assert state.status == "completed" and state.steps == 3
    observations = [(message["tool_name"], json.loads(message["content"]))
                    for message in state.messages if message.get("role") == "tool"]
    read_observation = observations[0][1]
    assert read_observation["success"] and read_observation["truncated"]
    artifact_id = read_observation["artifact_id"]
    for _, observation in observations:
        assert len(json.dumps(observation, ensure_ascii=False, separators=(",", ":"))) <= 512
    assert observations[1][0] == "read_artifact" and observations[1][1]["artifact_id"] == artifact_id
    text = reassemble(lambda **kwargs: registry.dispatch("read_artifact", kwargs), artifact_id, limit=352)
    full = json.loads(text)
    expected = "\n".join(f"{number}: {line}" for number, line in enumerate(lines, 1))
    assert full["available_output"]["content"] == expected
    assert len(expected) > len(read_observation["content"]) * 10
    artifact_path = Path(state.session_dir) / "artifacts" / f"{artifact_id}.json"
    assert artifact_path.stat().st_mode & 0o777 == 0o600


def test_new_artifact_ids_do_not_make_identical_reads_reset_failure_progress(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "large.py").write_text("source = '" + "x" * 5000 + "'\n")

    class RepeatingModel:
        context_size = 16384
        max_output_tokens = 512

        def chat(self, **kwargs):
            message = call("read_file", path="missing.py")
            message["tool_calls"].extend(call("read_file", path="large.py")["tool_calls"])
            return ModelResponse(message, 11, 5, 0, 0, 0, 0)

    state = Agent(RepeatingModel(), workspace=repo, config=HarnessConfig.preset("C"),
                  runs_dir=tmp_path / "runs", max_steps=5, max_tool_failures=2).run("Inspect the selected source.")
    assert state.stop_reason == "repeated_tool_failures"
    assert state.steps == 3 and state.tool_failures == 3


def test_artifact_pointer_remains_expandable_after_checkpoint_resume(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "large.py").write_text("value = '" + "x" * 5000 + "'\n")
    runs = tmp_path / "runs"
    paused = Agent(ExpandModel(), workspace=repo, config=HarnessConfig.preset("C"),
                   runs_dir=runs, pause_after_step=1).run("Inspect large.py and retrieve its omitted evidence.")
    assert paused.stop_reason == "paused"
    old_result = next(json.loads(message["content"]) for message in paused.messages
                      if message.get("tool_name") == "read_file")
    model = ExpandModel()
    model.calls = 1
    resumed = Agent(model, workspace=repo, config=HarnessConfig.preset("C"), runs_dir=runs).resume(paused.session_id)
    assert resumed.status == "completed" and resumed.session_id == paused.session_id
    expanded = next(json.loads(message["content"]) for message in resumed.messages
                    if message.get("tool_name") == "read_artifact")
    assert expanded["success"] and expanded["artifact_id"] == old_result["artifact_id"]


def test_exact_character_pages_cover_unicode_and_json_escapes_at_small_budget(tmp_path):
    store = store_at(tmp_path / "session", max_output_chars=511)
    payload = {"content": ("λ café 🐍 \"quoted\" \\\\ tab\tline\n" * 80), "nested": {"answer": 42}}
    artifact_id = store.save(payload)
    expected = (store.artifacts / f"{artifact_id}.json").read_text()
    actual = reassemble(store.read, artifact_id, max_chars=2000, limit=511)
    assert actual == expected
    assert json.loads(actual) == payload
    final = store.read(artifact_id, offset=len(actual), max_chars=1)
    assert final["content"] == "" and final["next_offset"] is None and not final["truncated"]


def test_artifact_and_full_memory_alias_redact_credentials(tmp_path):
    store = store_at(tmp_path / "session", max_output_chars=511)
    secret = "sk-abcdefghijklmnopqrstuvwxyz123456"
    payload = {"password": "private-password", "content": f"api_key = '{secret}'\nBearer abcdefghijklmnop"}
    artifact_id = store.save(payload)
    saved = reassemble(store.read, artifact_id, limit=511)
    assert secret not in saved and "private-password" not in saved and "abcdefghijklmnop" not in saved
    assert "[REDACTED]" in saved
    (store.directory / "state.json").write_text(json.dumps({"objective": f"password = '{secret}'", **payload}))
    memory = reassemble(store.read, "memory", limit=511)
    assert secret not in memory and "private-password" not in memory and "abcdefghijklmnop" not in memory
    assert "[REDACTED]" in memory


@pytest.mark.parametrize("artifact_id", ["../state.json", "/tmp/outside", "memory/../state.json", "tool-" + "g" * 24, ""])
def test_artifact_ids_cannot_select_paths(tmp_path, artifact_id):
    store = store_at(tmp_path / "session")
    with pytest.raises(ToolError) as error:
        store.read(artifact_id)
    assert error.value.code == "invalid_artifact"


def test_cross_session_artifact_id_is_not_readable(tmp_path):
    first = store_at(tmp_path / "one")
    second = store_at(tmp_path / "two")
    artifact_id = first.save({"content": "private first session"})
    with pytest.raises(ToolError) as error:
        second.read(artifact_id)
    assert error.value.code == "artifact_not_found"


@pytest.mark.parametrize("selection", ["artifact", "memory"])
def test_reader_rejects_symlinks_even_when_the_target_is_inside_session(tmp_path, selection):
    store = store_at(tmp_path / "session")
    target = store.directory / "private.json"
    target.write_text('{"content":"must not follow a symlink"}')
    artifact_id = "memory" if selection == "memory" else "tool-" + "a" * 24
    path = store.directory / "state.json" if selection == "memory" else store.artifacts / f"{artifact_id}.json"
    path.symlink_to(target)
    with pytest.raises(ToolError) as error:
        store.read(artifact_id)
    assert error.value.code == "invalid_artifact"


@pytest.mark.parametrize("kind", ["directory", "fifo", "oversized"])
def test_reader_rejects_nonregular_or_oversized_artifacts(tmp_path, kind):
    store = store_at(tmp_path / "session")
    artifact_id = "tool-" + "a" * 24
    path = store.artifacts / f"{artifact_id}.json"
    if kind == "directory":
        path.mkdir()
    elif kind == "fifo":
        os.mkfifo(path)
    else:
        with path.open("wb") as stream:
            stream.truncate(MAX_ARTIFACT_BYTES + 1)
    with pytest.raises(ToolError) as error:
        store.read(artifact_id)
    assert error.value.code == "artifact_limit"


def test_replaced_artifact_directory_cannot_redirect_a_save_outside_session(tmp_path):
    store = store_at(tmp_path / "session")
    outside = tmp_path / "outside"
    outside.mkdir()
    store.artifacts.rmdir()
    store.artifacts.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ToolError) as error:
        store.save({"content": "must remain inside the session"})
    assert error.value.code == "invalid_artifact"
    assert list(outside.iterdir()) == []


def test_save_byte_limit_includes_serialization_newline(tmp_path, monkeypatch):
    store = store_at(tmp_path / "session")
    payload = {"content": "λ small boundary payload"}
    encoded_bytes = len(json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"))
    monkeypatch.setattr(artifact_module, "MAX_ARTIFACT_BYTES", encoded_bytes)
    with pytest.raises(ToolError) as error:
        store.save(payload)
    assert error.value.code == "artifact_limit"
    assert list(store.artifacts.iterdir()) == []


@pytest.mark.parametrize("arguments", [
    {"offset": True}, {"offset": -1}, {"offset": "0"}, {"max_chars": 0},
    {"max_chars": 2001}, {"max_chars": "100"}, {"command": "cat"},
])
def test_artifact_registry_keeps_strict_runtime_argument_validation(tmp_path, arguments):
    store = store_at(tmp_path / "session")
    artifact_id = store.save({"content": "available"})
    repo = tmp_path / "repo"
    repo.mkdir()
    registry = ToolRegistry(Workspace(repo))
    registry.attach_artifacts(store)
    observation = registry.dispatch("read_artifact", {"artifact_id": artifact_id, **arguments})
    assert not observation["success"] and observation["error"]["code"] == "invalid_arguments"
