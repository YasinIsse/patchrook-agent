"""Unchanged replacements are errors and do not count as code changes."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from patchrook_agent.agent import Agent
from patchrook_agent.completion import TaskRequirements
from patchrook_agent.config import HarnessConfig
from patchrook_agent.models.ollama import ModelResponse
from patchrook_agent.tools import ToolConfig, ToolRegistry, Workspace
from patchrook_agent.tools.base import ToolError
from patchrook_agent.tools.files import edit_file


@pytest.mark.parametrize("old_text,new_text", [
    ("return 1", "return 1"),
    ("def answer():\r\n    return 1", "def answer():\n    return 1"),
])
def test_noop_rejects_identical_or_normalized_replacement_without_writing(
    tmp_path, monkeypatch, old_text, new_text,
):
    path = tmp_path / "answer.py"
    original = b"def answer():\r\n    return 1\r\n# Preserve this comment.\r\n"
    path.write_bytes(original)
    path.chmod(0o750)
    before = path.stat()

    def unexpected_write(*args):
        pytest.fail("A no-op must not replace the file")

    monkeypatch.setattr("patchrook_agent.tools.files._replace_bytes", unexpected_write)
    with pytest.raises(ToolError) as error:
        edit_file(Workspace(tmp_path), "answer.py", old_text, new_text)

    assert error.value.code == "no_change"
    assert "unchanged" in error.value.message and "different new_text" in error.value.message
    after = path.stat()
    assert (after.st_mtime_ns, after.st_ctime_ns, after.st_ino, after.st_mode) == (
        before.st_mtime_ns, before.st_ctime_ns, before.st_ino, before.st_mode,
    )
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("path,content,old_text,code", [
    ("missing.py", None, "same", "not_found"),
    ("example.py", "different", "same", "no_match"),
    ("example.py", "same same", "same", "ambiguous_match"),
    ("example.py", "same", "", "invalid_match"),
    ("../outside.py", None, "same", "outside_workspace"),
])
def test_noop_still_validates_file_path_and_exact_match(tmp_path, path, content, old_text, code):
    if content is not None:
        (tmp_path / path).write_text(content)
    with pytest.raises(ToolError) as error:
        edit_file(Workspace(tmp_path), path, old_text, old_text)
    assert error.value.code == code


def test_registry_reports_noop_as_error_without_change_evidence(tmp_path):
    (tmp_path / "example.py").write_text("value = 1\n")
    registry = ToolRegistry(Workspace(tmp_path))
    result = registry.dispatch("edit_file", {
        "path": "example.py", "old_text": "value = 1", "new_text": "value = 1",
    })
    assert result["success"] is False
    assert result["error"]["code"] == "no_change"
    assert "replacements" not in result and "modified" not in result


def edit(old_text, new_text):
    return {"role": "assistant", "content": "", "tool_calls": [{
        "function": {"name": "edit_file", "arguments": {
            "path": "calc.py", "old_text": old_text, "new_text": new_text,
        }},
    }]}


class ScriptedModel:
    context_size = 8192
    max_output_tokens = 512

    def __init__(self, *messages):
        self.messages = iter(messages)
        self.requests = []

    def chat(self, **options):
        self.requests.append(deepcopy(options))
        return ModelResponse(next(self.messages), 0, 0, None, None, None, 0,
                             prompt_tokens_reported=False, completion_tokens_reported=False)


def coding_agent(tmp_path, monkeypatch, *messages, max_steps=6):
    def reject_live_model(*args, **kwargs):
        pytest.fail("No-op regression tests must not contact Ollama")

    monkeypatch.setattr("patchrook_agent.models.ollama.httpx.Client", reject_live_model)
    root = tmp_path / "repo"
    root.mkdir()
    (root / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (root / "test_calc.py").write_text("from calc import add\ndef test_add():\n    assert add(3, 2) == 5\n")
    config = replace(HarnessConfig.preset("C"), workflow_mode="coding", action_protocol="native")
    return Agent(ScriptedModel(*messages),
                 registry=ToolRegistry(Workspace(root, ToolConfig(allow_tests=True))),
                 config=config, runs_dir=tmp_path / "runs", max_steps=max_steps)


def observations(state, name):
    return [json.loads(message["content"]) for message in state.messages
            if message.get("tool_name") == name]


def test_noop_does_not_trigger_tests_or_count_as_requested_change(tmp_path, monkeypatch):
    agent = coding_agent(tmp_path, monkeypatch, edit("a - b", "a - b"), max_steps=1)
    state = agent.run("Fix calc.py addition", requirements=TaskRequirements.fix())
    assert observations(state, "edit_file")[0]["error"]["code"] == "no_change"
    assert not observations(state, "run_tests")
    assert not state.modified_files and state.verification == "not_run"
    assert state.completion["evidence"]["changed"] is False


@pytest.mark.parametrize("noop_after_patch", [False, True])
def test_one_noop_allows_correction_and_preserves_passing_verification(
    tmp_path, monkeypatch, noop_after_patch,
):
    patch = edit("a - b", "a + b")
    noop_text = "a + b" if noop_after_patch else "a - b"
    noop = edit(noop_text, noop_text)
    messages = [patch, noop] if noop_after_patch else [noop, patch]
    agent = coding_agent(tmp_path, monkeypatch, *messages, {"role": "assistant", "content": "DONE"})
    state = agent.run("Fix calc.py addition", requirements=TaskRequirements.fix())
    edits = observations(state, "edit_file")
    assert len(edits) == 2
    assert sum(result.get("error", {}).get("code") == "no_change" for result in edits) == 1
    assert sum(result.get("modified", False) for result in edits) == 1
    assert state.status == "completed" and state.verification == "passed"
    assert state.modified_files == ["calc.py"] and state.completion["satisfied"]
    assert len(observations(state, "run_tests")) == 1
