"""Batch reads preserve observed facts, atomic context, and resume boundaries."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from patchrook_agent.agent import Agent, LEGACY_SYSTEM_PROMPT, PRE_BATCH_GUIDANCE_SYSTEM_PROMPT
from patchrook_agent.config import HarnessConfig
from patchrook_agent.context import ContextConfig, ContextManager
from patchrook_agent.context.compaction import SUMMARY_PREFIX
from patchrook_agent.memory.state import WorkingMemory
from patchrook_agent.models.ollama import ModelResponse, OllamaClient
from patchrook_agent.tools import ToolConfig, ToolRegistry, Workspace


@pytest.fixture(autouse=True)
def prohibit_live_http(monkeypatch):
    def unexpected_http(*args, **kwargs):
        raise AssertionError("Batch integration tests must not contact a live model")

    monkeypatch.setattr("patchrook_agent.models.ollama.httpx.Client", unexpected_http)


def call(name, **arguments):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": f"call-{name}", "function": {"name": name, "arguments": arguments}}]}


def final():
    return {"role": "assistant", "content": "Reported the observed file and test results."}


def exchange(name, arguments, result):
    return [call(name, **arguments), {"role": "tool", "tool_name": name,
                                    "tool_call_id": f"call-{name}", "content": json.dumps(result)}]


class ScriptedModel:
    model = "batch-unit-fixture"
    context_size = 8192
    max_output_tokens = 512

    def __init__(self, *responses):
        self.responses = iter(responses)
        self.requests = []

    def chat(self, **options):
        self.requests.append(deepcopy(options))
        return ModelResponse(next(self.responses), 11, 7, 0, 0, 0, 10_000_000)


def results(state, name):
    return [json.loads(message["content"]) for message in state.messages
            if message.get("role") == "tool" and message.get("tool_name") == name]


@pytest.fixture
def repository(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("VALUE = 1\n")
    (repo / "b.py").write_text("VALUE = 2\n")
    (repo / "test_values.py").write_text(
        "import a, b\n\ndef test_values():\n    assert a.VALUE == 11\n    assert b.VALUE == 22\n")
    return repo, tmp_path / "runs"


def make_agent(repository, model, **options):
    repo, runs = repository
    return Agent(model, registry=ToolRegistry(Workspace(repo, ToolConfig(allow_tests=True))),
                 runs_dir=runs, config=HarnessConfig.preset("C"), **options)


def test_partial_batch_updates_each_observed_file_and_exact_failure_memory(repository):
    model = ScriptedModel(call("read_files", files=[{"path": "a.py"}, {"path": "b.py"},
                                                   {"path": "missing.py"}]), final())
    agent = make_agent(repository, model)
    state = agent.run("Inspect both modules and report any unavailable selection.")
    batch = results(state, "read_files")[0]
    assert batch["success"] is False
    assert [item["success"] for item in batch["files"]] == [True, True, False]
    assert {"a.py", "b.py"} <= set(state.memory["relevant_files"])
    failure = state.memory["unresolved_errors"][0]
    assert failure["tool"] == "read_file" and failure["path"] == "missing.py"
    exact_error = batch["files"][-1]["error"]["message"]
    assert failure["error"]["message"] == exact_error
    assert json.dumps(model.requests[-1]["messages"]).count(exact_error) == 1
    assert state.tool_calls == 1 and state.tool_failures == 1
    assert {"a.py", "b.py"} <= set(agent._session.snapshots)


def test_batch_snapshots_returned_paths_even_if_prior_listing_recorded_the_inventory(repository):
    model = ScriptedModel(call("list_files", path="."),
                          call("read_files", files=[{"path": "a.py"}, {"path": "b.py"}]))
    state = make_agent(repository, model, pause_after_step=2).run("Inspect the two modules.")
    checkpoint = json.loads((Path(state.session_dir) / "checkpoint.json").read_text())
    assert state.stop_reason == "paused"
    assert {"a.py", "b.py"} <= set(checkpoint["file_snapshots"])
    assert all(checkpoint["file_snapshots"][path]["kind"] == "file" for path in ("a.py", "b.py"))


def test_resume_partial_batch_clears_only_successfully_reread_paths(repository):
    paused = make_agent(repository, ScriptedModel(call(
        "read_files", files=[{"path": "a.py"}, {"path": "b.py"}])), pause_after_step=1).run(
            "Update both values after inspecting their current source, then run test_values.py.")
    repo, _ = repository
    (repo / "a.py").write_text("VALUE = 10\n")
    (repo / "b.py").write_text("VALUE = 20\n")
    model = ScriptedModel(
        call("read_files", files=[{"path": "./a.py"}, {"path": "missing.py"}]),
        call("edit_file", path="a.py", old_text="VALUE = 10", new_text="VALUE = 11"),
        call("edit_file", path="b.py", old_text="VALUE = 20", new_text="VALUE = 22"),
        call("read_files", files=[{"path": "./b.py"}]),
        call("edit_file", path="b.py", old_text="VALUE = 20", new_text="VALUE = 22"),
        call("run_tests", path="test_values.py"), final(),
    )
    agent = make_agent(repository, model)
    state = agent.resume(paused.session_id, allow_tests=True)
    edits = results(state, "edit_file")
    assert edits[0]["success"] is True
    assert edits[1]["success"] is False and edits[1]["error"]["code"] == "workspace_changed"
    assert edits[2]["success"] is True
    assert state.status == "completed" and state.verification == "passed"
    assert agent._revalidate_paths == set()
    assert not any(note.startswith("Relevant files changed since checkpoint:") for note in state.memory["blockers"])


def test_batch_memory_errors_resolve_per_file_without_erasing_other_failures():
    memory = WorkingMemory(session_id="batch-unit", objective="Inspect the selected files.")
    memory.observe_tool("read_files", {}, {"success": False, "files": [
        {"success": True, "path": "a.py", "content": "1: VALUE = 1"},
        {"success": False, "path": "b.py", "error": {"code": "not_found", "message": "b missing"}},
        {"success": False, "path": "c.py", "error": {"code": "not_found", "message": "c missing"}},
    ]}, 1)
    memory.observe_tool("read_files", {}, {"success": True, "files": [
        {"success": True, "path": "b.py", "content": "1: VALUE = 2"},
    ]}, 2)
    assert set(memory.relevant_files) == {"a.py", "b.py"}
    assert [item["path"] for item in memory.unresolved_errors] == ["c.py"]
    assert memory.unresolved_errors[0]["error"]["message"] == "c missing"


def test_batch_sources_are_protected_as_one_atomic_exchange_during_compaction():
    messages = [{"role": "system", "content": "Inspect before editing."},
                {"role": "user", "content": "Trace api.py, parser.py and pricing.py."}]
    for index in range(6):
        messages += exchange("list_files", {"path": f"old{index}"}, {
            "success": True, "entries": [f"historical_{number}.py" for number in range(100)]})
    batch = exchange("read_files", {"files": [{"path": path} for path in ("api.py", "parser.py", "pricing.py")]}, {
        "success": True, "files": [{"success": True, "path": path, "content": f"1: {path}_source = 1"}
                                    for path in ("api.py", "parser.py", "pricing.py")]})
    messages += batch
    for index in range(3):
        messages += exchange("list_files", {"path": f"new{index}"}, {"success": True, "entries": ["test_api.py"]})
    raw = deepcopy(messages)
    result = ContextManager(ContextConfig(context_size=8192, trigger=0.4)).prepare(messages, [])
    assert result.compacted and result.safe
    assert all(message in result.messages for message in batch)
    assert result.messages[result.messages.index(batch[0]) + 1] == batch[1]
    assert messages == raw


def test_unresolved_batch_error_is_not_superseded_by_an_unrelated_successful_batch():
    messages = [{"role": "system", "content": "Inspect first."}, {"role": "user", "content": "Investigate failures."}]
    failure = exchange("read_files", {"files": [{"path": "missing.py"}]}, {
        "success": False, "files": [{"success": False, "path": "missing.py", "error": {
            "code": "not_found", "message": "Exact missing.py diagnostic\n" * 100}}]})
    messages += failure
    for index in range(6):
        messages += exchange("read_files", {"files": [{"path": f"old{index}.py"}]}, {
            "success": True, "files": [{"success": True, "path": f"old{index}.py",
                                        "content": "old_source = 1\n" * 100}]})
    result = ContextManager(ContextConfig(context_size=8192, trigger=0.4)).prepare(messages, [])
    assert result.compacted and result.safe
    assert all(message in result.messages for message in failure)
    summaries = [message for message in result.messages if message.get("content", "").startswith(SUMMARY_PREFIX)]
    assert any('"files"' in message["content"] and '"content_omitted"' in message["content"] for message in summaries)


def test_omitted_batch_results_remain_in_a_retrievable_session_artifact(repository):
    repo, runs = repository
    for name in ("a.py", "b.py"):
        (repo / name).write_text("VALUE = 1\n" * 100 + f"# COMPLETE_CAPTURE_{name}\n")

    class ArtifactModel(ScriptedModel):
        def chat(self, **options):
            self.requests.append(deepcopy(options))
            if len(self.requests) == 1:
                message = call("read_files", files=[{"path": "a.py"}, {"path": "b.py"}])
            elif len(self.requests) == 2:
                observed = json.loads(options["messages"][-2]["content"])
                message = call("read_artifact", artifact_id=observed["artifact_id"], max_chars=300)
            else:
                message = final()
            return ModelResponse(message, 11, 7, 0, 0, 0, 10_000_000)

    model = ArtifactModel()
    registry = ToolRegistry(Workspace(repo, ToolConfig(max_output_chars=700)))
    state = Agent(model, registry=registry, config=HarnessConfig.preset("C"), runs_dir=runs).run("Inspect selected files.")
    batch = results(state, "read_files")[0]
    assert state.status == "completed" and batch["truncated"]
    assert results(state, "read_artifact")[0]["success"] is True
    artifact = json.loads((Path(state.session_dir) / "artifacts" / f'{batch["artifact_id"]}.json').read_text())
    captures = artifact["available_output"]["files"]
    assert [item["path"] for item in captures] == ["a.py", "b.py"]
    assert all(f'COMPLETE_CAPTURE_{item["path"]}' in item["available_output"]["content"] for item in captures)


@pytest.mark.parametrize("trusted", [True, False])
def test_legacy_14b_16k_checkpoint_keeps_original_prompt_and_six_tool_subset(repository, trusted):
    class ScriptedOllama(OllamaClient, ScriptedModel):
        chat = ScriptedModel.chat

        def __init__(self, *responses):
            OllamaClient.__init__(self, model="qwen3:14b", context_size=16384)
            ScriptedModel.__init__(self, *responses)

    first_model = ScriptedOllama(call("read_file", path="a.py"))
    assert first_model.chat.__func__ is ScriptedModel.chat
    first = make_agent(repository, first_model, pause_after_step=1)
    paused = first.run("Inspect a.py without modifying it.")
    checkpoint_path = Path(paused.session_dir) / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    original_prompt = LEGACY_SYSTEM_PROMPT if trusted else LEGACY_SYSTEM_PROMPT + "Untrusted extra instruction."
    checkpoint["state"]["messages"][0] = {"role": "system", "content": original_prompt}
    checkpoint["state"]["active_messages"][0] = {"role": "system", "content": original_prompt}
    old_tools = {"list_files", "read_file", "search_code", "write_file", "edit_file", "run_tests"}
    checkpoint["execution"]["enabled_tools"] = sorted(old_tools)
    checkpoint_path.write_text(json.dumps(checkpoint))
    model = ScriptedOllama(final())
    resumed = make_agent(repository, model)
    if not trusted:
        with pytest.raises(ValueError, match="Invalid agent checkpoint"):
            resumed.resume(paused.session_id)
        assert model.requests == []
        return
    state = resumed.resume(paused.session_id)
    assert state.status == "completed"
    assert state.messages[0] == state.active_messages[0] == {"role": "system", "content": LEGACY_SYSTEM_PROMPT}
    assert model.requests[0]["messages"][0]["content"] == LEGACY_SYSTEM_PROMPT
    assert {item["function"]["name"] for item in model.requests[0]["tools"]} == old_tools
    assert model.context_size == state.metrics["context_window_tokens"] == 16384
    assert resumed._model_settings()["model"] == "qwen3:14b"


def test_previous_8b_checkpoint_keeps_its_original_prompt_and_enabled_tools(repository):
    paused = make_agent(repository, ScriptedModel(call("read_file", path="a.py")),
                        pause_after_step=1).run("Inspect a.py without modifying it.")
    checkpoint_path = Path(paused.session_dir) / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    previous = {"role": "system", "content": PRE_BATCH_GUIDANCE_SYSTEM_PROMPT}
    checkpoint["state"]["messages"][0] = previous
    checkpoint["state"]["active_messages"][0] = previous
    enabled = set(checkpoint["execution"]["enabled_tools"])
    checkpoint_path.write_text(json.dumps(checkpoint))
    model = ScriptedModel(final())
    state = make_agent(repository, model).resume(paused.session_id)
    assert state.status == "completed"
    assert state.messages[0] == state.active_messages[0] == previous
    assert model.requests[0]["messages"][0] == previous
    assert {tool["function"]["name"] for tool in model.requests[0]["tools"]} == enabled
    assert state.metrics["context_window_tokens"] == 8192
