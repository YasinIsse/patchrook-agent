"""Coding turns advertise only usable, observed session artifacts."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from patchrook_agent.agent import Agent
from patchrook_agent.completion import TaskRequirements
from patchrook_agent.config import HarnessConfig
from patchrook_agent.models.ollama import ModelResponse
from patchrook_agent.observability.artifacts import ArtifactStore
from patchrook_agent.tools import ToolConfig, ToolRegistry, Workspace


def call(name, **arguments):
    return {"type": "function", "function": {"name": name, "arguments": arguments}}


def action(*calls):
    return {"role": "assistant", "content": "", "tool_calls": list(calls)}


def final():
    return {"role": "assistant", "content": "DONE"}


class ScriptedModel:
    model = "coding-artifact-mechanics"
    context_size = 8192
    max_output_tokens = 512

    def __init__(self, *messages):
        self.messages = iter(messages)
        self.requests = []

    def chat(self, **options):
        self.requests.append(deepcopy(options))
        message = next(self.messages)
        if callable(message):
            message = message(options)
        return ModelResponse(message, 0, 0, None, None, None, 0,
                             prompt_tokens_reported=False, completion_tokens_reported=False)


@pytest.fixture(autouse=True)
def no_live_model(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("Artifact mechanics tests must not contact Ollama")
    monkeypatch.setattr("patchrook_agent.models.ollama.httpx.Client", fail)


def repository(tmp_path, *, large=False):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "small.py").write_text("def answer():\n    return 42\n")
    if large:
        (root / "large.py").write_text("payload = '" + "x" * 6000 + "'\n")
    return root


def make_agent(tmp_path, model, *, max_steps=5, **options):
    config = replace(HarnessConfig.preset("C"), workflow_mode="coding", action_protocol="native")
    registry = ToolRegistry(Workspace(tmp_path / "repo", ToolConfig(allow_tests=True)))
    return Agent(model=model, registry=registry, config=config, runs_dir=tmp_path / "runs",
                 max_steps=max_steps, **options)


def artifact_definition(request):
    return next((item["function"] for item in request["tools"]
                 if item["function"]["name"] == "read_artifact"), None)


def advertised_ids(request):
    definition = artifact_definition(request)
    return definition["parameters"]["properties"]["artifact_id"]["enum"] if definition else []


def observations(state, name):
    return [json.loads(item["content"]) for item in state.messages if item.get("tool_name") == name]


def source_artifact_ids(state):
    return {result["artifact_id"] for name in ("read_code", "read_file")
            for result in observations(state, name) if result.get("truncated") and result.get("artifact_id")}


def artifact_path(state, artifact_id):
    return Path(state.session_dir) / "artifacts" / f"{artifact_id}.json"


def test_small_coding_patch_never_advertises_artifact_reader(tmp_path):
    root = repository(tmp_path)
    (root / "small.py").write_text("def answer():\n    return 41\n")
    (root / "test_small.py").write_text("from small import answer\ndef test_answer():\n    assert answer() == 42\n")
    model = ScriptedModel(action(call("edit_file", path="small.py", old_text="return 41", new_text="return 42")),
                          final())
    state = make_agent(tmp_path, model).run("Fix small.py answer", requirements=TaskRequirements.fix())
    assert state.status == "completed" and state.verification == "passed"
    assert len(model.requests) == 2
    assert not source_artifact_ids(state)
    assert all(artifact_definition(request) is None for request in model.requests)


def test_truncated_source_exposes_observed_id_and_bounded_schema_then_reads_exact_page(tmp_path):
    repository(tmp_path, large=True)
    selected = []

    def read_omitted(request):
        definition = artifact_definition(request)
        assert definition is not None
        properties = definition["parameters"]["properties"]
        assert properties["offset"]["minimum"] == 0
        assert (properties["max_chars"].get("minimum") == 1
                or properties["max_chars"].get("exclusiveMinimum") == 0)
        assert properties["max_chars"]["maximum"] == 2000
        selected.extend(advertised_ids(request))
        assert selected and "memory" not in selected and "latest_test" not in selected
        return action(call("read_artifact", artifact_id=selected[0], offset=0, max_chars=2000))

    model = ScriptedModel(read_omitted, final())
    agent = make_agent(tmp_path, model)
    state = agent.run("Inspect small.py and large.py and retrieve omitted source evidence.",
                      requirements=TaskRequirements.inspect())
    assert state.status == "completed", (state.stop_reason, state.final_response)
    assert set(selected) == source_artifact_ids(state)
    page, = observations(state, "read_artifact")
    assert page["success"] and page["artifact_id"] == selected[0]
    original = artifact_path(state, selected[0]).read_text()
    assert page["content"] == original[:len(page["content"])]
    assert 0 < len(page["content"]) <= 2000 and page["truncated"]
    assert page["next_offset"] == len(page["content"])
    assert len(json.dumps(page, ensure_ascii=False, separators=(",", ":"))) <= agent.registry.workspace.config.max_output_chars


def test_artifact_reader_appears_after_a_read_and_disappears_after_artifact_deletion(tmp_path):
    repository(tmp_path, large=True)
    agent = None

    def request_missing_source(request):
        assert artifact_definition(request) is None
        return action(call("read_file", path="large.py"))

    def remove_observed_artifacts(request):
        ids = advertised_ids(request)
        assert ids
        for artifact_id in ids:
            (agent._session.artifacts.artifacts / f"{artifact_id}.json").unlink()
        return action(call("read_file", path="small.py"))

    def finish_without_artifact_reader(request):
        assert artifact_definition(request) is None
        return final()

    model = ScriptedModel(request_missing_source, remove_observed_artifacts, finish_without_artifact_reader)
    agent = make_agent(tmp_path, model)
    state = agent.run("Inspect small.py.", requirements=TaskRequirements.inspect())
    assert state.status == "completed", (state.stop_reason, state.final_response)
    assert len(model.requests) == 3 and source_artifact_ids(state)


@pytest.mark.parametrize("invented_id", ["latest_test", "memory", "tool-" + "f" * 24])
def test_invented_artifact_id_is_rejected_before_dispatch_when_reader_is_available(tmp_path, monkeypatch, invented_id):
    repository(tmp_path, large=True)
    model = ScriptedModel(action(call("read_artifact", artifact_id=invented_id, max_chars=2000)))
    agent = make_agent(tmp_path, model, max_steps=1)
    dispatched = []
    original = agent.registry.dispatch

    def record(name, arguments, **options):
        dispatched.append(name)
        return original(name, arguments, **options)

    monkeypatch.setattr(agent.registry, "dispatch", record)
    state = agent.run("Inspect small.py and large.py.", requirements=TaskRequirements.inspect())
    assert advertised_ids(model.requests[0])
    assert invented_id not in advertised_ids(model.requests[0])
    assert "read_artifact" not in dispatched
    result, = observations(state, "read_artifact")
    assert not result["success"] and result["error"]["code"] == "invalid_artifact"


def test_unadvertised_artifact_tool_cannot_dispatch_an_invented_alias(tmp_path, monkeypatch):
    repository(tmp_path)
    model = ScriptedModel(action(call("read_artifact", artifact_id="latest_test")))
    agent = make_agent(tmp_path, model, max_steps=1)
    original = agent.registry.dispatch

    def reject_artifact_dispatch(name, arguments, **options):
        assert name != "read_artifact"
        return original(name, arguments, **options)

    monkeypatch.setattr(agent.registry, "dispatch", reject_artifact_dispatch)
    state = agent.run("Inspect small.py.", requirements=TaskRequirements.inspect())
    assert artifact_definition(model.requests[0]) is None
    result, = observations(state, "read_artifact")
    assert not result["success"] and result["error"]["code"] == "unknown_tool"


@pytest.mark.parametrize("unobserved_kind", ["same_session", "other_session"])
def test_resume_restores_only_observed_artifact_ids_in_the_same_session(tmp_path, monkeypatch, unobserved_kind):
    repository(tmp_path, large=True)
    paused = make_agent(tmp_path, ScriptedModel(action(call("read_file", path="small.py"))),
                        pause_after_step=1).run("Inspect small.py and large.py.", requirements=TaskRequirements.inspect())
    assert paused.stop_reason == "paused"
    observed = source_artifact_ids(paused)
    assert observed
    same_store = ArtifactStore(Path(paused.session_dir))
    hidden_id = same_store.save({"content": "Saved without returning an observation"})
    foreign_directory = tmp_path / "foreign-session"
    foreign_directory.mkdir()
    foreign_id = ArtifactStore(foreign_directory).save({"content": "Evidence from another session"})
    unobserved_id = hidden_id if unobserved_kind == "same_session" else foreign_id

    def request_valid_and_invalid_pages(request):
        assert set(advertised_ids(request)) == observed
        assert hidden_id not in advertised_ids(request) and foreign_id not in advertised_ids(request)
        return action(call("read_artifact", artifact_id=next(iter(observed)), max_chars=100),
                      call("read_artifact", artifact_id=unobserved_id, max_chars=100))

    model = ScriptedModel(request_valid_and_invalid_pages, final())
    dispatched = []
    original = ToolRegistry.dispatch

    def record(registry, name, arguments, **options):
        if name == "read_artifact":
            dispatched.append(arguments["artifact_id"])
        return original(registry, name, arguments, **options)

    monkeypatch.setattr(ToolRegistry, "dispatch", record)
    state = Agent(model=model, workspace=tmp_path / "repo", runs_dir=tmp_path / "runs").resume(paused.session_id)
    assert state.status == "completed", (state.stop_reason, state.final_response)
    assert state.session_id == paused.session_id and len(dispatched) == 1
    assert dispatched[0] in observed
    pages = observations(state, "read_artifact")
    assert pages[0]["success"]
    assert [result["error"]["code"] for result in pages[1:]] == ["invalid_artifact"]


@pytest.mark.parametrize("invalid_arguments", [{"max_chars": 2001}, {"max_chars": 0}, {"offset": -1}])
def test_artifact_runtime_rejects_out_of_bounds_arguments_before_store_read(tmp_path, monkeypatch, invalid_arguments):
    repository(tmp_path, large=True)
    reads = []
    original = ArtifactStore.read

    def record(store, *args, **kwargs):
        reads.append((args, kwargs))
        return original(store, *args, **kwargs)

    def request_invalid_page(request):
        return action(call("read_artifact", artifact_id=advertised_ids(request)[0], **invalid_arguments))

    monkeypatch.setattr(ArtifactStore, "read", record)
    model = ScriptedModel(request_invalid_page)
    state = make_agent(tmp_path, model, max_steps=1).run("Inspect small.py and large.py.",
                                                       requirements=TaskRequirements.inspect())
    result, = observations(state, "read_artifact")
    assert not result["success"] and result["error"]["code"] == "invalid_arguments"
    assert not reads
