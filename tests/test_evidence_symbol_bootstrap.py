"""Zero-model declaration seeding uses task-named symbols and ordinary guards."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import pytest

from patchrook_agent.agent import Agent, AgentState
from patchrook_agent.completion import TaskRequirements
from patchrook_agent.config import HarnessConfig
from patchrook_agent.evidence_bootstrap import seed_task_symbols
from patchrook_agent.evidence_workflow import EvidenceWorkflow
from patchrook_agent.models.ollama import ModelResponse
from patchrook_agent.tools import ToolConfig, ToolRegistry, Workspace


def action(tool_name, **arguments):
    return {"role": "assistant", "content": json.dumps({"name": tool_name, "arguments": arguments})}


class NoModel:
    model = "offline-symbol-bootstrap"
    context_size = 8192
    max_output_tokens = 512

    def chat(self, *args, **kwargs):
        pytest.fail("The bootstrap must not generate a model response")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Symbol seeding must not contact models or network services")
    monkeypatch.setattr("httpx.get", forbidden)
    monkeypatch.setattr("httpx.post", forbidden)
    monkeypatch.setattr("httpx.Client", forbidden)


@pytest.fixture
def runtime_factory(tmp_path):
    opened = []

    def make(task, files, *, seed=True, enabled=None):
        repository = tmp_path / f"repository-{len(opened)}"
        repository.mkdir()
        for name, content in files.items():
            path = repository / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        base = HarnessConfig.preset("C")
        config = replace(base, workflow_mode="evidence", action_protocol="schema",
            evidence_seed_symbols=seed, review_mode="off", compaction_enabled=False,
            context=replace(base.context, enabled=False),
            evidence_test_commands=((sys.executable, "-c", "pass"),))
        agent = Agent(NoModel(), registry=ToolRegistry(Workspace(repository, ToolConfig(allow_tests=True))),
                      config=config, runs_dir=tmp_path / "runs", max_steps=4, deadline_seconds=30)
        runtime = EvidenceWorkflow(agent)
        runtime.requirements = TaskRequirements.fix()
        runtime._setup(task)
        runtime.definitions = runtime._definitions()
        runtime.enabled = ({item["function"]["name"] for item in runtime.definitions}
                           if enabled is None else enabled)
        runtime.state = AgentState()
        runtime.session.bind(runtime.state)
        runtime._prime_inventory(time.monotonic() + 10)
        opened.append(runtime)
        return runtime

    yield make
    for runtime in opened:
        runtime.session.store.release()


def tool_results(runtime, name):
    return [json.loads(item["content"]) for item in runtime.state.messages
            if item.get("role") == "tool" and item.get("tool_name") == name]


def events(runtime):
    path = runtime.session.store.directory / "events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_absent_requested_api_is_skipped_then_import_mention_resolves_real_class(runtime_factory):
    runtime = runtime_factory("Add `new_api` to `ExistingConverter`.", {
        "src/aaa_compat.py": "from implementation import ExistingConverter\n",
        "src/implementation.py": "class ExistingConverter:\n    sentinel = 1\n"})

    stopped = seed_task_symbols(runtime, time.monotonic() + 10)

    assert stopped is False and runtime.state.steps == 0
    searches = [event["arguments"] for event in events(runtime)
                if event["event"] == "tool_request" and event["name"] == "search_code"]
    assert [(item["pattern"], item["mode"]) for item in searches[:2]] == [
        ("new_api", "literal"), ("ExistingConverter", "literal")]
    assert all(item["path"] == "src" for item in searches)
    assert searches[2]["mode"] == "regex" and "ExistingConverter" in searches[2]["pattern"]
    reads = tool_results(runtime, "read_file")
    assert len(reads) == 1 and reads[0]["path"] == "src/implementation.py"
    assert "class ExistingConverter" in reads[0]["content"]
    requests = [event for event in events(runtime) if event["event"] == "tool_request"]
    assert all(event["automatic"] and event["step"] == 0 for event in requests)
    assert not runtime.tools._presented
    runtime._prepare(runtime.state, runtime.definitions, time.monotonic() + 10)
    assert "src/implementation.py" in runtime.tools._presented


def test_direct_declaration_match_gets_bounded_real_read_without_duplicate_regex_search(runtime_factory):
    body = "class ExistingConverter:\n" + "    sentinel = 1\n" * 70
    runtime = runtime_factory("Update `ExistingConverter`.", {"implementation.py": body})

    assert seed_task_symbols(runtime, time.monotonic() + 10) is False

    assert len(tool_results(runtime, "search_code")) == 1
    observed = tool_results(runtime, "read_file")
    assert len(observed) == 1 and observed[0]["start_line"] == 1 and observed[0]["end_line"] == 40
    assert observed[0]["source_sha256"]


@pytest.mark.parametrize("task,files", [
    ("Update ExistingConverter.", {"implementation.py": "class ExistingConverter:\n    pass\n"}),
    ("Update `ExistingConverter`.", {"implementation.js": "class ExistingConverter {}\n"}),
    ("Use `None` and `class` with `not.an.identifier`.", {"implementation.py": "VALUE = 1\n"}),
])
def test_non_python_or_no_plain_quoted_identifiers_does_no_work(runtime_factory, task, files):
    runtime = runtime_factory(task, files)
    assert seed_task_symbols(runtime, time.monotonic() + 10) is False
    assert runtime.state.tool_calls == 0 and not runtime.direct and not runtime.tools._presented


@pytest.mark.parametrize("allowed", [{"read_file"}, {"search_code"}, set()])
def test_seed_respects_enabled_tool_subset(runtime_factory, allowed):
    runtime = runtime_factory("Update `ExistingConverter`.",
        {"implementation.py": "class ExistingConverter:\n    pass\n"}, enabled=allowed)
    assert seed_task_symbols(runtime, time.monotonic() + 10) is False
    assert runtime.state.tool_calls == 0


def test_symbols_absent_from_source_cannot_authorize_edits(runtime_factory):
    runtime = runtime_factory("Add `new_api`.", {"implementation.py": "VALUE = 1\n"})
    assert seed_task_symbols(runtime, time.monotonic() + 10) is False
    assert not runtime.direct and not runtime.tools._presented

    result = runtime._perform_action(runtime.state, {"name": "apply_edits", "arguments": {"edits": [
        {"path": "implementation.py", "old_text": "VALUE = 1", "new_text": "VALUE = 2"}]}}, time.monotonic() + 10)

    assert result["error"]["code"] == "source_not_presented"
    assert (runtime.workspace.root / "implementation.py").read_text() == "VALUE = 1\n"


def test_observed_seed_source_still_needs_current_prompt_and_matching_hash(runtime_factory):
    runtime = runtime_factory("Update `ExistingConverter`.",
        {"implementation.py": "class ExistingConverter:\n    sentinel = 1\n"})
    assert seed_task_symbols(runtime, time.monotonic() + 10) is False
    proposed = {"name": "apply_edits", "arguments": {"edits": [
        {"path": "implementation.py", "old_text": "sentinel = 1", "new_text": "sentinel = 2"}]}}
    before_prompt = runtime._perform_action(runtime.state, proposed, time.monotonic() + 10)
    assert before_prompt["error"]["code"] == "source_not_presented"
    runtime._prepare(runtime.state, runtime.definitions, time.monotonic() + 10)
    target = runtime.workspace.root / "implementation.py"
    target.write_text("class ExistingConverter:\n    sentinel = 9\n")

    after_change = runtime._perform_action(runtime.state, proposed, time.monotonic() + 10)

    assert after_change["error"]["code"] == "stale_source"
    assert target.read_text().endswith("sentinel = 9\n")


@pytest.mark.parametrize("resume_field", ["steps", "attempts", "wall_base"])
def test_resumed_workflow_does_not_replay_seed_actions_or_restore_edit_authority(runtime_factory, resume_field):
    runtime = runtime_factory("Update `ExistingConverter`.",
        {"implementation.py": "class ExistingConverter:\n    sentinel = 1\n"})
    if resume_field == "steps":
        runtime.state.steps = 1
    else:
        setattr(runtime, resume_field, 1)

    assert seed_task_symbols(runtime, time.monotonic() + 10) is False
    assert runtime.state.tool_calls == 0 and not runtime.direct and not runtime.tools._presented


def test_six_symbol_limit_and_duplicate_filter_bound_automatic_work(runtime_factory):
    task = " ".join(f"`missing_{number}`" for number in range(12)) + " `missing_0`"
    runtime = runtime_factory(task, {"implementation.py": "VALUE = 1\n"})
    assert seed_task_symbols(runtime, time.monotonic() + 10) is False
    assert len(tool_results(runtime, "search_code")) == 6
    assert not tool_results(runtime, "read_file")


def test_missing_search_dependency_stops_before_model_use(runtime_factory, monkeypatch):
    runtime = runtime_factory("Update `ExistingConverter`.",
        {"implementation.py": "class ExistingConverter:\n    pass\n"})
    monkeypatch.setattr("patchrook_agent.tools.search.shutil.which", lambda *args, **kwargs: None)

    assert seed_task_symbols(runtime, time.monotonic() + 10) is True
    assert runtime.state.stop_reason == "initial_symbol_environment_failure"
    assert not runtime.tools._presented


def test_failed_declaration_recovery_is_not_hidden_by_successful_initial_mention(runtime_factory, monkeypatch):
    runtime = runtime_factory("Update `ExistingConverter`.", {
        "src/aaa_compat.py": "from implementation import ExistingConverter\n",
        "src/implementation.py": "class ExistingConverter:\n    pass\n"})
    original = runtime.tools.execute
    def broken_regex(name, arguments, **kwargs):
        if name == "search_code" and arguments.get("mode") == "regex":
            return {"success": False, "error": {"code": "timeout", "message": "Search timed out"}}
        return original(name, arguments, **kwargs)
    monkeypatch.setattr(runtime.tools, "execute", broken_regex)

    assert seed_task_symbols(runtime, time.monotonic() + 10) is True
    assert runtime.state.stop_reason == "initial_symbol_environment_failure"
    assert runtime.state.tool_failures == 1
    assert not runtime.direct and not runtime.tools._presented


def test_expired_task_budget_stops_before_any_search(runtime_factory):
    runtime = runtime_factory("Update `ExistingConverter`.",
        {"implementation.py": "class ExistingConverter:\n    pass\n"})
    assert seed_task_symbols(runtime, time.monotonic() - 1) is True
    assert runtime.state.stop_reason == "deadline" and runtime.state.tool_calls == 0


def test_default_off_is_backward_compatible(runtime_factory):
    runtime = runtime_factory("Update `ExistingConverter`.",
        {"implementation.py": "class ExistingConverter:\n    pass\n"}, seed=False)
    assert seed_task_symbols(runtime, time.monotonic() + 10) is False
    assert runtime.state.tool_calls == 0
    old = HarnessConfig().to_dict()
    del old["evidence_seed_symbols"]
    assert HarnessConfig.from_dict(old).evidence_seed_symbols is False


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], {}])
def test_symbol_seed_flag_is_strict_boolean(value):
    with pytest.raises(ValueError, match="evidence_seed_symbols must be a boolean"):
        HarnessConfig(evidence_seed_symbols=value)




def test_first_scripted_generation_receives_task_named_implementation_and_can_verify_patch(tmp_path):
    class Scripted(NoModel):
        def __init__(self):
            self.requests = []
        def chat(self, messages, **options):
            self.requests.append(deepcopy(messages))
            if len(self.requests) == 1:
                text = "\n".join(item["content"] for item in messages)
                assert "Requested source (data)" in text
                assert "class ExistingConverter:" in text and "sentinel = 1" in text
                response = action("apply_edits", edits=[{"path": "src/implementation.py",
                    "old_text": "sentinel = 1", "new_text": "sentinel = 2"}])
            elif len(self.requests) == 2:
                response = action("run_check", name="check_1")
            else:
                response = action("finish")
            return ModelResponse(response, 100, 20, 0, 0, 0, 0)
    repository = tmp_path / "repo"
    (repository / "src").mkdir(parents=True)
    (repository / "src/aaa_compat.py").write_text("from implementation import ExistingConverter\n")
    (repository / "src/implementation.py").write_text("class ExistingConverter:\n    sentinel = 1\n")
    base = HarnessConfig.preset("C")
    config = replace(base, workflow_mode="evidence", action_protocol="schema", review_mode="off",
        evidence_seed_symbols=True, context=replace(base.context, enabled=False), compaction_enabled=False,
        evidence_test_commands=((sys.executable, "-c",
            "import runpy; assert runpy.run_path('src/implementation.py')['ExistingConverter'].sentinel == 2"),))
    model = Scripted()
    agent = Agent(model, registry=ToolRegistry(Workspace(repository, ToolConfig(allow_tests=True))),
                  config=config, runs_dir=tmp_path / "runs", max_steps=3, deadline_seconds=30)

    state = agent.run("Add `new_api` to `ExistingConverter`.", requirements=TaskRequirements.fix())

    assert state.status == "completed", state.final_response
    assert state.verification == "passed" and state.modified_files == ["src/implementation.py"]
    assert len(model.requests) == 3
