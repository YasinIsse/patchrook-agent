"""Fixed coding workflow uses native requests, real file tools, and real pytest."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from coding_fixtures import create_fixture, verify_fixture
from patchrook_agent.agent import Agent
from patchrook_agent.completion import TaskRequirements
from patchrook_agent.config import HarnessConfig
from patchrook_agent.models.ollama import ModelResponse
from patchrook_agent.tools import ToolConfig, ToolRegistry, Workspace


def call(name, **arguments):
    return {"type": "function", "function": {"name": name, "arguments": arguments}}


def action(*calls):
    return {"role": "assistant", "content": "", "tool_calls": list(calls)}


def final():
    return {"role": "assistant", "content": "Made the requested patch and observed pytest pass."}


class ScriptedModel:
    model = "coding-mechanics"
    context_size = 8192
    max_output_tokens = 512

    def __init__(self, *messages, on_request=None):
        self.messages = iter(messages)
        self.requests = []
        self.on_request = on_request

    def chat(self, **options):
        self.requests.append(deepcopy(options))
        if self.on_request is not None:
            self.on_request(len(self.requests), options)
        return ModelResponse(next(self.messages), 0, 0, None, None, None, 0,
                             prompt_tokens_reported=False, completion_tokens_reported=False)


@pytest.fixture(autouse=True)
def no_live_model(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("Coding workflow integration tests must not contact Ollama")
    monkeypatch.setattr("patchrook_agent.models.ollama.httpx.Client", fail)


def make_agent(tmp_path, model, *, allow_tests=True, max_steps=10, **options):
    root = tmp_path / "repo"
    root.mkdir(exist_ok=True)
    config = replace(HarnessConfig.preset("C"), workflow_mode="coding", action_protocol="native")
    return Agent(model=model, registry=ToolRegistry(Workspace(root, ToolConfig(allow_tests=allow_tests))),
                 config=config, runs_dir=tmp_path / "runs", max_steps=max_steps, **options)


def calculator(tmp_path):
    root = tmp_path / "repo"
    root.mkdir(exist_ok=True)
    (root / "calc.py").write_text("# Preserve this comment.\ndef add(a, b):\n    return a - b\n")
    (root / "test_calc.py").write_text("from calc import add\ndef test_add():\n    assert add(3, 2) == 5\n")
    return root


def rendered(request):
    return "\n".join(message.get("content", "") for message in request["messages"])


def observations(state, name):
    return [json.loads(message["content"]) for message in state.messages if message.get("tool_name") == name]


def test_pipeline_source_is_gathered_before_first_request_and_patch_auto_verifies(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    task = create_fixture(root, "long-context")

    def check_request(number, request):
        assert "response_format" not in request
        names = {item["function"]["name"] for item in request["tools"]}
        assert {"edit_file", "write_file", "read_file"} <= names
        assert not names & {"run_tests", "list_files", "search_code"}
        if number == 1:
            assert "read_code" in names
            text = rendered(request)
            assert "pipeline/runner.py" in text
            for index in range(8):
                assert f"pipeline/stage_{index}.py" in text
                operator = "-" if index == 7 else "+"
                assert f"return value {operator} {index + 1}" in text
        elif number == 2:
            assert "read_code" not in names
            assert "Completion review:" in rendered(request)
            assert "passed" in rendered(request)

    model = ScriptedModel(action(call("edit_file", path="pipeline/stage_7.py",
                                      old_text="return value - 8", new_text="return value + 8")),
                          final(), on_request=check_request)
    state = make_agent(tmp_path, model).run(task.instructions, requirements=TaskRequirements.fix())
    acceptance = verify_fixture(root, task, state)
    assert acceptance["passed"], (state.stop_reason, state.final_response, acceptance)
    assert state.steps == 2 and len(model.requests) == 2
    assert state.modified_files == ["pipeline/stage_7.py"] and state.verification == "passed"
    assert len(observations(state, "edit_file")) == 1
    tests = observations(state, "run_tests")
    assert len(tests) == 1 and tests[0]["status"] == "passed" and tests[0]["exit_code"] == 0
    assert (root / "pipeline/stage_7.py").read_text() == task.files["pipeline/stage_7.py"].replace(
        "return value - 8", "return value + 8")


def test_failed_automatic_pytest_returns_to_model_for_patch_correction(tmp_path):
    root = calculator(tmp_path)

    def check_request(number, request):
        text = rendered(request)
        if number == 2:
            assert "failed" in text and "test_calc.py" in text and "assert 6 == 5" in text
            assert "return a * b" in text
        if number == 3:
            assert "passed" in text

    model = ScriptedModel(
        action(call("edit_file", path="calc.py", old_text="a - b", new_text="a * b")),
        action(call("edit_file", path="calc.py", old_text="a * b", new_text="a + b")), final(),
        on_request=check_request)
    state = make_agent(tmp_path, model).run("Fix calc.py addition", requirements=TaskRequirements.fix())
    assert state.status == "completed", (state.stop_reason, state.final_response)
    assert state.steps == 3 and state.verification == "passed"
    assert [item["status"] for item in observations(state, "run_tests")] == ["failed", "passed"]
    assert (root / "calc.py").read_text() == "# Preserve this comment.\ndef add(a, b):\n    return a + b\n"


@pytest.mark.parametrize("name,arguments", [
    ("list_files", {"path": "."}),
    ("search_code", {"path": ".", "pattern": "return"}),
    ("run_tests", {}),
])
def test_model_cannot_dispatch_harness_owned_navigation_or_tests(tmp_path, monkeypatch, name, arguments):
    calculator(tmp_path)
    model = ScriptedModel(action(call(name, **arguments)))
    agent = make_agent(tmp_path, model, max_steps=1)
    dispatched = []
    original = agent.registry.dispatch

    def record(tool_name, tool_arguments, **options):
        dispatched.append(tool_name)
        return original(tool_name, tool_arguments, **options)

    monkeypatch.setattr(agent.registry, "dispatch", record)
    state = agent.run("Fix calc.py", requirements=TaskRequirements.fix())
    assert name not in dispatched
    assert observations(state, name)[-1]["error"]["code"] == "unknown_tool"
    assert not state.modified_files and state.verification == "not_run"
    assert name not in {item["function"]["name"] for item in model.requests[0]["tools"]}


def test_read_and_edit_in_one_response_cannot_credit_source_model_has_not_seen(tmp_path):
    root = calculator(tmp_path)
    (root / "other.py").write_text("def subtract(a, b):\n    return a + b\n")
    model = ScriptedModel(action(call("read_file", path="other.py"),
                                  call("edit_file", path="other.py", old_text="a + b", new_text="a - b")))
    state = make_agent(tmp_path, model, max_steps=1).run("Fix calc.py", requirements=TaskRequirements.fix())
    assert "other.py" not in rendered(model.requests[0])
    assert observations(state, "read_file")[-1]["success"]
    assert observations(state, "edit_file")[-1]["error"]["code"] == "inspection_required"
    assert (root / "other.py").read_text() == "def subtract(a, b):\n    return a + b\n"
    assert not state.modified_files and not observations(state, "run_tests")


def test_existing_file_cannot_be_overwritten_through_write_file(tmp_path):
    root = calculator(tmp_path)
    original = (root / "calc.py").read_text()
    model = ScriptedModel(action(call("write_file", path="calc.py", content="removed = True\n", overwrite=True)))
    state = make_agent(tmp_path, model, max_steps=1).run("Fix calc.py", requirements=TaskRequirements.fix())
    assert (root / "calc.py").read_text() == original
    assert observations(state, "write_file")[-1]["error"]["code"] == "inspection_required"
    assert not state.modified_files


def test_new_file_can_be_created_after_source_gathering_and_is_automatically_tested(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "legacy.py").write_text("def original_answer():\n    return 42\n")
    (root / "test_new.py").write_text("from new import answer\ndef test_answer():\n    assert answer() == 42\n")
    model = ScriptedModel(action(call("write_file", path="new.py", content="def answer():\n    return 42\n")), final())
    state = make_agent(tmp_path, model).run("Create new.py using existing legacy.py", requirements=TaskRequirements.fix())
    assert "legacy.py" in rendered(model.requests[0])
    assert state.status == "completed" and state.verification == "passed"
    assert state.modified_files == ["new.py"]
    assert observations(state, "run_tests")[0]["exit_code"] == 0


def test_pause_resume_preserves_coding_workflow_budget_and_does_not_replay_patch(tmp_path):
    calculator(tmp_path)
    paused = make_agent(tmp_path, ScriptedModel(action(call("edit_file", path="calc.py",
                                                          old_text="a - b", new_text="a + b"))),
                        max_steps=2, pause_after_step=1).run("Fix calc.py", requirements=TaskRequirements.fix())
    assert paused.stop_reason == "paused" and paused.steps == 1 and paused.verification == "passed"
    model = ScriptedModel(final())
    agent = Agent(model=model, workspace=tmp_path / "repo", runs_dir=tmp_path / "runs", max_steps=100)
    state = agent.resume(paused.session_id, allow_tests=True)
    assert agent.config.workflow_mode == "coding" and agent.max_steps == 2
    assert state.status == "completed" and state.steps == 2
    assert len(model.requests) == 1
    assert len(observations(state, "edit_file")) == 1
    assert len(observations(state, "run_tests")) == 1


def test_resume_external_change_is_reinspected_and_retested_before_next_request(tmp_path):
    root = calculator(tmp_path)
    paused = make_agent(tmp_path, ScriptedModel(action(call("edit_file", path="calc.py",
                                                          old_text="a - b", new_text="a + b"))),
                        pause_after_step=1).run("Fix calc.py", requirements=TaskRequirements.fix())
    (root / "calc.py").write_text("# External comment.\ndef add(a, b):\n    return a * b\n")

    def check_request(number, request):
        if number == 1:
            text = rendered(request)
            assert "return a * b" in text and "return a - b" not in text
            assert "failed" in text and "assert 6 == 5" in text

    model = ScriptedModel(action(call("edit_file", path="calc.py", old_text="a * b", new_text="a + b")),
                          final(), on_request=check_request)
    state = make_agent(tmp_path, model).resume(paused.session_id, allow_tests=True)
    assert state.status == "completed", (state.stop_reason, state.final_response)
    assert [item["status"] for item in observations(state, "run_tests")] == ["passed", "failed", "passed"]
    assert (root / "calc.py").read_text() == "# External comment.\ndef add(a, b):\n    return a + b\n"


def test_unauthorized_pytest_stops_after_patch_without_claiming_completion(tmp_path):
    root = calculator(tmp_path)
    model = ScriptedModel(action(call("edit_file", path="calc.py", old_text="a - b", new_text="a + b")))
    state = make_agent(tmp_path, model, allow_tests=False).run("Fix calc.py", requirements=TaskRequirements.fix())
    assert "return a + b" in (root / "calc.py").read_text()
    assert state.status == "incomplete" and state.stop_reason == "verification_not_authorized"
    assert state.verification == "not_run" and len(model.requests) == 1
    assert not observations(state, "run_tests")


def test_multiple_file_patch_is_applied_before_one_automatic_test_run(tmp_path):
    root = calculator(tmp_path)
    (root / "offset.py").write_text("def offset():\n    return 0\n")
    (root / "calc.py").write_text("from offset import offset\ndef add(a, b):\n    return a - b + offset()\n")
    (root / "test_calc.py").write_text("from calc import add\ndef test_add():\n    assert add(3, 2) == 6\n")
    model = ScriptedModel(action(
        call("edit_file", path="calc.py", old_text="a - b", new_text="a + b"),
        call("edit_file", path="offset.py", old_text="return 0", new_text="return 1")), final())
    state = make_agent(tmp_path, model).run("Fix calc.py and its offset", requirements=TaskRequirements.fix())
    assert state.status == "completed", (state.stop_reason, state.final_response)
    assert state.modified_files == ["calc.py", "offset.py"]
    assert len(observations(state, "run_tests")) == 1
    tool_names = [message.get("tool_name") for message in state.messages if message.get("role") == "tool"]
    assert max(index for index, name in enumerate(tool_names) if name == "edit_file") < tool_names.index("run_tests")


def test_no_change_completion_still_observes_automatic_pytest(tmp_path):
    root = calculator(tmp_path)
    source = root / "calc.py"
    source.write_text(source.read_text().replace("a - b", "a + b"))
    model = ScriptedModel(final(), final())
    state = make_agent(tmp_path, model).run("Verify calc.py addition", requirements=TaskRequirements.code())
    assert state.status == "completed" and state.verification == "passed"
    assert not state.modified_files and len(model.requests) == 2
    assert len(observations(state, "run_tests")) == 1


def test_exact_edit_can_include_its_observed_trailing_newline(tmp_path):
    calculator(tmp_path)
    model = ScriptedModel(action(call("edit_file", path="calc.py",
                                      old_text="    return a - b\n", new_text="    return a + b\n")), final())
    state = make_agent(tmp_path, model).run("Fix calc.py", requirements=TaskRequirements.fix())
    assert state.status == "completed", (state.stop_reason, state.final_response)
    assert state.verification == "passed" and observations(state, "edit_file")[0]["success"]


def test_many_empty_imported_modules_do_not_stall_automatic_gathering(tmp_path):
    root = calculator(tmp_path)
    source = root / "calc.py"
    imports = "".join(f"import empty_{number}\n" for number in range(12))
    source.write_text(imports + source.read_text())
    for number in range(12):
        (root / f"empty_{number}.py").write_text("")
    model = ScriptedModel(action(call("edit_file", path="calc.py", old_text="a - b", new_text="a + b")), final())
    state = make_agent(tmp_path, model).run("Fix calc.py", requirements=TaskRequirements.fix())
    assert state.status == "completed", (state.stop_reason, state.final_response)
    assert state.steps == 2 and state.verification == "passed"
    read_paths = {record["path"] for observation in observations(state, "read_code")
                  for record in observation.get("files", [])}
    assert {f"empty_{number}.py" for number in range(12)} <= read_paths


def test_source_dropped_from_model_working_set_cannot_authorize_a_patch(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    paths = []
    for number in range(18):
        path = f"module_{number:02}.py"
        paths.append(path)
        lines = [f"    item_{index} = {index}" for index in range(22)]
        (root / path).write_text(f"def operation_{number}():\n" + "\n".join(lines) + f"\n    return {101 + number}\n")
    model = ScriptedModel(action(call("edit_file", path=paths[0], old_text="return 101", new_text="return 999")))
    model.context_size = 4096
    agent = make_agent(tmp_path, model, max_steps=1)
    agent.config = replace(agent.config, context=replace(agent.config.context, target=.3))
    state = agent.run("Fix " + ", ".join(paths), requirements=TaskRequirements.fix())
    assert len(model.requests) == 1
    assert "return 101" not in rendered(model.requests[0])
    # Raw observation history contains this source, but the model never saw
    # that omitted block. Only its actual prepared request may permit editing.
    assert any("return 101" in record.get("content", "") for observation in observations(state, "read_code")
               for record in observation.get("files", []))
    assert observations(state, "edit_file")[-1]["error"]["code"] == "inspection_required"
    assert "return 101" in (root / paths[0]).read_text()
    assert not state.modified_files


def test_clipped_line_prefix_is_not_complete_source_evidence_for_edit(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    original = 'payload = "' + "x" * 6000 + '"\n'
    (root / "large_line.py").write_text(original)
    model = ScriptedModel(action(call("edit_file", path="large_line.py",
                                      old_text='payload = "', new_text='renamed = "')))
    state = make_agent(tmp_path, model, max_steps=1).run("Fix large_line.py", requirements=TaskRequirements.fix())
    records = [record for observation in observations(state, "read_code") for record in observation.get("files", [])]
    assert records[0]["partial_line"] is True
    assert 'payload = "' in records[0]["content"]
    assert observations(state, "edit_file")[-1]["error"]["code"] == "inspection_required"
    assert (root / "large_line.py").read_text() == original and not state.modified_files


def test_interrupted_new_file_creation_is_reinspected_and_tested_before_resumed_model_request(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "legacy.py").write_text("def original_answer():\n    return 42\n")
    (root / "test_new.py").write_text("from new import answer\ndef test_answer():\n    assert answer() == 42\n")
    content = "def answer():\n    return 42\n"
    first = make_agent(tmp_path, ScriptedModel(action(call("write_file", path="new.py", content=content))))
    dispatch = first.registry.dispatch

    def interrupt_after_creation(name, arguments, **options):
        result = dispatch(name, arguments, **options)
        if name == "write_file":
            assert result["success"] and (root / "new.py").read_text() == content
            # The real mutation happened; no result or new-file snapshot was
            # recorded. Recovery must inspect its outcome, never replay it.
            raise KeyboardInterrupt
        return result

    monkeypatch.setattr(first.registry, "dispatch", interrupt_after_creation)
    paused = first.run("Create new.py using legacy.py", requirements=TaskRequirements.code())
    assert paused.stop_reason == "interrupted" and paused.verification == "not_run"
    checkpoint = json.loads((tmp_path / "runs" / paused.session_id / "checkpoint.json").read_text())
    assert "new.py" not in checkpoint["file_snapshots"]
    assert not checkpoint["execution"]["coding"]["test_pending"]

    def check_request(number, request):
        assert number == 1
        text = rendered(request)
        assert "Observed source new.py" in text and "return 42" in text
        assert any(message.get("tool_name") == "run_tests"
                   and json.loads(message["content"]).get("status") == "passed"
                   for message in request["messages"])

    model = ScriptedModel(final(), on_request=check_request)
    state = make_agent(tmp_path, model).resume(paused.session_id, allow_tests=True)
    assert state.status == "completed" and state.verification == "passed"
    assert len(model.requests) == 1 and (root / "new.py").read_text() == content
    assert len(observations(state, "run_tests")) == 1
    writes = [item for message in state.messages for item in message.get("tool_calls", [])
              if item["function"]["name"] == "write_file"]
    assert len(writes) == 1
    assert observations(state, "write_file")[0]["error"]["code"] == "operation_interrupted"
