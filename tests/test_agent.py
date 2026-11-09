from copy import deepcopy
import json
import time

import httpx
import pytest

from patchrook_agent.agent import Agent
from patchrook_agent.models.ollama import ModelResponse
from patchrook_agent.tools import ToolConfig, ToolRegistry, Workspace


def call(name, **arguments):
    return {"function": {"name": name, "arguments": arguments}}


def action(*calls):
    return {"role": "assistant", "content": "", "tool_calls": list(calls)}


def final(text="Finished inspection."):
    return {"role": "assistant", "content": text}


class MockModel:
    context_size = 4096

    def __init__(self, *messages, done_reason="stop"):
        self.responses = iter(messages)
        self.requests = []
        self.done_reason = done_reason

    def chat(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        message = next(self.responses)
        if isinstance(message, Exception):
            raise message
        return ModelResponse(message, 100, 20, 1, 2, 3, 1_000_000_000, self.done_reason)


def observations(state):
    return [json.loads(item["content"]) for item in state.messages if item["role"] == "tool"]


def test_multiple_calls_preserve_assistant_then_observations_and_metrics(tmp_path):
    (tmp_path / "code.py").write_text("x = 1\n")
    requested = action(call("list_files"), call("read_file", path="code.py"))
    requested["tool_calls"][1]["id"] = "call-2"
    model = MockModel(requested, final())
    state = Agent(model, workspace=tmp_path, think=True).run("Inspect the code")
    assert state.status == "completed"
    assert state.messages[2] == requested
    assert [m["role"] for m in state.messages] == ["system", "user", "assistant", "tool", "tool", "assistant"]
    assert state.messages[4]["tool_call_id"] == "call-2"
    assert model.requests[1]["messages"][3]["tool_name"] == "list_files"
    assert model.requests[0]["think"] is True
    assert len(model.requests[0]["tools"]) == 7
    assert state.steps == 2 and state.tool_calls == 2 and state.tool_failures == 0
    assert state.total_prompt_tokens == 200 and state.total_completion_tokens == 40
    assert state.total_inference_seconds == 2 and state.total_tool_seconds > 0


def test_invalid_arguments_and_missing_file_are_recoverable(tmp_path):
    (tmp_path / "good").write_text("hello")
    model = MockModel(action(call("read_file", path=5)), action(call("read_file", path="absent")),
                      action(call("read_file", path="good")), final())
    state = Agent(model, workspace=tmp_path).run("Inspect")
    assert state.status == "completed" and state.tool_failures == 2
    assert [result.get("error", {}).get("code") for result in observations(state)] == [
        "invalid_arguments", "not_found", None,
    ]


def test_max_steps_stops_tool_loop(tmp_path):
    state = Agent(MockModel(action(call("list_files"))), max_steps=1, workspace=tmp_path).run("Inspect")
    assert state.steps == 1 and state.tool_calls == 1
    assert state.status == "incomplete" and state.stop_reason == "max_steps"


def test_repeated_failures_stop_before_budget(tmp_path):
    model = MockModel(*(action(call("read_file", path="absent")) for _ in range(5)))
    state = Agent(model, workspace=tmp_path).run("Inspect")
    assert state.stop_reason == "repeated_tool_failures" and state.steps == 3


def test_final_response_stops_without_fabricating_verification(tmp_path):
    model = MockModel(final("No modifications needed."))
    state = Agent(model, workspace=tmp_path).run("Inspect")
    assert state.final_response == "No modifications needed."
    assert state.steps == 1 and state.verification == "not_run" and state.tool_calls == 0


def test_modifications_without_trust_are_reported_unverified(tmp_path):
    model = MockModel(action(call("write_file", path="new.py", content="x = 1\n")), final("Tests passed."))
    state = Agent(model, workspace=tmp_path).run("Add a file")
    assert state.modified_files == ["new.py"]
    assert state.status == "incomplete" and state.stop_reason == "verification_not_authorized"
    assert "not run" in state.final_response


def test_harness_attempts_omitted_verification_with_actual_pytest(tmp_path):
    (tmp_path / "test_new.py").write_text("def test_new():\n    assert True\n")
    model = MockModel(action(call("write_file", path="new.py", content="x = 1\n")),
                      final("Done."), final("Added new.py; pytest passed."))
    registry = ToolRegistry(Workspace(tmp_path, ToolConfig(allow_tests=True)))
    state = Agent(model, registry=registry).run("Add new.py")
    assert state.status == "completed" and state.verification == "passed"
    assert state.tool_calls == 2
    assert observations(state)[-1]["exit_code"] == 0


def test_tests_before_edit_do_not_verify_later_modification(tmp_path):
    (tmp_path / "test_example.py").write_text("def test_example():\n    assert True\n")
    model = MockModel(action(call("run_tests", path=".")),
                      action(call("write_file", path="new.py", content="x = 1\n")),
                      final("Done"), final("Added new.py; tests passed."))
    state = Agent(model, registry=ToolRegistry(Workspace(tmp_path, ToolConfig(allow_tests=True)))).run("Add")
    assert state.status == "completed" and state.verification == "passed"
    assert sum(m.get("tool_name") == "run_tests" for m in state.messages) == 2


def test_failing_tests_do_not_mark_task_successful(tmp_path):
    (tmp_path / "test_failure.py").write_text("def test_failure():\n    assert False\n")
    model = MockModel(action(call("run_tests", path=".")), final("The test is failing."))
    registry = ToolRegistry(Workspace(tmp_path, ToolConfig(allow_tests=True)))
    state = Agent(model, registry=registry).run("Test")
    assert state.status == "incomplete" and state.verification == "failed"
    assert state.stop_reason == "tests_failed"


def test_failed_pytest_can_be_repaired_and_verified(tmp_path):
    source = "def add_numbers(a, b):\n    return a - b\n\ndef untouched():\n    return 7\n"
    (tmp_path / "numbers_impl.py").write_text(source)
    (tmp_path / "test_numbers.py").write_text(
        "from numbers_impl import add_numbers\n\ndef test_add():\n    assert add_numbers(3, 2) == 5\n"
    )
    model = MockModel(action(call("read_file", path="numbers_impl.py")),
                      action(call("run_tests", path="test_numbers.py")),
                      action(call("edit_file", path="numbers_impl.py",
                                  old_text="return a - b", new_text="return a + b")),
                      action(call("run_tests", path="test_numbers.py")),
                      final("Changed subtraction to addition; the relevant pytest test now passes."))
    state = Agent(model, registry=ToolRegistry(Workspace(tmp_path, ToolConfig(allow_tests=True)))).run("Fix add_numbers")
    assert state.status == "completed" and state.verification == "passed"
    assert state.tool_failures == 1
    tests = [json.loads(m["content"]) for m in state.messages if m.get("tool_name") == "run_tests"]
    assert [result["exit_code"] for result in tests] == [1, 0]
    assert (tmp_path / "numbers_impl.py").read_text() == source.replace("a - b", "a + b")


def test_server_context_error_is_a_graceful_stop(tmp_path):
    request = httpx.Request("POST", "http://localhost:11434/api/chat")
    response = httpx.Response(400, request=request, json={"error": "input exceeds context length"})
    error = httpx.HTTPStatusError("too long", request=request, response=response)
    state = Agent(MockModel(error), workspace=tmp_path).run("Inspect")
    assert state.status == "incomplete" and state.stop_reason == "context_limit"


def test_task_deadline_blocks_tool_execution_after_slow_model(tmp_path):
    model = MockModel(action(call("write_file", path="x", content="bad")))
    original = model.chat

    def delayed(**kwargs):
        time.sleep(0.02)
        return original(**kwargs)

    model.chat = delayed
    state = Agent(model, workspace=tmp_path, deadline_seconds=0.01).run("Write")
    assert state.stop_reason == "deadline_exceeded"
    assert not (tmp_path / "x").exists() and state.tool_calls == 0


def test_context_overflow_fails_without_dropping_messages(tmp_path):
    model = MockModel()
    task = "x" * 30_000
    state = Agent(model, workspace=tmp_path).run(task)
    assert state.stop_reason == "context_limit" and state.steps == 0
    assert state.messages[1]["content"] == task
    assert not model.requests


@pytest.mark.parametrize("response,reason", [
    (httpx.ConnectError("offline"), "model_error: ConnectError"),
    (httpx.ReadTimeout("timeout"), "model_timeout"),
    ({"role": "user", "content": "invalid"}, "model_error: ValueError"),
    ({"role": "assistant", "content": "", "tool_calls": "bad"}, "invalid_tool_calls"),
    (final(""), "empty_response"),
])
def test_unrecoverable_model_failures_are_reported(tmp_path, response, reason):
    state = Agent(MockModel(response), workspace=tmp_path).run("Inspect")
    assert state.status != "completed" and state.stop_reason == reason


def test_truncated_generation_cannot_be_completed_or_executed(tmp_path):
    state = Agent(MockModel(action(call("write_file", path="x", content="bad")), done_reason="length"),
                  workspace=tmp_path).run("Write")
    assert state.stop_reason == "response_limit"
    assert state.tool_calls == 0 and not (tmp_path / "x").exists()


def test_disabled_tools_cannot_be_called(tmp_path):
    model = MockModel(action(call("write_file", path="x", content="bad")), final())
    registry = ToolRegistry(Workspace(tmp_path))
    subset = [item for item in registry.definitions() if item["function"]["name"] == "list_files"]
    state = Agent(model, registry=registry).run("Inspect", tools=subset)
    assert observations(state)[0]["error"]["code"] == "unknown_tool"
    assert not (tmp_path / "x").exists()
