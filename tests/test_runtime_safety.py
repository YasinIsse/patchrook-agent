"""Critical state remains factual across output reduction and uncertain edits."""

from copy import deepcopy
import json

from patchrook_agent.agent import Agent
from patchrook_agent.config import HarnessConfig
from patchrook_agent.models.ollama import ModelResponse
from patchrook_agent.session import Session
from patchrook_agent.tools import ToolConfig, ToolRegistry, Workspace


def call(name, **arguments):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": name, "arguments": arguments}},
    ]}


class Model:
    context_size = 4096
    max_output_tokens = 512

    def __init__(self, *messages):
        self.responses = iter(messages)

    def chat(self, **kwargs):
        return ModelResponse(next(self.responses), 5, 2, 0, 0, 0, 0)


def test_memory_projection_refers_to_protected_exact_error_without_repeating_it():
    result = {"success": False, "status": "failed", "exit_code": 1,
              "stdout": "EXACT_DEBUG_EVIDENCE: assertion failed", "stderr": ""}
    memory = {"unresolved_errors": [{"tool": "run_tests", "path": "tests", "step": 2,
                                      **{key: value for key, value in result.items() if key != "success"}}],
              "latest_test": {"path": "tests", "status": "failed", "exit_code": 1,
                              "stdout": result["stdout"], "stderr": ""}}
    original = deepcopy(memory)
    active = [{"role": "tool", "tool_name": "run_tests", "content": json.dumps(result)}]
    Session._deduplicate_error_evidence(memory, active)
    assert "EXACT_DEBUG_EVIDENCE" not in json.dumps(memory)
    assert "EXACT_DEBUG_EVIDENCE" in active[0]["content"]
    assert "EXACT_DEBUG_EVIDENCE" in original["unresolved_errors"][0]["stdout"]
    assert memory["unresolved_errors"][0]["evidence"]


def test_uncertain_first_edit_still_requires_verification_after_reread(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "impl.py").write_text("value = 1\n")
    (repo / "test_impl.py").write_text("from impl import value\ndef test_value():\n    assert value == 2\n")
    config = HarnessConfig.preset("C")
    registry = ToolRegistry(Workspace(repo, ToolConfig(allow_tests=True)))
    first = Agent(Model(call("read_file", path="impl.py"),
                        call("edit_file", path="impl.py", old_text="value = 1", new_text="value = 2")),
                  registry=registry, config=config, runs_dir=tmp_path / "runs")
    dispatch = registry.dispatch

    def interrupted(name, arguments, **kwargs):
        result = dispatch(name, arguments, **kwargs)
        if name == "edit_file":
            raise KeyboardInterrupt
        return result

    monkeypatch.setattr(registry, "dispatch", interrupted)
    paused = first.run("Change value to 2 and verify it.")
    assert paused.modified_files == [] and paused.stop_reason == "interrupted"
    final = {"role": "assistant", "content": "Inspected the updated value."}
    resumed = Agent(Model(call("read_file", path="impl.py"), final, final),
                    registry=ToolRegistry(Workspace(repo, ToolConfig(allow_tests=True))),
                    config=config, runs_dir=tmp_path / "runs").resume(paused.session_id, allow_tests=True)
    assert resumed.status == "completed" and resumed.verification == "passed"
    assert sum(message.get("tool_name") == "run_tests" for message in resumed.messages) == 1
    assert sum(message.get("tool_name") == "edit_file" for message in resumed.messages) == 1
    assert resumed.metrics["tools_by_name"]["edit_file"]["unknown"] == 1
    assert sum(row["calls"] for row in resumed.metrics["tools_by_name"].values()) == resumed.tool_calls
