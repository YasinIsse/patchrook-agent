"""Import navigation must acquire real source before a verified patch."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys

from patchrook_agent.agent import Agent
from patchrook_agent.completion import TaskRequirements
from patchrook_agent.config import HarnessConfig
from patchrook_agent.models.ollama import ModelResponse
from patchrook_agent.tools import ToolConfig, ToolRegistry, Workspace


def action(tool_name, **arguments):
    return {"role": "assistant", "content": json.dumps({"name": tool_name, "arguments": arguments})}


class Model:
    model = "offline-navigation-integration"
    context_size = 8192
    max_output_tokens = 1024

    def __init__(self, *responses):
        self.responses = iter(responses)
        self.requests = []

    def chat(self, messages, **options):
        self.requests.append(deepcopy(messages))
        response = next(self.responses)
        if callable(response):
            response = response(messages)
        return ModelResponse(response, 100, 20, 0, 0, 0, 0)


def make_agent(tmp_path, model, *, steps=4):
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg/__init__.py").write_text("")
    (repo / "pkg/shim.py").write_text('from pkg.implementation import Calculator\n__all__ = ["Calculator"]\n')
    (repo / "pkg/implementation.py").write_text("class Calculator:\n    def add(self, a, b):\n        return a - b\n")
    base = HarnessConfig.preset("C")
    config = replace(base, workflow_mode="evidence", action_protocol="schema", review_mode="off",
        context=replace(base.context, enabled=False), compaction_enabled=False,
        evidence_test_commands=((sys.executable, "-c", "from pkg.implementation import Calculator; assert Calculator().add(2,3)==5"),))
    return Agent(model, registry=ToolRegistry(Workspace(repo, ToolConfig(allow_tests=True))),
                 config=config, runs_dir=tmp_path / "runs", max_steps=steps, deadline_seconds=30)


def test_wrapper_read_follows_named_local_symbol_then_checks_guarded_patch(tmp_path):
    def patch(messages):
        text = "\n".join(m["content"] for m in messages)
        assert '"pure_reexport":true' in text
        assert '"path":"pkg/implementation.py"' in text
        assert "Requested source (data)" in text and "return a - b" in text
        assert "Information need: Read shim imports" not in text
        return action("apply_edits", edits=[{"path": "pkg/implementation.py", "old_text": "return a - b", "new_text": "return a + b"}])
    model = Model(action("read_file", path="pkg/shim.py", question="Read shim imports"), patch,
                  action("run_check", name="check_1"), action("finish"))
    agent = make_agent(tmp_path, model)
    state = agent.run("Fix Calculator addition and verify it.", requirements=TaskRequirements.fix())
    assert state.status == "completed", state.final_response
    assert state.verification == "passed" and state.modified_files == ["pkg/implementation.py"]
    assert len(model.requests) == 4
    events = [json.loads(line) for line in (Path(state.session_dir) / "events.jsonl").read_text().splitlines()]
    automatic = [e for e in events if e["event"] == "tool_request" and e.get("automatic")]
    assert [e["name"] for e in automatic] == ["search_code", "read_file"]
    assert automatic[0]["arguments"]["pattern"] == "class Calculator"
    assert automatic[1]["arguments"]["path"] == "pkg/implementation.py"


def test_old_navigation_observations_are_archived_but_not_replayed_as_optional_text(tmp_path):
    model = Model(action("read_file", path="pkg/shim.py"), action("list_files", path="pkg"),
                  action("list_files", path="pkg"), action("finish"))
    agent = make_agent(tmp_path, model)
    state = agent.run("Inspect Calculator; preserve source.", requirements=TaskRequirements())
    assert len(model.requests) == 4
    text = "\n".join(m["content"] for m in model.requests[-1])
    assert "navigation" in text
    assert "Observation " not in text  # old raw read/list receipts stay retrievable
    checkpoint = json.loads((Path(state.session_dir) / "checkpoint.json").read_text())
    assert checkpoint["execution"]["evidence"]["navigation"]["reads"]
    assert (Path(state.session_dir) / "evidence.json").exists()
