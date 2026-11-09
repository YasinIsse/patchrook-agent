"""A hard process death cannot replay an already-applied edit on resume."""

import json
from pathlib import Path
import subprocess
import sys
import time

from patchrook_agent.agent import Agent
from patchrook_agent.config import HarnessConfig
from patchrook_agent.models.ollama import ModelResponse
from patchrook_agent.tools import ToolConfig, ToolRegistry, Workspace


def test_sigkill_after_edit_preserves_unknown_outcome_and_resumes_safely(tmp_path):
    repo, runs = tmp_path / "repo", tmp_path / "runs"
    repo.mkdir()
    (repo / "sample_impl.py").write_text("value = 1\n")
    (repo / "test_impl.py").write_text("from sample_impl import value\ndef test_value():\n    assert value == 2\n")
    ready = repo / ".edit-applied"
    script = '''
from pathlib import Path
import time
from patchrook_agent.agent import Agent
from patchrook_agent.config import HarnessConfig
from patchrook_agent.models.ollama import ModelResponse
from patchrook_agent.tools import ToolConfig, ToolRegistry, Workspace

def call(name, **arguments):
    return {"role":"assistant","content":"","tool_calls":[{"function":{"name":name,"arguments":arguments}}]}
class Model:
    context_size = 4096
    max_output_tokens = 512
    def __init__(self):
        self.messages = iter([call("read_file",path="sample_impl.py"),
            call("edit_file",path="sample_impl.py",old_text="value = 1",new_text="value = 2")])
    def chat(self, **kwargs):
        return ModelResponse(next(self.messages),5,2,0,0,0,0)
repo, runs = Path(REPO), Path(RUNS)
registry = ToolRegistry(Workspace(repo, ToolConfig(allow_tests=True)))
dispatch = registry.dispatch
def stop_after_edit(name, arguments, **kwargs):
    result = dispatch(name,arguments,**kwargs)
    if name == "edit_file":
        (repo / ".edit-applied").write_text("ready")
        time.sleep(30)
    return result
registry.dispatch = stop_after_edit
Agent(Model(),registry=registry,config=HarnessConfig.preset("C"),runs_dir=runs).run("Change value to 2 and verify it.")
'''.replace("REPO", repr(str(repo))).replace("RUNS", repr(str(runs)))
    child = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and child.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists(), "Child failed before reaching the interrupted tool boundary"
        child.kill()
        child.wait(timeout=5)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)
        child.stderr.close()
    session_id = next(runs.iterdir()).name
    checkpoint = json.loads((runs / session_id / "checkpoint.json").read_text())
    assert checkpoint["execution"]["pending_operation"]["function"]["name"] == "edit_file"
    assert (repo / "sample_impl.py").read_text() == "value = 2\n"

    class ResumedModel:
        context_size = 4096
        max_output_tokens = 512

        def __init__(self):
            self.messages = iter([
                {"role": "assistant", "content": "", "tool_calls": [{"function": {
                    "name": "read_file", "arguments": {"path": "sample_impl.py"}}}]},
                {"role": "assistant", "content": "", "tool_calls": [{"function": {
                    "name": "run_tests", "arguments": {"path": "test_impl.py"}}}]},
                {"role": "assistant", "content": "Inspected the applied change; pytest passed."},
            ])

        def chat(self, **kwargs):
            return ModelResponse(next(self.messages), 5, 2, 0, 0, 0, 0)

    state = Agent(ResumedModel(), registry=ToolRegistry(Workspace(repo, ToolConfig(allow_tests=True))),
                  config=HarnessConfig.preset("C"), runs_dir=runs).resume(session_id, allow_tests=True)
    assert state.status == "completed" and state.verification == "passed"
    assert state.metrics["unknown_tool_calls"] == 1
    assert state.metrics["tools_by_name"]["edit_file"]["calls"] == 1
    assert sum(row["calls"] for row in state.metrics["tools_by_name"].values()) == state.tool_calls
    edits = [json.loads(message["content"]) for message in state.messages if message.get("tool_name") == "edit_file"]
    assert len(edits) == 1 and edits[0]["error"]["code"] == "operation_interrupted"
    assert (repo / "sample_impl.py").read_text() == "value = 2\n"
