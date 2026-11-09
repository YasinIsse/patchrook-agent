"""Legacy/custom tool sets receive advice only for available read functions."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

from patchrook_agent.agent import Agent, LEGACY_SYSTEM_PROMPT, SYSTEM_PROMPT
from patchrook_agent.config import HarnessConfig
from patchrook_agent.models.ollama import ModelResponse
from patchrook_agent.tools import ToolConfig, ToolRegistry, Workspace
from patchrook_agent.tools.base import with_read_guidance


def action(name, **arguments):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": name, "arguments": arguments}},
    ]}


def final():
    return {"role": "assistant", "content": "Inspection complete."}


class Model:
    model = "guidance-fixture"
    context_size = 16384
    max_output_tokens = 512

    def __init__(self, *messages):
        self.messages = iter(messages)
        self.requests = []

    def chat(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        return ModelResponse(next(self.messages), 10, 2, 0, 0, 0, 0)


def fixture_registry(tmp_path, *, max_output_chars=2000):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("a = 1\n")
    (repo / "b.py").write_text("b = 2\n")
    tools = ToolRegistry(Workspace(repo, ToolConfig(max_output_chars=max_output_chars)))
    matches = [{"path": f"stage{index}.py", "line_number": 14, "text": "def transform(value):"}
               for index in range(8)]
    tools.tools["search_code"] = replace(tools.tools["search_code"], execute=lambda workspace, **kwargs: {
        "success": True, "matches": matches, "offset": 0, "returned_count": 8, "truncated": False,
        "read_hint": "Use read_files", "suggested_read": {"name": "read_files", "arguments": {"files": []}},
    })
    return tools


def subset(names):
    return [{"type": "function", "function": {"name": name}} for name in names]


def observed(state, name):
    return [json.loads(message["content"]) for message in state.messages if message.get("tool_name") == name]


def test_new_six_tool_subset_uses_trusted_legacy_prompt_and_single_read_guidance(tmp_path):
    tools = fixture_registry(tmp_path)
    six = set(tools.tools) - {"read_files"}
    model = Model(action("search_code", pattern="transform"), action("list_files"), final())
    state = Agent(model, registry=tools, runs_dir=tmp_path / "runs").run("Inspect known sources.", tools=subset(six))
    assert state.status == "completed"
    for request in model.requests:
        assert request["messages"][0]["content"] == LEGACY_SYSTEM_PROMPT
        assert {item["function"]["name"] for item in request["tools"]} == six
        assert "read_files" not in json.dumps(request["tools"])
    search = observed(state, "search_code")[0]
    assert search["suggested_read"]["name"] == "read_file"
    assert "read_files" not in json.dumps(search)
    assert "read_file" in observed(state, "list_files")[0]["read_hint"]


def test_subset_without_read_tools_has_no_generated_read_advice_or_disabled_descriptions(tmp_path):
    tools = fixture_registry(tmp_path)
    enabled = {"search_code", "list_files"}
    model = Model(action("search_code", pattern="transform"), action("list_files"), final())
    state = Agent(model, registry=tools, runs_dir=tmp_path / "runs").run("Find relevant sources.", tools=subset(enabled))
    for name in ("search_code", "list_files"):
        result = observed(state, name)[0]
        assert "suggested_read" not in result
        assert "read_file" not in result.get("read_hint", "")
    assert "read_file" not in json.dumps(model.requests[0]["tools"])


def test_batch_only_reader_never_suggests_disabled_single_read(tmp_path):
    tools = fixture_registry(tmp_path)
    tools.guidance_tools = {"search_code", "read_files"}
    one = {"success": True, "matches": [{"path": "a.py", "line_number": 1, "text": "a = 1"}], "truncated": False}
    result = with_read_guidance(one, 2000, enabled_tools=tools.guidance_tools)
    assert result["suggested_read"] == {"name": "read_files", "arguments": {
        "files": [{"path": "a.py", "start_line": 1, "end_line": 6}],
    }}
    definitions = tools.definitions(enabled_tools=tools.guidance_tools)
    assert {item["function"]["name"] for item in definitions} == tools.guidance_tools
    assert "read_file for" not in json.dumps(definitions)


def test_restricted_raw_artifact_has_no_generated_advice_for_disabled_batch_tool(tmp_path):
    tools = fixture_registry(tmp_path, max_output_chars=512)
    enabled = (set(tools.tools) - {"read_files"}) | {"read_artifact"}
    state = Agent(Model(action("search_code", pattern="transform"), final()), registry=tools,
                  config=HarnessConfig.preset("C"), runs_dir=tmp_path / "runs").run("Locate sources.", tools=subset(enabled))
    result = observed(state, "search_code")[0]
    artifact = Path(state.session_dir) / "artifacts" / f"{result['artifact_id']}.json"
    payload = json.loads(artifact.read_text())
    assert "read_files" not in json.dumps(payload)
    suggestion = tools.last_execution["result"].get("suggested_read")
    assert suggestion is None or suggestion["name"] == "read_file"


def test_six_tool_checkpoint_resume_preserves_original_prompt_and_guidance_subset(tmp_path):
    tools = fixture_registry(tmp_path)
    six = set(tools.tools) - {"read_files"}
    runs = tmp_path / "runs"
    paused = Agent(Model(action("list_files")), registry=tools, runs_dir=runs,
                   config=HarnessConfig.preset("C"), pause_after_step=1).run("Inspect known sources.", tools=subset(six))
    assert paused.stop_reason == "paused" and paused.messages[0]["content"] == LEGACY_SYSTEM_PROMPT
    resumed_registry = ToolRegistry(Workspace(tools.workspace.root))
    model = Model(action("list_files"), final())
    resumed = Agent(model, registry=resumed_registry, runs_dir=runs).resume(paused.session_id)
    assert resumed.status == "completed"
    assert resumed.messages[:len(paused.messages)] == paused.messages
    assert model.requests[0]["messages"][0]["content"] == LEGACY_SYSTEM_PROMPT
    assert {item["function"]["name"] for item in model.requests[0]["tools"]} == six
    assert "read_files" not in json.dumps(model.requests[0]["tools"])
    assert "read_files" not in json.dumps(observed(resumed, "list_files")[-1])


def test_full_tool_definitions_guidance_and_raw_results_remain_identical(tmp_path):
    tools = fixture_registry(tmp_path)
    all_tools = set(tools.tools)
    assert tools.definitions() == tools.definitions(enabled_tools=all_tools)
    result = {"success": True, "matches": [
        {"path": name, "line_number": 14, "text": "def transform(value):"} for name in ("a.py", "b.py")
    ], "truncated": False}
    assert with_read_guidance(result, 2000) == with_read_guidance(result, 2000, enabled_tools=all_tools)
    original = tools.dispatch("search_code", {"pattern": "transform"})
    raw = deepcopy(tools.last_execution)
    tools.guidance_tools = all_tools
    assert tools.dispatch("search_code", {"pattern": "transform"}) == original
    assert tools.last_execution == raw
    tools.guidance_tools = None
    model = Model(final())
    state = Agent(model, registry=tools, runs_dir=tmp_path / "runs").run("Inspect sources.")
    assert state.messages[0]["content"] == SYSTEM_PROMPT
    assert model.requests[0]["messages"][0]["content"] == SYSTEM_PROMPT
