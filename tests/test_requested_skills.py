"""Explicit skill requests use the frozen trusted catalog in coding runs."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from patchrook_agent.agent import Agent
from patchrook_agent.completion import TaskRequirements
from patchrook_agent.config import HarnessConfig
from patchrook_agent.models.ollama import ModelResponse
from patchrook_agent.skills.requests import explicit_skill_names, requested_skills, skill_access_requested
from patchrook_agent.tools import ToolConfig, ToolRegistry, Workspace


INDEX = [{"name": name} for name in ("debugging", "feature-implementation", "investigation")]


@pytest.mark.parametrize("task,expected", [
    ("Use the debugging skill to fix calc.py.", ["debugging"]),
    ("Use `debugging` skill.", ["debugging"]),
    ("Use the skill named 'feature-implementation'.", ["feature-implementation"]),
    ("Apply $investigation before $debugging.", ["investigation", "debugging"]),
    ("Use skills/debugging/SKILL.md and investigation/SKILL.md.", ["debugging", "investigation"]),
    ("Use the DEBUGGING skill, then $debugging again.", ["debugging"]),
    ("Use debugging and investigation skills.", ["debugging", "investigation"]),
    ("Do not use debugging and investigation skills.", []),
    ("Debugging this parser requires investigation.", []),
    ("Use a relevant trusted skill.", []),
    ("List available skills and read their instructions.", []),
    ("Fix skill loading in loader.py.", []),
    ("Implement skill support.", []),
    ("Explain skill handling.", []),
    ("Use file metadata to improve skill loading.", []),
    ("Do not use the debugging skill.", []),
    ("Avoid $debugging.", []),
    ("Fix it without the debugging skill.", []),
    ("Do not use debugging skill; use investigation skill.", ["investigation"]),
    ("Do not change tests, but use the debugging skill.", ["debugging"]),
    ("Use the debugging-extra skill.", ["debugging-extra"]),
    ("Use $security-review.", ["security-review"]),
    ("Use $debug--tools and $trailing-.", ["debug--tools", "trailing-"]),
    ("$debugging to fix calc.py.", ["debugging"]),
    ("Fix config.py so $HOME, $PATH, and $PYTHONPATH expand correctly.", []),
    ("Fix config.py so $project expands correctly.", []),
    ("Use $debugging to update config.py so $project expands correctly.", ["debugging"]),
])
def test_explicit_name_requests_are_not_topic_routing(task, expected):
    assert explicit_skill_names(task) == expected
    assert requested_skills(task, INDEX) == [name for name in expected if name in {item["name"] for item in INDEX}]


def test_only_supplied_catalog_names_can_be_selected():
    assert requested_skills("Use $debugging and $investigation", [{"name": "debugging"}]) == ["debugging"]
    assert requested_skills("Use $debugging", []) == []
    with pytest.raises(ValueError, match="string"):
        explicit_skill_names(None)


@pytest.mark.parametrize("task,expected", [
    ("Use a relevant trusted skill to fix calc.py.", True),
    ("List available skills.", True),
    ("Show the trusted skills.", True),
    ("Discover skills for this task.", True),
    ("Read skill instructions before patching.", True),
    ("Which skills do we have?", True),
    ("What skill should be used?", True),
    ("Do not use skills.", False),
    ("Avoid $debugging.", False),
    ("Do not list skills; use $debugging instead.", True),
    ("Use $security-review.", True),
    ("Fix skill loading in loader.py.", False),
    ("Implement skill support.", False),
    ("Explain skill handling.", False),
    ("Read skill loading code before editing it.", False),
    ("What skill loader class handles metadata?", False),
    ("Investigate the bug with debugging evidence.", False),
    ("Fix config.py so $HOME and $project expand correctly.", False),
])
def test_access_exposes_skills_only_for_explicit_use_or_discovery(task, expected):
    assert skill_access_requested(task) is expected


class ScriptedModel:
    model = "requested-skill-mechanics"
    context_size = 8192
    max_output_tokens = 512

    def __init__(self, *messages, check=None):
        self.messages = iter(messages)
        self.requests = []
        self.check = check

    def chat(self, **options):
        self.requests.append(deepcopy(options))
        if self.check:
            self.check(len(self.requests), options)
        return ModelResponse(next(self.messages), 0, 0, None, None, None, 0,
                             prompt_tokens_reported=False, completion_tokens_reported=False)


def action(tool_name, **arguments):
    return {"role": "assistant", "content": "", "tool_calls": [{"function": {
        "name": tool_name, "arguments": arguments}}]}


PATCH = action("edit_file", path="calc.py", old_text="a - b", new_text="a + b")
DONE = {"role": "assistant", "content": "DONE"}


def setup_repo(tmp_path, *, body="Keep exact source evidence before every correction."):
    root = tmp_path / "repo"
    root.mkdir(exist_ok=True)
    (root / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (root / "test_calc.py").write_text("from calc import add\ndef test_add():\n    assert add(2, 3) == 5\n")
    skill_root = tmp_path / "trusted_skills"
    directory = skill_root / "debugging"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "SKILL.md").write_text(
        "---\nname: debugging\ndescription: Correct code using observed failures.\n"
        "activation: When explicitly requested.\n---\n" + body + "\n")
    return root, skill_root


def agent_for(tmp_path, model, *, skills=True, pause_after_step=None):
    root, skill_root = tmp_path / "repo", tmp_path / "trusted_skills"
    config = replace(HarnessConfig.preset("C"), workflow_mode="coding", skills_enabled=skills,
                     trusted_skills_dir=str(skill_root))
    return Agent(model=model, registry=ToolRegistry(Workspace(root, ToolConfig(allow_tests=True))),
                 config=config, runs_dir=tmp_path / "runs", max_steps=5, pause_after_step=pause_after_step)


def skill_results(state):
    return [json.loads(message["content"]) for message in state.messages if message.get("tool_name") == "read_skill"]


@pytest.fixture(autouse=True)
def no_live_inference(monkeypatch):
    monkeypatch.setattr("patchrook_agent.models.ollama.httpx.Client",
                        lambda *args, **kwargs: pytest.fail("Skill tests must not contact Ollama"))


def test_named_skill_is_controller_loaded_before_first_model_request_and_pinned_whole(tmp_path):
    body = "START_EXACT_INSTRUCTIONS\n" + "Preserve the observed source and unrelated behavior.\n" * 100 + "END_EXACT_INSTRUCTIONS"
    setup_repo(tmp_path, body=body)

    def check(number, options):
        assert any(body in message.get("content", "") for message in options["messages"])
        assert not any("Available trusted skills" in message.get("content", "") for message in options["messages"])
        definition = next(item for item in options["tools"] if item["function"]["name"] == "read_skill")
        assert definition["function"]["parameters"]["properties"]["name"]["enum"] == ["debugging"]

    model = ScriptedModel(PATCH, DONE, check=check)
    state = agent_for(tmp_path, model).run("Use the debugging skill to fix calc.py.", requirements=TaskRequirements.fix())
    assert state.status == "completed" and state.verification == "passed", state.final_response
    assert len(skill_results(state)) == 1 and skill_results(state)[0]["success"]
    assert state.memory["loaded_skills"] == ["debugging"]
    assert state.metrics["skills_loaded"] == state.metrics["skills_activated"] == 1


def test_loaded_skill_is_not_loaded_again_on_checkpoint_resume(tmp_path):
    setup_repo(tmp_path)
    paused = agent_for(tmp_path, ScriptedModel(PATCH), pause_after_step=1).run(
        "Use $debugging to fix calc.py.", requirements=TaskRequirements.fix())
    assert paused.stop_reason == "paused" and len(skill_results(paused)) == 1
    resumed_model = ScriptedModel(DONE)
    resumed = agent_for(tmp_path, resumed_model).resume(paused.session_id, allow_tests=True)
    assert resumed.status == "completed" and resumed.verification == "passed", resumed.final_response
    assert len(resumed_model.requests) == 1 and len(skill_results(resumed)) == 1
    assert any("Trusted skill document: debugging/SKILL.md" in message.get("content", "")
               for message in resumed_model.requests[0]["messages"])


@pytest.mark.parametrize("skills,name", [(True, "security-review"), (False, "debugging")])
def test_unknown_or_disabled_requested_skill_stops_before_inference(tmp_path, skills, name):
    setup_repo(tmp_path)
    model = ScriptedModel()
    state = agent_for(tmp_path, model, skills=skills).run(f"Use ${name} to fix calc.py.", requirements=TaskRequirements.fix())
    assert not model.requests and state.status == "incomplete"
    assert state.stop_reason == "requested_skill_unavailable"
    assert any(name in missing for missing in state.completion["missing"])
    assert name in state.final_response


def test_requested_skill_reference_is_loaded_as_text_and_never_executed(tmp_path):
    _, skill_root = setup_repo(tmp_path)
    reference = skill_root / "debugging" / "references" / "inspect.py"
    reference.parent.mkdir()
    marker = tmp_path / "script_was_executed"
    script = f"from pathlib import Path\nPath({str(marker)!r}).write_text('unexpected execution')\n"
    reference.write_text(script)

    def check(number, options):
        if number >= 2:
            assert any(script in message.get("content", "") for message in options["messages"])

    model = ScriptedModel(action("read_skill", name="debugging", reference="references/inspect.py"), PATCH, DONE, check=check)
    state = agent_for(tmp_path, model).run("Use the debugging skill to fix calc.py.", requirements=TaskRequirements.fix())
    assert state.status == "completed" and state.verification == "passed", state.final_response
    assert not marker.exists()
    assert len(skill_results(state)) == 2 and skill_results(state)[1]["is_reference"]


def test_configuration_variable_task_reaches_model_without_skill_routing(tmp_path):
    root, _ = setup_repo(tmp_path)
    (root / "calc.py").rename(root / "config.py")
    (root / "test_calc.py").rename(root / "test_config.py")
    (root / "test_config.py").write_text("from config import add\ndef test_add():\n    assert add(2, 3) == 5\n")

    def check(number, options):
        assert not any(item["function"]["name"] == "read_skill" for item in options["tools"])
        assert not any("Available trusted skills" in message.get("content", "") for message in options["messages"])

    model = ScriptedModel(action("edit_file", path="config.py", old_text="a - b", new_text="a + b"), DONE, check=check)
    state = agent_for(tmp_path, model).run(
        "Fix config.py so $HOME and $project expand correctly.", requirements=TaskRequirements.fix())
    assert state.status == "completed" and state.verification == "passed", state.final_response
    assert len(model.requests) == 2 and not skill_results(state)
