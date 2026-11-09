"""Formatting recovery stays opt-in, auditable, atomic and source-authorized."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import textwrap
import time

import pytest

from patchrook_agent.agent import Agent
from patchrook_agent.config import HarnessConfig
from patchrook_agent.evidence_tools import EvidenceTools
from patchrook_agent.evidence_workflow import EvidenceWorkflow
from patchrook_agent.tools import ToolConfig, ToolRegistry, Workspace
from test_evidence_tools import present


SOURCE = "def subject(value):\n    first = value + 1\n\n    return first\n"
PROPOSAL = "def subject(value):\n    first = value + 2\n    return first\n"


@pytest.fixture(autouse=True)
def no_models_or_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Anchor tests must not use model or network APIs")
    monkeypatch.setattr("httpx.Client", forbidden)
    monkeypatch.setattr("patchrook_agent.models.ollama.OllamaClient.chat", forbidden)


def editor(root, *, enabled=True):
    (root / "subject.py").write_text(SOURCE)
    return EvidenceTools(Workspace(root, ToolConfig(validate_python_edits=False)),
                         recover_python_function_anchors=enabled)


def apply(tools, old, new=PROPOSAL, *, visible=True):
    arguments = {"edits": [{"path": "subject.py", "old_text": old, "new_text": new}]}
    sources = [present(tools, "subject.py")] if visible else None
    result = tools.execute("apply_edits", arguments, deadline=time.monotonic() + 10,
                           visible_sources=sources)
    return result, arguments


@pytest.mark.parametrize("old,new", [
    (textwrap.indent(SOURCE, "    "), textwrap.indent(PROPOSAL, "    ")),
    (SOURCE.replace("\n\n", "\n"), PROPOSAL),
])
def test_recovery_uses_only_model_new_text_and_retains_original_proposal(tmp_path, old, new):
    tools = editor(tmp_path)
    result, arguments = apply(tools, old, new)
    original = deepcopy(arguments)
    assert result["success"] and result["python_syntax_validation"]["checked_paths"] == ["subject.py"]
    assert (tmp_path / "subject.py").read_text().startswith(textwrap.dedent(new))
    receipt = result["anchor_recovery"]
    assert receipt["kind"] == "python_function_formatting" and receipt["applied"] is True
    assert receipt["requires_fresh_presented_function"] is True
    assert receipt["original_proposal_sha256"] == hashlib.sha256(json.dumps(
        original, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    assert tools.last_execution["arguments"] == original == arguments
    assert arguments["edits"] == [{"path": "subject.py", "old_text": old, "new_text": new}]
    assert receipt["recoveries"][0]["function"] == "subject"
    assert not tools._presented


def test_disabled_and_unpresented_paths_keep_normal_guards(tmp_path):
    tools = editor(tmp_path, enabled=False)
    result, _ = apply(tools, textwrap.indent(SOURCE, "    "))
    assert result["error"]["code"] == "no_match" and "anchor_recovery" not in result
    tools = editor(tmp_path)
    result, _ = apply(tools, textwrap.indent(SOURCE, "    "), visible=False)
    assert result["error"]["code"] == "source_not_presented" and "anchor_recovery" not in result
    assert (tmp_path / "subject.py").read_text() == SOURCE


def test_partial_old_scope_does_not_acquire_the_rest_of_the_function(tmp_path):
    tools = editor(tmp_path)
    result, _ = apply(tools, "    def subject(value):\n        first = value + 1\n")
    assert result["error"]["code"] == "no_match" and "anchor_recovery" not in result
    assert (tmp_path / "subject.py").read_text() == SOURCE


def test_ordinary_exact_edits_do_not_inherit_recovery_metadata(tmp_path):
    tools = editor(tmp_path)
    bad = {"edits": [
        {"path": "subject.py", "old_text": textwrap.indent(SOURCE, "    "), "new_text": PROPOSAL},
        {"path": "subject.py", "old_text": "return first", "new_text": "return 9"},
    ]}
    result = tools.execute("apply_edits", bad, deadline=time.monotonic() + 10,
                           visible_sources=[present(tools, "subject.py")])
    assert result["error"]["code"] == "overlapping_edits"
    assert result["anchor_recovery"]["applied"] is False
    assert (tmp_path / "subject.py").read_text() == SOURCE
    result, _ = apply(tools, "value + 1", "value + 3")
    assert result["success"] and "anchor_recovery" not in result
    assert "python_syntax_validation" not in result


def test_unsupported_proposals_fall_back_without_creating_metadata(tmp_path):
    tools = editor(tmp_path)
    source = SOURCE.replace("return first", "return first # keep")
    (tmp_path / "subject.py").write_text(source)
    result, _ = apply(tools, textwrap.indent(SOURCE, "    "))
    assert result["error"]["code"] == "no_match" and "anchor_recovery" not in result
    assert (tmp_path / "subject.py").read_text() == source


def test_config_roundtrip_defaults_and_strict_opt_in():
    base = HarnessConfig(workflow_mode="evidence")
    legacy = base.to_dict()
    legacy.pop("evidence_recover_python_function_anchors")
    assert HarnessConfig.from_dict(legacy).evidence_recover_python_function_anchors is False
    enabled = replace(base, evidence_recover_python_function_anchors=True)
    assert HarnessConfig.from_dict(enabled.to_dict()) == enabled
    with pytest.raises(ValueError, match="boolean"):
        replace(base, evidence_recover_python_function_anchors=1)
    with pytest.raises(ValueError, match="workflow_mode evidence"):
        HarnessConfig(evidence_recover_python_function_anchors=True)


def test_workflow_constructs_only_evidence_editor_with_persisted_opt_in(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "subject.py").write_text(SOURCE)
    class NoModel:
        model = "offline-no-generation"
        context_size = 8192
        max_output_tokens = 1024
    config = HarnessConfig(workflow_mode="evidence", persist=False,
                           evidence_recover_python_function_anchors=True)
    agent = Agent(NoModel(), config=config, registry=ToolRegistry(Workspace(root, ToolConfig())),
                  runs_dir=tmp_path / "runs", max_steps=3, deadline_seconds=30)
    runtime = EvidenceWorkflow(agent)
    runtime._setup("Change the public function using its current source.")
    try:
        assert runtime.tools.recover_python_function_anchors is True
        result, _ = apply(runtime.tools, textwrap.indent(SOURCE, "    "))
        assert result["success"]
    finally:
        if runtime.session.store:
            runtime.session.store.release()


def test_two_recovered_repair_calls_remain_charged_and_require_real_checks(tmp_path, monkeypatch):
    from patchrook_agent.completion import TaskRequirements
    from test_model_routing_workflow import Coder, Router, action, check, edit, read, route, setup, tool_results
    small = Coder("qwen3:8b", read(), check(), action("finish"))
    before = "    def add(a, b):\n        return a - b\n"
    wrong = "    def add(a, b):\n        return a * b\n"
    after = "    def add(a, b):\n        return a + b\n"
    large = Coder("qwen3:14b", edit(old=before, new=wrong), edit(old=wrong, new=after))
    agent, _, _ = setup(tmp_path, monkeypatch, small, large, router=Router("large"),
        config_overrides={"evidence_recover_python_function_anchors": True})
    state = agent.run("Fix the addition fixture; verify the complete contract.", requirements=TaskRequirements.fix())
    assert state.status == "completed" and state.verification == "passed"
    patches = tool_results(state, "apply_edits")
    assert len(patches) == len(large.requests) == 2
    assert all(item["anchor_recovery"]["applied"] for item in patches)
    assert [item["exit_code"] for item in tool_results(state, "run_check")] == [1, 1, 0]
    repair = route(state)["repair"]
    assert repair["status"] == "verified"
    assert repair["calls_used"] == agent.config.routing_max_repair_calls == 2
    assert "a + b" in (tmp_path / "repo/calc.py").read_text()
