"""Declaration and read expansion stays inside the observed source scope."""

import ast
from dataclasses import replace
import sys
import time

import pytest

from patchrook_agent.agent import Agent, AgentState
from patchrook_agent.completion import TaskRequirements
from patchrook_agent.config import HarnessConfig
from patchrook_agent.evidence_bootstrap import seed_task_symbols
from patchrook_agent.evidence_workflow import EvidenceWorkflow
from patchrook_agent.tools import ToolConfig, ToolRegistry, Workspace


TASK = "Add `partial_structure` to `Target` with nested fields, defaults, errors, collections and validation."
SOURCE = ("CONSTANT = 1\n\nclass Target:\n    \"\"\"Keep the requested class.\"\"\"\n    __slots__ = ()\n\n"
          "    def relevant(self, value):\n        # guard_value must survive.\n"
          "        if value < 0:\n            raise ValueError('negative')\n        return value\n\n"
          "    def other(self, value):\n        if value == 0:\n            return CONSTANT\n"
          "        return value + 1\n\nclass Unrelated:\n    def __init__(self):\n"
          + "        constructor_only = 'nested fields defaults errors collections validation'\n" * 145)


class NoModel:
    model = "offline-expansion-focus"
    context_size = 8192
    max_output_tokens = 512
    def chat(self, *args, **kwargs):
        pytest.fail("Expansion checks must not call a model")


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Expansion checks must not access model endpoints")
    monkeypatch.setattr("httpx.Client", forbidden)
    root = tmp_path / "repository"
    (root / "src/pkg").mkdir(parents=True)
    (root / "src/pkg/core.py").write_text(SOURCE)
    (root / "src/aa_wrapper.py").write_text("from pkg.core import Target\n")
    base = HarnessConfig.preset("C")
    config = replace(base, workflow_mode="evidence", action_protocol="schema",
        evidence_seed_symbols=True, review_mode="off", persist=False, checkpoint_enabled=False,
        compaction_enabled=False, context=replace(base.context, enabled=False),
        evidence_test_commands=((sys.executable, "-c", "pass"),))
    agent = Agent(NoModel(), registry=ToolRegistry(Workspace(root, ToolConfig(allow_tests=True))),
                  config=config, runs_dir=tmp_path / "runs", max_steps=4, deadline_seconds=30)
    value = EvidenceWorkflow(agent)
    value.requirements = TaskRequirements.fix()
    value._setup(TASK)
    value.definitions = value._definitions()
    value.enabled = {item["function"]["name"] for item in value.definitions}
    value.state = AgentState()
    value.session.bind(value.state)
    value._prime_inventory(time.monotonic() + 10)
    return value


def code_cards(runtime):
    return [card for card in runtime.evidence.cards()
            if card.kind == "source" and card.path == "src/pkg/core.py"]


def perform(runtime, name, **arguments):
    return runtime._perform_action(runtime.state, {"name": name, "arguments": arguments},
                                   time.monotonic() + 10, automatic=True)


def node(name):
    return next(item for item in ast.walk(ast.parse(SOURCE))
                if isinstance(item, ast.FunctionDef) and item.name == name)


def test_complete_symbol_recovery_and_declaration_read_exclude_unrelated_constructor(runtime):
    assert not seed_task_symbols(runtime, time.monotonic() + 10)
    cards = code_cards(runtime)
    assert any(card.metadata["origin"] == "direct" and "class Target:" in card.text for card in cards)
    assert not any("constructor_only" in card.text for card in cards)
    assert not any("def __init__" in card.text for card in cards)
    assert runtime.query == TASK, "Completed auxiliary lookups must leave the coding objective intact"
    assert runtime.tools._presented == {}, "Acquisition is not edit authority"


def test_class_header_read_with_full_task_query_excludes_unrelated_semantic_match(runtime):
    runtime.query = TASK
    result = perform(runtime, "read_file", path="src/pkg/core.py", start_line=3, end_line=6)
    assert result["success"]
    cards = code_cards(runtime)
    assert any("__slots__" in card.text for card in cards)
    assert not any("constructor_only" in card.text for card in cards)


def test_search_body_hit_expands_whole_enclosing_function_and_class(runtime):
    result = perform(runtime, "search_code", path="src/pkg/core.py", mode="literal",
                     pattern="guard_value", max_results=1)
    assert result["success"]
    expected = node("relevant")
    cards = code_cards(runtime)
    body = next(card for card in cards if card.start_line == expected.lineno)
    assert body.end_line == expected.end_lineno
    assert "raise ValueError('negative')" in body.text and "return value" in body.text
    assert any("class Target:" in card.text for card in cards)
    assert not any("constructor_only" in card.text for card in cards)


def test_new_read_range_invalidates_expansion_cache_even_with_identical_question(runtime):
    for name in ("relevant", "other"):
        expected = node(name)
        result = perform(runtime, "read_file", path="src/pkg/core.py", start_line=expected.lineno + 1,
                         end_line=expected.lineno + 1, question="Inspect the same public class")
        assert result["success"]
        assert any(card.start_line == expected.lineno and card.end_line == expected.end_lineno
                   for card in code_cards(runtime))
    assert not any("constructor_only" in card.text for card in code_cards(runtime))


def test_omitted_constructor_cannot_authorize_an_existing_file_edit(runtime):
    perform(runtime, "read_file", path="src/pkg/core.py", start_line=3, end_line=6)
    runtime._prepare(runtime.state, runtime.definitions, time.monotonic() + 10)
    result = runtime.tools.execute("apply_edits", {"edits": [{"path": "src/pkg/core.py",
        "old_text": "constructor_only", "new_text": "changed"}]}, deadline=time.monotonic() + 10)
    assert not result["success"]
    assert "constructor_only" in (runtime.workspace.root / "src/pkg/core.py").read_text()
