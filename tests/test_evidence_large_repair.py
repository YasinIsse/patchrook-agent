"""Focused repair packets remain complete under the navigation selector cap."""

import ast
from types import SimpleNamespace
import time

import pytest

from patchrook_agent.evidence import ContextBudgetError, EvidenceStore, RepositoryIndex
from patchrook_agent.evidence_repair import collect_repair_sources
from patchrook_agent.evidence_tools import EvidenceTools
from patchrook_agent.tools import ToolConfig, Workspace
from patchrook_agent.tools.base import ToolError


def subject(root):
    workspace = Workspace(root, ToolConfig(max_read_lines=200))
    return SimpleNamespace(workspace=workspace, tools=EvidenceTools(workspace),
        evidence=EvidenceStore(root), index=RepositoryIndex(root), revision=0,
        query="decoy", task="Repair the observed definition", direct=[], originals={},
        state=SimpleNamespace(modified_files=[]), routing={"repair": {"source_ranges": []}})


def collect(value):
    return collect_repair_sources(value, time.monotonic() + 10)


def crowded_source(target="def target(value):\n    return value + 1\n"):
    return "CONSTANT = 1\n\n" + "\n".join(
        f"def decoy_{number}():\n    return {number}\n" for number in range(90)) + "\n" + target


def target_node(source):
    return next(node for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.FunctionDef) and node.name == "target")


def test_focus_prioritizes_complete_target_without_increasing_selector_cap(tmp_path):
    text = crowded_source()
    (tmp_path / "module.py").write_text(text)
    index = RepositoryIndex(tmp_path)
    plain = index.blocks("module.py", "decoy")
    assert "target" not in {item["symbol"] for item in plain["selectors"]}
    node = target_node(text)
    focused = index.blocks("module.py", "decoy", line_ranges=[(node.lineno + 1, node.lineno + 1)])
    assert len(focused["selectors"]) == len(plain["selectors"]) == 64
    assert focused["reasons"] == ["block_limit"]
    assert {"path": "module.py", "start_line": node.lineno,
            "end_line": node.end_lineno, "symbol": "target"} in focused["selectors"]
    assert any(item["symbol"] == "module_header" for item in focused["selectors"])


def test_repair_acquires_observed_complete_function_from_globally_capped_file(tmp_path):
    text = crowded_source()
    (tmp_path / "module.py").write_text(text)
    value = subject(tmp_path)
    node = target_node(text)
    value.evidence.observe_source("module.py", node.lineno + 1, node.lineno + 1, origin="direct")
    cards = collect(value)
    assert [(card.start_line, card.end_line) for card in cards] == [
        (1, 2), (node.lineno, node.end_lineno)]
    assert cards[-1].text == "def target(value):\n    return value + 1\n"
    assert value.tools._presented == {}


def test_class_header_contains_declaration_docstring_slots_and_excludes_methods(tmp_path):
    text = ("import os\n\n@decorator\nclass Desired(Base):\n    \"\"\"Keep this description.\"\"\"\n"
            "    __slots__ = ('value',)\n\n    @classmethod\n    def create(cls):\n"
            "        return cls()\n" + "\n" * 90)
    (tmp_path / "module.py").write_text(text)
    result = RepositoryIndex(tmp_path).blocks("module.py", "Desired")
    header = result["selectors"][0]
    assert header == {"path": "module.py", "start_line": 3, "end_line": 7,
                      "symbol": "Desired", "kind": "class_header", "scope_end_line": 10}
    assert text.splitlines()[header["end_line"]].strip() == "@classmethod"


def test_class_header_observation_does_not_acquire_an_unrelated_first_function(tmp_path):
    text = ("VALUE = 1\n\ndef unrelated():\n    return 0\n\nclass Desired:\n"
            "    \"\"\"Preserve the public class.\"\"\"\n    __slots__ = ()\n\n"
            "    def existing(self):\n        return 1\n" + "\n" * 90)
    (tmp_path / "module.py").write_text(text)
    value = subject(tmp_path)
    value.evidence.observe_source("module.py", 6, 8, origin="direct")
    cards = collect(value)
    assert [(card.start_line, card.end_line) for card in cards] == [(1, 2), (6, 9)]
    assert "__slots__" in cards[-1].text
    assert "def unrelated" not in "".join(card.text for card in cards)
    assert "def existing" not in "".join(card.text for card in cards)


def test_method_focus_retains_its_own_class_header_in_large_file(tmp_path):
    text = crowded_source("class Desired:\n    __slots__ = ()\n\n    def target(self):\n        return 1\n")
    (tmp_path / "module.py").write_text(text)
    value = subject(tmp_path)
    node = target_node(text)
    value.evidence.observe_source("module.py", node.lineno + 1, node.lineno + 1, origin="direct")
    cards = collect(value)
    assert [(card.start_line, card.end_line) for card in cards] == [
        (1, 2), (node.lineno - 3, node.lineno - 1), (node.lineno, node.end_lineno)]
    assert "class Desired:" in cards[1].text
    assert "def target(self):" in cards[2].text


def test_globally_capped_file_still_rejects_an_oversized_complete_function(tmp_path):
    text = crowded_source("def target(value):\n" + "    value += 1\n" * 205 + "    return value\n")
    (tmp_path / "module.py").write_text(text)
    value = subject(tmp_path)
    node = target_node(text)
    value.evidence.observe_source("module.py", node.lineno, node.lineno + 1, origin="direct")
    with pytest.raises(ContextBudgetError, match="max_read_lines=200"):
        collect(value)
    assert not any(card.metadata["origin"] == "repair_packet" for card in value.evidence.cards())


def test_focused_selector_overflow_rejects_before_recording_a_partial_packet(tmp_path):
    text = crowded_source()
    (tmp_path / "module.py").write_text(text)
    value = subject(tmp_path)
    value.evidence.observe_source("module.py", origin="direct")
    structure = value.index.blocks("module.py", line_ranges=[(1, len(text.splitlines()))])
    assert "focused_block_limit" in structure["reasons"]
    assert len(structure["selectors"]) == 64
    with pytest.raises(ContextBudgetError, match="bounded acquisition limit"):
        collect(value)
    assert not any(card.metadata["origin"] == "repair_packet" for card in value.evidence.cards())


def test_small_nested_target_cannot_hide_an_oversized_enclosing_function(tmp_path):
    text = crowded_source("def outer(value):\n" + "    value += 1\n" * 205
                          + "    def target():\n        return value\n    return target()\n")
    (tmp_path / "module.py").write_text(text)
    value = subject(tmp_path)
    node = target_node(text)
    value.evidence.observe_source("module.py", node.lineno, node.end_lineno, origin="direct")
    with pytest.raises(ContextBudgetError, match="max_read_lines=200"):
        collect(value)
    assert not any(card.metadata["origin"] == "repair_packet" for card in value.evidence.cards())


def test_oversized_class_header_is_rejected_instead_of_clipping_its_docstring(tmp_path):
    text = ("class Desired:\n    \"\"\"\n" + "    Keep this documentation.\n" * 205
            + "    \"\"\"\n    def target(self):\n        return 1\n")
    (tmp_path / "module.py").write_text(text)
    value = subject(tmp_path)
    value.evidence.observe_source("module.py", 1, 2, origin="direct")
    with pytest.raises(ContextBudgetError, match="max_read_lines=200"):
        collect(value)
    assert not any(card.metadata["origin"] == "repair_packet" for card in value.evidence.cards())


@pytest.mark.parametrize("reason", ["deadline", "ast_node_limit"])
def test_incomplete_ast_remains_fail_closed(tmp_path, monkeypatch, reason):
    text = crowded_source()
    (tmp_path / "module.py").write_text(text)
    value = subject(tmp_path)
    value.evidence.observe_source("module.py", 3, 3, origin="direct")
    original = value.index.blocks
    def incomplete(*args, **kwargs):
        result = original(*args, **kwargs)
        return {**result, "truncated": True, "reasons": ["block_limit", reason]}
    monkeypatch.setattr(value.index, "blocks", incomplete)
    with pytest.raises(ContextBudgetError, match="bounded acquisition limit"):
        collect(value)
    assert not any(card.metadata["origin"] == "repair_packet" for card in value.evidence.cards())


def test_focused_source_changed_after_structure_discovery_is_rejected(tmp_path, monkeypatch):
    text = crowded_source()
    path = tmp_path / "module.py"
    path.write_text(text)
    value = subject(tmp_path)
    node = target_node(text)
    value.evidence.observe_source("module.py", node.lineno, node.lineno + 1, origin="direct")
    original = value.index.blocks
    def change_after_discovery(*args, **kwargs):
        result = original(*args, **kwargs)
        path.write_text(text.replace("return value + 1", "return value + 2"))
        return result
    monkeypatch.setattr(value.index, "blocks", change_after_discovery)
    with pytest.raises(ToolError) as error:
        collect(value)
    assert error.value.code == "stale_source"
    assert not any(card.metadata["origin"] == "repair_packet" for card in value.evidence.cards())
