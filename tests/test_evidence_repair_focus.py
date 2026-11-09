"""Repair focus follows actual edits and direct reads, not archived expansion."""

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
        query="target", task="Repair the observed definition", direct=[], originals={},
        state=SimpleNamespace(modified_files=[]), routing={"repair": {"source_ranges": []}})


def collect(value):
    return collect_repair_sources(value, time.monotonic() + 10)


def functions(text):
    return {node.name: node for node in ast.walk(ast.parse(text))
            if isinstance(node, ast.FunctionDef)}


def observe(value, text, name, *, path="module.py", origin="direct"):
    node = functions(text)[name]
    card = value.evidence.observe_source(path, node.lineno, node.end_lineno, origin=origin)
    if origin == "direct":
        value.direct.append(card.id)
    return card


def module():
    return ("CONSTANT = 1\n\nclass Example:\n    __slots__ = ()\n\n"
            "    def target(self):\n        return CONSTANT\n\n"
            "    def ignored(self):\n        return 2\n" + "\n" * 90)


def test_unused_automatic_source_does_not_inflate_direct_function_focus(tmp_path):
    text = module() + "def enormous(value):\n" + "    value += 1\n" * 205 + "    return value\n"
    (tmp_path / "module.py").write_text(text)
    (tmp_path / "unrelated.py").write_text("EXTRA = 4\n")
    value = subject(tmp_path)
    observe(value, text, "target")
    observe(value, text, "enormous", origin="dependency")
    value.evidence.observe_source("unrelated.py", origin="dependency")
    cards = collect(value)
    joined = "".join(card.text for card in cards)
    assert {card.path for card in cards} == {"module.py"}
    assert "def target" in joined and "class Example" in joined
    assert "def enormous" not in joined and "def ignored" not in joined
    assert value.tools._presented == {}


def test_latest_actual_read_can_reuse_an_older_source_card(tmp_path):
    text = module()
    (tmp_path / "module.py").write_text(text)
    value = subject(tmp_path)
    first = observe(value, text, "target")
    observe(value, text, "ignored")
    value.direct.append(first.id)
    joined = "".join(card.text for card in collect(value))
    assert "def target" in joined and "def ignored" not in joined


def test_latest_direct_card_wins_without_a_runtime_read_ledger(tmp_path):
    text = module()
    (tmp_path / "module.py").write_text(text)
    value = subject(tmp_path)
    observe(value, text, "target")
    observe(value, text, "ignored")
    value.direct = []
    observe(value, text, "target", origin="dependency")
    joined = "".join(card.text for card in collect(value))
    assert "def ignored" in joined and "def target" not in joined


def test_all_changed_methods_override_a_later_unmodified_direct_read(tmp_path):
    original = module() + "def second(value):\n    return value - 1\n"
    current = original.replace("return CONSTANT", "return CONSTANT + 1").replace(
        "return value - 1", "return value + 1")
    (tmp_path / "module.py").write_text(current)
    value = subject(tmp_path)
    value.originals["module.py"] = original
    value.state.modified_files = ["module.py"]
    observe(value, current, "ignored")
    joined = "".join(card.text for card in collect(value))
    assert "def target" in joined and "return CONSTANT + 1" in joined
    assert "def second" in joined and "return value + 1" in joined
    assert "def ignored" not in joined


def test_deleted_line_anchors_the_complete_surviving_method(tmp_path):
    original = module().replace("return CONSTANT", "obsolete = 7\n        return CONSTANT")
    current = original.replace("        obsolete = 7\n", "")
    (tmp_path / "module.py").write_text(current)
    value = subject(tmp_path)
    value.originals["module.py"] = original
    observe(value, current, "ignored")
    joined = "".join(card.text for card in collect(value))
    assert "def target(self):\n        return CONSTANT\n" in joined
    assert "def ignored" not in joined


def test_inserted_method_preserves_complete_body_and_its_class_header(tmp_path):
    original = module()
    current = original.replace("    def ignored(self):", "    def added(self, value):\n"
                               "        if value < 0:\n            return -value\n"
                               "        return value\n\n    def ignored(self):")
    (tmp_path / "module.py").write_text(current)
    value = subject(tmp_path)
    value.originals["module.py"] = original
    observe(value, current, "target")
    joined = "".join(card.text for card in collect(value))
    assert "class Example:" in joined and "__slots__" in joined
    assert "def added" in joined and "return -value\n        return value" in joined
    assert "def ignored" not in joined and "def target" not in joined


def test_changed_oversized_function_cannot_hide_behind_a_short_direct_read(tmp_path):
    original = module() + "def enormous(value):\n" + "    value += 1\n" * 205 + "    return value\n"
    current = original.replace("return value\n", "return value + 1\n")
    (tmp_path / "module.py").write_text(current)
    value = subject(tmp_path)
    value.originals["module.py"] = original
    observe(value, current, "target")
    with pytest.raises(ContextBudgetError, match="max_read_lines=200"):
        collect(value)
    assert not any(card.metadata["origin"] == "repair_packet" for card in value.evidence.cards())


def test_helper_import_used_by_focused_method_is_ranked_before_unrelated_imports(tmp_path):
    text = ("import unused_a\nimport unused_b\nimport unused_c\n"
            "from helper import adjust as apply\n\nclass Example:\n    __slots__ = ()\n\n"
            "    def target(self, value):\n        return apply(value)\n" + "\n" * 90)
    (tmp_path / "module.py").write_text(text)
    for name in ("unused_a", "unused_b", "unused_c"):
        (tmp_path / f"{name}.py").write_text("EXTRA = 4\n")
    (tmp_path / "helper.py").write_text("def adjust(value):\n    return value + 1\n")
    value = subject(tmp_path)
    observe(value, text, "target")
    value.evidence.observe_source("unused_c.py", origin="dependency")
    cards = collect(value)
    paths = {card.path for card in cards}
    assert "helper.py" in paths and "unused_c.py" not in paths
    assert len(paths) == 3, "One implementation plus the existing two-import limit"
    assert next(card for card in cards if card.path == "helper.py").text.startswith("def adjust")


def test_old_packet_dependencies_do_not_displace_a_new_direct_focus(tmp_path):
    text = module() + "def enormous(value):\n" + "    value += 1\n" * 205 + "    return value\n"
    (tmp_path / "module.py").write_text(text)
    (tmp_path / "unrelated.py").write_text("EXTRA = 4\n")
    value = subject(tmp_path)
    node = functions(text)["enormous"]
    value.routing["repair"]["source_ranges"] = [
        {"path": "module.py", "start_line": node.lineno, "end_line": node.end_lineno},
        {"path": "unrelated.py", "start_line": 1, "end_line": 1}]
    observe(value, text, "target")
    cards = collect(value)
    assert {card.path for card in cards} == {"module.py"}
    assert "def enormous" not in "".join(card.text for card in cards)


def test_newest_automatic_source_is_a_fallback_when_no_direct_source_exists(tmp_path):
    text = module() + "def enormous(value):\n" + "    value += 1\n" * 205 + "    return value\n"
    (tmp_path / "module.py").write_text(text)
    value = subject(tmp_path)
    observe(value, text, "enormous", origin="dependency")
    observe(value, text, "target", origin="dependency")
    joined = "".join(card.text for card in collect(value))
    assert "def target" in joined and "def enormous" not in joined


def test_source_changed_after_diff_focus_is_computed_is_rejected(tmp_path, monkeypatch):
    original = module()
    current = original.replace("return CONSTANT", "return CONSTANT + 1")
    path = tmp_path / "module.py"
    path.write_text(current)
    value = subject(tmp_path)
    value.originals["module.py"] = original
    observe(value, current, "target")
    original_blocks = value.index.blocks
    def change_before_discovery(*args, **kwargs):
        path.write_text(current.replace("return CONSTANT + 1", "return CONSTANT + 2"))
        return original_blocks(*args, **kwargs)
    monkeypatch.setattr(value.index, "blocks", change_before_discovery)
    with pytest.raises(ToolError) as error:
        collect(value)
    assert error.value.code == "stale_source"
    assert not any(card.metadata["origin"] == "repair_packet" for card in value.evidence.cards())


def test_more_than_four_changed_files_fails_instead_of_silently_omitting_edits(tmp_path):
    value = subject(tmp_path)
    for number in range(5):
        path = f"module_{number}.py"
        (tmp_path / path).write_text("VALUE = 2\n")
        value.originals[path] = "VALUE = 1\n"
    with pytest.raises(ContextBudgetError, match="four implementation files"):
        collect(value)
    assert value.evidence.cards() == []
