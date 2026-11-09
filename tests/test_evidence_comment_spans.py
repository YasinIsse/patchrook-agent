"""Comment-only model edits must remain in complete, guarded repair source."""

import ast
from dataclasses import replace
import time

import pytest

from patchrook_agent.evidence import ContextBudgetError, RepositoryIndex
from patchrook_agent.evidence_repair import collect_repair_sources
from test_evidence_repair_focus import subject


SOURCE = ('class Result:\n    def refine(self, data):\n        return data\n\n\n'
          'class Handler:\n    def recover(self, data):\n        """Unimplemented."""\n'
          '        # body comment\n        # original implementation placeholder\n'
          '    """Class documentation follows."""\n    __slots__ = ()\n\n'
          '    def __init__(self):\n' + '        unrelated_constructor = True\n' * 95)


def selector(index, symbol, **kwargs):
    return next(item for item in index.blocks("module.py", **kwargs)["selectors"]
                if item.get("symbol") == symbol)


def test_function_tail_includes_indented_comments_but_no_class_statement_or_next_constructor(tmp_path):
    (tmp_path / "module.py").write_text(SOURCE)
    item = selector(RepositoryIndex(tmp_path), "recover", line_ranges=((9, 9),))
    assert (item["start_line"], item["end_line"]) == (7, 10)
    selected = "\n".join(SOURCE.splitlines()[item["start_line"] - 1:item["end_line"]])
    assert "# original implementation placeholder" in selected
    assert "Class documentation follows" not in selected and "unrelated_constructor" not in selected


def test_blank_lines_between_indented_comments_are_retained(tmp_path):
    source = "def target():\n    pass\n\n    # first\n\n    # second\n\ndef next_function():\n    pass\n"
    (tmp_path / "module.py").write_text(source)
    assert selector(RepositoryIndex(tmp_path), "target")["end_line"] == 6


def test_same_indent_comment_stops_body_extension(tmp_path):
    source = "class Target:\n    def method(self):\n        pass\n    # belongs to the class\n        # never cross the outer comment\n"
    (tmp_path / "module.py").write_text(source)
    assert selector(RepositoryIndex(tmp_path), "method")["end_line"] == 3


def test_blank_only_file_tail_is_not_promoted_into_function_body(tmp_path):
    (tmp_path / "module.py").write_text("def target():\n    pass\n" + "\n" * 90)
    assert selector(RepositoryIndex(tmp_path), "target")["end_line"] == 2


def test_tab_indentation_compares_visual_prefix_only(tmp_path):
    source = "class Target:\n\tdef method(self):\n\t\tpass\n\t\t# body\tcontent\n        # class indentation equals the method\n"
    (tmp_path / "module.py").write_text(source)
    assert selector(RepositoryIndex(tmp_path), "method")["end_line"] == 4


def test_comment_only_changed_method_has_complete_fresh_body_and_owner_header(tmp_path):
    current = SOURCE.replace("# body comment", "# newly added model comment")
    (tmp_path / "module.py").write_text(current)
    runtime = subject(tmp_path)
    runtime.originals["module.py"] = SOURCE
    runtime.state.modified_files = ["module.py"]
    runtime.query = "refine"
    cards = collect_repair_sources(runtime, time.monotonic() + 10)
    text = "\n".join(card.text for card in cards)
    assert "class Handler:" in text and "def recover(self, data):" in text
    assert "# newly added model comment" in text and "# original implementation placeholder" in text
    assert "def refine" not in text and "unrelated_constructor" not in text
    assert not runtime.tools._presented


def test_omitted_changed_region_fails_before_any_repair_authority_is_recorded(tmp_path, monkeypatch):
    current = SOURCE.replace("# body comment", "# changed comment")
    (tmp_path / "module.py").write_text(current)
    runtime = subject(tmp_path)
    runtime.originals["module.py"] = SOURCE
    original_blocks = runtime.index.blocks
    def incomplete(*args, **kwargs):
        result = original_blocks(*args, **kwargs)
        result["selectors"] = [item for item in result["selectors"] if item.get("symbol") == "refine"]
        return result
    monkeypatch.setattr(runtime.index, "blocks", incomplete)
    with pytest.raises(ContextBudgetError, match="Changed repair source.*missing"):
        collect_repair_sources(runtime, time.monotonic() + 10)
    assert not runtime.tools._presented
    assert not any(card.metadata.get("origin") == "repair_packet" for card in runtime.evidence.cards())


def test_complete_comment_body_still_obeys_read_limit_without_clipping(tmp_path):
    original = "def target():\n    pass\n" + "    # tail\n" * 50 + "\n" * 40
    current = original.replace("    # tail", "    # edited tail", 1)
    (tmp_path / "module.py").write_text(current)
    runtime = subject(tmp_path)
    runtime.workspace.config = replace(runtime.workspace.config, max_read_lines=20)
    runtime.originals["module.py"] = original
    with pytest.raises(ContextBudgetError, match="max_read_lines"):
        collect_repair_sources(runtime, time.monotonic() + 10)
    assert not runtime.evidence.cards()
