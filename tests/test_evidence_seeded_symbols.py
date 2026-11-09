"""Existing method stubs retain their small public result type, without spillover."""

import ast
from dataclasses import replace
import time

import pytest

from patchrook_agent.evidence_bootstrap import seed_task_symbols, small_task_class_ranges
from patchrook_agent.evidence_repair import collect_repair_sources
from test_evidence_symbol_bootstrap import runtime_factory, tool_results


TASK = "Implement `recover` on `Handler` (and top-level). Return and export `Outcome`."
SOURCE = ("class Outcome:\n    def __init__(self, value, complete):\n"
          "        self.value = value\n        self.complete = complete\n\n"
          "    def refine(self, data):\n        pass\n\n\n"
          "class Handler:\n    def recover(self, data):\n        \"\"\"TODO\"\"\"\n"
          "    # The following constructor is unrelated to the requested method.\n"
          "    def __init__(self):\n" + "        unrelated_constructor = 1\n" * 130)


def make(runtime_factory, source=SOURCE, task=TASK):
    return runtime_factory(task, {"src/pkg/core.py": source,
        "src/pkg/__init__.py": "from .core import Handler, Outcome\n"})


def reads(runtime):
    return [row for row in tool_results(runtime, "read_file") if row.get("path") == "src/pkg/core.py"]


def test_existing_method_uses_exact_ast_body_and_adjacent_class_header(runtime_factory):
    runtime = make(runtime_factory)
    assert not seed_task_symbols(runtime, time.monotonic() + 10)
    first = reads(runtime)[0]
    method = next(node for node in ast.walk(ast.parse(SOURCE))
                  if isinstance(node, ast.FunctionDef) and node.name == "recover")
    assert first["start_line"] == method.lineno - 1
    assert first["end_line"] == method.end_lineno
    assert "class Handler:" in first["content"] and "unrelated_constructor" not in first["content"]
    assert not any("unrelated_constructor" in card.text for card in runtime.evidence.cards())


def test_nearby_explicit_result_class_preserves_complete_constructor_and_primary_method(runtime_factory):
    runtime = make(runtime_factory)
    assert not seed_task_symbols(runtime, time.monotonic() + 10)
    latest = reads(runtime)[-1]
    assert "self.complete = complete" in latest["content"]
    assert "def refine(self, data):" in latest["content"]
    assert "def recover(self, data):" in latest["content"]
    assert not runtime.tools._presented, "Source discovery grants no edit authority"
    runtime._prepare(runtime.state, runtime.definitions, time.monotonic() + 10)
    assert "src/pkg/core.py" in runtime.tools._presented


def test_focused_repair_reacquires_unchanged_result_class_after_method_edit(runtime_factory):
    runtime = make(runtime_factory)
    limit = time.monotonic() + 10
    assert not seed_task_symbols(runtime, limit)
    runtime._prepare(runtime.state, runtime.definitions, limit)
    result = runtime._perform_action(runtime.state, {"name": "apply_edits", "arguments": {"edits": [{
        "path": "src/pkg/core.py", "old_text": '        """TODO"""',
        "new_text": '        """Still unfinished"""'}]}}, limit)
    assert result["success"] and result["modified_files"]
    cards = collect_repair_sources(runtime, limit)
    source = "\n".join(card.text for card in cards)
    assert "self.complete = complete" in source and "class Outcome:" in source
    assert "class Handler:" in source and "def recover(self, data):" in source
    assert "unrelated_constructor" not in source


def test_repair_covers_both_same_scope_methods_when_new_stub_shadows_or_is_shadowed(runtime_factory):
    runtime = make(runtime_factory)
    limit = time.monotonic() + 10
    assert not seed_task_symbols(runtime, limit)
    runtime._prepare(runtime.state, runtime.definitions, limit)
    result = runtime._perform_action(runtime.state, {"name": "apply_edits", "arguments": {"edits": [{
        "path": "src/pkg/core.py", "old_text": "class Handler:",
        "new_text": 'class Handler:\n    def recover(self, data):\n        """TODO"""\n        # Recorded stub gap'}]}}, limit)
    assert result["success"]
    cards = collect_repair_sources(runtime, limit)
    source = "\n".join(card.text for card in cards)
    assert source.count("def recover(self, data):") == 2
    assert "unrelated_constructor" not in source
    assert "self.complete = complete" in source
    assert "# Recorded stub gap" in source
    registered = runtime.tools.register_presented([{"path": card.path, "start_line": card.start_line,
        "end_line": card.end_line, "content": card.text, "source_sha256": card.source_sha256} for card in cards])
    assert not registered["rejected"]
    current = (runtime.workspace.root / "src/pkg/core.py").read_text()
    family = current[current.index("class Handler:"):current.index("    # The following")].rstrip("\n")
    before = current
    checked = runtime.tools.execute("apply_edits", {"edits": [{"path": "src/pkg/core.py",
        "old_text": family, "new_text": family}]}, deadline=limit)
    assert checked["error"]["code"] == "no_change", "The unique whole-family anchor passed source/ambiguity guards"
    assert (runtime.workspace.root / "src/pkg/core.py").read_text() == before


def test_changed_source_after_bootstrap_never_authorizes_old_anchor(runtime_factory):
    runtime = make(runtime_factory)
    limit = time.monotonic() + 10
    assert not seed_task_symbols(runtime, limit)
    path = runtime.workspace.root / "src/pkg/core.py"
    path.write_text(SOURCE.replace("self.complete = complete", "self.complete = False"))
    runtime._prepare(runtime.state, runtime.definitions, limit)
    result = runtime.tools.execute("apply_edits", {"edits": [{"path": "src/pkg/core.py",
        "old_text": "self.complete = complete", "new_text": "self.complete = True"}]}, deadline=limit)
    assert not result["success"]
    assert "self.complete = False" in path.read_text()


def test_unmentioned_small_class_is_not_promoted(runtime_factory):
    runtime = make(runtime_factory, task="Implement `recover` on `Handler` (and top-level).")
    assert not seed_task_symbols(runtime, time.monotonic() + 10)
    assert len(reads(runtime)) == 1
    assert "class Outcome:" not in reads(runtime)[0]["content"]


def test_distant_method_does_not_copy_intervening_methods(runtime_factory):
    source = "class Handler:\n    def unrelated(self):\n" + "        unrelated_marker = 1\n" * 30
    source += "\n    def recover(self, data):\n        return data\n"
    runtime = make(runtime_factory, source, "Implement `recover` on `Handler`.")
    assert not seed_task_symbols(runtime, time.monotonic() + 10)
    assert "unrelated_marker" not in reads(runtime)[0]["content"]
    assert reads(runtime)[0]["end_line"] - reads(runtime)[0]["start_line"] == 1


@pytest.mark.parametrize("prefix", ["def", "async def"])
def test_decorated_function_acquires_complete_body_without_next_definition(runtime_factory, prefix):
    source = f"@decorate\n{prefix} recover(data):\n    return data\n\ndef unrelated():\n    HIDDEN = 1\n"
    runtime = make(runtime_factory, source, "Update `recover`.")
    assert not seed_task_symbols(runtime, time.monotonic() + 10)
    observed = reads(runtime)[0]
    assert observed["start_line"] == 1 and observed["end_line"] == 3
    assert "HIDDEN" not in observed["content"]


def test_oversized_function_is_explicit_bounded_prefix_without_authority(runtime_factory):
    source = "def recover(data):\n" + "    result = data\n" * 45 + "    return result\n"
    runtime = make(runtime_factory, source, "Update `recover`.")
    runtime.workspace.config = replace(runtime.workspace.config, max_read_lines=20)
    assert not seed_task_symbols(runtime, time.monotonic() + 10)
    assert "only a declaration prefix" in runtime.correction
    assert "remaining current body" in runtime.correction
    assert not runtime.tools._presented
    assert reads(runtime)[0]["end_line"] == 20


@pytest.mark.parametrize("last,maximum,expected", [(40,40,[(1,40)]),(41,40,[]),(8,7,[]),(8,8,[(1,8)])])
def test_small_task_class_range_never_clips_a_large_body(last, maximum, expected):
    structure = {"selectors": [{"symbol": "Outcome", "kind": "class_header", "start_line": 1,
                                "end_line": 1, "scope_end_line": last}]}
    assert small_task_class_ranges("Return `Outcome`.", structure, maximum=maximum) == expected


def test_qualified_name_and_plain_prose_are_not_new_symbol_reads():
    structure = {"selectors": [{"symbol": "Outcome", "kind": "class_header", "start_line": 1,
                                "end_line": 1, "scope_end_line": 5}]}
    assert small_task_class_ranges("Return package.Outcome and `package.Outcome`.", structure) == []
