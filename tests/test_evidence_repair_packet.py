"""Repair source acquisition uses real files and ASTs without model inference."""

from types import SimpleNamespace
import time

import pytest

from patchrook_agent.evidence import ContextBudgetError, EvidenceStore, RepositoryIndex
from patchrook_agent.evidence_repair import collect_repair_sources
from patchrook_agent.evidence_tools import EvidenceTools
from patchrook_agent.tools import ToolConfig, Workspace
from patchrook_agent.tools.base import ToolError


RESOLVER = '''from tokens import TOKEN

# Keep unknown placeholders intact and preserve the display helper.

def expand(text, values):
    def replace(match):
        name, default = match.groups()
        value = values.get(name)
        if default is None:
            return match.group(0) if value is None else value
        return value or default
    return TOKEN.sub(replace, text)

def display_label(name):
    return "config:" + name
'''
TOKENS = 'import re\nTOKEN = re.compile(r"\\$\\{([A-Z_]+)(?::-([^}]+))?\\}")\n'


def workflow(tmp_path, *, max_read_lines=200):
    workspace = Workspace(tmp_path, ToolConfig(max_read_lines=max_read_lines))
    return SimpleNamespace(workspace=workspace, tools=EvidenceTools(workspace),
        evidence=EvidenceStore(tmp_path), index=RepositoryIndex(tmp_path), revision=0,
        query="expand interpolation fallback", task="Fix interpolation", direct=[], originals={},
        state=SimpleNamespace(modified_files=[]), routing={"repair": {"source_ranges": []}})


def collect(subject):
    return collect_repair_sources(subject, time.monotonic() + 10)


def test_partial_function_observation_expands_to_complete_small_file_and_import(tmp_path):
    (tmp_path / "resolver.py").write_text(RESOLVER)
    (tmp_path / "tokens.py").write_text(TOKENS)
    subject = workflow(tmp_path)
    card = subject.evidence.observe_source("resolver.py", 5, 10, origin="direct")
    subject.direct.append(card.id)
    result = collect(subject)
    by_path = {card.path: card for card in result}
    assert set(by_path) == {"resolver.py", "tokens.py"}
    assert by_path["resolver.py"].text == RESOLVER
    assert "return value or default" in by_path["resolver.py"].text
    assert by_path["tokens.py"].text == TOKENS
    assert all(card.metadata["origin"] == "repair_packet" for card in result)
    assert subject.tools._presented == {}, "Acquisition alone must not authorize edits"


def test_large_file_packet_keeps_full_enclosing_function_and_header(tmp_path):
    padding = "\n" * 82
    source = "CONSTANT = 3\n\ndef target(value):\n    if value < 0:\n        return -value\n    return value + CONSTANT\n" + padding
    source += "def unrelated():\n    return 'keep'\n"
    (tmp_path / "module.py").write_text(source)
    subject = workflow(tmp_path)
    subject.query = "target"
    subject.evidence.observe_source("module.py", 3, 4, origin="direct")
    result = collect(subject)
    assert [(card.start_line, card.end_line) for card in result] == [(1, 2), (3, 6)]
    assert "return value + CONSTANT" in result[-1].text
    assert "unrelated" not in "".join(card.text for card in result)


def test_nested_function_hints_return_complete_outer_function_without_duplicate_ranges(tmp_path):
    source = ("def outer(value):\n    def inner(number):\n        return number + 1\n"
              "    return inner(value)\n" + "\n" * 85)
    (tmp_path / "module.py").write_text(source)
    subject = workflow(tmp_path)
    subject.evidence.observe_source("module.py", 2, 3, origin="direct")
    subject.evidence.observe_source("module.py", 1, 4, origin="dependency")
    result = collect(subject)
    assert len(result) == 1
    assert result[0].start_line == 1 and result[0].end_line == 4


def test_retry_refreshes_stale_packet_from_current_source_after_edit(tmp_path):
    (tmp_path / "resolver.py").write_text(RESOLVER)
    (tmp_path / "tokens.py").write_text(TOKENS)
    subject = workflow(tmp_path)
    original_card = subject.evidence.observe_source("resolver.py", 5, 10, origin="direct")
    first = collect(subject)
    subject.routing["repair"]["source_ranges"] = [
        {"path": card.path, "start_line": card.start_line, "end_line": card.end_line} for card in first]
    patched = RESOLVER.replace("return value or default", "return default if value is None else value")
    (tmp_path / "resolver.py").write_text(patched)
    subject.revision = 1
    assert not subject.evidence.is_fresh(original_card)
    result = collect(subject)
    resolver = next(card for card in result if card.path == "resolver.py")
    assert resolver.text == patched and resolver.revision == 1
    assert resolver.source_sha256 != original_card.source_sha256
    assert subject.tools._presented == {}


def test_observed_test_can_seed_imported_implementation_but_is_not_editable_packet_source(tmp_path):
    (tmp_path / "resolver.py").write_text(RESOLVER)
    (tmp_path / "tokens.py").write_text(TOKENS)
    (tmp_path / "test_resolver.py").write_text("from resolver import expand\n\ndef test_expand():\n    assert expand('x', {}) == 'x'\n")
    subject = workflow(tmp_path)
    subject.evidence.observe_source("test_resolver.py", origin="direct")
    result = collect(subject)
    assert {card.path for card in result} == {"resolver.py", "tokens.py"}


def test_absent_implementation_evidence_returns_no_packet_and_does_not_use_task_or_oracle_paths(tmp_path):
    (tmp_path / "secret_solution.py").write_text("ANSWER = 42\n")
    subject = workflow(tmp_path)
    subject.task = "Modify secret_solution.py"
    subject.config = SimpleNamespace(evidence_oracle_locations=(("secret_solution.py", 1, 1),))
    subject.evidence.observe_source("secret_solution.py", origin="diagnostic_oracle")
    assert collect(subject) == []
    assert not any(card.metadata.get("origin") == "repair_packet" for card in subject.evidence.cards())


def test_oversized_function_rejected_instead_of_clipped(tmp_path):
    source = "def target(value):\n" + "    value += 1\n" * 205 + "    return value\n"
    (tmp_path / "module.py").write_text(source)
    subject = workflow(tmp_path)
    subject.evidence.observe_source("module.py", 1, 5, origin="direct")
    with pytest.raises(ContextBudgetError, match="max_read_lines=200"):
        collect(subject)
    assert not any(card.metadata.get("origin") == "repair_packet" for card in subject.evidence.cards())


@pytest.mark.parametrize("path", ["../outside.py", ".git/config", "/absolute.py"])
def test_protected_previous_packet_selectors_fail_at_existing_path_guard(tmp_path, path):
    subject = workflow(tmp_path)
    subject.routing["repair"]["source_ranges"] = [{"path": path, "start_line": 1, "end_line": 1}]
    with pytest.raises(ToolError):
        collect(subject)


def test_symlink_previous_packet_selector_cannot_rehydrate_external_source(tmp_path):
    outside = tmp_path.parent / (tmp_path.name + "-outside.py")
    outside.write_text("SECRET = 42\n")
    (tmp_path / "linked.py").symlink_to(outside)
    subject = workflow(tmp_path)
    subject.routing["repair"]["source_ranges"] = [{"path": "linked.py", "start_line": 1, "end_line": 1}]
    with pytest.raises(ToolError, match="Symlink"):
        collect(subject)
    assert subject.evidence.cards() == []


def test_source_and_dependency_acquisition_are_bounded(tmp_path):
    subject = workflow(tmp_path)
    for number in range(7):
        (tmp_path / f"module{number}.py").write_text(f"VALUE = {number}\n")
        subject.evidence.observe_source(f"module{number}.py", origin="direct")
    result = collect(subject)
    assert len({card.path for card in result}) == 4
    assert {card.path for card in result} == {f"module{number}.py" for number in (6, 5, 4, 3)}


def test_only_two_new_static_dependencies_are_acquired(tmp_path):
    subject = workflow(tmp_path)
    (tmp_path / "module.py").write_text("import first\nimport second\nimport third\nVALUE = 0\n")
    for name in ("first", "second", "third"):
        (tmp_path / f"{name}.py").write_text("VALUE = 1\n")
    subject.evidence.observe_source("module.py", origin="direct")
    assert {card.path for card in collect(subject)} == {"module.py", "first.py", "second.py"}


def test_more_than_sixteen_complete_ranges_is_rejected_before_recording_partial_packet(tmp_path):
    subject = workflow(tmp_path)
    source = "\n\n".join(f"def function{number}():\n    return {number}\n" for number in range(20))
    source += "\n" * 20
    (tmp_path / "module.py").write_text(source)
    subject.evidence.observe_source("module.py", origin="direct")
    with pytest.raises(ContextBudgetError, match="sixteen"):
        collect(subject)
    assert not any(card.metadata.get("origin") == "repair_packet" for card in subject.evidence.cards())


def test_changed_source_precedes_other_observed_files(tmp_path):
    subject = workflow(tmp_path)
    for number in range(6):
        (tmp_path / f"module{number}.py").write_text(f"VALUE = {number}\n")
        subject.evidence.observe_source(f"module{number}.py", origin="direct")
    subject.originals["module0.py"] = "VALUE = 0\n"
    (tmp_path / "module0.py").write_text("VALUE = 100\n")
    result = collect(subject)
    assert result[0].path == "module0.py" and result[0].text == "VALUE = 100\n"


def test_elapsed_task_deadline_stops_before_source_acquisition(tmp_path):
    subject = workflow(tmp_path)
    with pytest.raises(ToolError) as error:
        collect_repair_sources(subject, time.monotonic() - 1)
    assert error.value.code == "deadline_exceeded"
    assert subject.evidence.cards() == []

