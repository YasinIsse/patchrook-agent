"""Explicit public export requests can acquire guarded package API source."""

from dataclasses import replace
import time

import pytest

from patchrook_agent.evidence_bootstrap import _seed_public_api, seed_task_symbols
from test_evidence_symbol_bootstrap import runtime_factory, tool_results, events


DECLARATION = "class ExistingConverter:\n    pass\n"
API = 'from .implementation import ExistingConverter\n__all__ = ["ExistingConverter"]\n'


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr("httpx.Client", lambda *a, **k: pytest.fail("Bootstrap must not contact models"))


@pytest.mark.parametrize("phrase", ["export", "exports", "exported", "exporting", "top-level", "top level"])
def test_explicit_export_task_preserves_class_and_reads_nearest_package_api(runtime_factory, phrase):
    runtime = runtime_factory(f"Update `ExistingConverter` and its {phrase} public API.", {
        "src/pkg/implementation.py": DECLARATION, "src/pkg/__init__.py": API})
    assert seed_task_symbols(runtime, time.monotonic() + 10) is False
    reads = tool_results(runtime, "read_file")
    assert [read["path"] for read in reads] == ["src/pkg/implementation.py", "src/pkg/__init__.py"]
    assert reads[1]["start_line"] == 1
    assert not runtime.tools._presented
    assert {runtime.evidence.get(key).path for key in runtime.direct} == {
        "src/pkg/implementation.py", "src/pkg/__init__.py"}
    runtime._prepare(runtime.state, runtime.definitions, time.monotonic() + 10)
    assert {"src/pkg/implementation.py", "src/pkg/__init__.py"}.issubset(runtime.tools._presented)


def test_nested_package_uses_nearest_existing_api_only(runtime_factory):
    runtime = runtime_factory("Export `ExistingConverter` at the top-level.", {
        "pkg/sub/implementation.py": DECLARATION, "pkg/sub/__init__.py": "NEAREST = True\n",
        "pkg/__init__.py": "OUTER = True\n"})
    assert seed_task_symbols(runtime, time.monotonic() + 10) is False
    assert [read["path"] for read in tool_results(runtime, "read_file")] == [
        "pkg/sub/implementation.py", "pkg/sub/__init__.py"]


def test_namespace_parent_can_use_nearest_existing_ancestor_api(runtime_factory):
    runtime = runtime_factory("Export `ExistingConverter`.", {
        "pkg/sub/implementation.py": DECLARATION, "pkg/__init__.py": "API = True\n"})
    assert seed_task_symbols(runtime, time.monotonic() + 10) is False
    assert tool_results(runtime, "read_file")[-1]["path"] == "pkg/__init__.py"


@pytest.mark.parametrize("task", ["Update `ExistingConverter`.", "Fix the public `ExistingConverter` class."])
def test_class_only_tasks_keep_existing_bootstrap_reads(runtime_factory, task):
    runtime = runtime_factory(task, {"pkg/implementation.py": DECLARATION, "pkg/__init__.py": API})
    assert seed_task_symbols(runtime, time.monotonic() + 10) is False
    assert [read["path"] for read in tool_results(runtime, "read_file")] == ["pkg/implementation.py"]


def test_no_api_file_skips_extra_read(runtime_factory):
    runtime = runtime_factory("Export `ExistingConverter`.", {"pkg/implementation.py": DECLARATION})
    assert seed_task_symbols(runtime, time.monotonic() + 10) is False
    assert len(tool_results(runtime, "read_file")) == 1


def test_api_source_requires_presentation_and_matching_full_file_hash(runtime_factory):
    runtime = runtime_factory("Export `ExistingConverter`.", {"pkg/implementation.py": DECLARATION,
        "pkg/__init__.py": API})
    assert seed_task_symbols(runtime, time.monotonic() + 10) is False
    action = {"name": "apply_edits", "arguments": {"edits": [{"path": "pkg/__init__.py",
        "old_text": '__all__ = ["ExistingConverter"]', "new_text": '__all__ = []'}]}}
    before = runtime._perform_action(runtime.state, action, time.monotonic() + 10)
    assert before["error"]["code"] == "source_not_presented"
    runtime._prepare(runtime.state, runtime.definitions, time.monotonic() + 10)
    target = runtime.workspace.root / "pkg/__init__.py"
    target.write_text(API + "EXTERNAL = True\n")
    stale = runtime._perform_action(runtime.state, action, time.monotonic() + 10)
    assert stale["error"]["code"] == "stale_source"
    assert target.read_text() == API + "EXTERNAL = True\n"


def test_large_api_read_is_bounded_to_workspace_line_limit(runtime_factory):
    runtime = runtime_factory("Export `ExistingConverter`.", {"pkg/implementation.py": DECLARATION,
        "pkg/__init__.py": "VALUE = 1\n" * 400})
    assert seed_task_symbols(runtime, time.monotonic() + 10) is False
    api = tool_results(runtime, "read_file")[-1]
    assert 1 <= api["end_line"] <= runtime.workspace.config.max_read_lines
    assert api["end_line"] < 400
    request = [event for event in events(runtime) if event["event"] == "tool_request"
               and event["name"] == "read_file"][-1]
    assert request["arguments"]["end_line"] == runtime.workspace.config.max_read_lines


def test_file_byte_limit_stops_before_coding_with_class_read_preserved(runtime_factory):
    runtime = runtime_factory("Export `ExistingConverter`.", {"pkg/implementation.py": DECLARATION,
        "pkg/__init__.py": "VALUE = 1\n" * 400})
    runtime.workspace.config = replace(runtime.workspace.config, max_file_bytes=2048)
    assert seed_task_symbols(runtime, time.monotonic() + 10) is True
    assert runtime.state.stop_reason == "initial_symbol_environment_failure"
    reads = tool_results(runtime, "read_file")
    assert reads[0]["success"] and reads[0]["path"] == "pkg/implementation.py"
    assert reads[1]["error"]["code"] == "file_too_large"
    assert any(runtime.evidence.get(key).path == "pkg/implementation.py" for key in runtime.direct)


def test_expired_api_budget_does_no_extra_tool_work(runtime_factory):
    runtime = runtime_factory("Export `ExistingConverter`.", {"pkg/implementation.py": DECLARATION,
        "pkg/__init__.py": API})
    assert _seed_public_api(runtime, {"path": "pkg/implementation.py"}, time.monotonic() - 1) is True
    assert runtime.state.stop_reason == "deadline" and runtime.state.tool_calls == 0


@pytest.mark.parametrize("path", ["../outside.py", "/tmp/outside.py", "pkg/alias.py"])
def test_untrusted_declaration_paths_cannot_expand_api_reads(runtime_factory, path):
    runtime = runtime_factory("Export `ExistingConverter`.", {"pkg/implementation.py": DECLARATION,
        "pkg/__init__.py": API})
    (runtime.workspace.root / "pkg/alias.py").symlink_to(runtime.workspace.root / "pkg/implementation.py")
    assert _seed_public_api(runtime, {"path": path}, time.monotonic() + 10) is False
    assert runtime.state.tool_calls == 0 and not runtime.direct


def test_symlink_api_is_skipped_without_reading_target(runtime_factory, tmp_path):
    runtime = runtime_factory("Export `ExistingConverter`.", {"pkg/implementation.py": DECLARATION})
    outside = tmp_path / "outside.py"
    outside.write_text("DO_NOT_READ = True\n")
    (runtime.workspace.root / "pkg/__init__.py").symlink_to(outside)
    assert seed_task_symbols(runtime, time.monotonic() + 10) is False
    assert len(tool_results(runtime, "read_file")) == 1
    assert "DO_NOT_READ" not in "\n".join(card.text for card in runtime.evidence.cards(fresh=False))


def test_declaration_already_in_api_does_not_repeat_the_same_file(runtime_factory):
    runtime = runtime_factory("Export `ExistingConverter`.", {"pkg/__init__.py": DECLARATION})
    assert seed_task_symbols(runtime, time.monotonic() + 10) is False
    assert len(tool_results(runtime, "read_file")) == 1
