"""Inspection coverage and guidance use only complete lines visibly returned."""

import json

import pytest
from pydantic import ValidationError

from patchrook_agent.completion import inspection_observed
from patchrook_agent.inspection import InspectionLedger, source_read_boundaries, visible_lines
from patchrook_agent.memory.state import WorkingMemory
from patchrook_agent.progress import pending_local_imports, pending_read_selections
from patchrook_agent.tools import Workspace


def exchange(name, arguments, result, identifier="read-1"):
    return [{"role": "assistant", "content": "", "tool_calls": [{"id": identifier, "function": {
        "name": name, "arguments": arguments}}]},
        {"role": "tool", "tool_name": name, "tool_call_id": identifier, "content": json.dumps(result)}]


def prefix(path="module.py"):
    return {"success": True, "path": path, "start_line": 1, "end_line": 2,
            "total_lines": 15, "content": "1: # Preserve the module\n2: # More documentation",
            "truncated": True, "next_start_line": 3}


def narrow(path="module.py"):
    return {"success": True, "path": path, "start_line": 14, "end_line": 15,
            "total_lines": 15, "content": "14: def transform(value):\n15:     return value + 1",
            "truncated": False}


def test_comment_prefix_is_partial_evidence_with_a_real_continuation(tmp_path):
    (tmp_path / "module.py").write_text("must never be read by coverage helpers")
    messages = exchange("read_file", {"path": "module.py", "end_line": 100}, prefix())
    ledger = InspectionLedger.from_messages(messages)
    evidence = ledger.files["module.py"]
    assert evidence.visible_ranges == [[1, 2]] and not evidence.code_like_ranges
    assert evidence.pending_ranges == [[3, 15]] and not evidence.inspected
    assert not inspection_observed(messages)
    assert pending_read_selections(messages, Workspace(tmp_path)) == [
        {"path": "module.py", "start_line": 3, "end_line": 15}]


def test_complete_narrow_code_range_can_replace_a_broad_partial_request(tmp_path):
    (tmp_path / "module.py").write_text("unread contents")
    messages = exchange("read_file", {"path": "module.py", "end_line": 100}, prefix())
    messages += exchange("read_file", {"path": "module.py", "start_line": 14, "end_line": 15}, narrow(), "read-2")
    ledger = InspectionLedger.from_messages(messages)
    evidence = ledger.files["module.py"]
    assert evidence.visible_ranges == [[1, 2], [14, 15]]
    assert evidence.complete_selections == [[14, 15]] and not evidence.pending_ranges
    assert evidence.inspected and inspection_observed(messages)
    assert not pending_read_selections(messages, Workspace(tmp_path))
    assert "transform" not in json.dumps(ledger.to_dict())


def test_narrow_selection_does_not_require_reading_a_large_file():
    record = {**narrow(), "total_lines": 1_000_000}
    ledger = InspectionLedger.from_messages(exchange("read_file", {
        "path": "module.py", "start_line": 14, "end_line": 15}, record))
    assert ledger.files["module.py"].inspected
    assert not ledger.files["module.py"].pending_ranges


def test_read_code_selected_function_is_complete_without_whole_file_metadata():
    record = {key: value for key, value in narrow().items() if key != "total_lines"}
    record["omitted_ranges"] = [[1, 13], [16, 200]]
    messages = exchange("read_code", {"files": [{"path": "module.py", "symbol": "transform"}]},
                        {"success": True, "selection": "code", "files": [record]})
    evidence = InspectionLedger.from_messages(messages).files["module.py"]
    assert evidence.inspected and evidence.visible_ranges == [[14, 15]]
    assert not evidence.pending_ranges and inspection_observed(messages)
    memory = WorkingMemory(session_id="code", objective="Inspect function")
    memory.observe_tool("read_code", {"files": [{"path": "module.py"}]},
                        {"success": True, "files": [record]}, 1)
    assert memory.inspection.files["module.py"].inspected
    assert memory.relevant_files == ["module.py"]


def test_clipped_read_code_keeps_chosen_endpoint_pending():
    record = {"success": True, "path": "module.py", "start_line": 14, "end_line": 14,
              "selected_end_line": 15, "content": "14: def transform(value):", "truncated": True}
    messages = exchange("read_code", {"files": [{"path": "module.py", "symbol": "transform"}]},
                        {"success": True, "files": [record], "truncated": True})
    evidence = InspectionLedger.from_messages(messages).files["module.py"]
    assert not evidence.inspected and evidence.pending_ranges == [[15, 15]]


def test_omitted_read_code_file_stays_pending_without_full_source_capture():
    record = {key: value for key, value in narrow().items() if key != "total_lines"}
    arguments = {"files": [{"path": "module.py"}, {"path": "other.py", "start_line": 9, "end_line": 10}]}
    messages = exchange("read_code", arguments, {"success": True, "files": [record], "omitted_file_count": 1})
    ledger = InspectionLedger.from_messages(messages)
    assert ledger.files["module.py"].inspected
    assert ledger.files["other.py"].pending_ranges == [[9, 10]]
    assert not ledger.files["other.py"].inspected


def test_read_code_path_only_batch_tracks_omitted_paths():
    record = {key: value for key, value in narrow().items() if key != "total_lines"}
    arguments = {"paths": ["module.py", "other.py"]}
    messages = exchange("read_code", arguments, {"success": True, "files": [record], "omitted_file_count": 1})
    ledger = InspectionLedger.from_messages(messages)
    assert ledger.files["module.py"].inspected
    assert ledger.files["other.py"].pending_ranges == [[1, None]]
    assert not ledger.files["other.py"].inspected


def test_narrow_read_supersedes_only_its_containing_request_not_disjoint_omitted_selection():
    arguments = {"files": [{"path": "module.py", "start_line": 1, "end_line": 10},
                            {"path": "module.py", "start_line": 100, "end_line": 110}]}
    first = {"success": True, "path": "module.py", "start_line": 1, "end_line": 3,
             "total_lines": 200, "content": "1: # Header\n2: def first():\n3:     return 1", "truncated": True}
    messages = exchange("read_files", arguments, {"success": True, "files": [first],
                        "truncated": True, "omitted_file_count": 1, "next_file_index": 1})
    ledger = InspectionLedger.from_messages(messages)
    assert ledger.files["module.py"].pending_requests == [[1, 10], [100, 110]]
    assert ledger.files["module.py"].pending_ranges == [[4, 10], [100, 110]]
    # Lines 2–3 were already in the visible prefix. Their explicit complete
    # reread changes focus without cancelling the separate omitted selection.
    narrowed = {**first, "start_line": 2, "end_line": 3,
                "content": "2: def first():\n3:     return 1", "truncated": False}
    messages += exchange("read_file", {"path": "module.py", "start_line": 2, "end_line": 3}, narrowed, "narrow")
    ledger = InspectionLedger.from_messages(messages)
    evidence = ledger.files["module.py"]
    assert evidence.inspected
    assert evidence.pending_requests == [[100, 110]]
    assert evidence.pending_ranges == [[100, 110]]
    assert ledger.pending_selections() == [{"path": "module.py", "start_line": 100, "end_line": 110}]
    assert InspectionLedger.from_dict(ledger.to_dict()) == ledger


@pytest.mark.parametrize("record", [
    {**narrow(), "partial_line": True, "truncated": True,
     "start_line": 14, "end_line": 14, "content": "14: def trans"},
    {"success": True, "path": "module.py", "total_lines": 15, "truncated": True},
    {**narrow(), "content": "13: wrong declared range"},
    {**narrow(), "content": "source without numbered lines"},
    {**narrow(), "content": "²: invalid line number"},
])
def test_partial_lines_and_metadata_are_not_inspection(record):
    messages = exchange("read_file", {"path": "module.py", "start_line": 14, "end_line": 15}, record)
    assert not inspection_observed(messages)
    assert not visible_lines(record)
    assert InspectionLedger.from_messages(messages).files["module.py"].pending_ranges


def test_metadata_only_empty_file_is_not_credited_but_an_observed_empty_file_is():
    metadata = {"success": True, "path": "empty.py", "total_lines": 0}
    assert not inspection_observed(exchange("read_file", {"path": "empty.py"}, metadata))
    assert inspection_observed(exchange("read_file", {"path": "empty.py"}, {**metadata, "content": ""}))


def test_full_documentation_selection_remains_legitimate_inspection():
    record = {"success": True, "path": "NOTES.md", "start_line": 1, "end_line": 2,
              "total_lines": 20, "content": "1: # Instructions\n2: // A selected note", "truncated": False}
    messages = exchange("read_file", {"path": "NOTES.md", "start_line": 1, "end_line": 2}, record)
    assert inspection_observed(messages)


def test_omitted_batch_selection_is_pending_without_invented_source(tmp_path):
    for name in ("a.py", "b.py"):
        (tmp_path / name).write_text("unread contents")
    args = {"files": [{"path": "a.py", "start_line": 14, "end_line": 15},
                      {"path": "b.py", "start_line": 10, "end_line": 12}]}
    result = {"success": True, "files": [narrow("a.py")], "truncated": True,
              "omitted_file_count": 1, "next_file_index": 1}
    ledger = InspectionLedger.from_messages(exchange("read_files", args, result))
    assert ledger.files["a.py"].inspected
    assert not ledger.files["b.py"].visible_ranges and not ledger.files["b.py"].inspected
    assert ledger.files["b.py"].pending_ranges == [[10, 12]]
    assert pending_read_selections(exchange("read_files", args, result), Workspace(tmp_path)) == [
        {"path": "b.py", "start_line": 10, "end_line": 12}]


def test_import_with_partial_comments_stays_pending_until_code_is_observed(tmp_path, monkeypatch):
    (tmp_path / "module.py").write_text("raise RuntimeError('do not execute or read this')")
    importer = {"success": True, "path": "entry.py", "content": "1: from module import transform"}
    messages = exchange("read_file", {"path": "entry.py"}, importer)
    messages += exchange("read_file", {"path": "module.py", "end_line": 100}, prefix(), "prefix")
    monkeypatch.setattr(type(tmp_path), "read_text", lambda *args, **kwargs: pytest.fail("Hidden source read"))
    assert pending_local_imports(messages, Workspace(tmp_path)) == ["module.py"]
    messages += exchange("read_file", {"path": "module.py", "start_line": 14, "end_line": 15}, narrow(), "code")
    assert not pending_local_imports(messages, Workspace(tmp_path))


def test_observed_edit_removes_stale_import_guidance_without_changing_history(tmp_path):
    for path in ("a.py", "old_dep.py"):
        (tmp_path / path).write_text("contents are not inspection evidence")
    workspace = Workspace(tmp_path)
    read = {"success": True, "path": "a.py", "start_line": 1, "end_line": 1,
            "content": "1: import old_dep", "truncated": False}
    messages = exchange("read_file", {"path": "a.py", "start_line": 1, "end_line": 1}, read)
    assert pending_local_imports(messages, workspace) == ["old_dep.py"]
    messages += exchange("edit_file", {"path": "a.py", "old_text": "import old_dep", "new_text": "value = 1"},
                         {"success": True, "path": "a.py", "modified": True}, "edit")
    original = json.dumps(messages)
    boundaries = source_read_boundaries(messages, workspace)
    assert boundaries == {"a.py": 4}
    assert not pending_local_imports(messages, workspace, source_boundaries=boundaries)
    assert not InspectionLedger.from_messages(messages, workspace, source_boundaries=boundaries).files
    assert json.dumps(messages) == original
    # The optional behavior does not change historical auditing defaults.
    assert pending_local_imports(messages, workspace) == ["old_dep.py"]


def test_external_source_change_makes_task_path_unread_until_a_fresh_read(tmp_path):
    (tmp_path / "a.py").write_text("unread current version")
    workspace = Workspace(tmp_path)
    old = {**narrow("a.py"), "content": "14: def transform(value):\n15:     return value - 1"}
    arguments = {"path": "a.py", "start_line": 14, "end_line": 15}
    messages = exchange("read_file", arguments, old)
    boundaries = source_read_boundaries(messages, workspace, external={"./a.py": len(messages)})
    assert boundaries == {"a.py": 2}
    assert pending_local_imports(messages, workspace, task="Inspect a.py", source_boundaries=boundaries) == ["a.py"]
    messages += exchange("read_file", arguments, narrow("a.py"), "fresh")
    assert not pending_local_imports(messages, workspace, task="Inspect a.py", source_boundaries=boundaries)
    assert InspectionLedger.from_messages(messages, workspace, source_boundaries=boundaries).files["a.py"].visible_ranges == [[14, 15]]


def test_mixed_batch_invalidation_keeps_current_peers_and_discards_old_omitted_debt(tmp_path):
    for path in ("a.py", "b.py", "old_dep.py", "peer_dep.py"):
        (tmp_path / path).write_text("metadata checks only")
    workspace = Workspace(tmp_path)
    def imported(path, dependency):
        return {"success": True, "path": path, "start_line": 1, "end_line": 1,
                "content": f"1: import {dependency}", "truncated": False}
    arguments = {"files": [{"path": "a.py", "start_line": 1, "end_line": 1},
                            {"path": "b.py", "start_line": 1, "end_line": 1},
                            {"path": "a.py", "start_line": 20, "end_line": 30}]}
    messages = exchange("read_code", arguments, {"success": True, "files": [
        imported("a.py", "old_dep"), imported("b.py", "peer_dep")], "omitted_file_count": 1})
    assert pending_read_selections(messages, workspace) == [{"path": "a.py", "start_line": 20, "end_line": 30}]
    boundaries = source_read_boundaries(messages, workspace, external={"a.py": 2})
    ledger = InspectionLedger.from_messages(messages, workspace, source_boundaries=boundaries)
    assert set(ledger.files) == {"b.py"} and ledger.files["b.py"].inspected
    assert not pending_read_selections(messages, workspace, source_boundaries=boundaries)
    assert pending_local_imports(messages, workspace, source_boundaries=boundaries) == ["peer_dep.py"]


def test_source_boundaries_ignore_failed_and_noop_writes_and_keep_latest_external_value(tmp_path):
    (tmp_path / "a.py").write_text("unread")
    workspace = Workspace(tmp_path)
    messages = exchange("write_file", {"path": "./a.py"}, {"success": True, "modified": False})
    messages += exchange("edit_file", {"path": "a.py"}, {"success": False, "modified": True}, "failed")
    assert not source_read_boundaries(messages, workspace)
    messages += exchange("write_file", json.dumps({"path": "./a.py"}), {"success": True, "modified": True}, "changed")
    assert source_read_boundaries(messages, workspace) == {"a.py": 6}
    assert source_read_boundaries(messages, workspace, external={"./a.py": 20, "a.py": 3}) == {"a.py": 20}


@pytest.mark.parametrize("external", [{"a.py": True}, {"a.py": -1}, {"../outside.py": 2}, []])
def test_invalid_source_boundaries_are_rejected(tmp_path, external):
    with pytest.raises(ValueError, match="boundar"):
        source_read_boundaries([], Workspace(tmp_path), external=external)


def test_artifact_capture_does_not_credit_inspection_or_imports(tmp_path):
    (tmp_path / "module.py").write_text("value = 1")
    messages = [{"role": "tool", "tool_name": "read_artifact", "content": json.dumps(narrow())}]
    assert not inspection_observed(messages)
    assert not pending_read_selections(messages, Workspace(tmp_path))


def test_memory_roundtrip_and_legacy_migration_reconstruct_actual_visible_ranges():
    memory = WorkingMemory(session_id="coverage", objective="Inspect module.py")
    memory.relevant_files = ["module.py"]
    memory.completed_steps = ["Read module.py."]
    old = memory.to_dict()
    old.pop("inspection")
    restored = WorkingMemory.from_dict(old)
    assert not restored.inspection.files
    messages = exchange("read_file", {"path": "module.py", "end_line": 100}, prefix())
    restored.restore_inspection(messages)
    assert restored.inspection.files["module.py"].visible_ranges == [[1, 2]]
    assert not restored.inspection.files["module.py"].inspected
    assert "Read module.py." not in restored.completed_steps
    assert any("lines 1-2; partial selection remains" in note for note in restored.completed_steps)
    assert WorkingMemory.from_dict(restored.to_dict()) == restored


def test_memory_observes_batch_ranges_and_omissions_incrementally():
    memory = WorkingMemory(session_id="coverage", objective="Inspect selected files")
    args = {"files": [{"path": "a.py", "start_line": 1, "end_line": 100},
                      {"path": "b.py", "start_line": 14, "end_line": 15}]}
    result = {"success": True, "files": [prefix("a.py")], "truncated": True,
              "omitted_file_count": 1, "next_file_index": 1}
    memory.observe_tool("read_files", args, result, 1)
    assert memory.inspection.files["a.py"].pending_ranges == [[3, 15]]
    assert memory.inspection.files["b.py"].pending_ranges == [[14, 15]]
    assert not any(note == "Read a.py." for note in memory.completed_steps)
    assert not any("Read b.py" in note for note in memory.completed_steps)


def test_memory_aliases_and_completed_continuation_replace_stale_progress_notes():
    memory = WorkingMemory(session_id="coverage", objective="Inspect module.py")
    memory.observe_tool("read_file", {"path": "./module.py", "end_line": 100}, prefix("./module.py"), 1)
    assert list(memory.inspection.files) == ["module.py"]
    assert any("partial selection remains" in note for note in memory.completed_steps)
    memory.observe_tool("read_file", {"path": "module.py", "start_line": 14, "end_line": 15}, narrow(), 2)
    reads = [note for note in memory.completed_steps if note.startswith("Read ")]
    assert reads == ["Read module.py lines 1-2, 14-15."]


def test_range_progress_is_bounded_while_full_coverage_remains_serializable():
    memory = WorkingMemory(session_id="coverage", objective="Inspect selected functions")
    for step, number in enumerate(range(1, 20, 2), 1):
        memory.observe_tool("read_file", {"path": "module.py", "start_line": number, "end_line": number},
                            {"success": True, "path": "module.py", "start_line": number, "end_line": number,
                             "total_lines": 100, "content": f"{number}: value = {number}", "truncated": False}, step)
    reads = [note for note in memory.completed_steps if note.startswith("Read ")]
    assert len(reads) == 1 and "+6 ranges in inspection ledger" in reads[0]
    assert len(memory.inspection.files["module.py"].visible_ranges) == 10
    assert WorkingMemory.from_dict(memory.to_dict()) == memory


@pytest.mark.parametrize("field,value", [("visible_ranges", [[-1, 3]]), ("pending_ranges", [[5, 2]]),
                                        ("code_like_ranges", [[True, 3]])])
def test_inspection_checkpoint_ranges_are_strictly_validated(field, value):
    with pytest.raises(ValidationError):
        InspectionLedger.from_dict({"schema_version": 1, "files": {"x.py": {field: value}}})


@pytest.mark.parametrize("path", ["", "bad\x00path", "x" * 4097])
def test_inspection_checkpoint_paths_are_bounded(path):
    with pytest.raises(ValidationError):
        InspectionLedger.from_dict({"schema_version": 1, "files": {path: {}}})
