"""Selective empty-search detection, pure recovery guidance, and saved state."""

from copy import deepcopy
import json
import re

import pytest

from patchrook_agent.progress import SearchProgress, empty_search_guidance


EMPTY = {"success": True, "matches": [], "truncated": False}
QUERY = {"pattern": r"stage\d+\."}


def observe_empty(progress, arguments=None, **extra):
    return progress.observe("search_code", arguments if arguments is not None else QUERY, {**EMPTY, **extra})


def test_second_identical_empty_result_recovers_and_third_stops_without_mutating_result():
    progress = SearchProgress()
    original = deepcopy(EMPTY)
    assert progress.observe("search_code", QUERY, EMPTY) is None
    assert progress.action is None
    assert progress.observe("search_code", QUERY, EMPTY) == progress.action == "recover"
    assert progress.observe("search_code", QUERY, EMPTY) == progress.action == "stop"
    assert progress.observe("search_code", QUERY, EMPTY) == "stop"
    assert progress.empty_count == 3
    assert EMPTY == original and EMPTY["success"] is True


def test_default_paths_offsets_json_order_and_limits_do_not_create_false_new_queries():
    progress = SearchProgress()
    assert observe_empty(progress) is None
    equivalent = json.dumps({"max_results": 100, "offset": 0, "path": "./", "pattern": QUERY["pattern"]})
    assert observe_empty(progress, equivalent, duration_seconds=99) == "recover"
    assert observe_empty(progress, {"pattern": QUERY["pattern"], "path": ".", "max_results": 1}) == "stop"


@pytest.mark.parametrize("changed", [
    {"pattern": "stage_"}, {"pattern": QUERY["pattern"], "path": "pipeline"},
    {"pattern": QUERY["pattern"], "offset": 1},
])
def test_changed_query_starts_a_new_sequence(changed):
    progress = SearchProgress()
    observe_empty(progress)
    assert observe_empty(progress) == "recover"
    assert observe_empty(progress, changed) is None
    assert progress.empty_count == 1 and progress.action is None


@pytest.mark.parametrize("name,result", [
    ("read_file", {"success": True, "content": "1: return value + 8"}),
    ("read_files", {"success": False, "files": [{"success": True, "path": "runner.py"}, {"success": False}]}),
    ("read_artifact", {"success": True, "content": "observed source"}),
    ("list_files", {"success": True, "entries": ["runner.py"]}),
    ("edit_file", {"success": True, "modified": True}),
    ("write_file", {"success": True, "modified": True}),
    ("run_tests", {"success": True, "status": "passed", "exit_code": 0}),
    ("run_tests", {"success": False, "status": "failed", "exit_code": 1}),
])
def test_new_observed_evidence_resets_empty_search_sequence(name, result):
    progress = SearchProgress()
    observe_empty(progress)
    observe_empty(progress)
    assert progress.observe(name, {}, result) is None
    assert progress.action is None and progress.empty_count == 0
    assert observe_empty(progress) is None


def test_nonempty_search_resets_and_repeated_file_reads_are_always_allowed():
    progress = SearchProgress()
    observe_empty(progress)
    observe_empty(progress)
    assert progress.observe("search_code", QUERY, {"success": True, "matches": [{"path": "stage_0.py"}]}) is None
    for _ in range(5):
        assert progress.observe("read_file", {"path": "runner.py"}, {"success": True, "path": "runner.py"}) is None
    assert progress.empty_count == 0 and progress.action is None
    assert observe_empty(progress) is None


@pytest.mark.parametrize("name,result", [
    ("search_code", {"success": False, "error": {"code": "unavailable"}}),
    ("read_file", {"success": False, "error": {"code": "not_found"}}),
    ("edit_file", {"success": True, "modified": False}),
    ("write_file", {"success": False, "modified": True}),
    ("read_files", {"success": False, "files": [{"success": False}]}),
    ("run_tests", {"success": False, "error": {"code": "unavailable"}}),
    ("unknown_tool", {"success": True}),
])
def test_errors_and_non_evidence_do_not_count_or_erase_prior_empty_search(name, result):
    progress = SearchProgress()
    observe_empty(progress)
    assert progress.observe(name, {}, result) is None
    assert progress.empty_count == 1
    assert observe_empty(progress) == "recover"


@pytest.mark.parametrize("result", [
    {"success": True, "matches": [], "truncated": True},
    {"success": True, "matches": [], "metadata_omitted": True},
    {"success": True}, {"success": True, "matches": "[]"},
    {"success": 1, "matches": []},
])
def test_partial_or_malformed_search_observations_are_not_genuine_empty_results(result):
    progress = SearchProgress()
    assert progress.observe("search_code", QUERY, result) is None
    assert progress.empty_count == 0


@pytest.mark.parametrize("arguments", [None, "invalid JSON", "[]", {}, {"pattern": ""},
                                       {"pattern": "x", "offset": True}, {"pattern": "x", "path": None}])
def test_malformed_search_arguments_are_ignored(arguments):
    progress = SearchProgress()
    assert progress.observe("search_code", arguments, EMPTY) is None
    assert progress.empty_count == 0


def test_state_roundtrip_preserves_recovery_and_copies_serialized_data():
    progress = SearchProgress()
    observe_empty(progress)
    observe_empty(progress)
    saved = progress.to_dict()
    restored = SearchProgress.from_dict(json.loads(json.dumps(saved)))
    assert restored.action == "recover"
    assert observe_empty(restored) == "stop"
    saved["last_search"]["pattern"] = "different"
    assert progress.to_dict()["last_search"]["pattern"] == QUERY["pattern"]
    progress.reset()
    assert progress.to_dict() == {"schema_version": 1, "last_search": None, "empty_count": 0}


@pytest.mark.parametrize("change", [
    {"schema_version": True}, {"schema_version": 2}, {"extra": 1},
    {"empty_count": -1}, {"empty_count": 4}, {"empty_count": True},
    {"empty_count": 1}, {"last_search": {"pattern": "x", "path": ".", "offset": 0}},
    {"last_search": {"pattern": "x", "path": "./", "offset": 0}, "empty_count": 1},
    {"last_search": {"pattern": "x", "path": ".", "offset": "0"}, "empty_count": 1},
])
def test_corrupt_saved_state_is_rejected(change):
    saved = SearchProgress().to_dict()
    saved.update(change)
    with pytest.raises(ValueError):
        SearchProgress.from_dict(saved)


def test_recovery_guidance_uses_only_bounded_task_paths_and_performs_no_reads(monkeypatch):
    def forbidden_read(*args, **kwargs):
        raise AssertionError("Guidance must not open files")

    monkeypatch.setattr("builtins.open", forbidden_read)
    task = "Fix `pipeline/runner.py` and stage_7.py. Reference https://example.com/private.py; avoid ../outside.py."
    guidance = empty_search_guidance(task)
    assert '"pipeline/runner.py"' in guidance and '"stage_7.py"' in guidance
    assert "private.py" not in guidance and "outside.py" not in guidance
    assert "read_files" in guidance
    assert len(guidance) < 1000
    assert "incomplete" in empty_search_guidance(task, stopping=True)


def test_guidance_without_known_paths_proposes_discovery_and_limits_large_inventories():
    assert "list_files" in empty_search_guidance("Investigate the failing behavior.")
    guidance = empty_search_guidance(" ".join(f"module_{index}.py" for index in range(100)))
    assert '"module_3.py"' in guidance and "module_4.py" not in guidance
    assert len(guidance) < 1000


@pytest.mark.parametrize("enabled,expected,excluded", [
    ({"read_file", "list_files", "search_code", "write_file", "edit_file", "run_tests"},
     {"read_file", "search_code"}, {"read_files"}),
    ({"read_files"}, {"read_files"}, {"read_file", "list_files", "search_code"}),
    ({"search_code"}, {"search_code"}, {"read_file", "read_files", "list_files"}),
    ({"list_files"}, {"list_files"}, {"read_file", "read_files", "search_code"}),
    (set(), set(), {"read_file", "read_files", "list_files", "search_code"}),
])
def test_recovery_guidance_only_suggests_enabled_tools(enabled, expected, excluded):
    guidance = empty_search_guidance("Inspect pipeline/runner.py and pipeline/stage_7.py.", enabled_tools=enabled)
    for name in expected:
        assert re.search(r"\b" + name + r"\b", guidance)
    for name in excluded:
        assert re.search(r"\b" + name + r"\b", guidance) is None
    if not enabled:
        assert "incomplete" in guidance


def test_guidance_default_preserves_all_recovery_capabilities_and_never_adds_discovery():
    task = "Inspect runner.py and parser.py."
    all_recovery = {"read_file", "read_files", "list_files", "search_code"}
    assert empty_search_guidance(task) == empty_search_guidance(task, enabled_tools=all_recovery)
    guidance = empty_search_guidance("Investigate the issue.", enabled_tools={"read_file"})
    assert "read_file" in guidance and "list_files" not in guidance and "search_code" not in guidance
