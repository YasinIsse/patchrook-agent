"""Only new evidence or an accepted mutation repays rejected-action debt."""

from copy import deepcopy
import json

import pytest

from patchrook_agent.completion import TaskRequirements
from patchrook_agent.config import HarnessConfig
from patchrook_agent.model_routing import RoutingProgress, restore_routing
from test_model_routing_state import checkpoint
from test_model_routing_workflow import Coder, LARGE, SMALL, action, edit, read, route, setup, tool_results

BAD = {"success": False, "error": {"code": "overlapping_edits"}}


def source(first=2, last=3, digest="a" * 64):
    lines = {2: "def add(a, b):", 3: "    return a - b", 4: "# more source"}
    return {"success": True, "path": "calc.py", "start_line": first, "source_sha256": digest,
            "content": "\n".join(f"{n}: {lines[n]}" for n in range(first, last + 1))}


def failed_check(**options):
    return {"success": False, "exit_code": 1, "stdout": "FAILED test_add.py::test_add\n",
            "verification_kind": "configured_command", **options}


def rejected_batch():
    return action("apply_edits", edits=[{"path": "calc.py", "old_text": "a - b", "new_text": "a + b"}] * 2)


@pytest.fixture(autouse=True)
def no_model_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid-debt tests must not contact model endpoints")
    monkeypatch.setattr("httpx.Client", forbidden)


@pytest.mark.parametrize("name,arguments,result", [
    ("read_file", {"path": "calc.py"}, source()),
    ("search_code", {"path": ".", "pattern": "add"}, {"success": True, "matches": [
        {"path": "calc.py", "line_number": 2, "text": "def add(a, b):"}]}),
    ("list_files", {"path": "."}, {"success": True, "path": ".", "entries": ["calc.py"]}),
    ("read_evidence", {"id": "same"}, {"success": True, "id": "same", "content": "prior output", "stale": False}),
])
def test_identical_read_only_observation_does_not_clear_invalid_debt(name, arguments, result):
    config, progress = HarnessConfig(), RoutingProgress()
    assert progress.observe(name, arguments, result, 0, config) is None
    assert progress.observe("apply_edits", {}, BAD, 0, config) is None
    assert progress.observe(name, arguments | {"question": "different wording"},
                            result | {"evidence_id": "new", "duration_seconds": 99}, 0, config) is None
    assert progress.invalid_responses == 1
    assert progress.observe("apply_edits", {}, BAD, 0, config) == "invalid_responses"


@pytest.mark.parametrize("new_source", [source(2, 4), source(digest="b" * 64)])
def test_new_lines_or_changed_current_source_break_invalid_debt(new_source):
    config, progress = HarnessConfig(), RoutingProgress()
    progress.observe("read_file", {}, source(), 0, config)
    progress.invalid(config)
    progress.observe("read_file", {}, new_source, 0, config)
    assert progress.invalid_responses == 0
    assert progress.observe("apply_edits", {}, BAD, 0, config) is None


def test_subset_of_seen_source_keeps_debt():
    config, progress = HarnessConfig(), RoutingProgress()
    progress.observe("read_file", {}, source(2, 4), 0, config)
    progress.invalid(config)
    progress.observe("read_file", {}, source(2, 3), 0, config)
    assert progress.invalid_responses == 1


def test_negative_search_is_a_fact_but_repeating_the_same_query_is_not_new():
    config, progress = HarnessConfig(), RoutingProgress()
    empty = {"success": True, "matches": []}
    progress.observe("search_code", {"pattern": "absent"}, empty, 0, config)
    progress.invalid(config)
    progress.observe("search_code", {"pattern": "absent", "question": "try again"}, empty, 0, config)
    assert progress.invalid_responses == 1
    progress.observe("search_code", {"pattern": "different_absent"}, empty, 0, config)
    assert progress.invalid_responses == 0


@pytest.mark.parametrize("name,result", [
    ("read_file", {"success": False, "error": {"code": "not_found"}}),
    ("apply_edits", {"success": False, "error": {"code": "stale_source"}}),
    ("apply_edits", {"success": True, "modified_files": []}),
    ("read_evidence", {"success": True, "content": "old source", "stale": True}),
    ("run_check", failed_check(error={"code": "workspace_changed_during_check"})),
    ("run_check", failed_check(timed_out=True)),
])
def test_errors_noops_and_stale_archive_content_do_not_repay_invalid_debt(name, result):
    config, progress = HarnessConfig(), RoutingProgress()
    progress.invalid(config)
    progress.observe(name, {}, result, 0, config)
    assert progress.invalid_responses == 1


def test_only_new_noninfrastructure_configured_check_result_clears_debt():
    config, progress = HarnessConfig(), RoutingProgress()
    args = {"name": "check_1"}
    progress.invalid(config)
    progress.observe("run_check", args, failed_check(), 0, config)
    assert progress.invalid_responses == 0
    progress.invalid(config)
    progress.observe("run_check", args, failed_check(duration_seconds=123), 0, config)
    assert progress.invalid_responses == 1
    progress.observe("run_check", args, failed_check(), 1, config)
    assert progress.invalid_responses == 0
    progress.invalid(config)
    progress.observe("run_check", args, failed_check(success=True, exit_code=0, stdout=""), 1, config)
    assert progress.invalid_responses == 0


def test_successful_mutation_resets_invalid_debt():
    config, progress = HarnessConfig(), RoutingProgress()
    progress.invalid(config)
    progress.observe("apply_edits", {}, {"success": True, "modified_files": ["calc.py"]}, 1, config)
    assert progress.invalid_responses == progress.repeated_actions == 0


def test_seen_facts_and_invalid_debt_round_trip_with_unchanged_checkpoint_fields():
    config, progress = HarnessConfig(), RoutingProgress()
    progress.observe("read_file", {}, source(), 0, config)
    progress.invalid(config)
    serialized = json.loads(json.dumps(progress.to_dict()))
    assert set(serialized) == {"invalid_responses", "repeated_actions", "observations", "failed_repairs"}
    restored = RoutingProgress.from_dict(serialized)
    restored.observe("read_file", {}, source(), 0, config)
    assert restored.invalid(config) == "invalid_responses"


def test_automatic_seed_only_adds_hashes_without_consuming_counters_or_repairs():
    config, progress = HarnessConfig(), RoutingProgress()
    progress.invalid_responses, progress.repeated_actions = 1, 2
    progress.seed("read_file", {}, source(), 0)
    progress.seed("run_check", {"name": "check_1"}, failed_check(), 1)
    assert progress.invalid_responses == 1 and progress.repeated_actions == 2
    assert progress.failed_repairs == [] and len(progress.observations) == 3
    assert RoutingProgress.from_dict(progress.to_dict()).to_dict() == progress.to_dict()
    progress.observe("read_file", {}, source(), 0, config)
    assert progress.invalid_responses == 1 and progress.repeated_actions == 3
    progress.observe("run_check", {"name": "check_1"}, failed_check(), 0, config)
    assert progress.invalid_responses == 0, "A different revision is new check evidence"


@pytest.mark.parametrize("name,result", [
    ("read_file", {"success": False, "error": {"code": "not_found"}}),
    ("apply_edits", BAD),
    ("run_check", failed_check(timed_out=True)),
    ("run_check", failed_check(verification_kind="untrusted")),
])
def test_seed_ignores_errors_mutations_and_untrusted_checks(name, result):
    progress = RoutingProgress()
    before = deepcopy(progress.to_dict())
    progress.seed(name, {}, result, 0)
    assert progress.to_dict() == before


def test_bootstrap_header_reread_between_rejections_reaches_verified_focused_repair(tmp_path, monkeypatch):
    small = Coder(SMALL, rejected_batch(), read(start_line=2, end_line=3), rejected_batch(), action("finish"))
    large = Coder(LARGE, edit())
    agent, loads, unloads = setup(tmp_path, monkeypatch, small, large,
        config_overrides={"evidence_initial_checks": True, "evidence_seed_symbols": True})
    state = agent.run("Fix `add` for integers.", requirements=TaskRequirements.fix())
    assert state.status == "completed" and state.verification == "passed", state.final_response
    assert route(state)["repair"]["reason"] == "invalid_responses"
    assert route(state)["repair"]["calls_used"] == 1
    assert loads == [LARGE, SMALL] and unloads == [SMALL, LARGE]
    assert len(small.requests) == 4 and len(large.requests) == 1
    assert [r["error"]["code"] for r in tool_results(state, "apply_edits") if not r["success"]] == [
        "overlapping_edits", "overlapping_edits"]
    assert [r["exit_code"] for r in tool_results(state, "run_check")] == [1, 0]


def test_repeated_seed_source_never_replaces_required_failed_check(tmp_path, monkeypatch):
    small = Coder(SMALL, rejected_batch(), read(start_line=2, end_line=3), rejected_batch(), action("finish"))
    large = Coder(LARGE)
    agent, loads, unloads = setup(tmp_path, monkeypatch, small, large, max_steps=4,
        config_overrides={"evidence_initial_checks": False, "evidence_seed_symbols": True})
    state = agent.run("Fix `add` for integers.", requirements=TaskRequirements.fix())
    assert route(state)["repair"]["status"] == "requested"
    assert route(state)["repair"]["calls_used"] == 0
    assert not loads and not unloads and not large.requests
    assert not state.modified_files and state.verification == "not_run"


def test_invalid_debt_without_any_source_cannot_load_repair(tmp_path, monkeypatch):
    small = Coder(SMALL, rejected_batch(), rejected_batch())
    large = Coder(LARGE)
    agent, loads, unloads = setup(tmp_path, monkeypatch, small, large, max_steps=3,
        config_overrides={"evidence_initial_checks": True, "evidence_seed_symbols": False})
    state = agent.run("Fix addition.", requirements=TaskRequirements.fix())
    assert route(state)["repair"]["status"] == "requested"
    assert not loads and not unloads and not large.requests and not state.modified_files
    assert [r["error"]["code"] for r in tool_results(state, "apply_edits")] == [
        "source_not_presented", "source_not_presented"]


def test_source_change_does_not_allow_repair_from_stale_check_snapshot(tmp_path, monkeypatch):
    invalid = {"role": "assistant", "content": "Not an action"}
    def change_outside_harness(messages):
        path = tmp_path / "repo" / "calc.py"
        path.write_text(path.read_text() + "# external source changed\n")
        return invalid
    small = Coder(SMALL, invalid, read(start_line=2, end_line=3), change_outside_harness, action("finish"))
    large = Coder(LARGE)
    agent, loads, unloads = setup(tmp_path, monkeypatch, small, large, max_steps=4,
        config_overrides={"evidence_initial_checks": True, "evidence_seed_symbols": True})
    state = agent.run("Fix `add` for integers.", requirements=TaskRequirements.fix())
    assert route(state)["repair"]["status"] == "requested"
    assert not loads and not unloads and not large.requests
    assert not state.modified_files and state.verification == "not_run"


def test_two_call_repair_allowance_still_applies_after_redundant_read_trigger(tmp_path, monkeypatch):
    small = Coder(SMALL, rejected_batch(), read(start_line=2, end_line=3), rejected_batch())
    large = Coder(LARGE, rejected_batch(), rejected_batch())
    agent, loads, unloads = setup(tmp_path, monkeypatch, small, large,
        config_overrides={"evidence_initial_checks": True, "evidence_seed_symbols": True})
    state = agent.run("Fix `add` for integers.", requirements=TaskRequirements.fix())
    assert state.stop_reason == "repair_call_limit"
    assert route(state)["repair"]["calls_used"] == len(large.requests) == 2
    assert loads == [LARGE] and unloads == [SMALL] and not state.modified_files


def test_bootstrap_fact_hashes_are_valid_with_zero_completed_steps():
    config, progress = HarnessConfig(), RoutingProgress()
    progress.seed("read_file", {}, source(), 0)
    progress.seed("run_check", {"name": "check_1"}, failed_check(), 0)
    value = checkpoint(decided=False)
    value["progress"] = progress.to_dict()
    _, restored = restore_routing(value, config, SMALL, 0, 0)
    assert restored.to_dict() == progress.to_dict()
    assert restored.invalid_responses == restored.repeated_actions == 0
    restored.invalid(config)
    restored.observe("read_file", {}, source(), 0, config)
    assert restored.invalid_responses == 1


def test_bootstrap_large_view_is_bounded_and_identical_reread_keeps_debt():
    config, progress = HarnessConfig(), RoutingProgress()
    observed = source()
    observed["content"] = "\n".join(f"{n}: source_{n}" for n in range(1, 201))
    observed["start_line"] = 1
    progress.seed("read_file", {}, observed, 0)
    progress.seed("read_file", {}, observed, 0)
    assert len(progress.observations) == 1
    assert progress.repeated_actions == 0
    progress.invalid(config)
    progress.observe("read_file", {}, observed, 0, config)
    assert progress.invalid_responses == 1 and progress.repeated_actions == 1
