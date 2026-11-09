"""Explicit resume can requalify one unused repair call without replay/refund."""

from pathlib import Path
import json
import sys

import pytest

from patchrook_agent.completion import TaskRequirements
from patchrook_agent.evidence_workflow import EvidenceWorkflow
from patchrook_agent.model_routing import infrastructure_failure
from test_model_routing_workflow import Coder, SMALL, LARGE, Router, action, read, check, route, setup, text, tool_results


IMPORT = 'from missing_repair_resume_fixture import absent\n'
HEADER = 'def add(a, b):'


def bad_import():
    return action('apply_edits', edits=[{'path':'calc.py', 'old_text':HEADER, 'new_text':IMPORT + HEADER}])


def repair_import(*, correct=True):
    return action('apply_edits', edits=[{'path':'calc.py', 'old_text':IMPORT, 'new_text':''},
        {'path':'calc.py', 'old_text':'a - b', 'new_text':'a + b' if correct else 'a * b'}])


@pytest.fixture(autouse=True)
def no_model_network(monkeypatch):
    monkeypatch.setattr('httpx.Client', lambda *a, **k: pytest.fail('Resume tests must not contact models'))


def old_blocked_run(tmp_path, monkeypatch, *, commands=None, config_overrides=None):
    small = Coder(SMALL, read(), check(), action('finish'))
    large = Coder(LARGE, bad_import(), repair_import())
    agent, loads, unloads = setup(tmp_path, monkeypatch, small, large, persist=True,
        router=Router('large'), commands=commands, config_overrides=config_overrides)
    current_classifier = EvidenceWorkflow._check_infrastructure_failure
    # Simulate the pre-fix classification that produced the saved real failure.
    monkeypatch.setattr(EvidenceWorkflow, '_check_infrastructure_failure',
        lambda self, name, result, deadline: infrastructure_failure(result))
    stopped = agent.run('Fix integer addition.', requirements=TaskRequirements.fix())
    monkeypatch.setattr(EvidenceWorkflow, '_check_infrastructure_failure', current_classifier)
    assert stopped.stop_reason == 'repair_environment_failure'
    assert route(stopped)['repair']['status'] == 'blocked'
    assert route(stopped)['repair']['calls_used'] == len(large.requests) == 1
    assert agent.model is large
    return agent, small, large, loads, unloads, stopped


def checkpoint(stopped):
    return Path(stopped.session_dir) / 'checkpoint.json'


def test_explicit_resume_refreshes_source_and_checks_before_only_remaining_call(tmp_path, monkeypatch):
    agent, small, large, loads, unloads, stopped = old_blocked_run(tmp_path, monkeypatch)
    first_history = route(stopped)['switches'][:]
    before = json.loads(checkpoint(stopped).read_text())
    resumed = agent.resume(stopped.session_id, allow_tests=True)
    assert resumed.status == 'completed' and resumed.verification == 'passed', resumed.final_response
    assert len(large.requests) == 2 and len(small.requests) == 3
    assert route(resumed)['repair']['calls_used'] == 2
    assert route(resumed)['switches'][:1] == first_history
    assert loads == [LARGE, SMALL] and unloads == [SMALL, LARGE]
    assert IMPORT.strip() in text(large.requests[1]), 'Remaining call must see the actual accepted broken import'
    assert 'ModuleNotFoundError' in text(large.requests[1])
    results = tool_results(resumed, 'run_check')
    assert [r['exit_code'] for r in results] == [1, 1, 1, 0], 'Resume must recheck, not reuse old failure'
    saved = json.loads(checkpoint(resumed).read_text())
    assert set(saved['execution']['routing']['repair']) == {
        'status','reason','calls_used','call_inflight','pending_checks','last_revision','source_ranges'}
    assert '_resume_environment_repair' not in saved['execution']
    assert saved['execution']['deadline_seconds'] == before['execution']['deadline_seconds']
    assert saved['metrics']['wall_seconds'] >= before['metrics']['wall_seconds']


def test_persistent_real_environment_failure_rechecks_without_spending_remaining_call(tmp_path, monkeypatch):
    command = ((sys.executable, '-c', 'import os;\nif os.getenv("PATCHROOK_RESUME_TEST_OUTAGE"): raise ConnectionError("offline")\nfrom calc import add; assert add(2, 3) == 5'),)
    agent, small, large, loads, unloads, stopped = old_blocked_run(tmp_path, monkeypatch, commands=command)
    monkeypatch.setenv('PATCHROOK_RESUME_TEST_OUTAGE', '1')
    resumed = agent.resume(stopped.session_id, allow_tests=True)
    assert resumed.stop_reason == 'repair_environment_failure'
    assert route(resumed)['repair']['status'] == 'blocked'
    assert route(resumed)['repair']['calls_used'] == len(large.requests) == 1
    assert loads == [LARGE] and unloads == [SMALL]
    assert 'ConnectionError' in tool_results(resumed, 'run_check')[-1]['stderr']
    assert len(tool_results(resumed, 'run_check')) == 3


def test_no_unused_allowance_never_reopens_a_blocked_environment_repair(tmp_path, monkeypatch):
    agent, small, large, loads, unloads, stopped = old_blocked_run(tmp_path, monkeypatch,
        config_overrides={'routing_max_repair_calls':1})
    resumed = agent.resume(stopped.session_id, allow_tests=True)
    assert resumed.stop_reason == 'repair_blocked'
    assert route(resumed)['repair']['calls_used'] == len(large.requests) == 1
    assert len(tool_results(resumed, 'run_check')) == 2
    assert loads == [LARGE] and unloads == [SMALL]


@pytest.mark.parametrize('reason', ['context_limit', 'repair_packet_unavailable', 'rollback_failed'])
def test_other_blocked_reasons_never_requalify(tmp_path, monkeypatch, reason):
    agent, small, large, loads, unloads, stopped = old_blocked_run(tmp_path, monkeypatch)
    path = checkpoint(stopped)
    saved = json.loads(path.read_text())
    saved['state']['stop_reason'] = reason
    path.write_text(json.dumps(saved))
    resumed = agent.resume(stopped.session_id, allow_tests=True)
    assert resumed.stop_reason == 'repair_blocked'
    assert route(resumed)['repair']['calls_used'] == len(large.requests) == 1
    assert len(tool_results(resumed, 'run_check')) == 2


def test_missing_complete_source_blocks_before_remaining_model_call(tmp_path, monkeypatch):
    agent, small, large, loads, unloads, stopped = old_blocked_run(tmp_path, monkeypatch)
    monkeypatch.setattr('patchrook_agent.evidence_workflow.collect_repair_sources', lambda *a, **k: None)
    resumed = agent.resume(stopped.session_id, allow_tests=True)
    assert resumed.stop_reason == 'repair_packet_unavailable'
    assert route(resumed)['repair']['calls_used'] == len(large.requests) == 1
    assert loads == [LARGE] and unloads == [SMALL]


def test_second_bad_repair_still_exhausts_without_replaying_the_first(tmp_path, monkeypatch):
    agent, small, large, loads, unloads, stopped = old_blocked_run(tmp_path, monkeypatch)
    large.steps = iter([repair_import(correct=False)])
    resumed = agent.resume(stopped.session_id, allow_tests=True)
    assert resumed.stop_reason == 'repair_call_limit'
    assert route(resumed)['repair']['status'] == 'exhausted'
    assert route(resumed)['repair']['calls_used'] == len(large.requests) == 2
    assert loads == [LARGE] and unloads == [SMALL]
    assert IMPORT not in (tmp_path/'repo/calc.py').read_text(), 'First generation was not replayed'


def test_exhausted_two_call_checkpoint_never_runs_a_third_call(tmp_path, monkeypatch):
    agent, small, large, loads, unloads, stopped = old_blocked_run(tmp_path, monkeypatch)
    large.steps = iter([repair_import(correct=False)])
    exhausted = agent.resume(stopped.session_id, allow_tests=True)
    assert route(exhausted)['repair']['calls_used'] == 2
    again = agent.resume(exhausted.session_id, allow_tests=True)
    assert again.stop_reason == 'repair_call_limit'
    assert route(again)['repair']['calls_used'] == len(large.requests) == 2
