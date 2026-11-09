"""Order admitted repair source without changing selection, authority, or focus."""
from collections import Counter
from dataclasses import replace
import json
import time

import pytest

from patchrook_agent.evidence.compiler import estimate_tokens
from patchrook_agent.evidence_workflow import _source_message
from test_evidence_symbol_bootstrap import runtime_factory


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Source ordering must not contact models or execute checks')
    monkeypatch.setattr('httpx.Client', forbidden)
    monkeypatch.setattr('httpx.get', forbidden)
    monkeypatch.setattr('httpx.post', forbidden)


def configured(runtime_factory):
    runtime = runtime_factory('Repair observed behavior.', {
        'z.py': 'def target():\n    return 1\n',
        'a.py': 'VALUE = 2\n', 'optional.py': 'OPTIONAL = 3\n'})
    runtime.routing = {'initial_model': 'small', 'active_model': 'large', 'repair': {
        'status': 'active', 'calls_used': 1, 'source_ranges': []}}
    runtime.correction = 'Exact observed guard rejection; do not drop or rewrite.'
    runtime.session.memory.test_results = [{'name': 'check_1', 'success': False,
        'exit_code': 1, 'stdout': 'specific assertion: expected 4, got 3', 'stderr': ''}]
    runtime.checks = {'check_1': {'passed': False, 'revision': 0}}
    return runtime


def observe(runtime, path, first=1, last=None, role='required_source'):
    return runtime.evidence.observe_source(path, first, last, origin='repair_packet',
        metadata={'repair_role': role})


def prepare(runtime):
    return runtime._prepare_repair(runtime.state, runtime._repair_definitions(), time.monotonic()+10)


def test_different_collector_order_gives_identical_rendered_source_and_keeps_focus(runtime_factory, monkeypatch):
    runtime = configured(runtime_factory)
    z = observe(runtime, 'z.py')
    a = observe(runtime, 'a.py')
    before_files = {p: (runtime.workspace.root/p).read_bytes() for p in ('a.py', 'z.py')}
    monkeypatch.setattr('patchrook_agent.evidence_workflow.enrich_owner_context',
                        lambda runtime, cards, messages, *args, **kwargs: (cards, messages))
    monkeypatch.setattr('patchrook_agent.evidence_workflow.collect_repair_sources', lambda *args: [z, a])
    first = prepare(runtime)
    assert runtime.routing['repair']['source_ranges'] == [
        {'path': c.path, 'start_line': c.start_line, 'end_line': c.end_line} for c in (z, a)]
    monkeypatch.setattr('patchrook_agent.evidence_workflow.collect_repair_sources', lambda *args: [a, z])
    second = prepare(runtime)
    assert first == second
    assert first[2:-1] == [_source_message(a), _source_message(z)]
    assert first[-1]['content'].count('specific assertion: expected 4, got 3') == 1
    assert runtime.correction in first[-1]['content']
    assert runtime.routing['repair']['calls_used'] == 1 and runtime.revision == 0
    assert before_files == {p: (runtime.workspace.root/p).read_bytes() for p in before_files}


def test_complete_ranges_sort_by_path_then_start_after_optional_admission(runtime_factory, monkeypatch):
    runtime = configured(runtime_factory)
    first = observe(runtime, 'z.py', 1, 1)
    second = observe(runtime, 'z.py', 2, 2)
    optional = observe(runtime, 'a.py', role='optional_owner_context')
    selected = [second, first]
    monkeypatch.setattr('patchrook_agent.evidence_workflow.collect_repair_sources', lambda *args: selected)
    def enrich(runtime, cards, messages, *args, **kwargs):
        return [*cards, optional], [*messages[:-1], _source_message(optional), messages[-1]]
    monkeypatch.setattr('patchrook_agent.evidence_workflow.enrich_owner_context', enrich)
    messages = prepare(runtime)
    assert messages[2:-1] == [_source_message(optional), _source_message(first), _source_message(second)]
    # Optional same-path or extra-path additions are not promoted to required focus.
    assert runtime.routing['repair']['source_ranges'] == [
        {'path': c.path, 'start_line': c.start_line, 'end_line': c.end_line} for c in selected]
    assert runtime.context_records[-1]['presented_source_ids'] == [optional.id, first.id, second.id]
    assert set(runtime.tools._presented) == {'a.py', 'z.py'}
    assert all(runtime.evidence.is_fresh(c) for c in (first, second, optional))


def test_message_multiset_and_estimated_budget_unchanged(runtime_factory, monkeypatch):
    runtime = configured(runtime_factory)
    cards = [observe(runtime, 'z.py'), observe(runtime, 'a.py')]
    before = []
    monkeypatch.setattr('patchrook_agent.evidence_workflow.collect_repair_sources', lambda *args: cards)
    def enrich(runtime, admitted, messages, *args, **kwargs):
        before.extend(messages)
        return admitted, messages
    monkeypatch.setattr('patchrook_agent.evidence_workflow.enrich_owner_context', enrich)
    result = prepare(runtime)
    serialize = lambda message: json.dumps(message, ensure_ascii=False, sort_keys=True)
    assert Counter(map(serialize, before)) == Counter(map(serialize, result))
    assert estimate_tokens(before) == estimate_tokens(result)
    assert runtime.context_records[-1]['estimated_tokens'] == estimate_tokens(result)
    assert result[:2] == before[:2] and result[-1] == before[-1]


def test_optional_drop_is_decided_before_presentation_order(runtime_factory, monkeypatch):
    runtime = configured(runtime_factory)
    runtime.config = replace(runtime.config, evidence_optional_tokens=0)
    z = observe(runtime, 'z.py')
    optional = observe(runtime, 'a.py', role='optional_dependency')
    monkeypatch.setattr('patchrook_agent.evidence_workflow.collect_repair_sources', lambda *args: [optional, z])
    monkeypatch.setattr('patchrook_agent.evidence_workflow.enrich_owner_context',
                        lambda runtime, cards, messages, *args, **kwargs: (cards, messages))
    result = prepare(runtime)
    assert result[2:-1] == [_source_message(z)]
    assert runtime.routing['repair']['source_ranges'] == [
        {'path': 'z.py', 'start_line': z.start_line, 'end_line': z.end_line}]
    assert set(runtime.tools._presented) == {'z.py'}
