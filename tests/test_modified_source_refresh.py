"""Fresh current source in the next real serialized base request; no inference."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

import httpx
import pytest

from patchrook_agent.agent import Agent, AgentState
from patchrook_agent.completion import TaskRequirements
from patchrook_agent.config import HarnessConfig
from patchrook_agent.evidence_modified_source import KIND, refresh_modified_sources
from patchrook_agent.evidence_workflow import EvidenceWorkflow
from patchrook_agent.models.ollama import OllamaClient, ModelResponse
from patchrook_agent.tools import ToolConfig, ToolRegistry, Workspace


SOURCE = 'def total(a, b):\n    return a - b\n'
BAD = SOURCE.replace('a - b', 'a * b')


def action(tool, **arguments):
    return {'name': tool, 'arguments': arguments}


def edit(old='a - b', new='a * b'):
    return action('apply_edits', edits=[{'path':'subject.py', 'old_text':old, 'new_text':new}])


class Coder:
    model='qwen3:8b'
    context_size=8192
    max_output_tokens=512
    def __init__(self, *actions):self.actions=iter(actions);self.requests=[]
    def chat(self, messages, **kwargs):
        self.requests.append(deepcopy(messages))
        return ModelResponse({'role':'assistant','content':json.dumps(next(self.actions))},100,20,0,0,0,0)


def agent(tmp_path, model, *, enabled=True, config=None, tool_config=None, pause=None):
    root=tmp_path/'repo';root.mkdir(exist_ok=True)
    if not (root/'subject.py').exists():(root/'subject.py').write_text(SOURCE)
    base=HarnessConfig.preset('C')
    settings=replace(base, workflow_mode='evidence', action_protocol='schema', review_mode='off',
        evidence_refresh_modified_python=enabled, evidence_auto_check_edits=True,
        evidence_test_commands=((sys.executable,'-c','from subject import total; assert total(2,3)==5'),),
        evidence_optional_tokens=1536, evidence_context_tokens=6500,
        context=replace(base.context,enabled=False,context_size=8192,reserved_output_tokens=512),
        compaction_enabled=False, persist=True, checkpoint_enabled=True)
    return Agent(model,config=config or settings,registry=ToolRegistry(Workspace(root,
        tool_config or ToolConfig(allow_tests=True,validate_python_edits=True,max_output_chars=12000))),
        runs_dir=tmp_path/'runs',max_steps=8,deadline_seconds=30,pause_after_step=pause)


def runtime(tmp_path, **kwargs):
    value=EvidenceWorkflow(agent(tmp_path,Coder(),**kwargs));value.requirements=TaskRequirements.fix()
    value._setup('Fix total for integer addition.')
    value.definitions=value._definitions();value.enabled={x['function']['name'] for x in value.definitions}
    value.state=AgentState(messages=[{'role':'system','content':'unused'},{'role':'user','content':value.task}])
    value.session.bind(value.state)
    value.originals={'subject.py':SOURCE};(value.workspace.root/'subject.py').write_text(BAD)
    value.revision=1;value._update_state(value.state)
    return value


def records(state):
    return [json.loads(line) for line in (Path(state.session_dir)/'events.jsonl').read_text().splitlines()]


def test_next_base_request_after_accepted_bad_patch_has_current_numbered_source_and_sha(tmp_path,monkeypatch):
    actions=iter([action('read_file',path='subject.py'),edit(),edit('a * b','a + b'),action('finish')])
    requests=[]
    def transport(request):
        payload=json.loads(request.content);requests.append(payload)
        return httpx.Response(200,json={'message':{'role':'assistant','content':json.dumps(next(actions))},
            'prompt_eval_count':200,'eval_count':20,'done_reason':'stop'})
    original=httpx.Client
    monkeypatch.setattr(httpx,'Client',lambda **options:original(transport=httpx.MockTransport(transport),**options))
    value=agent(tmp_path,OllamaClient())
    state=value.run('Fix total for integer addition.',requirements=TaskRequirements.fix())
    assert state.status=='completed' and state.verification=='passed',state.final_response
    sources=[m['content'] for m in requests[2]['messages'] if m['content'].startswith('Requested source (data)')]
    assert len(sources)==1 and '2:     return a * b' in sources[0]
    assert hashlib.sha256(BAD.encode()).hexdigest() in sources[0] and 'return a - b' not in sources[0]
    assert state.steps==4 and state.metrics['inference_calls']==4
    reads=[x for x in records(state) if x['event']=='tool_request' and x['name']=='read_file']
    assert len(reads)==3 and sum(x['automatic'] for x in reads)==2
    contexts=state.metrics['evidence_contexts']
    assert contexts[2]['source_views'][0]['source_sha256']==hashlib.sha256(BAD.encode()).hexdigest()
    assert contexts[2]['modified_source_refresh_tokens']+contexts[2]['optional_estimated_tokens']<=1536


def test_option_off_preserves_source_gap(tmp_path):
    model=Coder(action('read_file',path='subject.py'),edit(),action('finish'))
    state=agent(tmp_path,model,enabled=False,pause=3).run('Fix total.',requirements=TaskRequirements.fix())
    assert state.metrics['evidence_contexts'][2]['source_views']==[]
    assert not any(x['event']==KIND for x in records(state))


def test_receipt_reuses_complete_fresh_source_without_repeated_tool_reads_or_debt_reset(tmp_path):
    value=runtime(tmp_path);value.failures=2;value.correction='keep actual failure';value.state.steps=3;value.attempts=3
    value.routing_progress.invalid_responses=1;value.routing_progress.repeated_actions=1
    before=value.routing_progress.to_dict()
    first=refresh_modified_sources(value,time.monotonic()+10)
    tool_calls=value.state.tool_calls
    second=refresh_modified_sources(value,time.monotonic()+10)
    assert [c.id for c in first]==[c.id for c in second] and len(first)==1
    assert value.state.tool_calls==tool_calls==1 and value.failures==2
    assert value.correction=='keep actual failure' and value.routing_progress.to_dict()==before
    assert value.state.steps==3 and value.attempts==3
    assert value.tools._presented=={},'Acquisition must not grant edit authority before presentation'


@pytest.mark.parametrize('whole',[False,True])
def test_only_complete_existing_fresh_card_can_skip_read(tmp_path,whole):
    value=runtime(tmp_path)
    value.evidence.observe_source('subject.py',1,2 if whole else 1,revision=1)
    cards=refresh_modified_sources(value,time.monotonic()+10)
    assert len(cards)==1 and cards[0].text==BAD
    assert value.state.tool_calls==(0 if whole else 1)


def test_stale_cached_receipt_is_not_replayed_or_used_as_authority(tmp_path):
    value=runtime(tmp_path);assert refresh_modified_sources(value,time.monotonic()+10)
    (value.workspace.root/'subject.py').write_text(BAD.replace('a * b','a / b'))
    calls=value.state.tool_calls
    assert refresh_modified_sources(value,time.monotonic()+10)==[]
    assert value.state.tool_calls==calls and not value.tools._presented


@pytest.mark.parametrize('text,tool_config,status',[
    ('x=1\n'*4,ToolConfig(allow_tests=True,max_read_lines=3),'line_limit'),
    ('x="'+'a'*800+'"\n',ToolConfig(allow_tests=True,max_output_chars=512),'output_limit'),
    ('x="'+'a'*800+'"\n',ToolConfig(allow_tests=True,max_file_bytes=100),'read_error'),
    ('',ToolConfig(allow_tests=True),'empty_file')])
def test_ineligible_whole_files_are_explicitly_skipped(tmp_path,text,tool_config,status):
    value=runtime(tmp_path,tool_config=tool_config);(value.workspace.root/'subject.py').write_text(text)
    assert refresh_modified_sources(value,time.monotonic()+10)==[] and value.state.tool_calls==0
    receipt=[c for c in value.evidence.cards(fresh=False) if c.kind==KIND][-1]
    assert json.loads(receipt.text)['files'][0]['status']==status


def test_at_most_two_python_paths_and_no_automatic_dependency_expansion(tmp_path,monkeypatch):
    value=runtime(tmp_path)
    for name in ('a.py','b.py','c.py','notes.txt'):
        (value.workspace.root/name).write_text('VALUE = 1\n');value.originals[name]='VALUE = 0\n'
    value._update_state(value.state)
    monkeypatch.setattr(value,'_expand',lambda *a,**k:pytest.fail('Automatic refresh must not read dependencies'))
    cards=refresh_modified_sources(value,time.monotonic()+10)
    assert [c.path for c in cards]==['a.py','b.py'] and value.state.tool_calls==2
    receipt=json.loads([c for c in value.evidence.cards(fresh=False) if c.kind==KIND][-1].text)
    assert [r['path'] for r in receipt['files'] if r['status']=='file_limit']==['c.py','subject.py']
    assert all(r['path']!='notes.txt' for r in receipt['files'])


def test_refresh_read_error_is_reported_without_spending_model_failure_debt(tmp_path,monkeypatch):
    value=runtime(tmp_path);value.failures=2;value.correction='original error'
    original=value.tools.execute
    monkeypatch.setattr(value.tools,'execute',lambda name,*a,**k: {'success':False,'error':{'code':'io_error','message':'bounded'}} if name=='read_file' else original(name,*a,**k))
    assert refresh_modified_sources(value,time.monotonic()+10)==[]
    assert value.failures==2 and value.correction=='original error' and value.state.tool_failures==1
    receipt=json.loads([c for c in value.evidence.cards(fresh=False) if c.kind==KIND][-1].text)
    assert receipt['files'][0]['error_code']=='io_error'


def test_tight_optional_budget_drops_whole_refresh_without_source_authority(tmp_path):
    value=runtime(tmp_path);value.config=replace(value.config,evidence_optional_tokens=0)
    messages=value._prepare(value.state,value.definitions,time.monotonic()+10)
    assert not any('Requested source (data)' in m['content'] for m in messages)
    assert not value.context_records[-1]['source_views'] and not value.tools._presented
    assert value.context_records[-1]['modified_source_refresh_tokens']==0


def test_resume_reuses_receipt_source_without_replaying_automatic_reads(tmp_path):
    first=Coder(action('read_file',path='subject.py'),edit(),action('run_check',name='check_1'))
    paused=agent(tmp_path,first,pause=3).run('Fix total.',requirements=TaskRequirements.fix())
    assert paused.stop_reason=='paused'
    prior_reads=sum(x['event']=='tool_request' and x['name']=='read_file' for x in records(paused))
    following=Coder(action('finish'))
    resumed=agent(tmp_path,following).resume(paused.session_id,allow_tests=True)
    assert sum(x['event']=='tool_request' and x['name']=='read_file' for x in records(resumed))==prior_reads
    assert 'return a * b' in '\n'.join(x['content'] for x in following.requests[0])
    assert resumed.metrics['inference_calls']==4 and resumed.steps==4


@pytest.mark.parametrize('bad',[None,0,1,'false'])
def test_config_is_strict_boolean_and_legacy_default_off(bad):
    assert HarnessConfig().evidence_refresh_modified_python is False
    with pytest.raises(ValueError,match='evidence_refresh_modified_python'):
        replace(HarnessConfig(),evidence_refresh_modified_python=bad)


def test_flag_round_trip_and_workflow_scope():
    config=replace(HarnessConfig(),workflow_mode='evidence',evidence_refresh_modified_python=True)
    assert HarnessConfig.from_dict(config.to_dict()).evidence_refresh_modified_python is True
    with pytest.raises(ValueError,match='evidence workflow'):
        replace(HarnessConfig(),evidence_refresh_modified_python=True)


def test_truncated_read_never_creates_or_presents_partial_refresh_source(tmp_path,monkeypatch):
    value=runtime(tmp_path)
    original=value.tools.execute
    def truncate(name,arguments,**kwargs):
        result=original(name,arguments,**kwargs)
        return {**result,'truncated':True,'end_line':1} if name=='read_file' else result
    monkeypatch.setattr(value.tools,'execute',truncate)
    assert refresh_modified_sources(value,time.monotonic()+10)==[]
    assert not [c for c in value.evidence.cards(fresh=False) if c.kind=='source']
    assert not value.tools._presented


def test_interrupted_automatic_read_is_not_replayed_from_persisted_receipt(tmp_path,monkeypatch):
    value=runtime(tmp_path)
    calls=[]
    def interrupt(*a,**k):calls.append(True);raise KeyboardInterrupt
    monkeypatch.setattr(value,'_perform_action',interrupt)
    with pytest.raises(KeyboardInterrupt):refresh_modified_sources(value,time.monotonic()+10)
    assert refresh_modified_sources(value,time.monotonic()+10)==[] and len(calls)==1
    assert not value.tools._presented


def test_symlink_replacement_is_skipped_without_reading_outside_source(tmp_path):
    value=runtime(tmp_path);outside=tmp_path/'outside.py';outside.write_text('SECRET=1\n')
    target=value.workspace.root/'subject.py';target.unlink();target.symlink_to(outside)
    assert refresh_modified_sources(value,time.monotonic()+10)==[] and value.state.tool_calls==0
    receipt=json.loads([c for c in value.evidence.cards(fresh=False) if c.kind==KIND][-1].text)
    assert receipt['files'][0]['status']=='read_error' and 'SECRET' not in json.dumps(receipt)


def test_no_refresh_for_rejected_batch_or_active_repair_and_no_deadline_overspend(tmp_path):
    value=runtime(tmp_path);value.revision=0
    assert refresh_modified_sources(value,time.monotonic()+10)==[] and value.state.tool_calls==0
    value.revision=1;value.routing={'active_model':value.config.escalation_model}
    assert refresh_modified_sources(value,time.monotonic()+10)==[] and value.state.tool_calls==0
    value.routing=None
    from patchrook_agent.tools.base import ToolError
    with pytest.raises(ToolError,match='Deadline'):
        value._prepare(value.state,value.definitions,time.monotonic()-1)
    assert value.attempts==0 and value.state.tool_calls==0


def test_whole_input_bound_can_omit_refresh_without_granting_authority(tmp_path):
    from patchrook_agent.evidence.compiler import estimate_tokens
    value=runtime(tmp_path);baseline=value._prepare(value.state,value.definitions,time.monotonic()+10)
    # A new runtime with the same mandatory task/check structure but almost no
    # residual room must omit source as a whole instead of truncating it.
    first=next(c for c in value.evidence.cards(fresh=False) if c.kind=='source')
    full_cost=value.context_records[-1]['modified_source_refresh_tokens']
    value.config=replace(value.config,evidence_context_tokens=value.context_records[-1]['estimated_tokens']-full_cost+5)
    # Current progress metadata differs slightly on retries; give enough room
    # for those receipts, still less than a whole source message.
    value.config=replace(value.config,evidence_context_tokens=value.config.evidence_context_tokens+40)
    messages=value._prepare(value.state,value.definitions,time.monotonic()+10)
    assert first.id not in value.context_records[-1]['presented_source_ids']
    assert not value.tools._presented and not any(m['content'].startswith('Requested source (data)') for m in messages)


def test_truncated_refresh_body_does_not_bypass_zero_optional_budget_in_recent_facts(tmp_path,monkeypatch):
    value=runtime(tmp_path);value.config=replace(value.config,evidence_optional_tokens=0)
    original=value.tools.execute
    def truncate(name,arguments,**kwargs):
        result=original(name,arguments,**kwargs)
        return {**result,'truncated':True,'end_line':1} if name=='read_file' else result
    monkeypatch.setattr(value.tools,'execute',truncate)
    messages=value._prepare(value.state,value.definitions,time.monotonic()+10)
    shown='\n'.join(message['content'] for message in messages)
    assert 'return a * b' not in shown and '1: def total' not in shown
    assert not value.context_records[-1]['source_views'] and not value.tools._presented
    archived=[card for card in value.evidence.cards(fresh=False) if card.kind=='read_file']
    assert archived and 'return a * b' in archived[-1].text
