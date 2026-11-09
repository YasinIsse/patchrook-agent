"""Positive completion guidance keeps the existing finish and routing guards."""
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
from patchrook_agent.evidence import ContextBudgetError
from patchrook_agent.evidence.compiler import estimate_tokens
from patchrook_agent.evidence_workflow import COMPLETION_GUIDANCE, EvidenceWorkflow
from patchrook_agent.models.ollama import ModelResponse, OllamaClient
from patchrook_agent.tools import ToolConfig, ToolRegistry, Workspace


SOURCE='def total(a, b):\n    return a - b\n'
FIXED=SOURCE.replace('a - b','a + b')


def action(name,arguments=None):return {'name':name,'arguments':arguments or {}}
READ=action('read_file',{'path':'subject.py'})
EDIT=action('apply_edits',{'edits':[{'path':'subject.py','old_text':'a - b','new_text':'a + b'}]})
CHECK=action('run_check',{'name':'check_1'})
FINISH=action('finish')


class Coder:
    model='qwen3:8b';context_size=8192;max_output_tokens=1024
    def __init__(self,*actions):self.actions=iter(actions);self.requests=[]
    def chat(self,messages,**kwargs):
        self.requests.append(deepcopy(messages))
        return ModelResponse({'role':'assistant','content':json.dumps(next(self.actions))},100,20,0,0,0,0)


def agent(tmp_path,model,max_steps=4):
    root=tmp_path/'repo';root.mkdir(exist_ok=True)
    if not (root/'subject.py').exists():(root/'subject.py').write_text(SOURCE)
    base=HarnessConfig.preset('C')
    cfg=replace(base,workflow_mode='evidence',action_protocol='schema',review_mode='off',
        evidence_auto_check_edits=True,evidence_refresh_modified_python=True,
        evidence_test_commands=((sys.executable,'-c','from subject import total; assert total(2,3)==5'),),
        evidence_optional_tokens=1536,evidence_context_tokens=6500,
        context=replace(base.context,enabled=False,context_size=8192,reserved_output_tokens=1024),
        compaction_enabled=False,persist=True,checkpoint_enabled=True)
    return Agent(model,config=cfg,registry=ToolRegistry(Workspace(root,ToolConfig(allow_tests=True,
        validate_python_edits=True,max_output_chars=12000))),runs_dir=tmp_path/'runs',max_steps=max_steps,deadline_seconds=30)


def runtime(tmp_path):
    value=EvidenceWorkflow(agent(tmp_path,Coder()));value.requirements=TaskRequirements.fix()
    value._setup('Implement full integer addition; public examples are incomplete.')
    value.definitions=value._definitions();value.enabled={x['function']['name'] for x in value.definitions}
    value.state=AgentState(messages=[{'role':'system','content':'unused'},{'role':'user','content':value.task}])
    value.session.bind(value.state);value.inspected=True;value.originals={'subject.py':SOURCE}
    (value.workspace.root/'subject.py').write_text(FIXED);value.revision=1
    value.checks={'check_1':{'passed':True,'revision':1,'exit_code':0,'timed_out':False}}
    value._update_state(value.state)
    return value


def cue(messages):return [m['content'] for m in messages if m['content'].startswith('Completion guidance:')]


def test_real_wire_ready_request_offers_finish_with_full_specification_caveat(tmp_path,monkeypatch):
    proposals=iter([READ,EDIT,FINISH]);requests=[]
    def transport(request):
        payload=json.loads(request.content);requests.append(payload)
        return httpx.Response(200,json={'message':{'role':'assistant','content':json.dumps(next(proposals))},
            'prompt_eval_count':300,'eval_count':20,'done_reason':'stop'})
    original=httpx.Client
    monkeypatch.setattr(httpx,'Client',lambda **kwargs:original(transport=httpx.MockTransport(transport),**kwargs))
    state=agent(tmp_path,OllamaClient()).run('Implement full integer addition.',requirements=TaskRequirements.fix())
    assert state.status=='completed' and state.stop_reason=='verified_finish'
    assert not cue(requests[0]['messages']) and not cue(requests[1]['messages'])
    assert cue(requests[2]['messages'])==['Completion guidance: '+COMPLETION_GUIDANCE]
    assert 'If the full requested behavior is implemented, call finish now.' in cue(requests[2]['messages'])[0]
    assert 'public examples alone do not prove the full specification' in cue(requests[2]['messages'])[0]
    assert 'finish' in json.dumps(requests[2]['format']) and requests[2]['think'] is False
    assert state.steps==3 and state.metrics['inference_calls']==3
    assert all(c['estimated_tokens']<=c['input_limit'] for c in state.metrics['evidence_contexts'])


@pytest.mark.parametrize('condition',['failed','stale_revision','missing_change','missing_inspection'])
def test_not_ready_source_or_check_states_do_not_receive_ready_cue(tmp_path,condition):
    value=runtime(tmp_path)
    if condition=='failed':value.checks['check_1']['passed']=False
    if condition=='stale_revision':value.checks['check_1']['revision']=0
    if condition=='missing_change':(value.workspace.root/'subject.py').write_text(SOURCE)
    if condition=='missing_inspection':value.inspected=False
    value._update_state(value.state)
    rejected=value._execute(value.state,FINISH,time.monotonic()+10)
    assert not rejected['success'] and rejected['error']['code']=='completion_requirements'
    assert value.state.status!='completed'
    messages=value._prepare(value.state,value.definitions,time.monotonic()+10)
    assert not cue(messages)


def test_passing_check_does_not_finish_without_an_actual_model_finish_action(tmp_path):
    model=Coder(READ,EDIT,CHECK,CHECK)
    state=agent(tmp_path,model).run('Implement integer addition.',requirements=TaskRequirements.fix())
    assert state.verification=='passed' and state.status=='incomplete' and state.stop_reason=='max_steps'
    assert state.steps==4 and state.metrics['inference_calls']==4
    assert cue(model.requests[2]) and cue(model.requests[3])


def test_satisfied_base_coder_removes_obsolete_repair_request_correction_without_touching_debt(tmp_path):
    value=runtime(tmp_path);snapshot=value._fingerprint(time.monotonic()+10)
    value.checks['check_1']['snapshot']=snapshot;value.snapshot=snapshot
    value.routing={'active_model':'qwen3:8b','repair':{'status':'requested','reason':'repeated_actions','calls_used':0}}
    value.correction='A focused stronger-model repair is requested. Read the relevant implementation and reproduce the unresolved behavior with configured checks first.'
    value.session.memory.test_results=[{'name':'check_1','success':True,'exit_code':0,'verification_kind':'configured_command'}]
    value.failures=2;value.attempts=4;value.state.steps=4;value.agent.max_steps=12
    value.routing_progress.repeated_actions=2;value.routing_progress.invalid_responses=1
    before=value.routing_progress.to_dict();cfg=value.config.to_dict()
    assert not value._maybe_start_repair(time.monotonic()+10)
    assert value.routing['repair']=={'status':'idle','reason':'satisfied_by_base_coder','calls_used':0}
    assert value.correction=='' and value.failures==2 and value.routing_progress.to_dict()==before
    assert value.state.steps==value.attempts==4 and value.config.to_dict()==cfg
    value.routing=None
    messages=value._prepare(value.state,value.definitions,time.monotonic()+10)
    assert cue(messages) and 'focused stronger-model repair is requested' not in json.dumps(messages)


def test_finish_still_checks_workspace_snapshot_and_context_budget(tmp_path):
    value=runtime(tmp_path);deadline=time.monotonic()+10
    value.snapshot=value._fingerprint(deadline);value.checks['check_1']['snapshot']=value.snapshot
    (value.workspace.root/'subject.py').write_text(FIXED+'UNRELATED = 1\n')
    result=value._execute(value.state,FINISH,deadline)
    assert not result['success'] and result['error']['code']=='workspace_changed'
    assert value.state.status!='completed' and value.checks=={}
    value.config=replace(value.config,evidence_context_tokens=512)
    with pytest.raises(ContextBudgetError):value._prepare(value.state,value.definitions,deadline)
