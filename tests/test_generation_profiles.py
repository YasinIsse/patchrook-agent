"""Opt-in phase profiles preserve routing authority, budgets and call limits."""

from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
import sys

import httpx
import pytest

from patchrook_agent.agent import Agent
from patchrook_agent.completion import TaskRequirements
from patchrook_agent.config import HarnessConfig
from patchrook_agent.evidence_workflow import EvidenceWorkflow
from patchrook_agent.models.ollama import OllamaClient
from patchrook_agent.tools import ToolConfig, ToolRegistry, Workspace
from test_model_routing_workflow import Coder, SMALL, LARGE, Router, action, read, check, edit, route, setup


PROFILES = {'base':dict(think=False,max_output_tokens=1024,temperature=.7,top_p=.8,top_k=20,min_p=0.),
            'repair':dict(think=True,max_output_tokens=2048,temperature=.6,top_p=.95,top_k=20,min_p=0.)}


def config(**options):
    return replace(HarnessConfig.preset('C'),workflow_mode='evidence',action_protocol='schema',
        model_routing='tev',decision_model='on',routing_generation_profiles=deepcopy(PROFILES),
        evidence_context_tokens=6963,**options)


class ProfileCoder(Coder, OllamaClient):
    def __init__(self,name,*actions):
        Coder.__init__(self,name,*actions)
        profile=PROFILES['repair' if name==LARGE else 'base']
        for key,value in profile.items():
            if key!='think':setattr(self,key,value)
        self.request_timeout_seconds=1800.


def profiled_agent(tmp_path,monkeypatch,small,large,**options):
    settings={'routing_generation_profiles':deepcopy(PROFILES),'evidence_context_tokens':6963}
    settings.update(options.pop('config_overrides',{}))
    return setup(tmp_path,monkeypatch,small,large,config_overrides=settings,**options)


def test_config_roundtrip_and_old_missing_field_default():
    c=config()
    assert c.to_dict()['routing_generation_profiles']==PROFILES
    assert HarnessConfig.from_dict(json.loads(json.dumps(c.to_dict())))==c
    old=HarnessConfig().to_dict();old.pop('routing_generation_profiles')
    assert HarnessConfig.from_dict(old).routing_generation_profiles is None


@pytest.mark.parametrize('mutation', ['missing_phase','missing_field','extra_field','think_type','output_bool',
                                      'output_too_big','temperature_bool','unknown_phase'])
def test_profile_shapes_and_values_fail_closed(mutation):
    p=deepcopy(PROFILES)
    if mutation=='missing_phase':p.pop('base')
    elif mutation=='missing_field':p['repair'].pop('top_k')
    elif mutation=='extra_field':p['repair']['context_size']=16384
    elif mutation=='think_type':p['repair']['think']='true'
    elif mutation=='output_bool':p['repair']['max_output_tokens']=True
    elif mutation=='output_too_big':p['repair']['max_output_tokens']=8192
    elif mutation=='temperature_bool':p['repair']['temperature']=True
    else:p['other']=p.pop('base')
    with pytest.raises(ValueError):replace(config(),routing_generation_profiles=p)


def test_future_repair_budget_is_conservative_before_switch(tmp_path,monkeypatch):
    small,large=ProfileCoder(SMALL),ProfileCoder(LARGE)
    agent,_,_=profiled_agent(tmp_path,monkeypatch,small,large)
    runtime=EvidenceWorkflow(agent)
    assert runtime._input_limit()==6963
    assert runtime._input_limit(repair=True)==5939
    assert agent.model.max_output_tokens==1024
    assert 5327 <= runtime._input_limit(repair=True)
    assert runtime._input_limit(repair=True)+2048 < 8192


def test_real_wire_switches_profiles_and_restores_base_after_verified_patch(tmp_path,monkeypatch):
    root=tmp_path/'repo';root.mkdir();(root/'calc.py').write_text('def add(a, b):\n    return a - b\n')
    actions=iter([read(),check(),edit(),action('finish')]);requests=[];saved_phases=[]
    agent=None
    def server(request):
        body=json.loads(request.content);requests.append(body)
        phase='repair' if body['model']==LARGE else 'base';profile=PROFILES[phase]
        assert body['think'] is profile['think'] and body['options']['num_ctx']==8192
        assert body['options']['num_predict']==profile['max_output_tokens']
        for key in ('temperature','top_p','top_k','min_p'):assert body['options'][key]==profile[key]
        assert body['truncate'] is False and body['shift'] is False and body['format']['oneOf']
        saved=json.loads((agent._session.store.directory/'checkpoint.json').read_text())
        saved_phases.append(saved['execution'])
        message=next(actions)
        if phase=='repair':message={**message,'thinking':'private reasoning should not be replayed'}
        return httpx.Response(200,json={'message':message,'prompt_eval_count':100,'eval_count':30,'done_reason':'stop'})
    original=httpx.Client
    monkeypatch.setattr(httpx,'Client',lambda **kw: original(transport=httpx.MockTransport(server),**kw))
    monkeypatch.setattr(EvidenceWorkflow,'_unload_model',lambda *a:None)
    c=replace(config(),evidence_test_commands=((sys.executable,'-c','from calc import add; assert add(2,3)==5'),),
              persist=True,checkpoint_enabled=True)
    agent=Agent(OllamaClient(model=SMALL,max_output_tokens=1024),config=c,
        registry=ToolRegistry(Workspace(root,ToolConfig(allow_tests=True))),decision_client=Router('large'),
        runs_dir=tmp_path/'runs',deadline_seconds=30,max_steps=8)
    state=agent.run('Fix addition.',requirements=TaskRequirements.fix())
    assert state.status=='completed' and state.verification=='passed',state.final_response
    assert [b['model'] for b in requests]==[SMALL,SMALL,LARGE,SMALL]
    assert saved_phases[2]['think'] is True and saved_phases[2]['model_settings']['max_output_tokens']==2048
    assert saved_phases[-1]['think'] is False and saved_phases[-1]['model_settings']['max_output_tokens']==1024
    assert agent.think is False and agent.model.sampling_settings=={k:v for k,v in PROFILES['base'].items() if k not in {'think','max_output_tokens'}}
    assert 'private reasoning' not in json.dumps(requests[-1]['messages'])
    assert route(state)['repair']['calls_used']==1


def test_active_repair_resume_retains_thinking_profile_and_consumes_only_remaining_call(tmp_path,monkeypatch):
    small=ProfileCoder(SMALL,read(),check(),action('finish'))
    def interrupted(messages):raise KeyboardInterrupt()
    large=ProfileCoder(LARGE,interrupted,edit())
    agent,loads,unloads=profiled_agent(tmp_path,monkeypatch,small,large,router=Router('large'),persist=True)
    stopped=agent.run('Fix addition.',requirements=TaskRequirements.fix())
    assert route(stopped)['repair']['calls_used']==1
    saved=json.loads((Path(stopped.session_dir)/'checkpoint.json').read_text())
    assert saved['execution']['think'] is True
    assert saved['execution']['model_settings']['max_output_tokens']==2048
    resumed=agent.resume(stopped.session_id,allow_tests=True)
    assert resumed.status=='completed' and resumed.verification=='passed',resumed.final_response
    assert route(resumed)['repair']['calls_used']==2 and len(large.requests)==2
    assert agent.think is False and agent.model is small
    assert loads==[LARGE,SMALL] and unloads==[SMALL,LARGE]


@pytest.mark.parametrize('field,value',[('think',False),('max_output_tokens',1024),('temperature',.7)])
def test_resume_rejects_saved_active_profile_mismatch_before_management(tmp_path,monkeypatch,field,value):
    small=ProfileCoder(SMALL,read(),check())
    def interrupted(messages):raise KeyboardInterrupt()
    large=ProfileCoder(LARGE,interrupted)
    agent,loads,unloads=profiled_agent(tmp_path,monkeypatch,small,large,router=Router('large'),persist=True)
    state=agent.run('Fix addition.',requirements=TaskRequirements.fix())
    path=Path(state.session_dir)/'checkpoint.json';saved=json.loads(path.read_text())
    if field=='think':saved['execution']['think']=value
    else:saved['execution']['model_settings'][field]=value
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError,match='profile|inference settings'):agent.resume(state.session_id,allow_tests=True)
    assert loads==[LARGE] and unloads==[SMALL] and len(large.requests)==1


def test_two_invalid_thinking_responses_still_exhaust_without_patch(tmp_path,monkeypatch):
    small=ProfileCoder(SMALL,read(),check())
    bad={'role':'assistant','content':'','thinking':'reasoning without a final action'}
    large=ProfileCoder(LARGE,bad,bad)
    agent,loads,_=profiled_agent(tmp_path,monkeypatch,small,large,router=Router('large'))
    state=agent.run('Fix addition.',requirements=TaskRequirements.fix())
    assert state.stop_reason=='repair_call_limit' and route(state)['repair']['calls_used']==2
    assert len(large.requests)==2 and state.modified_files==[] and loads==[LARGE]


def test_unseen_source_remains_uneditable_under_repair_profiles(tmp_path,monkeypatch):
    small=ProfileCoder(SMALL,check(),edit(),edit())
    large=ProfileCoder(LARGE)
    agent,loads,_=profiled_agent(tmp_path,monkeypatch,small,large,max_steps=3)
    state=agent.run('Fix addition.',requirements=TaskRequirements.fix())
    assert not state.modified_files and not loads and not large.requests




@pytest.mark.parametrize('protocol',['schema','native'])
def test_required_packet_fits_old_limit_but_blocks_before_profile_switch(tmp_path,monkeypatch,protocol):
    from test_evidence_repair_optional import runtime, PRIMARY
    from patchrook_agent.evidence_workflow import estimate_tokens
    import time
    value=runtime(tmp_path,protocol=protocol,primary='REQUIRED = "'+'x'*15000+'"\n'+PRIMARY)
    value.config=replace(value.config,evidence_context_tokens=6963,evidence_optional_tokens=0)
    deadline=time.monotonic()+20
    messages=value._prepare_repair(value.state,value._repair_definitions(),deadline,record=False)
    native_cost = (math.ceil(len(json.dumps(value._repair_definitions(), ensure_ascii=False).encode()) / 3) + 64
                   if protocol == "native" else 0)
    assert 5939 < estimate_tokens(messages) + native_cost < 6963
    value.config=replace(value.config,model_routing='tev',decision_model='on',routing_generation_profiles=PROFILES)
    value.routing['repair'].update(status='requested',reason='invalid_responses')
    value.snapshot=value._fingerprint(deadline)
    value.checks['check_1'].update(revision=value.revision,snapshot=value.snapshot)
    value.session.memory.test_results[0]['verification_kind']='configured_command'
    monkeypatch.setattr(value,'_make_routed_model',lambda *a:pytest.fail('Oversized packet constructed a model client'))
    monkeypatch.setattr(value,'_unload_model',lambda *a:pytest.fail('Oversized packet unloaded the base model'))
    assert value._maybe_start_repair(deadline) is True
    assert value.state.stop_reason=='context_limit' and value.routing['repair']['status']=='blocked'
    assert value.routing['repair']['calls_used']==0 and value.attempts==0
    assert value.tools._presented=={}




