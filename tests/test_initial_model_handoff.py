"""Initial routing is honored before generation only after real qualification."""

from dataclasses import replace
import json
import sys

import httpx
import pytest

from patchrook_agent.agent import Agent
from patchrook_agent.completion import TaskRequirements
from patchrook_agent.evidence_workflow import EvidenceWorkflow
from patchrook_agent.models.ollama import OllamaClient
from patchrook_agent.tools import ToolConfig, ToolRegistry, Workspace
from test_generation_profiles import config
from test_model_routing_workflow import LARGE, SMALL, Router, action, check, edit, route


def run(tmp_path, monkeypatch, label, *, initial_checks=True, symbols=True, qualified=True):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    requests, unloads = [], []
    scripted = iter([edit(), action("finish")] if label == "large" and qualified
                    else [edit(), check(), action("finish")] if qualified else [action("finish")])
    router = Router(label)
    real_client = httpx.Client
    def server(request):
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(200, json={"message": next(scripted), "done_reason": "stop",
                                        "prompt_eval_count": 100, "eval_count": 20})
    monkeypatch.setattr(httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(server), **kw))
    monkeypatch.setattr(EvidenceWorkflow, "_unload_model", lambda self, name, deadline: unloads.append(name))
    settings = replace(config(), evidence_initial_checks=initial_checks, evidence_seed_symbols=symbols,
        persist=False, checkpoint_enabled=False, evidence_context_tokens=10444,
        evidence_test_commands=((sys.executable, "-c", "from calc import add; assert add(2,3)==5"),))
    settings = replace(settings, context=replace(settings.context, context_size=12288))
    agent = Agent(OllamaClient(model=SMALL, context_size=12288, max_output_tokens=1024), config=settings,
        registry=ToolRegistry(Workspace(root, ToolConfig(allow_tests=True))), decision_client=router,
        runs_dir=tmp_path / "runs", deadline_seconds=30, max_steps=8 if qualified else 1)
    state = agent.run("Fix `add` while preserving its API.", requirements=TaskRequirements.fix())
    return state, requests, unloads, router


def test_qualified_initial_large_first_real_transport_request_is_14b(tmp_path, monkeypatch):
    state, requests, unloads, router = run(tmp_path, monkeypatch, "large")
    assert state.status == "completed" and state.verification == "passed"
    assert [row["model"] for row in requests] == [LARGE, SMALL]
    first = requests[0]
    assert first["think"] is True and first["options"]["num_ctx"] == 12288
    assert first["options"]["num_predict"] == 2048
    assert first["format"]["oneOf"][0]["properties"]["name"]["const"] == "apply_edits"
    assert "return a - b" in json.dumps(first["messages"]) and "AssertionError" in json.dumps(first["messages"])
    assert route(state)["switches"][0]["step"] == 0
    assert route(state)["repair"]["calls_used"] == 1
    assert len(router.requests) == state.metrics["decision_calls"] == 1
    assert unloads == [SMALL, LARGE]


@pytest.mark.parametrize("label", ["small", "uncertain"])
def test_small_and_uncertain_keep_the_first_request_on_base(tmp_path, monkeypatch, label):
    state, requests, unloads, router = run(tmp_path, monkeypatch, label)
    assert state.status == "completed" and state.verification == "passed"
    assert requests and all(row["model"] == SMALL for row in requests)
    assert route(state)["repair"]["calls_used"] == 0 and not unloads
    assert len(router.requests) == 1


@pytest.mark.parametrize("missing", ["source", "failed_check"])
def test_initial_large_without_qualification_stays_on_base(tmp_path, monkeypatch, missing):
    state, requests, unloads, router = run(tmp_path, monkeypatch, "large", qualified=False,
        initial_checks=missing != "failed_check", symbols=missing != "source")
    assert len(requests) == 1 and requests[0]["model"] == SMALL
    assert route(state)["repair"]["calls_used"] == 0 and not unloads
    assert len(router.requests) == 1
