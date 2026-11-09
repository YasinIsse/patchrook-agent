"""The real transport sees the current phase allowance, without changing it."""

from dataclasses import replace
import json
import re
import sys
import time

import httpx
import pytest

from patchrook_agent.agent import Agent
from patchrook_agent.completion import TaskRequirements
from patchrook_agent.evidence_workflow import EvidenceWorkflow
from patchrook_agent.models.ollama import OllamaClient
from patchrook_agent.tools import ToolConfig, ToolRegistry, Workspace
from test_generation_profiles import PROFILES, config
from test_model_routing_workflow import LARGE, SMALL, Router, action, read, check, edit, route


@pytest.mark.parametrize("protocol", ["schema", "native"])
def test_wire_budget_follows_base_repair_and_verified_return(tmp_path, monkeypatch, protocol):
    root = tmp_path / "repository"
    root.mkdir()
    (root / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    scripted = iter([read(), check(), edit(), action("finish")])
    requests = []
    real_client = httpx.Client
    def server(request):
        body = json.loads(request.content)
        requests.append(body)
        system = body["messages"][0]["content"]
        allowance = int(re.search(r"Reply budget: (\d+) generated tokens", system).group(1))
        assert allowance == body["options"]["num_predict"]
        assert ("Thinking shares this allowance" in system) is body["think"]
        assert "unique old_text anchors" in system and "one working increment" in system
        assert body["options"]["num_ctx"] == 12288
        assert body["truncate"] is False and body["shift"] is False
        message = next(scripted)
        if protocol == "native":
            proposed = json.loads(message["content"])
            message = {"role": "assistant", "tool_calls": [{"function": proposed}]}
        return httpx.Response(200, json={"message": message, "prompt_eval_count": 100,
                                         "eval_count": 20, "done_reason": "stop"})
    monkeypatch.setattr(httpx, "Client", lambda **options: real_client(transport=httpx.MockTransport(server), **options))
    monkeypatch.setattr(EvidenceWorkflow, "_unload_model", lambda *args: None)
    settings = replace(config(), action_protocol=protocol, persist=False, checkpoint_enabled=False,
        evidence_test_commands=((sys.executable, "-c", "from calc import add; assert add(2,3)==5"),))
    settings = replace(settings, context=replace(settings.context, context_size=12288),
                       evidence_context_tokens=10444)
    agent = Agent(OllamaClient(model=SMALL, context_size=12288, max_output_tokens=1024), config=settings,
        registry=ToolRegistry(Workspace(root, ToolConfig(allow_tests=True))), decision_client=Router("large"),
        runs_dir=tmp_path / "runs", deadline_seconds=30, max_steps=8)
    state = agent.run("Fix addition.", requirements=TaskRequirements.fix())
    assert state.status == "completed" and state.verification == "passed"
    assert [(row["model"], row["options"]["num_predict"], row["think"]) for row in requests] == [
        (SMALL, 1024, False), (SMALL, 1024, False), (LARGE, 2048, True), (SMALL, 1024, False)]
    assert route(state)["repair"]["calls_used"] == 1
    assert agent.model.max_output_tokens == 1024 and agent.think is False


def test_future_repair_preflight_states_target_allowance_before_switch(tmp_path):
    from test_evidence_repair_optional import runtime
    value = runtime(tmp_path)
    value.config = replace(value.config, model_routing="tev", decision_model="on",
                           routing_generation_profiles=PROFILES)
    assert value.agent.model.max_output_tokens == 1024 and value.agent.think is False
    messages = value._prepare_repair(value.state, value._repair_definitions(), time.monotonic() + 10, record=False)
    assert "Reply budget: 2048 generated tokens" in messages[0]["content"]
    assert "Thinking shares this allowance" in messages[0]["content"]
    assert value.agent.model.max_output_tokens == 1024 and value.agent.think is False
    assert value.routing["repair"]["calls_used"] == 0 and value.attempts == 0
    assert not value.tools._presented, "Budget preflight grants no source authority"


def test_no_profiles_uses_actual_client_allowance_without_hardcoded_phase_limit(tmp_path):
    from test_evidence_repair_optional import runtime
    value = runtime(tmp_path)
    value.agent.model.max_output_tokens = 717
    messages = value._prepare(value.state, value.definitions, time.monotonic() + 10)
    assert "Reply budget: 717 generated tokens" in messages[0]["content"]
    assert "Thinking shares this allowance" not in messages[0]["content"]
    assert value.config.routing_generation_profiles is None
    assert value.agent.model.max_output_tokens == 717
