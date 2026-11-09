"""Real HTTP serialization + guarded patch/check execution; inference is mocked.

These prove transport and evidence accounting, not coding model intelligence.
The independent assertions do not inspect the model's prose or expected IDs.
"""

from dataclasses import replace
import json
import sys

import httpx
import pytest

from patchrook_agent.agent import Agent
from patchrook_agent.completion import TaskRequirements
from patchrook_agent.config import HarnessConfig
from patchrook_agent.models.ollama import OllamaClient
from patchrook_agent.tools import ToolConfig, ToolRegistry, Workspace


@pytest.mark.parametrize("protocol", ["schema", "native"])
def test_wire_protocol_stays_fixed_through_real_failed_check_patch_and_passing_check(tmp_path, monkeypatch, protocol):
    root = tmp_path / "repo"
    root.mkdir()
    source = "# preserve this contract\ndef total(a, b):\n    return a - b\n"
    (root / "calc.py").write_text(source)
    tests = "from calc import total\ndef test_general_addition():\n    for a, b in [(2,3),(-2,3),(0,0),(100,-40)]:\n        assert total(a,b) == a+b\n"
    (root / "test_calc.py").write_text(tests)
    actions = iter([
        {"name": "run_check", "arguments": {"name": "check_1"}},
        {"name": "read_file", "arguments": {"path": "calc.py"}},
        {"name": "apply_edits", "arguments": {"edits": [
            {"path": "calc.py", "old_text": "return a - b", "new_text": "return a + b"}]}},
        {"name": "run_check", "arguments": {"name": "check_1"}},
        {"name": "finish", "arguments": {}},
    ])
    requests = []
    def server(request):
        payload = json.loads(request.content)
        requests.append(payload)
        action = next(actions)
        if protocol == "schema":
            assert "format" in payload and "tools" not in payload
            assert payload["format"]["oneOf"]
            message = {"role": "assistant", "content": json.dumps(action)}
        else:
            assert "format" not in payload and payload["tools"]
            message = {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": action["name"], "arguments": action["arguments"]}}]}
        assert payload["think"] is False and payload["truncate"] is False
        assert not any("propose_next_steps" in row.get("content", "") for row in payload["messages"])
        if len(requests) == 3:
            assert "return a - b" in json.dumps(payload["messages"])
            assert "Unresolved observed check failure" in json.dumps(payload["messages"])
        return httpx.Response(200, json={"message": message, "prompt_eval_count": 300,
                                         "eval_count": 30, "total_duration": 1000000, "done_reason": "stop"})
    original = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original(transport=httpx.MockTransport(server), **kwargs))
    config = replace(HarnessConfig.preset("C"), workflow_mode="evidence", action_protocol=protocol,
                     review_mode="off", evidence_test_commands=((sys.executable, "-m", "pytest", "-q"),))
    model = OllamaClient()
    state = Agent(model=model, config=config, registry=ToolRegistry(Workspace(root, ToolConfig(allow_tests=True))),
                  runs_dir=tmp_path / "runs", max_steps=8).run("Fix total for all integers. Preserve tests and comments.",
                                                              requirements=TaskRequirements.fix())
    assert state.status == "completed", state.final_response
    assert state.verification == "passed" and state.modified_files == ["calc.py"]
    assert (root / "calc.py").read_text() == source.replace("a - b", "a + b")
    assert (root / "test_calc.py").read_text() == tests
    results = [json.loads(item["content"]) for item in state.messages
               if item.get("tool_name") == "run_check"]
    assert [row["exit_code"] for row in results] == [1, 0]
    assert "failed" in results[0]["stdout"] and "1 passed" in results[1]["stdout"]
    assert len({request["messages"][0]["content"] for request in requests}) == 1
    assert state.metrics["decision_calls"] == 0
    assert state.completion["independent_acceptance_evaluated"] is False


@pytest.mark.parametrize("policy,decision", [("window", "off"), ("rules", "off"), ("decision", "on")])
def test_config_round_trip_retains_evidence_authorization(policy, decision):
    config = replace(HarnessConfig.preset("C"), workflow_mode="evidence", action_protocol="schema",
                     evidence_policy=policy, decision_model=decision,
                     evidence_test_commands=((sys.executable, "-m", "pytest", "-q"),))
    assert HarnessConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config


def test_oracle_requires_explicit_diagnostic_and_policy_cannot_hide_a_model():
    with pytest.raises(ValueError, match="diagnostic"):
        HarnessConfig(evidence_oracle_locations=(("calc.py", 1, 2),))
    with pytest.raises(ValueError, match="decision_model"):
        HarnessConfig(workflow_mode="evidence", evidence_policy="decision")


def test_cli_check_rejects_empty_argv_before_model_setup(monkeypatch, capsys):
    from patchrook_agent import cli
    monkeypatch.setattr(cli, "Agent", lambda **kwargs: pytest.fail("Invalid checks must not construct an agent"))
    with pytest.raises(SystemExit) as error:
        cli.main(["fix it", "--check", ""])
    assert error.value.code == 2
    assert "check" in capsys.readouterr().err.lower()


def test_trusted_project_pytest_plugin_runs_in_check_environment(tmp_path, monkeypatch):
    """Disabling project plugins can silently skip or invalidate real tests."""
    from patchrook_agent.evidence_tools import EvidenceTools
    import time
    monkeypatch.delenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", raising=False)
    (tmp_path / "project_plugin.py").write_text(
        "import pytest\n@pytest.fixture\ndef project_contract():\n    return 41\n")
    distribution = tmp_path / "project_test_plugin-0.1.dist-info"
    distribution.mkdir()
    (distribution / "METADATA").write_text("Name: project-test-plugin\nVersion: 0.1\n")
    (distribution / "entry_points.txt").write_text("[pytest11]\nproject_contract = project_plugin\n")
    (tmp_path / "test_project.py").write_text(
        "def test_contract(project_contract):\n    assert project_contract + 1 == 42\n")
    tools = EvidenceTools(Workspace(tmp_path, ToolConfig(allow_tests=True)),
                          commands={"check_1": [sys.executable, "-m", "pytest", "-q"]})
    result = tools.execute("run_check", {"name": "check_1"}, deadline=time.monotonic() + 30)
    assert result["exit_code"] == 0, result
    assert "1 passed" in result["stdout"]
