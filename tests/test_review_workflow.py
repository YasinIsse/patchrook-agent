"""Selective review preserves operation scope, bounded rounds and durable choices."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from patchrook_agent.agent import Agent
from patchrook_agent.completion import TaskRequirements
from patchrook_agent.config import HarnessConfig
from patchrook_agent.models.decision import DecisionResult
from patchrook_agent.models.ollama import ModelResponse
from patchrook_agent.review_workflow import ReviewWorkflow
from patchrook_agent.proposal_generation import generation_schema
from patchrook_agent.repair_actions import RepairAction
from patchrook_agent.tools import ToolConfig, ToolRegistry, Workspace


TASK = "Reproduce the failing tests, fix addition in calc.py and downward rounding in rounding.py. Preserve existing tests."
DONE = {"role": "assistant", "content": "DONE"}


def tool(name, **arguments):
    return {"function": {"name": name, "arguments": arguments}}


def action(*calls):
    return {"role": "assistant", "content": "", "tool_calls": list(calls)}


def review_request():
    return action(tool("request_review", path="calc.py", reason=(
        "The assertion permits several arithmetic changes; compare their behavior before choosing a correction.")))


def proposals(*, suffix=""):
    def respond(options):
        assert options["response_format"] == generation_schema() and not options.get("tools")
        approaches = [
            {"approach": "Return a + b for general integer addition", "expected": "Addition works for all integer operands"},
            {"approach": "Return a * b - 1 to fit the supplied example", "expected": "The observed addition example passes"},
        ]
        if suffix:
            approaches[0]["approach"] = "Return sum((a, b)) for integer addition"
            approaches[1]["approach"] = "Return abs(a - b) as the arithmetic result"
        return {"role": "assistant", "content": json.dumps({"approaches": approaches})}
    return respond


def malformed_drafts(options):
    """Malformed drafts cannot omit required fields or supply authority."""
    message = proposals()(options)
    draft = json.loads(message["content"])
    draft["approaches"][0].pop("expected")
    draft["approaches"][1]["path"] = "calc.py"
    message["content"] = json.dumps(draft)
    return message


def inspection_range(monkeypatch, start, end):
    action = RepairAction("test_read", "inspect_helper", "test_item", "calc.py", start, end,
                          "Inspect the supplied missing policy range")
    monkeypatch.setattr("patchrook_agent.review_workflow.build_repair_actions", lambda plan: (action,))


class Model:
    model = "offline-selective-review"
    context_size = 8192
    max_output_tokens = 512

    def __init__(self, *script, check=None):
        self.script = iter(script)
        self.requests = []
        self.check = check

    def chat(self, **options):
        self.requests.append(deepcopy(options))
        if self.check:
            self.check(len(self.requests), options)
        item = next(self.script)
        message = item(options) if callable(item) else item
        return ModelResponse(message, 0, 0, None, None, None, 0,
                             prompt_tokens_reported=False, completion_tokens_reported=False)


class Decisions:
    def __init__(self, *choices):
        self.choices = iter(choices)
        self.calls = []

    def rank(self, context, candidates, **options):
        assert options["purpose"] == "proposal_review", "Adaptive review must not perform startup routing"
        selected = next(self.choices)
        assert selected in {candidate.id for candidate in candidates}
        self.calls.append((context, deepcopy(candidates), deepcopy(options)))
        return DecisionResult((selected, *(candidate.id for candidate in candidates if candidate.id != selected)),
                              {}, called=False)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Review integration tests must never contact a model server")
    monkeypatch.setattr("httpx.Client", forbidden)
    monkeypatch.setattr("httpx.get", forbidden)


def repository(tmp_path, *, rounding_correct=True):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (root / "rounding.py").write_text("def floor_cents(value):\n    return " + (
        "value // 100\n" if rounding_correct else "(value + 99) // 100\n"))
    (root / "test_calc.py").write_text(
        "from calc import add\nfrom rounding import floor_cents\n\n"
        "def test_add():\n    assert add(3, 2) == 5\n\n"
        "def test_floor():\n    assert floor_cents(151) == 1\n")
    return root


def agent(tmp_path, model, client=None, *, pause=None, dispatched=None):
    instance = Agent(model, registry=ToolRegistry(Workspace(tmp_path / "repo", ToolConfig(allow_tests=True))),
                     config=replace(HarnessConfig.preset("C"), workflow_mode="coding", action_protocol="native",
                                    review_mode="adaptive", decision_model="on"),
                     decision_client=client or Decisions(), runs_dir=tmp_path / "runs",
                     pause_after_step=pause, max_steps=10, deadline_seconds=60)
    if dispatched is not None:
        original = instance.registry.dispatch
        def record(name, arguments, **options):
            dispatched.append((name, deepcopy(arguments)))
            return original(name, arguments, **options)
        instance.registry.dispatch = record
    return instance


def observations(state, name):
    return [json.loads(message["content"]) for message in state.messages if message.get("tool_name") == name]


def checkpoint(state):
    return Path(state.session_dir) / "checkpoint.json"


def test_two_none_decisions_stop_without_executing_proposed_changes(tmp_path):
    root = repository(tmp_path)
    original = {path.name: path.read_text() for path in root.glob("*.py")}
    model = Model(review_request(), proposals(), proposals(suffix="second"))
    client = Decisions("none", "none")
    state = agent(tmp_path, model, client).run(TASK, requirements=TaskRequirements.fix())
    assert state.stop_reason == "review_exhausted" and state.status == "incomplete"
    assert len(model.requests) == 3 and len(client.calls) == 2
    assert state.memory["task_state"]["rounds"] == 2
    assert state.memory["task_state"]["phase"] == "stopped"
    assert not state.modified_files and observations(state, "edit_file") == []
    assert observations(state, "run_tests")[0]["status"] == "failed"
    assert {path.name: path.read_text() for path in root.glob("*.py")} == original
    for request in model.requests[1:]:
        assert request["response_format"] == generation_schema() and not request.get("tools")


def test_forced_review_rejects_mixed_proposal_and_mutation_batches(tmp_path):
    root = repository(tmp_path)
    def smuggle(options):
        message = proposals()(options)
        message["tool_calls"] = [tool("edit_file", path="calc.py", old_text="a - b", new_text="a + b")]
        return message
    client = Decisions()
    model = Model(review_request(), smuggle, smuggle)
    state = agent(tmp_path, model, client).run(TASK, requirements=TaskRequirements.fix())
    assert state.stop_reason == "review_exhausted"
    assert state.memory["task_state"]["rounds"] == 0
    assert state.memory["task_state"]["protocol_retries"] == 1
    assert len(model.requests) == 3 and client.calls == []
    assert (root / "calc.py").read_text().endswith("return a - b\n")
    assert not state.modified_files
    assert observations(state, "edit_file") == []
    assert all(result["error"]["code"] == "invalid_proposal" for result in observations(state, "propose_next_steps"))


def test_direct_mixed_review_batches_enter_bounded_review_without_mutating(tmp_path):
    root = repository(tmp_path)
    edit = tool("edit_file", path="calc.py", old_text="a - b", new_text="a + b")
    mixed = review_request()
    mixed["tool_calls"].append(edit)
    def smuggle(options):
        message = proposals()(options)
        message["tool_calls"] = [edit]
        return message
    client = Decisions()
    model = Model(mixed, smuggle)
    state = agent(tmp_path, model, client).run(TASK, requirements=TaskRequirements.fix())
    assert state.stop_reason == "review_exhausted"
    ledger = state.memory["task_state"]
    assert ledger["phase"] == "stopped" and ledger["rounds"] == 0 and ledger["episodes"] == 1
    assert ledger["protocol_retries"] == 1
    assert len(model.requests) == 2 and client.calls == []
    assert not state.modified_files and (root / "calc.py").read_text().endswith("return a - b\n")
    assert len(observations(state, "edit_file")) == 1
    assert observations(state, "edit_file")[0]["error"]["code"] == "review_required"
    assert observations(state, "propose_next_steps")[0]["error"]["code"] == "invalid_proposal"


def test_missing_draft_fields_are_corrected_before_review_and_execution(tmp_path):
    root = repository(tmp_path)
    client = Decisions("p1")
    model = Model(review_request(), malformed_drafts, proposals(),
                  action(tool("edit_file", path="calc.py", old_text="a - b", new_text="a + b")), DONE)
    state = agent(tmp_path, model, client).run(TASK, requirements=TaskRequirements.fix())
    assert state.status == "completed" and state.verification == "passed", state.final_response
    errors = observations(state, "propose_next_steps")
    assert not errors[0]["success"] and errors[0]["retry_allowed"] is True
    assert "expected" in errors[0]["error"]["message"]
    assert errors[1]["success"] and len(client.calls) == 1
    ledger = state.memory["task_state"]
    assert ledger["rounds"] == 1 and ledger["protocol_retries"] == 1
    assert (root / "calc.py").read_text().endswith("return a + b\n")


def test_repeated_malformed_drafts_stop_before_decider_without_spending_review_rounds(tmp_path):
    repository(tmp_path)
    client = Decisions()
    model = Model(review_request(), malformed_drafts, malformed_drafts)
    state = agent(tmp_path, model, client).run(TASK, requirements=TaskRequirements.fix())
    assert state.stop_reason == "review_exhausted" and client.calls == []
    assert not state.modified_files and len(model.requests) == 3
    ledger = state.memory["task_state"]
    assert ledger["rounds"] == 0 and ledger["protocol_retries"] == 1
    assert [item["retry_allowed"] for item in observations(state, "propose_next_steps")] == [True, False]


def test_visible_inspection_is_removed_before_decider_sees_alternatives(tmp_path, monkeypatch):
    repository(tmp_path)
    inspection_range(monkeypatch, 1, 2)
    client = Decisions("p1")
    state = agent(tmp_path, Model(review_request(), proposals(),
                                 action(tool("edit_file", path="calc.py", old_text="a - b", new_text="a + b")), DONE),
                  client).run(TASK, requirements=TaskRequirements.fix())
    assert state.status == "completed", state.final_response
    assert {candidate.id for candidate in client.calls[0][1]} == {"p1", "p2", "none"}
    assert observations(state, "propose_next_steps")[0]["proposal_ids"] == ["p1", "p2"]
    assert observations(state, "read_file") == []


def test_useful_inspection_then_empty_response_can_recover_and_execute(tmp_path, monkeypatch):
    inspection_range(monkeypatch, 3, 3)
    root = repository(tmp_path)
    with (root / "calc.py").open("a") as source:
        source.write("# Integer operands are supported by this helper.\n")
    def check(number, options):
        if number >= 3 and number <= 4:
            assert options["response_format"] == generation_schema() and not options.get("tools")
            assert "implementation ideas" in json.dumps(options["messages"])

    client = Decisions("inspect_evidence", "p1")
    model = Model(review_request(), proposals(), {"role": "assistant", "content": ""},
                  proposals(), action(tool("edit_file", path="calc.py", old_text="a - b", new_text="a + b")), DONE,
                  check=check)
    state = agent(tmp_path, model, client).run(TASK, requirements=TaskRequirements.fix())
    assert state.status == "completed" and state.verification == "passed", state.final_response
    ledger = state.memory["task_state"]
    assert ledger["rounds"] == 2 and ledger["protocol_retries"] == 1 and ledger["evidence_reads"] == 1
    assert len(client.calls) == 2 and len(observations(state, "read_file")) == 1


def test_protocol_correction_budget_survives_resume(tmp_path):
    repository(tmp_path)
    first = agent(tmp_path, Model(review_request(), malformed_drafts), pause=2).run(
        TASK, requirements=TaskRequirements.fix())
    assert first.stop_reason == "paused" and first.memory["task_state"]["protocol_retries"] == 1
    client = Decisions()
    model = Model(malformed_drafts)
    state = agent(tmp_path, model, client).resume(first.session_id, allow_tests=True)
    assert state.stop_reason == "review_exhausted" and len(model.requests) == 1 and client.calls == []
    assert not state.modified_files and state.memory["task_state"]["rounds"] == 0


def test_final_review_round_reserves_implementation_after_failed_selected_patch(tmp_path):
    repository(tmp_path)
    def wrong_approach(options):
        message = proposals()(options)
        draft = json.loads(message["content"])
        draft["approaches"][1]["approach"] = "Return a * b"
        message["content"] = json.dumps(draft)
        return message
    def check(number, options):
        if number == 4:
            assert options["response_format"] == generation_schema() and not options.get("tools")

    model = Model(review_request(), wrong_approach,
                  action(tool("edit_file", path="calc.py", old_text="a - b", new_text="a * b")),
                  proposals(), action(tool("edit_file", path="calc.py", old_text="a * b", new_text="a + b")), DONE,
                  check=check)
    client = Decisions("p2", "p1")
    state = agent(tmp_path, model, client).run(TASK, requirements=TaskRequirements.fix())
    assert state.status == "completed" and state.verification == "passed", state.final_response
    ledger = state.memory["task_state"]
    assert ledger["rounds"] == 2 and ledger["evidence_reads"] == 0 and ledger["protocol_retries"] == 0
    assert [result["status"] for result in observations(state, "run_tests")] == ["failed", "failed", "passed"]


def test_selected_implementation_rejects_other_path_and_accepts_selected_path(tmp_path):
    root = repository(tmp_path, rounding_correct=False)
    model = Model(review_request(), proposals(),
                  action(tool("edit_file", path="rounding.py", old_text="(value + 99) // 100", new_text="value // 100")),
                  action(tool("edit_file", path="calc.py", old_text="a - b", new_text="a + b")))
    instance = agent(tmp_path, model, Decisions("p1"), pause=4)
    state = instance.run(TASK, requirements=TaskRequirements.fix())
    assert state.stop_reason == "paused", state.final_response
    edits = observations(state, "edit_file")
    assert not edits[0]["success"] and "different file" in edits[0]["error"]["message"]
    assert edits[1]["success"] and edits[1]["modified"]
    assert state.modified_files == ["calc.py"]
    assert (root / "calc.py").read_text().endswith("return a + b\n")
    assert "(value + 99) // 100" in (root / "rounding.py").read_text()
    assert state.verification == "failed"  # selecting a plan cannot declare the task complete


def test_specific_ambiguity_can_request_review_before_a_failed_edit(tmp_path):
    repository(tmp_path)
    client = Decisions()
    model = Model(review_request())
    instance = agent(tmp_path, model, client, pause=1)
    state = instance.run(TASK, requirements=TaskRequirements.fix())
    ledger = instance._session.memory.task_state
    assert state.stop_reason == "paused" and ledger.phase == "propose"
    assert ledger.active_goal == "calc.py" and "arithmetic changes" in ledger.review_reason
    assert ledger.episodes == 1 and ledger.rounds == 0
    assert client.calls == [] and not state.modified_files
    names = {item["function"]["name"] for item in model.requests[0]["tools"]}
    assert "request_review" in names and "edit_file" in names


def test_pending_implementation_resume_preserves_choice_without_another_decider_call(tmp_path):
    root = repository(tmp_path)
    first_client = Decisions("p1")
    first_model = Model(review_request(), proposals())
    first = agent(tmp_path, first_model, first_client, pause=2).run(TASK, requirements=TaskRequirements.fix())
    saved = json.loads(checkpoint(first).read_text())
    selected = saved["memory"]["task_state"]["selected"]
    assert first.stop_reason == "paused" and selected["id"] == "p1" and selected["status"] == "selected"
    second_client = Decisions()
    model = Model(action(tool("edit_file", path="calc.py", old_text="a - b", new_text="a + b")), DONE)
    result = agent(tmp_path, model, second_client).resume(first.session_id, allow_tests=True)
    assert result.status == "completed" and result.verification == "passed", result.final_response
    assert len(first_client.calls) == 1 and second_client.calls == []
    assert len(model.requests) == 2 and "Selected next step" in json.dumps(model.requests[0]["messages"])
    assert (root / "calc.py").read_text().endswith("return a + b\n")
    assert observations(result, "run_tests")[-1]["status"] == "passed"


def test_pending_inspection_resume_reads_before_qwen_without_implementation_permission(tmp_path, monkeypatch):
    inspection_range(monkeypatch, 3, 3)
    root = repository(tmp_path)
    with (root / "calc.py").open("a") as source:
        source.write("# Integer operands are supported by this helper.\n")
    first_client = Decisions("inspect_evidence")
    first_model = Model(review_request(), proposals())
    def interrupt_before_read(self, state, deadline):
        raise KeyboardInterrupt
    with monkeypatch.context() as patch:
        patch.setattr(ReviewWorkflow, "execute_inspection", interrupt_before_read)
        first = agent(tmp_path, first_model, first_client).run(TASK, requirements=TaskRequirements.fix())
    saved = json.loads(checkpoint(first).read_text())
    assert first.stop_reason == "interrupted"
    assert saved["memory"]["task_state"]["selected"]["kind"] == "inspect_source"
    assert saved["memory"]["task_state"]["selected"]["status"] == "selected"
    reads_before = len(observations(first, "read_file"))
    def check(number, options):
        assert number == 1
        assert options["response_format"] == generation_schema() and not options.get("tools")
        assert "Integer operands are supported" in json.dumps(options["messages"])

    second_client = Decisions("p1")
    model = Model(proposals(), check=check)
    result = agent(tmp_path, model, second_client, pause=3).resume(first.session_id, allow_tests=True)
    assert result.stop_reason == "paused", result.final_response
    assert len(observations(result, "read_file")) == reads_before + 1
    assert len(first_client.calls) == 1 and len(second_client.calls) == 1
    assert not result.modified_files
    assert "return a - b\n" in (root / "calc.py").read_text()


def test_malformed_selected_checkpoint_is_rejected_before_model_or_decision_calls(tmp_path):
    repository(tmp_path)
    first = agent(tmp_path, Model(review_request(), proposals()), Decisions("p1"), pause=2).run(
        TASK, requirements=TaskRequirements.fix())
    path = checkpoint(first)
    original = json.loads(path.read_text())
    cases = [
        {"kind": "delete_everything"}, {"status": "approved_forever"}, {"selected_step": -1},
        {"path": "unobserved.py"}, {"path": "../outside.py"},
        {"kind": "inspect_source", "start_line": 1, "end_line": 100},
    ]
    for changes in cases:
        saved = deepcopy(original)
        saved["memory"]["task_state"]["selected"].update(changes)
        path.write_text(json.dumps(saved))
        model, client = Model(), Decisions()
        with pytest.raises(ValueError):
            agent(tmp_path, model, client).resume(first.session_id, allow_tests=True)
        assert model.requests == [] and client.calls == []
    path.write_text(json.dumps(original))
