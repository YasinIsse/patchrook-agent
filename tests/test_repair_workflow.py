"""Repair routing changes coder work while verification remains authoritative."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import time

import pytest

from patchrook_agent.agent import Agent
from patchrook_agent.completion import TaskRequirements
from patchrook_agent.config import HarnessConfig
from patchrook_agent.decision_assistance import DecisionAssistance
from patchrook_agent.models.decision import DecisionResult
from patchrook_agent.models.ollama import ModelResponse
from patchrook_agent.repair import RepairItem, RepairPlan
from patchrook_agent.tools import ToolConfig, ToolRegistry, Workspace


def call(name, **arguments):
    return {"role": "assistant", "content": "", "tool_calls": [{"function": {
        "name": name, "arguments": arguments}}]}


DONE = {"role": "assistant", "content": "DONE"}
TASK = "Reproduce the existing failing tests, fix calc.py and rounding.py with targeted edits, and rerun tests."


class Model:
    model = "offline-repair"
    context_size = 8192
    max_output_tokens = 512

    def __init__(self, *script):
        self.script = iter(script)
        self.requests = []

    def chat(self, **options):
        self.requests.append(deepcopy(options))
        return ModelResponse(next(self.script), 0, 0, None, None, None, 0,
                             prompt_tokens_reported=False, completion_tokens_reported=False)


class Decisions:
    def __init__(self, model=None, insufficient=False):
        self.calls = []
        self.model = model
        self.insufficient = insufficient

    def rank(self, context, candidates, **options):
        if self.model:
            assert not self.model.requests, "A warm coder must not be swapped for generic recovery advice"
        self.calls.append((context, deepcopy(candidates), options))
        selected = "insufficient_evidence" if self.insufficient else next(
            item.id for item in candidates if item.description.startswith("calc.py;"))
        order = (selected, *(item.id for item in candidates if item.id != selected))
        return DecisionResult(order, {}, called=False)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr("httpx.Client", lambda *args, **kwargs: pytest.fail("No HTTP in repair regression tests"))


def repository(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (root / "rounding.py").write_text("def floor_cents(value):\n    return (value + 99) // 100\n")
    (root / "test_calc.py").write_text(
        "from calc import add\nfrom rounding import floor_cents\n\n"
        "def test_add():\n    assert add(3, 2) == 5\n\n"
        "def test_floor():\n    assert floor_cents(151) == 1\n")
    return root


def agent(tmp_path, model, client=None, *, decision="on", pause=None):
    return Agent(model, registry=ToolRegistry(Workspace(tmp_path / "repo", ToolConfig(allow_tests=True))),
                 config=replace(HarnessConfig.preset("C"), workflow_mode="coding", decision_model=decision,
                                review_mode="off"),
                 decision_client=client, runs_dir=tmp_path / "runs", pause_after_step=pause, max_steps=8)


def edits():
    return (call("edit_file", path="calc.py", old_text="a - b", new_text="a + b"),
            call("edit_file", path="rounding.py", old_text="(value + 99) // 100", new_text="value // 100"))


def test_failure_selection_happens_once_before_coder_and_continues_without_swaps(tmp_path):
    repository(tmp_path)
    model = Model(*edits(), DONE)
    client = Decisions(model)
    result = agent(tmp_path, model, client).run(TASK, requirements=TaskRequirements.fix())
    assert result.status == "completed", result.final_response
    assert result.verification == "passed" and result.completion["satisfied"]
    assert len(client.calls) == 1 and client.calls[0][2]["purpose"] == "repair"
    assert "assert" in client.calls[0][0]
    assert all("Repair phase:" in json.dumps(request["messages"]) for request in model.requests[:2])
    assert set(result.modified_files) == {"calc.py", "rounding.py"}
    events = [json.loads(line) for line in (Path(result.session_dir) / "events.jsonl").read_text().splitlines()]
    assert next(event for event in events if event["event"] == "repair_packet")["selected_path"] == "calc.py"
    assert all(row["estimated_tokens"] <= 4000 for row in events if row["event"] == "context"
               and row["reason"] == "repair_working_set")


def test_rules_and_production_off_use_same_compact_repair_without_decision_service(tmp_path):
    repository(tmp_path)
    model = Model(*edits(), DONE)
    result = agent(tmp_path, model, decision="off").run(TASK, requirements=TaskRequirements.fix())
    assert result.status == "completed" and result.metrics["decision_calls"] == 0
    assert "Repair phase:" in json.dumps(model.requests[0]["messages"])


def test_insufficient_evidence_executes_bounded_read_before_coder(tmp_path):
    repository(tmp_path)
    model = Model(*edits(), DONE)
    result = agent(tmp_path, model, Decisions(model, insufficient=True)).run(TASK, requirements=TaskRequirements.fix())
    assert result.status == "completed", result.final_response
    calls = [message for message in result.messages if message.get("tool_name") == "read_file"]
    assert len(calls) == 1
    observed = json.loads(calls[0]["content"])
    assert observed["path"] in {"calc.py", "rounding.py"} and observed["end_line"] - observed["start_line"] < 80


def test_resume_reuses_selection_and_narrow_recovery_state(tmp_path):
    repository(tmp_path)
    model = Model(call("read_file", path="calc.py", start_line=1, end_line=2))
    first_client = Decisions(model)
    first = agent(tmp_path, model, first_client, pause=1).run(TASK, requirements=TaskRequirements.fix())
    assert first.stop_reason == "paused"
    second_model = Model(*edits(), DONE)
    second_client = Decisions(second_model)
    result = agent(tmp_path, second_model, second_client).resume(first.session_id, allow_tests=True)
    assert result.status == "completed", result.final_response
    assert second_client.calls == []
    assert len(first_client.calls) == 1


def test_header_expansion_cannot_reset_recovery_across_resume(tmp_path):
    root = repository(tmp_path)
    with (root / "calc.py").open("a") as handle:
        handle.write("# unrelated helper follows\ndef label(value):\n    return str(value)\n")
    read = call("read_file", path="calc.py", start_line=1, end_line=2)
    header = call("read_file", path="calc.py", start_line=3, end_line=4)
    model = Model(read, read, header)
    first_client = Decisions(model)
    first = agent(tmp_path, model, first_client, pause=3).run(TASK, requirements=TaskRequirements.fix())
    assert first.stop_reason == "paused"
    saved = json.loads((Path(first.session_dir) / "checkpoint.json").read_text())
    assert saved["execution"]["coding"]["repair"]["narrow"]
    second_client = Decisions()
    result = agent(tmp_path, Model(read), second_client).resume(first.session_id, allow_tests=True)
    assert result.stop_reason == "stalled_progress" and result.steps == 4
    assert second_client.calls == []
    saved = json.loads((Path(result.session_dir) / "checkpoint.json").read_text())
    assert saved["execution"]["coding"]["repair"]["narrow"]


def test_two_missing_middle_reads_remain_available_for_the_following_patch(tmp_path):
    root = repository(tmp_path)
    (root / "rounding.py").write_text("def floor_cents(value):\n    return value // 100\n")
    lines = ["def add(a, b):", *[f"    pad_{index} = a" for index in range(20)],
             "    value = a - b", *[f"    pad_{index} = b" for index in range(20, 40)], "    return value"]
    (root / "calc.py").write_text("\n".join(lines) + "\n")
    middle = lines.index("    value = a - b") + 1
    model = Model(call("read_file", path="calc.py", start_line=middle, end_line=middle),
                  call("read_file", path="calc.py", start_line=middle + 1, end_line=middle + 1),
                  call("edit_file", path="calc.py", old_text="value = a - b", new_text="value = a + b"), DONE)
    result = agent(tmp_path, model, decision="off").run(TASK, requirements=TaskRequirements.fix())
    assert result.status == "completed", result.final_response
    instance = agent(tmp_path, Model(), decision="off")
    assert middle not in instance._coding_visible_lines("calc.py", model.requests[0]["messages"])
    assert middle in instance._coding_visible_lines("calc.py", model.requests[2]["messages"])
    assert middle + 1 in instance._coding_visible_lines("calc.py", model.requests[2]["messages"])


def test_semantic_decision_memo_survives_old_ttl_and_checkpoint(tmp_path):
    root = repository(tmp_path)
    config = replace(HarnessConfig.preset("C"), workflow_mode="coding", decision_model="on")
    client = Decisions()
    plan = RepairPlan("a" * 64, "test_calc.py:5 assert add(3, 2) == 5", (
        RepairItem("work_a", "calc.py", "add", "calc.py; add", (("calc.py", 1, 2),), ("test_add",)),
        RepairItem("work_b", "rounding.py", "floor", "rounding.py; floor", (("rounding.py", 1, 2),), ("test_floor",)),
    ))
    controller = DecisionAssistance(config, Model(), Workspace(root), client=client)
    assert controller.select_repair(TASK, plan, time.monotonic() + 60) == "work_a"
    saved = controller.to_dict()
    restored = DecisionAssistance(config, Model(), Workspace(root), client=Decisions())
    restored.restore(saved)
    assert restored.select_repair(TASK, plan, time.monotonic() + 60, allow_call=False) == "work_a"
    assert restored.client.calls == []
    assert saved["repair_calls"] == 1


def test_old_decision_checkpoint_restores_without_new_repair_fields(tmp_path):
    root = repository(tmp_path)
    config = replace(HarnessConfig.preset("C"), workflow_mode="coding", decision_model="on")
    controller = DecisionAssistance(config, Model(), Workspace(root), client=Decisions())
    saved = controller.to_dict()
    saved.pop("repair_calls")
    saved.pop("repair_choices")
    controller.restore(saved)
    assert controller.repair_calls == 0 and controller.repair_choices == {}


def test_unknown_decision_ids_do_not_choose_unobserved_work(tmp_path):
    root = repository(tmp_path)
    class Untrusted:
        def rank(self, *args, **kwargs):
            return DecisionResult(("unseen",), {}, called=False)
    controller = DecisionAssistance(replace(HarnessConfig.preset("C"), workflow_mode="coding", decision_model="on"),
                                    Model(), Workspace(root), client=Untrusted())
    plan = RepairPlan("b" * 64, "assertion failed", (
        RepairItem("a", "calc.py", "add", "calc.py; add", (), ()),
        RepairItem("b", "rounding.py", "floor", "rounding.py; floor", (), ()),
    ))
    assert controller.select_repair(TASK, plan, time.monotonic() + 60) == "a"


def test_completion_review_retains_two_requested_ranges_for_remaining_work(tmp_path):
    root = repository(tmp_path)
    (root / "rounding.py").write_text(
        "def floor_cents(value):\n    return value // 100\n\n\n"
        "def left(value):\n    return value - 11\n\n\n"
        "def right(value):\n    return value - 22\n")
    remaining = call("edit_file", path="rounding.py", old_text="value - 11", new_text="value + 11")
    remaining["tool_calls"].extend(call(
        "edit_file", path="rounding.py", old_text="value - 22", new_text="value + 22")["tool_calls"])
    model = Model(edits()[0], call("read_file", path="rounding.py", start_line=5, end_line=6),
                  call("read_file", path="rounding.py", start_line=9, end_line=10), remaining, DONE)
    instance = agent(tmp_path, model, decision="off")
    result = instance.run(
        "Reproduce the failing tests and fix addition in calc.py. Also correct left to add 11 "
        "and right to add 22 in rounding.py. Preserve the existing tests.",
        requirements=TaskRequirements.fix())
    assert result.status == "completed", result.final_response
    assert result.verification == "passed"
    observed = instance._coding_visible_lines("rounding.py", model.requests[3]["messages"])
    assert observed[6] == "    return value - 11"
    assert observed[10] == "    return value - 22"
    assert "return value + 11" in (root / "rounding.py").read_text()
    assert "return value + 22" in (root / "rounding.py").read_text()
    assert sum(message.get("tool_name") == "edit_file" for message in result.messages) == 3
