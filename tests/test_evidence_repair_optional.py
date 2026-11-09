"""Optional import enrichment must not block complete, required repair source."""

from dataclasses import replace
from types import SimpleNamespace
import sys
import time

import pytest

from patchrook_agent.agent import Agent, AgentState
from patchrook_agent.completion import TaskRequirements
from patchrook_agent.config import HarnessConfig
from patchrook_agent.evidence import ContextBudgetError, EvidenceStore, RepositoryIndex
from patchrook_agent.evidence_repair import collect_repair_sources
from patchrook_agent.evidence_tools import EvidenceTools
from patchrook_agent.evidence_workflow import EvidenceWorkflow
from patchrook_agent.tools import ToolConfig, ToolRegistry, Workspace
from patchrook_agent.tools.base import ToolError


PRIMARY = "import helper\n\ndef target(value):\n    return helper.adjust(value)\n"


def subject(root, *, maximum=1_048_576):
    workspace = Workspace(root, ToolConfig(max_file_bytes=maximum))
    events = []
    value = SimpleNamespace(workspace=workspace, tools=EvidenceTools(workspace),
        evidence=EvidenceStore(root), index=RepositoryIndex(root), revision=0,
        query="target", task="Repair the observed definition", direct=[], originals={},
        state=SimpleNamespace(modified_files=[], steps=0), routing={"repair": {"source_ranges": []}},
        session=SimpleNamespace(emit=lambda event, step=0, **data: events.append({"event": event, **data})))
    value.events = events
    return value


def collect(value):
    return collect_repair_sources(value, time.monotonic() + 10)


def observed(value, path, first=1, last=None, *, origin="direct"):
    card = value.evidence.observe_source(path, first, last, origin=origin)
    if origin == "direct":
        value.direct.append(card.id)
    return card


def oversized():
    return "def adjust(value):\n" + "    value += 1\n" * 205 + "    return value\n"


def files(root, helper):
    (root / "module.py").write_text(PRIMARY)
    (root / "helper.py").write_text(helper)


def test_optional_oversized_body_is_skipped_with_trace_and_required_body_is_complete(tmp_path):
    files(tmp_path, oversized())
    value = subject(tmp_path)
    observed(value, "module.py")
    cards = collect(value)
    assert [card.path for card in cards] == ["module.py"]
    assert cards[0].text == PRIMARY and cards[0].metadata["repair_role"] == "required_source"
    assert value.events[-1]["event"] == "repair_dependency_skipped"
    assert value.events[-1]["path"] == "helper.py" and "max_read_lines=200" in value.events[-1]["detail"]
    assert value.tools._presented == {}


def test_explicit_oversized_body_remains_required_and_blocks(tmp_path):
    files(tmp_path, oversized())
    value = subject(tmp_path)
    observed(value, "module.py")
    observed(value, "helper.py", 1, 2)
    with pytest.raises(ContextBudgetError, match="max_read_lines=200"):
        collect(value)
    assert value.events == []


def test_optional_malformed_python_is_skipped_instead_of_promoted_to_required_fallback(tmp_path):
    files(tmp_path, "def broken(\n")
    value = subject(tmp_path)
    observed(value, "module.py")
    assert [card.path for card in collect(value)] == ["module.py"]
    assert value.events[-1]["detail"] == "Optional dependency AST is unavailable"


@pytest.mark.parametrize("reason", ["ast_node_limit", "focused_block_limit"])
def test_optional_incomplete_structure_is_skipped_but_explicit_structure_still_blocks(tmp_path, monkeypatch, reason):
    files(tmp_path, "def adjust(value):\n    return value\n")
    value = subject(tmp_path)
    observed(value, "module.py")
    original = value.index.blocks
    def limited(path, *args, **kwargs):
        result = original(path, *args, **kwargs)
        return {**result, "truncated": True, "reasons": [reason]} if path == "helper.py" else result
    monkeypatch.setattr(value.index, "blocks", limited)
    assert [card.path for card in collect(value)] == ["module.py"]
    observed(value, "helper.py")
    with pytest.raises(ContextBudgetError, match="bounded acquisition limit"):
        collect(value)


def test_optional_byte_limit_is_skipped_without_increasing_limits(tmp_path):
    files(tmp_path, 'DESCRIPTION = "' + "x" * 5000 + '"\n')
    value = subject(tmp_path, maximum=4096)
    observed(value, "module.py")
    assert [card.path for card in collect(value)] == ["module.py"]
    assert value.events[-1]["reason"] == "file_too_large"


def test_optional_source_integrity_failure_is_never_silently_skipped(tmp_path, monkeypatch):
    files(tmp_path, "def adjust(value):\n    return value\n")
    value = subject(tmp_path)
    observed(value, "module.py")
    original = value.index.blocks
    def changed(path, *args, **kwargs):
        if path == "helper.py":
            (tmp_path / path).write_text("def adjust(value):\n    return value + 1\n")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(value.index, "blocks", changed)
    with pytest.raises(ToolError) as error:
        collect(value)
    assert error.value.code == "stale_source"
    assert value.events == []


def test_optional_path_guard_remains_fail_closed(tmp_path, monkeypatch):
    files(tmp_path, "def adjust(value):\n    return value\n")
    value = subject(tmp_path)
    observed(value, "module.py")
    original = value.index.blocks
    def swapped_after_discovery(path, *args, **kwargs):
        result = original(path, *args, **kwargs)
        if path == "module.py":
            (tmp_path / "helper.py").rename(tmp_path / "real_helper.py")
            (tmp_path / "helper.py").symlink_to(tmp_path / "real_helper.py")
        return result
    monkeypatch.setattr(value.index, "blocks", swapped_after_discovery)
    with pytest.raises(ToolError) as error:
        collect(value)
    assert error.value.code == "symlink_path"
    assert value.events == []


def test_prior_optional_packet_does_not_become_required_after_retry(tmp_path):
    files(tmp_path, "def adjust(value):\n    return value\n")
    value = subject(tmp_path)
    observed(value, "module.py", origin="dependency")
    first = collect(value)
    assert next(card for card in first if card.path == "helper.py").metadata["repair_role"] == "optional_dependency"
    value.routing["repair"]["source_ranges"] = [
        {"path": card.path, "start_line": card.start_line, "end_line": card.end_line} for card in first]
    (tmp_path / "helper.py").write_text(oversized())
    assert [card.path for card in collect(value)] == ["module.py"]
    assert value.events[-1]["path"] == "helper.py"


class NoModel:
    model = "offline-optional-repair"
    context_size = 8192
    max_output_tokens = 1024
    def chat(self, *args, **kwargs):
        pytest.fail("Repair source preparation must not call a model")


def runtime(root, *, protocol="schema", primary=PRIMARY, failure="assert actual == expected"):
    files(root, 'DESCRIPTION = "' + "optional filler " * 900 + '"\n\ndef adjust(value):\n    return value\n')
    (root / "module.py").write_text(primary)
    base = HarnessConfig.preset("C")
    config = replace(base, workflow_mode="evidence", action_protocol=protocol,
        review_mode="off", persist=False, checkpoint_enabled=False, evidence_context_tokens=3000,
        compaction_enabled=False, context=replace(base.context, enabled=False),
        evidence_test_commands=((sys.executable, "-c", "pass"),))
    agent = Agent(NoModel(), registry=ToolRegistry(Workspace(root, ToolConfig(allow_tests=True))),
                  config=config, runs_dir=root.parent / "runs", max_steps=4, deadline_seconds=30)
    value = EvidenceWorkflow(agent)
    value.requirements = TaskRequirements.fix()
    value._setup("Repair target while preserving unrelated behavior.")
    value.definitions = value._definitions()
    value.enabled = {item["function"]["name"] for item in value.definitions}
    value.state = AgentState()
    value.session.bind(value.state)
    value.routing = {"repair": {"source_ranges": [], "calls_used": 0}}
    value.checks = {"check_1": {"passed": False}}
    value.session.memory.test_results = [{"name": "check_1", "success": False,
                                         "exit_code": 1, "stdout": failure, "stderr": ""}]
    value.events = []
    value.session.emit = lambda event, step=0, **data: value.events.append({"event": event, **data})
    observed(value, "module.py")
    return value


@pytest.mark.parametrize("protocol", ["schema", "native"])
def test_optional_budget_fallback_keeps_complete_required_source_exact_failure_and_schema(tmp_path, protocol):
    value = runtime(tmp_path, protocol=protocol)
    messages = value._prepare_repair(value.state, value._repair_definitions(), time.monotonic() + 10)
    text = "\n".join(message["content"] for message in messages)
    assert "def target(value):" in text and "return helper.adjust(value)" in text
    assert "assert actual == expected" in text and "apply_edits" in text
    assert "optional filler" not in text
    assert {item["path"] for item in value.routing["repair"]["source_ranges"]} == {"module.py"}
    assert set(value.tools._presented) == {"module.py"}
    assert any(item["reason"] == "input_budget" for item in value.events if item["event"] == "repair_dependency_skipped")
    result = value.tools.execute("apply_edits", {"edits": [{"path": "helper.py",
        "old_text": "DESCRIPTION", "new_text": "CHANGED"}]}, deadline=time.monotonic() + 10)
    assert result["error"]["code"] == "source_not_presented"


@pytest.mark.parametrize("protocol", ["schema", "native"])
def test_required_source_overflow_still_blocks_before_edit_authority(tmp_path, protocol):
    primary = 'REQUIRED = "' + "mandatory source " * 1200 + '"\n' + PRIMARY
    value = runtime(tmp_path, protocol=protocol, primary=primary)
    with pytest.raises(ContextBudgetError, match="input budget"):
        value._prepare_repair(value.state, value._repair_definitions(), time.monotonic() + 10)
    assert value.tools._presented == {}


def test_exact_failed_check_is_never_dropped_to_make_a_packet_fit(tmp_path):
    value = runtime(tmp_path, failure="Keep the full observed assertion " * 500)
    with pytest.raises(ContextBudgetError, match="input budget"):
        value._prepare_repair(value.state, value._repair_definitions(), time.monotonic() + 10)
    assert value.tools._presented == {}


@pytest.mark.parametrize("cap, retained", [(None, True), (0, False), (1536, False), (10000, True)])
def test_optional_only_cap_applies_even_when_the_complete_packet_fits(tmp_path, cap, retained):
    value = runtime(tmp_path)
    # Also exercises older HarnessConfig versions without the staged field.
    value.config = SimpleNamespace(**{**vars(value.config), "evidence_context_tokens": 6963,
                                     "evidence_optional_tokens": cap})
    messages = value._prepare_repair(value.state, value._repair_definitions(), time.monotonic() + 10)
    text = "\n".join(message["content"] for message in messages)
    assert ("optional filler" in text) is retained
    assert "def target(value):" in text and "assert actual == expected" in text
    assert value.context_records[-1]["estimated_tokens"] <= 6963
    skipped = [item for item in value.events if item["event"] == "repair_dependency_skipped"]
    assert bool(skipped) is (not retained)
    assert all(item["reason"] == "optional_budget" for item in skipped)


def test_optional_only_cap_never_becomes_a_limit_on_required_source_or_failure(tmp_path):
    primary = 'PRESERVE = "' + "required context " * 100 + '"\n' + PRIMARY
    failure = "Keep this exact observed assertion " * 30
    value = runtime(tmp_path, primary=primary, failure=failure)
    value.config = SimpleNamespace(**{**vars(value.config), "evidence_context_tokens": 6963,
                                     "evidence_optional_tokens": 0})
    messages = value._prepare_repair(value.state, value._repair_definitions(), time.monotonic() + 10)
    text = "\n".join(message["content"] for message in messages)
    assert "required context " * 100 in text and failure in text
    assert "optional filler" not in text
    assert value.context_records[-1]["estimated_tokens"] > 0
