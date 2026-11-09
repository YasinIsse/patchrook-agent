"""Rejected header anchors must not displace complete focused-repair source."""

from copy import deepcopy
from dataclasses import replace
import time

import pytest

from patchrook_agent.evidence_repair import collect_repair_sources
from test_evidence_repair_focus import subject, module, observe, collect
from test_evidence_symbol_bootstrap import runtime_factory


@pytest.fixture(autouse=True)
def no_models(monkeypatch):
    monkeypatch.setattr("httpx.Client", lambda *a, **k: pytest.fail("No model or HTTP calls"))


def test_active_repair_keeps_previous_complete_function_after_header_only_pin(tmp_path):
    text = module()
    (tmp_path / "module.py").write_text(text)
    runtime = subject(tmp_path)
    runtime.routing["repair"].update(status="active", calls_used=1, source_ranges=[
        {"path": "module.py", "start_line": 6, "end_line": 7}])
    header = runtime.evidence.observe_source("module.py", 3, 5, origin="repair_packet",
                                             metadata={"repair_role": "required_source"})
    runtime.direct = [header.id]
    cards = collect(runtime)
    source = "\n".join(card.text for card in cards)
    assert "def target" in source and "return CONSTANT" in source
    assert "def ignored" not in source
    assert runtime.routing["repair"]["calls_used"] == 1
    assert runtime.tools._presented == {}


@pytest.mark.parametrize("status", ["idle", "requested", "verified", "exhausted"])
def test_nonactive_navigation_still_follows_latest_direct_read(tmp_path, status):
    text = module()
    (tmp_path / "module.py").write_text(text)
    runtime = subject(tmp_path)
    runtime.routing["repair"].update(status=status, source_ranges=[
        {"path": "module.py", "start_line": 6, "end_line": 7}])
    observe(runtime, text, "ignored")
    source = "\n".join(card.text for card in collect(runtime))
    assert "def ignored" in source and "def target" not in source


def test_actual_mutation_overrides_old_focus_and_header_pin(tmp_path):
    original = module()
    current = original.replace("return 2", "return 99")
    (tmp_path / "module.py").write_text(current)
    runtime = subject(tmp_path)
    runtime.routing["repair"].update(status="active", source_ranges=[
        {"path": "module.py", "start_line": 6, "end_line": 7}])
    runtime.originals = {"module.py": original}
    runtime.state.modified_files = ["module.py"]
    header = runtime.evidence.observe_source("module.py", 3, 5, origin="direct")
    runtime.direct = [header.id]
    source = "\n".join(card.text for card in collect(runtime))
    assert "def ignored" in source and "return 99" in source
    assert "def target" not in source


def test_prior_selectors_are_reread_and_do_not_replay_stale_source(tmp_path):
    original = module()
    (tmp_path / "module.py").write_text(original)
    runtime = subject(tmp_path)
    runtime.routing["repair"].update(status="active", source_ranges=[
        {"path": "module.py", "start_line": 6, "end_line": 7}])
    header = runtime.evidence.observe_source("module.py", 3, 5, origin="direct")
    runtime.direct = [header.id]
    (tmp_path / "module.py").write_text(original.replace("return CONSTANT", "return CONSTANT + 2"))
    cards = collect(runtime)
    source = "\n".join(card.text for card in cards)
    assert "return CONSTANT + 2" in source
    assert all(runtime.evidence.is_fresh(card) for card in cards)
    assert runtime.tools._presented == {}


def test_real_rejected_header_insertion_preserves_method_and_optional_examples(runtime_factory):
    source = ("from .helpers import fields\n\nclass Outcome:\n    pass\n\nclass Handler:\n"
        "    def partial_convert(self, data):\n        \"\"\"Missing behavior.\"\"\"\n\n"
        "    def convert(self, data):\n        return data\n\n"
        "    def convert_dict(self, data):\n        return fields(data)\n\n"
        "    def unrelated(self):\n" + "        keep = 1\n" * 90)
    runtime = runtime_factory("Implement `partial_convert` on `Handler`, returning `Outcome` for dict input.", {
        "pkg/module.py": source, "pkg/helpers.py": "def fields(data):\n    return data\n"})
    runtime.workspace.config = replace(runtime.workspace.config, validate_python_edits=True)
    runtime.routing = {"initial_model": "small", "active_model": "large", "repair": {
        "status": "active", "calls_used": 1, "source_ranges": []}}
    deadline = time.monotonic() + 10
    runtime._perform_action(runtime.state, {"name": "read_file", "arguments": {
        "path": "pkg/module.py", "start_line": 6, "end_line": 8}}, deadline, automatic=True)
    definitions = runtime._repair_definitions()
    first = runtime._prepare_repair(runtime.state, definitions, deadline)
    before_ranges = deepcopy(runtime.routing["repair"]["source_ranges"])
    result = runtime._perform_action(runtime.state, {"name": "apply_edits", "arguments": {"edits": [{
        "path": "pkg/module.py", "old_text": "class Handler:",
        "new_text": "class Handler:\n    def partial_convert(self, data):\n        return data\n"}]}}, deadline)
    assert not result["success"] and result["error"]["code"] == "python_duplicate_function"
    assert (runtime.workspace.root / "pkg/module.py").read_text() == source
    second = runtime._prepare_repair(runtime.state, definitions, deadline)
    requested = "\n".join(message["content"] for message in second if message["content"].startswith("Requested source"))
    assert 'def partial_convert(self, data):' in requested and '"""Missing behavior."""' in requested
    assert "def convert_dict" in requested
    assert "keep = 1" not in requested
    assert runtime.routing["repair"]["calls_used"] == 1
    assert runtime.revision == 0 and runtime.state.modified_files == []
    assert runtime.routing["repair"]["source_ranges"] == before_ranges
    assert runtime.context_records[-1]["estimated_tokens"] <= runtime.context_records[-1]["input_limit"]
