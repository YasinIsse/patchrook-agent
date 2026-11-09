"""Optional implementation examples are static, budgeted and hash-fresh."""

from dataclasses import replace
import json
from pathlib import Path
import time

import pytest

from patchrook_agent.evidence.compiler import estimate_tokens
from patchrook_agent.evidence_owner_context import enrich_owner_context, owner_context_candidates
from patchrook_agent.evidence_workflow import _source_message
from patchrook_agent.tools.base import ToolError
from test_evidence_repair_optional import subject
from test_evidence_symbol_bootstrap import runtime_factory


TASK = "Implement `partial_convert` on `Handler` for dict input and attrs fields."
SOURCE = ("from .helpers import fields\n\nclass Handler:\n"
          "    def partial_convert(self, data):\n        pass\n\n"
          "    def convert(self, data):\n        return self.dispatch(data)\n\n"
          "    def convert_tuple(self, data):\n        return tuple(data)\n\n"
          "    def convert_dict(self, data):\n        return fields(data)\n")
HELPER = "raise RuntimeError('Repository modules must never execute')\n\ndef fields(data):\n    return data.items()\n"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr("httpx.Client", lambda *a, **k: pytest.fail("No model or network access"))


def make(tmp_path, source=SOURCE, helper=HELPER, task=TASK, *, view=(3, 5)):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg/api.py").write_text(source)
    (tmp_path / "pkg/helpers.py").write_text(helper)
    runtime = subject(tmp_path)
    runtime.task = task
    card = runtime.evidence.observe_source("pkg/api.py", *view, origin="repair_packet",
                                         metadata={"repair_role": "required_source"})
    return runtime, [card]


def candidates(runtime, cards):
    return owner_context_candidates(runtime, cards, time.monotonic() + 10, _source_message)


def packet(runtime, cards):
    return [{"role": "system", "content": "Repair using current source."},
            {"role": "user", "content": runtime.task}, *[_source_message(card) for card in cards],
            {"role": "user", "content": "Observed failures: ordinary assertion."}]


def enrich(runtime, cards, **options):
    messages = packet(runtime, cards)
    return enrich_owner_context(runtime, cards, messages, time.monotonic() + 10,
        render=_source_message, input_limit=options.pop("input_limit", 9420),
        optional_limit=options.pop("optional_limit", 1536), **options)


def test_generic_suffix_task_overlap_and_one_import_hop_select_complete_implementations(tmp_path):
    runtime, cards = make(tmp_path)
    plans = candidates(runtime, cards)
    assert [plan.key for plan in plans] == ["base", "sibling", "callee"]
    assert "def convert(self" in plans[0].text
    assert "def convert_dict" in plans[1].text and "convert_tuple" not in plans[1].text
    assert plans[2].path == "pkg/helpers.py" and plans[2].text == "def fields(data):\n    return data.items()\n"
    assert "raise RuntimeError" not in plans[2].text
    assert runtime.tools._presented == {}


def test_longest_existing_suffix_takes_precedence_over_a_shorter_one(tmp_path):
    source = SOURCE.replace("def partial_convert(self", "def new_partial_convert(self")
    source += "\n    def partial_convert(self, data):\n        return self.convert(data)\n"
    runtime, cards = make(tmp_path, source, task="Implement `new_partial_convert` on `Handler`.")
    plans = candidates(runtime, cards)
    assert "def partial_convert(self" in plans[0].text and "new_partial_convert" not in plans[0].text


@pytest.mark.parametrize("change", ["parameter", "local", "module", "nested"])
def test_shadowed_or_nested_names_are_not_resolved_as_module_imported_callees(tmp_path, change):
    source = SOURCE
    if change == "parameter":
        source = source.replace("def convert_dict(self, data):", "def convert_dict(self, data, fields):")
    elif change == "local":
        source = source.replace("        return fields(data)", "        fields = lambda value: value\n        return fields(data)")
    elif change == "module":
        source += "\nfields = lambda value: value\n"
    else:
        source = source.replace("        return fields(data)", "        def nested():\n            return fields(data)\n        return data")
    runtime, cards = make(tmp_path, source)
    assert [plan.key for plan in candidates(runtime, cards)] == ["base", "sibling"]


def test_ambiguous_absolute_import_is_omitted_without_executing_either_candidate(tmp_path):
    source = SOURCE.replace("from .helpers", "from helpers")
    runtime, cards = make(tmp_path, source)
    (tmp_path / "helpers.py").write_text(HELPER)
    (tmp_path / "src").mkdir()
    (tmp_path / "src/helpers.py").write_text(HELPER)
    assert [plan.key for plan in candidates(runtime, cards)] == ["base", "sibling"]


def test_absolute_import_cannot_bind_a_same_package_sibling(tmp_path):
    runtime, cards = make(tmp_path, SOURCE.replace("from .helpers", "from helpers"))
    assert [plan.key for plan in candidates(runtime, cards)] == ["base", "sibling"]


def test_star_import_does_not_claim_a_known_imported_binding(tmp_path):
    runtime, cards = make(tmp_path, SOURCE + "\nfrom other import *\n")
    assert [plan.key for plan in candidates(runtime, cards)] == ["base", "sibling"]


def test_nested_owner_closure_is_conservatively_omitted(tmp_path):
    source = "from .helpers import fields\n\ndef factory(fields):\n" + "\n".join(
        "    " + line for line in SOURCE.splitlines()[2:]) + "\n"
    runtime, cards = make(tmp_path, source, view=(1, len(source.splitlines())))
    assert candidates(runtime, cards) == []


@pytest.mark.parametrize("change", ["plain_task", "implemented", "ambiguous_owner", "oversized_base"])
def test_only_one_observed_task_named_stub_owner_can_enrich(tmp_path, change):
    source, task, view = SOURCE, TASK, (3, 5)
    if change == "plain_task":
        task = "Implement partial_convert on Handler for dict input."
    elif change == "implemented":
        source = source.replace("        pass", "        return self.convert(data)", 1)
    elif change == "ambiguous_owner":
        source += "\nclass Other:\n    def partial_convert(self, data):\n        pass\n"
        view = (1, len(source.splitlines()))
    else:
        source = source.replace("        return self.dispatch(data)", "        value = data\n" * 45 + "        return value")
    runtime, cards = make(tmp_path, source, task=task, view=view)
    assert candidates(runtime, cards) == []


def test_large_constructor_and_oversized_callee_are_never_clipped_or_selected(tmp_path):
    source = SOURCE + "\n    def __init__(self):\n" + "        self.option = True\n" * 150
    helper = "def fields(data):\n" + "    value = data\n" * 21 + "    return value\n"
    runtime, cards = make(tmp_path, source, helper)
    plans = candidates(runtime, cards)
    assert [plan.key for plan in plans] == ["base", "sibling"]
    assert all("__init__" not in plan.text for plan in plans)


@pytest.mark.parametrize("change", ["symlink", "test_path"])
def test_guarded_or_test_import_paths_cannot_supply_optional_source(tmp_path, change):
    runtime, cards = make(tmp_path, SOURCE.replace("from .helpers", "from .tests.helpers") if change == "test_path" else SOURCE)
    if change == "symlink":
        target = tmp_path.parent / "outside-owner-context.py"
        target.write_text(HELPER)
        (tmp_path / "pkg/helpers.py").unlink()
        (tmp_path / "pkg/helpers.py").symlink_to(target)
    else:
        (tmp_path / "pkg/tests").mkdir()
        (tmp_path / "pkg/tests/helpers.py").write_text(HELPER)
    assert [plan.key for plan in candidates(runtime, cards)] == ["base", "sibling"]


def test_budget_counts_existing_optional_messages_native_schema_and_correction(tmp_path):
    runtime, cards = make(tmp_path)
    existing = runtime.evidence.observe_source("pkg/helpers.py", 1, 1, origin="repair_packet",
                                               metadata={"repair_role": "optional_dependency"})
    cards.append(existing)
    original = packet(runtime, cards)
    required_cost = estimate_tokens([*original[:2], _source_message(cards[0]), original[-1]])
    existing_optional = estimate_tokens(original) - required_cost
    base_cost = estimate_tokens([_source_message(candidates(runtime, cards)[0].preview())]) - 4
    enriched, messages = enrich(runtime, cards, optional_limit=existing_optional + base_cost,
                                native_cost=100, correction=17)
    assert len(enriched) == len(cards) + 1
    assert enriched[-1].metadata["context_key"] == "base"
    assert estimate_tokens(messages) - required_cost <= existing_optional + base_cost
    assert messages[:2] == original[:2] and messages[-1] == original[-1]
    assert all(_source_message(card) in messages for card in cards)
    none, unchanged = enrich(runtime, cards, input_limit=estimate_tokens(original) + 116,
                             native_cost=100, correction=17)
    assert none == cards and unchanged == original


def test_acquisition_needs_actual_presentation_and_stale_hashes_reject_edits(tmp_path):
    runtime, cards = make(tmp_path)
    enriched, messages = enrich(runtime, cards)
    edit = {"edits": [{"path": "pkg/api.py", "old_text": "return self.dispatch(data)", "new_text": "return data"}]}
    assert runtime.tools.execute("apply_edits", edit, deadline=time.monotonic() + 10)["error"]["code"] == "source_not_presented"
    registration = runtime.tools.register_presented([{"path": card.path, "start_line": card.start_line,
        "end_line": card.end_line, "content": card.text, "source_sha256": card.source_sha256} for card in enriched])
    assert not registration.get("rejected")
    (tmp_path / "pkg/api.py").write_text(SOURCE + "\nEXTERNAL = True\n")
    assert runtime.tools.execute("apply_edits", edit, deadline=time.monotonic() + 10)["error"]["code"] == "stale_source"


def test_edited_required_snapshot_is_not_combined_with_current_optional_source(tmp_path):
    runtime, cards = make(tmp_path)
    (tmp_path / "pkg/api.py").write_text(SOURCE + "\nEXTERNAL = True\n")
    with pytest.raises(ToolError, match="Required source changed"):
        enrich(runtime, cards)


@pytest.mark.parametrize("change", ["deleted", "hash_changed"])
def test_separate_optional_import_failure_skips_without_recording_a_partial_card(tmp_path, monkeypatch, change):
    runtime, cards = make(tmp_path)
    original = runtime.evidence._read
    before = len(runtime.evidence.cards(fresh=False))
    def changed(path, deadline=None):
        if path == "pkg/helpers.py":
            if change == "deleted":
                (tmp_path / path).unlink()
            else:
                (tmp_path / path).write_text(HELPER + "\nEXTERNAL = True\n")
        return original(path, deadline)
    monkeypatch.setattr(runtime.evidence, "_read", changed)
    enriched, messages = enrich(runtime, cards)
    assert [card.metadata.get("context_key") for card in enriched[1:]] == ["base", "sibling"]
    assert len(runtime.evidence.cards(fresh=False)) == before + 2
    assert all("pkg/helpers.py" not in message["content"] for message in messages)
    assert runtime.events[-1]["reason"] == "optional_owner_context_unavailable"


def test_expired_optional_discovery_does_no_file_work(tmp_path, monkeypatch):
    runtime, cards = make(tmp_path)
    monkeypatch.setattr(runtime.tools, "_bytes", lambda *a: pytest.fail("No optional I/O after deadline"))
    assert owner_context_candidates(runtime, cards, time.monotonic() - 1, _source_message) == []


def test_workflow_preserves_required_focus_and_optional_authority_on_second_packet(runtime_factory):
    source = SOURCE.replace("    def convert(self", "    def unrelated(self):\n" + "        unused = 1\n" * 90 + "\n    def convert(self")
    runtime = runtime_factory(TASK, {"pkg/api.py": source, "pkg/helpers.py": HELPER})
    runtime.config = replace(runtime.config, evidence_optional_tokens=1536)
    runtime.routing = {"initial_model": "small", "repair": {"calls_used": 0, "source_ranges": []}}
    deadline = time.monotonic() + 10
    runtime._perform_action(runtime.state, {"name": "read_file", "arguments": {
        "path": "pkg/api.py", "start_line": 3, "end_line": 5}}, deadline, automatic=True)
    definitions = runtime._repair_definitions()
    first = runtime._prepare_repair(runtime.state, definitions, deadline)
    first_focus = list(runtime.routing["repair"]["source_ranges"])
    assert any("def convert_dict" in message["content"] for message in first)
    record = runtime.context_records[-1]
    owner = [runtime.evidence.get(identifier) for identifier in record["presented_source_ids"]
             if runtime.evidence.get(identifier).metadata.get("repair_role") == "optional_owner_context"]
    assert owner and all({"path": card.path, "start_line": card.start_line, "end_line": card.end_line}
                         not in first_focus for card in owner)
    second = runtime._prepare_repair(runtime.state, definitions, deadline)
    assert runtime.routing["repair"]["source_ranges"] == first_focus
    assert any("def convert_dict" in message["content"] for message in second)
    assert record["estimated_tokens"] <= record["input_limit"]
    assert "pkg/api.py" in runtime.tools._presented
