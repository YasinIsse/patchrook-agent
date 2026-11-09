"""Exact provenance, whole-checkout coverage, and bounded semantic admission."""

from dataclasses import FrozenInstanceError
from contextlib import contextmanager
from types import SimpleNamespace
import json
from pathlib import Path
import time

import pytest

from patchrook_agent.evidence import ContextBudgetError, ContextCompiler, EvidenceStore, RepositoryIndex
from patchrook_agent.evidence.compiler import estimate_tokens
from patchrook_agent.models.decision import DecisionResult


def source(tmp_path, text="def target():\n    return 1\n", name="module.py"):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def crowded_store(tmp_path, state_path=None):
    store = EvidenceStore(tmp_path, state_path)
    for number in range(12):
        name = f"module_{number}.py"
        source(tmp_path, f"def target_{number}():\n    return {number}  # " + "target " * 20 + "\n", name)
        store.observe_source(name)
    return store


class Decision:
    timeout_seconds = 20

    def __init__(self, label="keep", fail=False):
        self.label, self.fail = label, fail
        self.requests = []

    def rank(self, context, candidates, **options):
        self.requests.append((context, candidates, options))
        if self.fail:
            raise OSError("unavailable")
        others = [value for value in ("keep", "drop", "uncertain") if value != self.label]
        return DecisionResult((self.label, *others),
                              {self.label: .9, others[0]: .06, others[1]: .04},
                              elapsed_seconds=1.5, inference_seconds=1, management_seconds=.5,
                              input_tokens=120, called=True)


def test_card_preserves_exact_source_and_is_immutable(tmp_path):
    source(tmp_path, "# keep comment\ndef target():\n    return 1\n")
    card = EvidenceStore(tmp_path).observe_source("module.py", 2, 3, revision=4)
    assert card.text == "def target():\n    return 1\n"
    assert (card.start_line, card.end_line, card.revision) == (2, 3, 4)
    assert len(card.source_sha256) == 64
    with pytest.raises(FrozenInstanceError):
        card.text = "changed"
    metadata = card.metadata
    metadata["origin"] = "changed"
    assert card.metadata["origin"] == "read"


def test_full_file_hash_invalidates_card_when_unseen_lines_change(tmp_path):
    path = source(tmp_path, "first\nsecond\n")
    store = EvidenceStore(tmp_path)
    card = store.observe_source("module.py", 1, 1)
    path.write_text("first\nchanged elsewhere\n")
    assert store.get(card.id) is None
    assert store.get(card.id, fresh=False) == card
    assert store.observe_source("module.py", 1, 1).id != card.id


def test_store_atomic_persistence_and_revision_resume(tmp_path):
    source(tmp_path)
    state_path = tmp_path / "state" / "evidence.json"
    store = EvidenceStore(tmp_path, state_path)
    card = store.observe_source("module.py", revision=2)
    store.policy_state["decision_calls"] = 1
    store.save()
    resumed = EvidenceStore(tmp_path, state_path)
    assert resumed.get(card.id) == card
    assert resumed.sequence == 1
    assert resumed.policy_state["decision_calls"] == 1
    assert not list(state_path.parent.glob(".evidence-*"))


def test_store_rejects_wrong_repository_checkpoint(tmp_path):
    original = tmp_path / "one"
    original.mkdir()
    other = tmp_path / "two"
    other.mkdir()
    state = tmp_path / "saved.json"
    EvidenceStore(original, state).save()
    with pytest.raises(ValueError, match="different schema or repository"):
        EvidenceStore(other, state)


def test_source_range_cannot_claim_unobserved_or_missing_lines(tmp_path):
    source(tmp_path)
    store = EvidenceStore(tmp_path)
    with pytest.raises(ValueError):
        store.observe_source("module.py", 1, 5)
    with pytest.raises(ValueError):
        store.add_observation("source", "invented", path="module.py")


def test_store_and_index_reject_symlink_escape_and_excluded_paths(tmp_path):
    repo, outside = tmp_path / "repo", tmp_path / "outside.py"
    repo.mkdir()
    outside.write_text("secret\n")
    (repo / "escape.py").symlink_to(outside)
    source(repo, "hidden\n", ".git/secret.py")
    store = EvidenceStore(repo)
    for path in ["escape.py", "../outside.py", ".git/secret.py"]:
        with pytest.raises(ValueError):
            store.observe_source(path)
    index = RepositoryIndex(repo)
    assert index.refresh()["paths"] == []


def test_inventory_covers_files_beyond_old_128_path_cap(tmp_path):
    for number in range(160):
        source(tmp_path, "value = 1\n", f"pkg/module_{number:03}.py")
    source(tmp_path, "target_unique = 2\n", "z_last.py")
    result = RepositoryIndex(tmp_path).refresh()
    assert result["count"] == 161
    assert "z_last.py" in result["paths"]
    assert not result["truncated"]
    result = RepositoryIndex(tmp_path).search("target_unique")
    assert result["matches"][0]["path"] == "z_last.py"


def test_inventory_explicitly_reports_byte_budget(tmp_path):
    source(tmp_path)
    result = RepositoryIndex(tmp_path, inventory_byte_limit=3).refresh()
    assert result["truncated"]
    assert "output_byte_limit" in result["reasons"]
    assert result["paths"] == []


def test_inventory_ignores_gitignore_for_eligible_checkout_files(tmp_path):
    source(tmp_path, "ignored.py\n", ".gitignore")
    source(tmp_path, "target_unique = 1\n", "ignored.py")
    index = RepositoryIndex(tmp_path)
    assert "ignored.py" in index.refresh()["paths"]
    assert index.search("target_unique")["matches"][0]["path"] == "ignored.py"


def test_index_has_bounded_fallback_without_rg(tmp_path, monkeypatch):
    monkeypatch.setattr("patchrook_agent.evidence.index.shutil.which", lambda name: None)
    source(tmp_path, "needle = 1\n")
    index = RepositoryIndex(tmp_path)
    assert index.refresh()["count"] == 1
    assert index.search("needle")["matches"] == [{"path": "module.py", "line_number": 1,
                                                    "text": "needle = 1\n"}]


def test_lazy_ast_preserves_guard_comment_and_dependency(tmp_path):
    source(tmp_path, "from helper import calculate\n\n@decorator\ndef target(value):\n    # preserve\n    if value < 0:\n        raise ValueError\n    return calculate(value)\n")
    source(tmp_path, "def calculate(value):\n    return value\n", "helper.py")
    result = RepositoryIndex(tmp_path).blocks("module.py", "target")
    assert result["selectors"][0] == {"path": "module.py", "start_line": 3,
                                       "end_line": 8, "symbol": "target"}
    assert result["dependencies"] == ["helper.py"]
    card = EvidenceStore(tmp_path).observe_source("module.py", 3, 8)
    assert "# preserve" in card.text and "raise ValueError" in card.text


def test_index_deadline_and_oversize_are_visible(tmp_path):
    source(tmp_path, "x" * 200)
    index = RepositoryIndex(tmp_path, max_file_bytes=100)
    assert index.refresh(deadline=time.monotonic() - 1)["truncated"]
    result = index.blocks("module.py")
    assert result["reasons"] == ["file_byte_limit"]


def test_search_result_limit_explicit(tmp_path):
    source(tmp_path, "needle\n" * 10)
    result = RepositoryIndex(tmp_path).search("needle", limit=2)
    assert len(result["matches"]) == 2
    assert result["truncated"] and "result_limit" in result["reasons"]


def test_compiler_pins_mandatory_and_recent_exactly(tmp_path):
    store = crowded_store(tmp_path)
    mandatory = [{"role": "user", "content": "Required: preserve validation."}]
    recent = [{"role": "user", "content": "Fresh failure: AssertionError"}]
    result = ContextCompiler(store).prepare("system", "task", mandatory, recent, query="target", budget=600)
    assert result.messages[:4] == [{"role": "system", "content": "system"},
                                   {"role": "user", "content": "task"}, *mandatory, *recent]
    assert result.estimated_tokens == estimate_tokens(result.messages) <= 600
    assert result.selected_cards and result.dropped_ids
    assert "sha256=" in result.messages[-1]["content"]
    assert all(card.text in {candidate.text for candidate in store.cards()} for card in result.selected_cards)


def test_mandatory_overflow_fails_before_model_decision(tmp_path):
    decision = Decision()
    compiler = ContextCompiler(crowded_store(tmp_path), "decision", decision)
    with pytest.raises(ContextBudgetError):
        compiler.prepare("system", "task", [{"role": "user", "content": "x" * 3000}], budget=100)
    assert not decision.requests


def test_compiler_excludes_stale_source(tmp_path):
    path = source(tmp_path)
    store = EvidenceStore(tmp_path)
    card = store.observe_source("module.py")
    path.write_text("changed\n")
    result = ContextCompiler(store).prepare("system", "task")
    assert not result.selected_cards
    assert result.metadata["stale_ids"] == [card.id]
    assert card.text not in "\n".join(message["content"] for message in result.messages)


def test_rules_and_window_make_no_decision_calls(tmp_path):
    store, decision = crowded_store(tmp_path), Decision()
    for policy in ["rules", "window"]:
        result = ContextCompiler(store, policy, decision).prepare("s", "t", query="target", budget=400)
        assert result.estimated_tokens <= 400
    assert not decision.requests


def test_decision_classifies_actual_evidence_and_records_full_cost(tmp_path):
    store, decision, events = crowded_store(tmp_path), Decision("drop"), []
    compiler = ContextCompiler(store, "decision", decision, emit=lambda event, **value: events.append((event, value)))
    result = compiler.prepare("s", "t", query="target", budget=500, deadline=time.monotonic() + 100)
    assert len(decision.requests) == 2
    for context, candidates, options in decision.requests:
        assert "def target_" in context and "return " in context
        assert options["purpose"] == "evidence_admission"
        assert {candidate.id for candidate in candidates} == {"keep", "drop", "uncertain"}
        assert "approach" not in context
    records = result.metadata["decisions"]
    assert records[0]["elapsed_seconds"] == 1.5
    assert records[0]["management_seconds"] == .5
    assert records[0]["input_tokens"] == 120
    assert records[0]["called"] is True
    dropped = {record["card_id"] for record in records}
    assert not dropped & {card.id for card in result.selected_cards}
    assert any(event == "evidence_admission" for event, _ in events)


def test_decision_budget_cache_and_checkpoint_resume(tmp_path):
    state = tmp_path / "state.json"
    store, decision = crowded_store(tmp_path, state), Decision()
    compiler = ContextCompiler(store, "decision", decision)
    compiler.prepare("s", "t", query="target", budget=500)
    compiler.prepare("s", "t", query="target", budget=500)
    assert len(decision.requests) == 2
    resumed = ContextCompiler(EvidenceStore(tmp_path, state), "decision", decision)
    result = resumed.prepare("s", "t", query="target", budget=500)
    assert len(decision.requests) == 2
    assert all(record["cache_hit"] for record in result.metadata["decisions"])
    resumed.prepare("s", "t", query="different question", budget=500)
    assert len(decision.requests) == 2


def test_expired_deadline_never_calls_decision_client(tmp_path):
    store, decision = crowded_store(tmp_path), Decision()
    ContextCompiler(store, "decision", decision).prepare("s", "t", query="target", budget=400,
                                                        deadline=time.monotonic() - 1)
    assert not decision.requests


def test_decision_bypassed_when_all_optional_evidence_fits(tmp_path):
    source(tmp_path)
    store, decision = EvidenceStore(tmp_path), Decision()
    store.observe_source("module.py")
    result = ContextCompiler(store, "decision", decision).prepare("s", "t", budget=4000)
    assert result.selected_cards and not decision.requests


def test_decision_error_retains_rules_fallback(tmp_path):
    store, decision = crowded_store(tmp_path), Decision(fail=True)
    rules = ContextCompiler(store).prepare("s", "t", query="target", budget=500)
    result = ContextCompiler(store, "decision", decision).prepare("s", "t", query="target", budget=500)
    assert result.selected_cards == rules.selected_cards
    assert all(record["fallback_reason"] == "evidence_decision_error" for record in result.metadata["decisions"])


def test_partial_classifier_excerpt_cannot_reject_unseen_remainder(tmp_path):
    store, decision = crowded_store(tmp_path), Decision("drop")
    source(tmp_path, "# target " + "many " * 1000 + "\n", "huge.py")
    large = store.observe_source("huge.py")
    result = ContextCompiler(store, "decision", decision).prepare("s", "t", query="target", budget=2000)
    record = next(item for item in result.metadata["decisions"] if item["card_id"] == large.id)
    assert record["evidence_excerpt_truncated"]
    assert record["label"] == "uncertain"


def test_archive_card_limit_is_explicit_and_does_not_evict(tmp_path):
    source(tmp_path)
    store = EvidenceStore(tmp_path, max_cards=1)
    first = store.observe_source("module.py")
    with pytest.raises(ValueError, match="card limit"):
        store.add_observation("test", "failed")
    assert store.cards() == [first]
    assert store.truncated


def test_symlink_cannot_alias_excluded_directory(tmp_path):
    source(tmp_path, "secret\n", ".git/private.py")
    (tmp_path / "alias").symlink_to(tmp_path / ".git", target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink|excluded"):
        EvidenceStore(tmp_path).observe_source("alias/private.py")


def test_search_includes_new_paths_after_inventory_snapshot(tmp_path):
    source(tmp_path)
    index = RepositoryIndex(tmp_path)
    index.refresh()
    source(tmp_path, "new_needle = 1\n", "new.py")
    assert index.search("new_needle")["matches"][0]["path"] == "new.py"


@pytest.mark.parametrize("result", [
    DecisionResult(("invented", "keep", "drop"), {"invented": .9, "keep": .06, "drop": .04}, called=True),
    DecisionResult(("drop", "keep", "uncertain"), {"drop": .9, "keep": .9, "uncertain": .9}, called=True),
])
def test_malformed_semantic_labels_cannot_change_rule_view(tmp_path, result):
    store = crowded_store(tmp_path)
    class Malformed(Decision):
        def rank(self, *args, **kwargs):
            return result
    rules = ContextCompiler(store).prepare("s", "t", query="target", budget=500)
    compiled = ContextCompiler(store, "decision", Malformed()).prepare("s", "t", query="target", budget=500)
    assert compiled.selected_cards == rules.selected_cards
    assert all(item.get("fallback_reason") for item in compiled.metadata["decisions"])


def test_adjacent_classifications_share_residency_and_account_cleanup(tmp_path):
    class PhaseDecision(Decision):
        def __init__(self):
            super().__init__()
            self.phases = 0
            self.inside = False
        @contextmanager
        def decision_phase(self, *, timeout_seconds):
            self.phases += 1
            assert 0 < timeout_seconds <= self.timeout_seconds
            phase = SimpleNamespace(result=None)
            self.inside = True
            try:
                yield phase
            finally:
                self.inside = False
                phase.result = DecisionResult((), {}, elapsed_seconds=.25, management_seconds=.25,
                                              management_events=("decision_phase_unloaded",))
        def rank(self, *args, **kwargs):
            assert self.inside
            return super().rank(*args, **kwargs)
    decision, events = PhaseDecision(), []
    result = ContextCompiler(crowded_store(tmp_path), "decision", decision,
                             emit=lambda event, **value: events.append((event, value))).prepare(
                                 "s", "t", query="target", budget=500)
    assert decision.phases == 1 and len(decision.requests) == 2
    assert result.metadata["phase_management"]["management_seconds"] == .25
    assert sum(item["elapsed_seconds"] for item in result.metadata["decisions"]) == 3
    assert any(event == "evidence_admission_phase" for event, _ in events)


def test_pinned_numbered_source_excludes_other_origin_duplicate(tmp_path):
    source(tmp_path, "def target(value):\n    if value < 0:\n        raise ValueError\n    return value\n")
    store, decision = EvidenceStore(tmp_path), Decision()
    direct = store.observe_source("module.py", origin="direct")
    duplicate = store.observe_source("module.py", origin="dependency")
    assert direct.id != duplicate.id
    numbered = "\n".join(f"{number}: {line}" for number, line in enumerate(direct.text.splitlines(), 1))
    required = [{"role": "user", "content": f"Requested source {direct.id}:\n{numbered}"}]
    result = ContextCompiler(store, "decision", decision).prepare(
        "s", "t", mandatory=required, mandatory_card_ids=(direct.id,), budget=600)
    assert not result.selected_cards and not decision.requests
    assert duplicate.id in result.metadata["covered_by_mandatory_ids"]
    assert result.messages[-1] == required[0]


def test_recent_observation_ids_avoid_replaying_full_raw_output(tmp_path):
    source(tmp_path)
    store, decision = EvidenceStore(tmp_path), Decision()
    card = store.observe_source("module.py")
    observation = store.add_observation("read_file", json.dumps({"content": card.text}))
    recent = [{"role": "user", "content": f"Read succeeded; source {card.id}."}]
    result = ContextCompiler(store, "decision", decision).prepare(
        "s", "t", recent=recent, excluded_card_ids=(observation.id,), budget=600)
    assert result.selected_cards == (card,)
    assert observation.id in result.metadata["excluded_card_ids"]
    assert not decision.requests


def test_oversized_archive_cannot_create_artificial_decision_pressure(tmp_path):
    source(tmp_path)
    store, decision = EvidenceStore(tmp_path), Decision()
    card = store.observe_source("module.py")
    huge = store.add_observation("inventory", "target " * 2000)
    result = ContextCompiler(store, "decision", decision).prepare("s", "t", query="target", budget=600)
    assert result.selected_cards == (card,)
    assert result.metadata["oversized_ids"] == [huge.id]
    assert not result.metadata["optional_pressure"] and not decision.requests
    assert store.get(huge.id, fresh=False) == huge


def test_semantic_admission_changes_actual_whole_code_context_under_pressure(tmp_path):
    source(tmp_path, "def preserve_guard(value):\n    if value < 0:\n        raise ValueError\n    return value\n", "guard.py")
    source(tmp_path, "def unrelated_cache():\n    # decode parser normalize\n    # " + "padding " * 45 + "\n    return 'cache'\n", "parser/normalize.py")
    source(tmp_path, "def decode(raw):\n    if not raw:\n        raise ValueError('empty')\n    # " + "padding " * 45 + "\n    return raw.strip()\n", "decode.py")
    store = EvidenceStore(tmp_path)
    guard = store.observe_source("guard.py", origin="direct")
    wrong = store.observe_source("parser/normalize.py")
    right = store.observe_source("decode.py")
    required = [{"role": "user", "content": guard.text}]
    class Semantic(Decision):
        def rank(self, context, candidates, **options):
            self.label = "drop" if "def unrelated_cache" in context else "keep"
            return super().rank(context, candidates, **options)
    options = {"mandatory": required, "mandatory_card_ids": (guard.id,),
               "query": "decode parser normalize", "budget": 420}
    baseline = ContextCompiler(store).prepare("s", "t", **options)
    decision = Semantic()
    intervention = ContextCompiler(store, "decision", decision).prepare("s", "t", **options)
    assert baseline.selected_cards == (wrong,)
    assert intervention.selected_cards == (right,)
    assert len(decision.requests) == 2
    assert intervention.messages[2] == baseline.messages[2] == required[0]
    assert "if not raw:" in intervention.messages[-1]["content"]
    assert "raise ValueError('empty')" in intervention.messages[-1]["content"]
    assert intervention.estimated_tokens <= options["budget"]


def test_internal_symlink_files_and_directory_aliases_are_ineligible(tmp_path):
    source(tmp_path, "needle = 1\n")
    source(tmp_path, "needle = 2\n", "real/helper.py")
    (tmp_path / "alias.py").symlink_to(tmp_path / "module.py")
    (tmp_path / "directory_alias").symlink_to(tmp_path / "real", target_is_directory=True)
    store = EvidenceStore(tmp_path)
    for path in ("alias.py", "directory_alias/helper.py"):
        with pytest.raises(ValueError, match="Symlink"):
            store.observe_source(path)
    index = RepositoryIndex(tmp_path)
    assert index.refresh()["paths"] == ["module.py", "real/helper.py"]
    assert {match["path"] for match in index.search("needle")["matches"]} == {"module.py", "real/helper.py"}
