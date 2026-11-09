"""Optional archive caps never consume or shorten explicitly pinned context."""

from dataclasses import replace
import json
from pathlib import Path

import pytest

from patchrook_agent.config import HarnessConfig
from patchrook_agent.evidence import ContextBudgetError, ContextCompiler, EvidenceStore
from patchrook_agent.evidence.compiler import estimate_tokens
from patchrook_agent.evidence_tools import EvidenceTools
from patchrook_agent.tools import ToolConfig, Workspace


def make_store(root):
    store = EvidenceStore(root)
    for number in range(8):
        path = root / f"source_{number}.py"
        path.write_text(f"def target_{number}():\n    # " + "relevant details " * 30 + f"\n    return {number}\n")
        store.observe_source(path.name)
    return store


def test_default_optional_budget_preserves_existing_request_bytes(tmp_path):
    compiler = ContextCompiler(make_store(tmp_path))
    default = compiler.prepare("system", "task", query="target", budget=4000)
    explicit = compiler.prepare("system", "task", query="target", budget=4000, optional_budget=None)
    assert default.messages == explicit.messages
    assert default.selected_cards == explicit.selected_cards
    assert default.metadata["optional_budget"] is None
    assert HarnessConfig().evidence_optional_tokens is None


@pytest.mark.parametrize("policy", ["rules", "window"])
def test_optional_archive_cost_is_capped_independently_from_pinned_messages(tmp_path, policy):
    compiler = ContextCompiler(make_store(tmp_path), policy=policy)
    mandatory = [{"role": "user", "content": "Required exact behavior: " + "preserve semantics " * 100}]
    recent = [{"role": "user", "content": "Actual failed assertion: expected 2 but got 1"}]
    uncapped = compiler.prepare("system", "task", mandatory, recent, query="target", budget=6000)
    capped = compiler.prepare("system", "task", mandatory, recent, query="target", budget=6000, optional_budget=512)
    assert capped.messages[:4] == uncapped.messages[:4] == [{"role": "system", "content": "system"},
        {"role": "user", "content": "task"}, *mandatory, *recent]
    assert 0 < capped.metadata["optional_estimated_tokens"] <= 512
    assert capped.estimated_tokens == estimate_tokens(capped.messages)
    assert capped.estimated_tokens <= 6000
    assert len(capped.selected_cards) < len(uncapped.selected_cards)
    assert all(compiler.store.get(card.id, fresh=False) is not None for card in uncapped.selected_cards)


def test_zero_optional_budget_retains_full_direct_source_and_failed_check(tmp_path):
    store = make_store(tmp_path)
    direct = store.cards()[0]
    pinned = {"role": "user", "content": "Requested source (data):\n" + direct.text * 3}
    failed = {"role": "user", "content": "Unresolved actual failure: " + "assertion detail " * 60}
    result = ContextCompiler(store).prepare("system", "task", mandatory=[pinned], recent=[failed],
        mandatory_card_ids=(direct.id,), budget=6000, optional_budget=0)
    assert result.messages == [{"role": "system", "content": "system"}, {"role": "user", "content": "task"}, pinned, failed]
    assert result.metadata["optional_estimated_tokens"] == 0
    assert not result.selected_cards
    assert result.estimated_tokens > 512


def test_optional_cap_cannot_hide_mandatory_overflow(tmp_path):
    compiler = ContextCompiler(make_store(tmp_path))
    with pytest.raises(ContextBudgetError, match="Mandatory"):
        compiler.prepare("system", "task", mandatory=[{"role": "user", "content": "required " * 1000}],
                         budget=500, optional_budget=0)


def test_stale_optional_source_never_reappears_when_cap_is_relaxed(tmp_path):
    store = make_store(tmp_path)
    stale = store.cards()[0]
    (tmp_path / stale.path).write_text("VALUE = 99\n")
    for cap in (0, 512, 1536, None):
        result = ContextCompiler(store).prepare("system", "task", budget=6000, optional_budget=cap)
        assert stale.id in result.metadata["stale_ids"]
        assert stale not in result.selected_cards


def test_selected_source_still_requires_current_hash_for_edit_authority(tmp_path):
    store = make_store(tmp_path)
    result = ContextCompiler(store).prepare("system", "task", budget=4000, optional_budget=512)
    card = result.selected_cards[0]
    (tmp_path / card.path).write_text("VALUE = 99\n")
    tools = EvidenceTools(Workspace(tmp_path, ToolConfig()))
    registration = tools.register_presented([{"path": card.path, "start_line": card.start_line,
        "end_line": card.end_line, "content": card.text, "source_sha256": card.source_sha256}])
    assert registration["accepted"] == 0
    assert registration["rejected"][0]["code"] == "stale_source"


@pytest.mark.parametrize("value", [-1, True, 1.5, "512"])
def test_invalid_optional_budgets_rejected_before_compilation(tmp_path, value):
    with pytest.raises(ValueError, match="Optional archive budget"):
        ContextCompiler(make_store(tmp_path)).prepare("system", "task", optional_budget=value)
    with pytest.raises(ValueError, match="evidence_optional_tokens"):
        replace(HarnessConfig(), evidence_optional_tokens=value)


def test_optional_budget_round_trips_through_config():
    config = replace(HarnessConfig(), evidence_optional_tokens=1536)
    assert config.to_dict()["evidence_optional_tokens"] == 1536
    assert HarnessConfig.from_dict(config.to_dict()).evidence_optional_tokens == 1536


def test_numbered_source_policy_does_not_require_a_literal_render_label():
    from patchrook_agent.evidence_workflow import EVIDENCE_SYSTEM_PROMPT
    assert "fresh numbered source blocks" in EVIDENCE_SYSTEM_PROMPT
    assert "fresh numbered Requested source blocks" not in EVIDENCE_SYSTEM_PROMPT


def test_zero_archive_cap_does_not_spend_optional_decision_calls(tmp_path):
    class ForbiddenDecision:
        def rank(self, *args, **kwargs):
            pytest.fail("Oversized optional evidence must not trigger a decision call")
    result = ContextCompiler(make_store(tmp_path), policy="decision", decision_client=ForbiddenDecision()).prepare(
        "system", "task", budget=6000, optional_budget=0)
    assert not result.selected_cards and not result.metadata["decisions"]


