"""A rejected patch may retain only matching source shown in the last request."""

from dataclasses import replace
import json
from pathlib import Path
import time

import pytest

from patchrook_agent.agent import Agent, AgentState
from patchrook_agent.completion import TaskRequirements
from patchrook_agent.config import HarnessConfig
from patchrook_agent.evidence_workflow import EVIDENCE_SYSTEM_PROMPT, EvidenceWorkflow
from patchrook_agent.tools import ToolConfig, ToolRegistry, Workspace


SOURCE = ('from typing import Any\n\nclass Subject:\n    """Keep the public contract."""\n\n'
          '    def existing(self):\n        return 1\n')
ANCHOR = 'class Subject:\n    """Keep the public contract."""'


@pytest.fixture(autouse=True)
def no_inference_or_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Source retention must not run models or network requests")
    monkeypatch.setattr("httpx.Client", forbidden)
    monkeypatch.setattr("httpx.get", forbidden)
    monkeypatch.setattr("httpx.post", forbidden)
    monkeypatch.setattr("patchrook_agent.models.ollama.OllamaClient.chat", forbidden)


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "subject.py").write_text(SOURCE)
    (root / "dependency.py").write_text("DEPENDENCY = True\n")
    (root / "unseen.py").write_text("SECRET = 1\nUNSHOWN = 'must require a read'\n")
    class NoModel:
        model = "offline-no-generation"
        context_size = 8192
        max_output_tokens = 1024
    base = HarnessConfig.preset("C")
    config = replace(base, workflow_mode="evidence", action_protocol="schema", review_mode="off",
        persist=False, checkpoint_enabled=False, compaction_enabled=False,
        context=replace(base.context, enabled=False), evidence_policy="rules", model_routing="off")
    agent = Agent(NoModel(), config=config, registry=ToolRegistry(Workspace(root, ToolConfig())),
                  runs_dir=tmp_path / "runs", max_steps=10, deadline_seconds=30)
    value = EvidenceWorkflow(agent)
    value.requirements = TaskRequirements.inspect()
    value._setup("Inspect Subject and preserve its public contract.")
    value.definitions = value._definitions()
    value.enabled = {item["function"]["name"] for item in value.definitions}
    value.state = AgentState(messages=[{"role": "system", "content": EVIDENCE_SYSTEM_PROMPT},
                                      {"role": "user", "content": value.task}])
    value.session.bind(value.state)
    # Suppress OPTIONAL source to prove the retained source is mandatory. The
    # real compiler still budgets, renders, hashes and registers all exact pins.
    original = value.compiler.prepare
    def only_explicit_pins(*args, **options):
        options["excluded_card_ids"] = tuple({*options.get("excluded_card_ids", ()),
            *(card.id for card in value.evidence.cards(fresh=False))})
        return original(*args, **options)
    monkeypatch.setattr(value.compiler, "prepare", only_explicit_pins)
    yield value
    if value.session.store:
        value.session.store.release()


def card(runtime, path="subject.py", start=3, end=5):
    return runtime.evidence.observe_source(path, start, end, origin="direct")


def prepare(runtime, cards=None):
    if cards is not None:
        runtime.direct = [item.id for item in cards]
    return runtime._prepare(runtime.state, runtime.definitions, time.monotonic() + 10)


def replace_edit(path="subject.py", old=ANCHOR, new='class Subject:\n    """Updated contract."""'):
    return {"path": path, "old_text": old, "new_text": new}


def apply(runtime, edits):
    return runtime._perform_action(runtime.state, {"name": "apply_edits", "arguments": {"edits": edits}},
                                   time.monotonic() + 10)


def requested(messages):
    return "\n".join(item["content"] for item in messages if item["content"].startswith("Requested source (data) "))


def test_duplicate_rejection_retains_only_prior_exact_header_and_allows_a_new_corrected_batch(runtime):
    header = card(runtime)
    imports = card(runtime, start=1, end=1)
    dependency = card(runtime, "dependency.py", 1, 1)
    prepare(runtime, [header, imports, dependency])
    assert runtime.direct == []
    first = apply(runtime, [replace_edit(), replace_edit()])
    assert first["error"]["code"] == "overlapping_edits"
    assert runtime.revision == 0 and runtime.state.modified_files == []
    assert (runtime.workspace.root / "subject.py").read_text() == SOURCE
    assert runtime.direct == [header.id]
    messages = prepare(runtime)
    assert runtime.context_records[-1]["presented_source_ids"] == [header.id]
    assert '3: class Subject:\n4:     """Keep the public contract."""' in requested(messages)
    assert "from typing" not in requested(messages) and "DEPENDENCY" not in requested(messages)
    corrections = [message["content"] for message in messages
                   if message["content"].startswith("Action correction:")]
    assert len(corrections) == 1
    assert "Correct the reported error" in corrections[0]
    assert "read_file only for missing source" in corrections[0]
    assert "Read fresh source covering each" not in corrections[0]
    assert runtime.direct == []
    # Retention alone never replays the rejected edits.
    assert (runtime.workspace.root / "subject.py").read_text() == SOURCE
    second = apply(runtime, [replace_edit()])
    assert second["success"] and second["modified_files"] == ["subject.py"]
    assert "Updated contract" in (runtime.workspace.root / "subject.py").read_text()


def test_stale_full_file_hash_discards_retained_header_even_if_header_text_is_unchanged(runtime):
    header = card(runtime)
    prepare(runtime, [header])
    rejected = apply(runtime, [replace_edit(), replace_edit()])
    assert rejected["error"]["code"] == "overlapping_edits" and runtime.direct == [header.id]
    path = runtime.workspace.root / "subject.py"
    path.write_text(SOURCE.replace("return 1", "return 999"))
    messages = prepare(runtime)
    assert requested(messages) == "" and runtime.context_records[-1]["presented_source_ids"] == []
    result = apply(runtime, [replace_edit()])
    assert result["error"]["code"] == "source_not_presented"
    assert "return 999" in path.read_text() and "Updated contract" not in path.read_text()


def test_never_read_file_cannot_gain_authority_from_rejected_edit_or_its_preview(runtime):
    prepare(runtime, [card(runtime)])
    result = apply(runtime, [replace_edit("unseen.py", "SECRET = 1", "SECRET = 2")])
    assert result["error"]["code"] == "source_not_presented"
    assert runtime.direct == []
    messages = prepare(runtime)
    assert "UNSHOWN" not in "\n".join(item["content"] for item in messages)
    assert runtime.context_records[-1]["presented_source_ids"] == []
    assert (runtime.workspace.root / "unseen.py").read_text().startswith("SECRET = 1\n")


def test_card_from_an_older_request_does_not_qualify_when_omitted_from_last_request(runtime):
    header = card(runtime)
    prepare(runtime, [header])
    prepare(runtime, [card(runtime, "dependency.py", 1, 1)])
    result = apply(runtime, [replace_edit()])
    assert result["error"]["code"] == "source_not_presented"
    assert header.id not in runtime.direct
    assert requested(prepare(runtime)) == ""


def test_unshown_anchor_in_a_previously_presented_file_still_requires_a_fresh_read(runtime):
    imports = card(runtime, start=1, end=1)
    prepare(runtime, [imports])
    result = apply(runtime, [replace_edit()])
    assert result["error"]["code"] == "source_not_presented"
    assert runtime.direct == []
    assert requested(prepare(runtime)) == ""


def test_incomplete_anchor_does_not_expand_the_previous_view(runtime):
    header_line = card(runtime, start=3, end=3)
    prepare(runtime, [header_line])
    result = apply(runtime, [replace_edit()])
    assert result["error"]["code"] == "source_not_presented"
    assert runtime.direct == []
    assert runtime.evidence.get(header_line.id).end_line == 3


@pytest.mark.parametrize("kind", ["existing", "new_duplicate"])
def test_rejected_creations_never_retain_old_source_or_create_directories(runtime, kind):
    prepare(runtime, [card(runtime)])
    if kind == "existing":
        edits = [{"path": "subject.py", "content": "VALUE = 2\n"}]
    else:
        edits = [{"path": "new/deep/file.py", "content": "VALUE = 2\n"}] * 2
    result = apply(runtime, edits)
    assert result["error"]["code"] == "already_exists"
    assert runtime.direct == []
    assert requested(prepare(runtime)) == ""
    assert not (runtime.workspace.root / "new").exists()
    assert (runtime.workspace.root / "subject.py").read_text() == SOURCE


def test_rejection_text_and_search_receipts_cannot_be_promoted_to_source(runtime):
    observation = runtime.evidence.add_observation("search_code", json.dumps({
        "path": "unseen.py", "text": "SECRET = 1", "line_number": 1}))
    prepare(runtime, [])
    # Even an invalid context ID referencing an observation remains non-source.
    runtime.context_records[-1]["presented_source_ids"] = [observation.id]
    result = apply(runtime, [replace_edit("unseen.py", "SECRET = 1", "SECRET = 2")])
    assert result["error"]["code"] == "source_not_presented"
    assert runtime.direct == []
    assert requested(prepare(runtime)) == ""


def test_retention_is_consumed_after_one_next_request_not_a_permanent_pin(runtime):
    header = card(runtime)
    prepare(runtime, [header])
    apply(runtime, [replace_edit(), replace_edit()])
    assert header.id in runtime.context_records[-1]["presented_source_ids"]
    assert header.id in runtime.direct
    prepare(runtime)
    assert runtime.direct == []
    # A later unrelated action does not refresh or retain the old source again.
    runtime._perform_action(runtime.state, {"name": "list_files", "arguments": {"path": "."}}, time.monotonic() + 10)
    assert requested(prepare(runtime)) == ""


def test_noop_rejection_can_retain_exact_source_without_replacing_it(runtime):
    header = card(runtime)
    prepare(runtime, [header])
    result = apply(runtime, [replace_edit(new=ANCHOR)])
    assert result["error"]["code"] == "no_change"
    assert runtime.direct == [header.id]
    assert (runtime.workspace.root / "subject.py").read_text() == SOURCE


def test_crlf_anchor_normalization_matches_the_existing_edit_guard(runtime):
    target = runtime.workspace.root / "subject.py"
    target.write_bytes(SOURCE.replace("\n", "\r\n").encode())
    header = card(runtime)
    prepare(runtime, [header])
    result = apply(runtime, [replace_edit(), replace_edit()])
    assert result["error"]["code"] == "overlapping_edits"
    assert runtime.direct == [header.id]
    prepare(runtime)
    assert apply(runtime, [replace_edit()])["success"]
    assert b'"""Updated contract."""\r\n' in target.read_bytes()


def test_source_ids_from_focused_repair_context_use_the_same_exact_retention_boundary(runtime):
    header = card(runtime)
    prepare(runtime, [header])
    # Normal and repair records share this concrete presented-ID field.
    runtime.context_records[-1] = {"phase": "focused_repair", "presented_source_ids": [header.id]}
    result = apply(runtime, [replace_edit(), replace_edit()])
    assert result["error"]["code"] == "overlapping_edits"
    assert runtime.direct == [header.id]


def test_missing_or_malformed_context_record_is_not_an_authority_source(runtime):
    for records in ([], [None], [{"presented_source_ids": None}], [{"presented_source_ids": [None, {}]}]):
        runtime.context_records = records
        assert runtime._retain_rejected_source({"edits": [replace_edit()]}) == []
        assert runtime.direct == []


def test_optional_source_that_was_actually_presented_can_become_a_rejected_anchor_pin(runtime, monkeypatch):
    from patchrook_agent.evidence import ContextCompiler
    header = card(runtime)
    card(runtime, start=1, end=1)
    card(runtime, "dependency.py", 1, 1)
    runtime.compiler = ContextCompiler(runtime.evidence, policy="rules")
    prepare(runtime, [])
    assert header.id in runtime.context_records[-1]["presented_source_ids"]
    assert runtime.context_records[-1]["mandatory_card_ids"] == []
    original = runtime.compiler.prepare
    def no_optional(*args, **options):
        options["excluded_card_ids"] = tuple(card.id for card in runtime.evidence.cards(fresh=False))
        return original(*args, **options)
    monkeypatch.setattr(runtime.compiler, "prepare", no_optional)
    rejected = apply(runtime, [replace_edit(), replace_edit()])
    assert rejected["error"]["code"] == "overlapping_edits" and runtime.direct == [header.id]
    messages = prepare(runtime)
    assert runtime.context_records[-1]["presented_source_ids"] == [header.id]
    assert runtime.context_records[-1]["mandatory_card_ids"] == [header.id]
    assert "class Subject:" in requested(messages) and "from typing" not in requested(messages)


def test_race_after_retention_hash_check_fails_the_existing_registration_guard(runtime, monkeypatch):
    from patchrook_agent.tools.base import ToolError
    header = card(runtime)
    prepare(runtime, [header])
    apply(runtime, [replace_edit(), replace_edit()])
    original = runtime.compiler.prepare
    def changed_during_compilation(*args, **options):
        prepared = original(*args, **options)
        (runtime.workspace.root / "subject.py").write_text(SOURCE.replace("return 1", "return 999"))
        return prepared
    monkeypatch.setattr(runtime.compiler, "prepare", changed_during_compilation)
    with pytest.raises(ToolError, match="Source changed during context preparation"):
        prepare(runtime)
    assert "Updated contract" not in (runtime.workspace.root / "subject.py").read_text()


def two_path_rejection(runtime):
    edits = [replace_edit(), replace_edit("dependency.py", "DEPENDENCY = True", "DEPENDENCY = False"),
             replace_edit()]
    result = apply(runtime, edits)
    assert result["error"]["code"] == "overlapping_edits"
    assert runtime.revision == 0 and runtime.state.modified_files == []
    return result


def test_two_path_rejection_preserves_exact_unchanged_source_prefix_order(runtime):
    header, dependency = card(runtime), card(runtime, "dependency.py", 1, 1)
    # Force original presentation to differ from the sorted ID allowlist.
    ordered = sorted([header, dependency], key=lambda item: item.id, reverse=True)
    first = prepare(runtime, ordered)
    context = runtime.context_records[-1]
    assert context["presented_source_ids"] == sorted(item.id for item in ordered)
    assert [view["id"] for view in context["source_views"]] == [item.id for item in ordered]
    two_path_rejection(runtime)
    assert runtime.direct == [item.id for item in ordered]
    second = prepare(runtime)
    assert first[:4] == second[:4]
    assert all(message["content"].startswith("Requested source (data) ") for message in second[2:4])
    assert (runtime.workspace.root / "subject.py").read_text() == SOURCE
    assert (runtime.workspace.root / "dependency.py").read_text() == "DEPENDENCY = True\n"


@pytest.mark.parametrize("views", [None, "bad", {}, [None, {}, {"id": []}, {"id": 42}]])
def test_malformed_source_views_fall_back_to_actual_presented_id_order(runtime, views):
    header, dependency = card(runtime), card(runtime, "dependency.py", 1, 1)
    ordered = sorted([header, dependency], key=lambda item: item.id, reverse=True)
    prepare(runtime, ordered)
    runtime.context_records[-1] = {"presented_source_ids": [item.id for item in ordered],
                                   "source_views": views}
    two_path_rejection(runtime)
    assert runtime.direct == [item.id for item in ordered]


def test_repair_source_views_without_ids_keep_actual_presented_id_order(runtime):
    header, dependency = card(runtime), card(runtime, "dependency.py", 1, 1)
    ordered = sorted([header, dependency], key=lambda item: item.id, reverse=True)
    first = prepare(runtime, ordered)
    runtime.context_records[-1] = {"phase": "focused_repair",
        "presented_source_ids": [item.id for item in ordered],
        "source_views": [{"path": item.path, "start_line": item.start_line, "end_line": item.end_line}
                         for item in ordered]}
    two_path_rejection(runtime)
    assert runtime.direct == [item.id for item in ordered]
    assert prepare(runtime)[:4] == first[:4]


def test_source_view_order_cannot_introduce_unpresented_ids_or_duplicate_pins(runtime):
    header = card(runtime)
    unseen = card(runtime, "unseen.py", 1, 2)
    prepare(runtime, [header])
    runtime.context_records[-1]["source_views"] = [
        {"id": unseen.id}, {"id": header.id}, {"id": header.id}, {"id": []}]
    result = apply(runtime, [replace_edit(), replace_edit(),
        replace_edit("unseen.py", "SECRET = 1", "SECRET = 2")])
    assert result["error"]["code"] == "overlapping_edits"
    assert runtime.direct == [header.id]
    messages = prepare(runtime)
    assert runtime.context_records[-1]["presented_source_ids"] == [header.id]
    assert "UNSHOWN" not in "\n".join(item["content"] for item in messages)
    denied = apply(runtime, [replace_edit("unseen.py", "SECRET = 1", "SECRET = 2")])
    assert denied["error"]["code"] == "source_not_presented"


def test_retained_order_drops_a_stale_path_without_authorizing_changed_source(runtime):
    header, dependency = card(runtime), card(runtime, "dependency.py", 1, 1)
    ordered = sorted([header, dependency], key=lambda item: item.id, reverse=True)
    prepare(runtime, ordered)
    two_path_rejection(runtime)
    target = runtime.workspace.root / "dependency.py"
    target.write_text("DEPENDENCY = False\n")
    messages = prepare(runtime)
    assert runtime.context_records[-1]["presented_source_ids"] == [header.id]
    source_messages = [item for item in messages if item["content"].startswith("Requested source (data) ")]
    assert len(source_messages) == 1 and "subject.py:" in source_messages[0]["content"]
    denied = apply(runtime, [replace_edit("dependency.py", "DEPENDENCY = True", "DEPENDENCY = None")])
    assert denied["error"]["code"] == "source_not_presented"
    assert target.read_text() == "DEPENDENCY = False\n"
