"""One cost model drives compaction and final requests, without inference."""

from copy import deepcopy
import json

from patchrook_agent.context import ContextConfig
from patchrook_agent.context.budget import estimate_tokens
from patchrook_agent.context.compaction import (
    ContextManager, SUMMARY_MARKER, SUMMARY_PREFIX, _groups, _summarize,
)


def exchange(name, arguments, result):
    return [{"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": name, "arguments": arguments}}]},
            {"role": "tool", "tool_name": name, "content": json.dumps(result)}]


def start():
    return [{"role": "system", "content": "Inspect before editing."}, {"role": "user", "content": "Preserve existing behavior."}]


def test_request_cost_can_trigger_capacity_compaction_when_native_estimate_is_below_trigger():
    messages = start() + exchange("search_code", {"pattern": "target"}, {
        "success": True, "matches": [{"path": "old.py", "line_number": 1, "text": "x" * 6600}]})
    original = deepcopy(messages)
    config = ContextConfig(context_size=4096, trigger=.7, keep_recent_turns=0)
    native_before = estimate_tokens(messages, [])
    assert native_before < config.context_size * config.trigger
    cost = lambda records, tools: estimate_tokens(records, tools) + 1100
    assert cost(messages, []) > config.input_limit
    prepared = ContextManager(config).prepare(messages, [], request_cost=cost)
    assert prepared.attempted and prepared.attempt_reason == "capacity"
    assert prepared.compacted and prepared.safe and prepared.target_reached
    assert prepared.tokens_before == native_before + 1100
    assert prepared.tokens_after == cost(prepared.messages, [])
    assert prepared.tokens_after <= config.compaction_goal
    assert messages == original
    _groups(prepared.active_messages)


def test_compaction_keeps_cheap_exact_read_snippets_and_drops_duplicated_requested_ranges():
    source = "14: def transform(value):\n15:     return value + 1"
    result = {"success": True, "path": "stage.py", "start_line": 14, "end_line": 15,
              "total_lines": 15, "truncated": False, "content": source}
    group = exchange("read_file", {"path": "stage.py", "start_line": 14, "end_line": 100}, result)
    original = deepcopy(group)
    record = _summarize(group)[0]
    assert record["content"] == source
    assert "arguments" not in record and "content_omitted" not in record
    assert (record["start_line"], record["end_line"], record["truncated"]) == (14, 15, False)
    assert group == original


def test_partial_line_fragments_are_not_promoted_to_exact_source_snippets():
    group = exchange("read_file", {"path": "wide.py"}, {
        "success": True, "path": "wide.py", "start_line": 1, "end_line": 1,
        "total_lines": 1, "truncated": True, "partial_line": True,
        "line_chars_returned": 5, "next_start_line": 1, "content": "1: very_"})
    record = _summarize(group)[0]
    assert record["content_omitted"] is True and "content" not in record
    assert record["partial_line"] is True and record["truncated"] is True
    assert record["next_start_line"] == 1 and record["line_chars_returned"] == 5


def test_repeated_old_read_summary_normalization_is_bounded_and_keeps_omissions_explicit():
    # Older checkpoints duplicated requested/observed ranges and prose for each
    # read. Same observed range should not grow with different requested ends.
    old_records = [{"tool": "read_file", "arguments": {"path": "old.py", "start_line": 1, "end_line": end},
                    "success": True, "path": "old.py", "start_line": 1, "end_line": 10, "total_lines": 10,
                    "truncated": False, "content_omitted": "Read the file again before relying on exact source."}
                   for end in range(10, 70)]
    messages = start() + [{"role": "assistant", SUMMARY_MARKER: True,
                           "content": SUMMARY_PREFIX + json.dumps(old_records)}]
    original = deepcopy(messages)
    config = ContextConfig(context_size=4096, trigger=.6, keep_recent_turns=0)
    manager = ContextManager(config)
    first = manager.prepare(messages, [])
    assert first.safe and first.compacted
    summaries = [m for m in first.active_messages if m.get(SUMMARY_MARKER)]
    records = json.loads(summaries[0]["content"][len(SUMMARY_PREFIX):])
    assert len(records) == 1 and records[0]["content_omitted"] is True
    assert "arguments" not in records[0]
    resumed = ContextManager.from_dict(manager.to_dict())
    repeated = resumed.prepare(first.active_messages, [])
    assert repeated.active_messages == first.active_messages
    assert repeated.tokens_after == first.tokens_after
    assert messages == original
