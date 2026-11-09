"""Capacity and retry policy, independent of model/tokenizer availability."""

from copy import deepcopy
import json

import pytest

from patchrook_agent.context import ContextConfig, ContextManager, estimate_tokens
from patchrook_agent.context.compaction import SUMMARY_MARKER


def exchange(name, arguments, result, identifier="call"):
    return [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": identifier, "function": {"name": name, "arguments": arguments}}]},
        {"role": "tool", "tool_name": name, "tool_call_id": identifier,
         "content": json.dumps(result)},
    ]


def protected_history():
    return [{"role": "system", "content": "Inspect first."},
            {"role": "user", "content": "Preserve public APIs. " * 450}]


def test_selected_defaults_use_8k_with_headroom_policy():
    config = ContextConfig()
    assert config.context_size == 8192
    assert config.trigger == 0.80 and config.hard_limit == 0.85
    assert config.target is None
    assert config.reserved_output_tokens == 512
    assert config.estimation_margin_tokens == 128
    assert config.compaction_goal < config.context_size * config.trigger < config.input_limit
    assert config.compaction_goal > config.context_size * 0.35


@pytest.mark.parametrize("size", [4096, 8192, 16384])
def test_effective_ceiling_accounts_for_output_and_estimation_margin(size):
    config = ContextConfig(context_size=size)
    assert config.input_limit == min(int(size * 0.85), size - 512 - 128)
    assert config.fits(config.input_limit)
    assert not config.fits(config.input_limit + 1)
    assert config.context_size * config.trigger < config.input_limit


def test_larger_generation_reserve_requires_an_earlier_trigger():
    with pytest.raises(ValueError, match="trigger must precede"):
        ContextConfig(context_size=4096, reserved_output_tokens=1024)
    config = ContextConfig(context_size=4096, reserved_output_tokens=1024, trigger=0.6)
    assert config.input_limit == 4096 - 1024 - 128


@pytest.mark.parametrize("kwargs", [
    {"trigger": 0.85}, {"target": True}, {"target": float("nan")},
    {"estimation_margin_tokens": -1}, {"headroom_tokens": True},
    {"estimation_margin_tokens": 20000},
])
def test_invalid_budget_controls_are_rejected(kwargs):
    with pytest.raises(ValueError):
        ContextConfig(**kwargs)


def test_complete_request_includes_stable_pins_schemas_and_memory_tail():
    manager = ContextManager(ContextConfig(context_size=4096))
    messages = [{"role": "system", "content": "Stable instructions."},
                {"role": "user", "content": "Fix addition."}]
    pins = [{"role": "system", "content": "Loaded debugging skill."}]
    tail = [{"role": "system", "content": "Current verification: not run."}]
    tools = [{"type": "function", "function": {"name": "read_file", "parameters": {}}}]
    original = deepcopy((messages, pins, tail, tools))
    result = manager.prepare(messages, tools, pinned_messages=pins, tail_messages=tail)
    assert result.messages == messages[:1] + pins + messages[1:] + tail
    assert result.active_messages == messages
    assert result.estimated_tokens == estimate_tokens(result.messages, tools)
    assert (messages, pins, tail, tools) == original
    assert result.preparation_seconds >= result.compaction_seconds == 0


def test_ineffective_attempt_is_recorded_once_and_survives_checkpoint():
    manager = ContextManager(ContextConfig(context_size=4096))
    first = manager.prepare(protected_history(), [])
    assert first.attempted and first.safe and not first.compacted
    assert first.reason == "headroom_unavailable"
    assert manager.compaction_attempts == 1 and manager.compaction_count == 0
    restored = ContextManager.from_dict(json.loads(json.dumps(manager.to_dict())))
    second = restored.prepare(first.active_messages, [])
    assert second.safe and not second.attempted
    assert second.reason == "unchanged_reclaimable_history"
    assert second.compaction_seconds == 0
    assert restored.compaction_attempts == 1


def test_small_new_history_waits_but_meaningful_reclaimable_history_retries():
    manager = ContextManager(ContextConfig(context_size=8192, trigger=0.4, keep_recent_turns=0))
    first = manager.prepare(protected_history(), [])
    assert first.attempted and first.safe and not first.compacted
    small = first.active_messages + exchange("list_files", {"path": "."}, {
        "success": True, "entries": ["a.py"], "truncated": False})
    second = manager.prepare(small, [])
    assert second.safe and not second.attempted
    assert second.reason == "waiting_for_reclaimable_history"
    expanded = second.active_messages + exchange("read_file", {"path": "old.py"}, {
        "success": True, "path": "old.py", "content": "previous_source = 1\n" * 120})
    for index in range(3):
        expanded += exchange("read_file", {"path": f"active{index}.py"}, {
            "success": True, "path": f"active{index}.py", "content": "1: return a - b"})
    third = manager.prepare(expanded, [])
    assert third.attempted and third.compacted and third.safe
    assert third.attempt_reason == "new_reclaimable_history"
    assert third.tokens_after < third.tokens_before
    assert manager.compaction_attempts == 2 and manager.compaction_count == 1


def test_capacity_check_and_unsafe_retry_are_never_bypassed_by_cooldown():
    manager = ContextManager(ContextConfig(context_size=4096))
    first = manager.prepare(protected_history(), [])
    assert first.safe and first.attempted
    tail = [{"role": "system", "content": "Required current evidence. " * 50}]
    second = manager.prepare(first.active_messages, [], tail_messages=tail)
    assert second.attempted and not second.safe
    assert second.attempt_reason == "capacity" and second.reason == "context_limit"
    assert str(manager.config.input_limit) in second.limit_message
    assert "Narrow loaded skills" in second.limit_message
    assert tail[0] in second.messages
    # An identical impossible request reports the same stop without repeating work.
    third = manager.prepare(second.active_messages, [], tail_messages=tail)
    assert not third.attempted and not third.safe and third.reason == "context_limit"
    assert third.compaction_seconds == 0


def test_different_memory_tail_alone_does_not_repeat_safe_ineffective_work():
    manager = ContextManager(ContextConfig(context_size=8192, trigger=0.4))
    first = manager.prepare(protected_history(), [])
    second = manager.prepare(first.active_messages, [], tail_messages=[{
        "role": "system", "content": "Next step: verify addition."}])
    assert second.safe and not second.attempted
    assert manager.compaction_attempts == 1


def test_latest_exact_source_for_a_modified_file_and_debugging_evidence_survive():
    messages = [{"role": "system", "content": "Inspect before editing."},
                {"role": "user", "content": "Repair auth.py; preserve public APIs."}]
    source = exchange("read_file", {"path": "auth.py", "start_line": 20, "end_line": 23}, {
        "success": True, "path": "auth.py", "content": "20: def valid(token):\n21:     return token is None"})
    messages += source + exchange("edit_file", {"path": "auth.py"}, {
        "success": True, "modified": True, "path": "auth.py"})
    failure = exchange("run_tests", {"path": "tests/test_auth.py"}, {
        "success": False, "status": "failed", "exit_code": 1,
        "stdout": "tests/test_auth.py:42: AssertionError: expected token='abc'\nExact frame.",
        "stderr": "Exact diagnostic: invalid token."})
    messages += failure
    for index in range(8):
        messages += exchange("read_file", {"path": f"old{index}.py"}, {
            "success": True, "path": f"old{index}.py", "content": "historical_source = 1\n" * 100})
    raw = deepcopy(messages)
    result = ContextManager(ContextConfig(context_size=4096, keep_recent_turns=0)).prepare(messages, [])
    assert result.compacted and result.safe
    assert all(message in result.messages for message in source + failure)
    assert messages == raw
    assert any(message.get(SUMMARY_MARKER) for message in result.active_messages)
    assert all(SUMMARY_MARKER not in message for message in result.messages)


def test_latest_source_remains_exact_after_navigation_ages_it_beyond_recent_turns():
    messages = [{"role": "system", "content": "Inspect before editing."},
                {"role": "user", "content": "Find and repair addition; preserve multiplication."}]
    for index in range(6):
        messages += exchange("read_file", {"path": f"historical{index}.py"}, {
            "success": True, "path": f"historical{index}.py", "content": "historical_source = 1\n" * 100})
    source = exchange("read_file", {"path": "math_utils.py", "start_line": 1, "end_line": 4}, {
        "success": True, "path": "math_utils.py",
        "content": "1: def add_numbers(a, b):\n2:     return a - b\n3: def multiply(a, b):\n4:     return a * b"})
    messages += source
    for index in range(3):
        messages += exchange("list_files", {"path": f"tests/subdir{index}"}, {
            "success": True, "entries": ["test_math_utils.py"], "truncated": False})
    result = ContextManager(ContextConfig(context_size=4096)).prepare(messages, [])
    assert result.safe and result.compacted
    assert all(message in result.messages for message in source)
    # Keeping a source exchange keeps its assistant request and observation together.
    index = result.messages.index(source[0])
    assert result.messages[index + 1] == source[1]


@pytest.mark.parametrize("read_path,edit_path", [
    ("math.py", "./math.py"),
    ("./math.py", "math.py"),
    ("alias.py", "math.py"),
])
def test_modified_source_protection_uses_canonical_observed_path(read_path, edit_path):
    messages = [{"role": "system", "content": "Inspect first."},
                {"role": "user", "content": "Fix math.py; preserve unrelated behavior."}]
    source = exchange("read_file", {"path": read_path}, {
        "success": True, "path": "math.py", "content": "1: def add(a, b):\n2:     return a - b"})
    messages += source + exchange("edit_file", {"path": edit_path}, {
        "success": True, "path": "math.py", "modified": True})
    for index in range(6):
        messages += exchange("read_file", {"path": f"historical{index}.py"}, {
            "success": True, "path": f"historical{index}.py", "content": "historical_source = 1\n" * 90})
    result = ContextManager(ContextConfig(context_size=4096)).prepare(messages, [])
    assert result.safe and result.compacted
    assert all(message in result.messages for message in source)
    index = result.messages.index(source[0])
    assert result.messages[index + 1] == source[1]


def source_working_set():
    messages = [{"role": "system", "content": "Inspect first."},
                {"role": "user", "content": "Repair the parser and preserve other modules."}]
    sources = []
    for path in ("api.py", "parser.py", "pricing.py"):
        group = exchange("read_file", {"path": path}, {
            "success": True, "path": path, "content": "return_value = 1\n" * 80})
        sources.append(group)
        messages += group
    failure = exchange("run_tests", {"path": "tests.py"}, {
        "success": False, "status": "failed", "exit_code": 1,
        "stdout": "tests.py:4 AssertionError: exact debugging evidence"})
    messages += failure
    for index in range(3):
        messages += exchange("list_files", {"path": f"tests{index}"}, {
            "success": True, "entries": ["test_parser.py"]})
    return messages, sources, failure


def request_components(pin_repetitions):
    pins = [{"role": "system", "content": "Stable instruction. " * pin_repetitions}]
    tail = [{"role": "system", "content": "Required memory. " * 80}]
    tools = [{"function": {"name": "read_file", "description": "schema " * 100}}]
    return pins, tail, tools


def test_8k_retains_three_aged_source_modules_and_persists_matching_cooldown():
    messages, sources, failure = source_working_set()
    for index in range(6):
        messages[2:2] = exchange("list_files", {"path": f"historical{index}"}, {
            "success": True, "entries": [f"previous_{index}_{number}.py" for number in range(100)]})
    raw = deepcopy(messages)
    pins, tail, tools = request_components(180)
    manager = ContextManager(ContextConfig(context_size=8192, trigger=0.4))
    result = manager.prepare(messages, tools, pinned_messages=pins, tail_messages=tail)
    assert result.safe and result.compacted
    assert all(message in result.messages for group in sources for message in group)
    assert all(message in result.messages for message in failure)
    assert result.tokens_after > 8192 * 0.4  # Useful exact evidence may exceed the soft goal.
    assert messages == raw
    restored = ContextManager.from_dict(manager.to_dict())
    repeated = restored.prepare(result.active_messages, tools, pinned_messages=pins, tail_messages=tail)
    assert repeated.safe and not repeated.attempted
    assert repeated.reason == "unchanged_reclaimable_history"
    assert restored.compaction_attempts == 1


def test_4k_source_admission_counts_pins_tail_and_schemas():
    messages, sources, failure = source_working_set()
    pins, tail, tools = request_components(165)
    manager = ContextManager(ContextConfig(context_size=4096))
    result = manager.prepare(messages, tools, pinned_messages=pins, tail_messages=tail)
    assert result.safe and result.compacted
    assert all(message in result.messages for message in sources[-1] + failure)
    assert any(message not in result.messages for group in sources[:2] for message in group)
    assert all(message in result.messages for message in pins + tail)
    assert result.estimated_tokens == estimate_tokens(result.messages, tools)
    assert result.estimated_tokens <= manager.config.input_limit


def test_soft_source_groups_are_released_once_when_retained_history_exceeds_capacity():
    from patchrook_agent.context.compaction import _groups, _protected

    messages, sources, failure = source_working_set()
    pins, tail, tools = request_components(15)
    manager = ContextManager(ContextConfig(context_size=4096))
    groups = _groups(messages)
    hard = _protected(groups, manager.config.keep_recent_turns)
    admitted = manager._protected_for_request(groups, tools, pins, tail)
    assert len(admitted - hard) == 2  # All three exact sources fit the essential subset.
    result = manager.prepare(messages, tools, pinned_messages=pins, tail_messages=tail)
    assert result.tokens_before > manager.config.input_limit
    assert result.safe and result.compacted and result.attempted
    assert all(message in result.messages for message in sources[-1] + failure)
    assert all(message in result.messages for index in hard for message in groups[index])
    assert any(message not in result.messages for message in sources[0])
    assert all(message in result.messages for message in sources[1])
    assert manager.compaction_attempts == 1  # Fallback is part of one bounded attempt.
    assert result.estimated_tokens <= manager.config.input_limit


def test_source_aliases_share_a_slot_and_superseded_reads_can_be_compacted():
    messages, sources, failure = source_working_set()
    # Three canonical modules were read, then the API was reread through an alias.
    # The extra alias must not displace parser.py from the three-file working set.
    latest_api = exchange("read_file", {"path": "./api.py"}, {
        "success": True, "path": "api.py", "content": "current_api_source = 1\n" * 60})
    messages = messages[:2] + sources[1] + sources[2] + sources[0] + latest_api + messages[8:]
    for index in range(6):
        messages[2:2] = exchange("list_files", {"path": f"historical{index}"}, {
            "success": True, "entries": [f"previous_{index}_{number}.py" for number in range(100)]})
    pins, tail, tools = request_components(180)
    manager = ContextManager(ContextConfig(context_size=8192, trigger=0.4))
    result = manager.prepare(messages, tools, pinned_messages=pins, tail_messages=tail)
    assert result.safe and result.compacted
    assert all(message in result.messages for group in sources[1:] + [latest_api] for message in group)
    assert any(message not in result.messages for message in sources[0])
    assert all(message in result.messages for message in failure)


def test_schema_one_old_checkpoint_retains_explicit_legacy_goal():
    restored = ContextManager.from_dict({
        "schema_version": 1, "compaction_count": 2,
        "config": {"context_size": 4096, "trigger": 0.6, "target": 0.35,
                   "hard_limit": 0.85, "reserved_output_tokens": 512,
                   "enabled": True, "keep_recent_turns": 2},
    })
    assert restored.config.target == 0.35
    assert restored.config.compaction_goal == int(4096 * 0.35)
    assert restored.config.estimation_margin_tokens == 128
    assert restored.compaction_attempts == restored.compaction_count == 2


@pytest.mark.parametrize("change", [
    {"compaction_attempts": -1}, {"last_reclaimable": {"hash": True}},
    {"last_request_fingerprint": []},
])
def test_corrupt_persisted_retry_state_is_rejected(change):
    state = ContextManager().to_dict()
    state.update(change)
    with pytest.raises(ValueError):
        ContextManager.from_dict(state)
