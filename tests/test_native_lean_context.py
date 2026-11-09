"""Native lean feedback preserves the latest atomic tool exchange and its cost."""

from copy import deepcopy
import json

from patchrook_agent.context.budget import ContextConfig
from patchrook_agent.context.compaction import _groups
from patchrook_agent.context.lean import LEAN_SYSTEM_PROMPT, WORKING_SET_PREFIX, LeanContext, content_tokens


def history():
    return [{"role": "system", "content": LEAN_SYSTEM_PROMPT},
            {"role": "user", "content": "Fix calc.py and run its tests."}]


def exchange(records, identity="latest", content=""):
    calls = [{"id": f"{identity}-{index}", "type": "function",
              "function": {"name": name, "arguments": arguments}}
             for index, (name, arguments, result) in enumerate(records)]
    return [{"role": "assistant", "content": content, "tool_calls": calls},
            *[{"role": "tool", "tool_name": name, "tool_call_id": calls[index]["id"],
               "content": json.dumps(result)}
              for index, (name, arguments, result) in enumerate(records)]]


def read(content, *, path="calc.py", start=1):
    return {"success": True, "path": path, "start_line": start,
            "end_line": start + len(content.splitlines()) - 1, "content": content, "truncated": False}


def facts(prepared):
    rendered = next(message["content"] for message in prepared.messages
                    if message["content"].startswith(WORKING_SET_PREFIX))
    return json.loads(rendered[len(WORKING_SET_PREFIX):].split("\n", 1)[0])


def text(prepared):
    return "\n".join(message["content"] for message in prepared.messages)


def test_latest_native_exchange_is_atomic_exact_and_canonical_history_is_unchanged():
    old = exchange([("read_file", {"path": "other.py"}, read("1: preserved = True", path="other.py"))], "old")
    latest = exchange([
        ("read_file", {"path": "calc.py"}, read("1: def calc(a, b):\n2:     return a - b")),
        ("run_tests", {"path": "tests/test_calc.py"}, {
            "success": False, "status": "failed", "exit_code": 1, "stderr": "expected addition, got subtraction",
        }),
    ], content="I will inspect and verify this function.")
    raw = history() + old + latest
    original = deepcopy(raw)
    prepared = LeanContext(ContextConfig()).prepare(raw, definitions=[], protocol_instructions="Produce the patch next.")
    assert prepared.safe
    replay = next(group for group in _groups(prepared.messages) if group[0].get("tool_calls"))
    assert len(replay) == 3 and replay[0]["content"] == ""
    assert replay[0]["tool_calls"] == latest[0]["tool_calls"] and replay[1:] == latest[1:]
    assert raw == original == prepared.active_messages
    assert "I will inspect" not in text(prepared)
    assert "Observed source other.py" in text(prepared)
    assert "Observed source calc.py" not in text(prepared)
    assert text(prepared).count("return a - b") == 1
    assert text(prepared).count("expected addition, got subtraction") == 1
    assert facts(prepared)["latest_test"] == {"after_last_change": True, "result_ref": "native_tool_messages"}
    assert prepared.messages[-1] == {"role": "system", "content": "Produce the patch next."}


def test_older_source_ranges_survive_without_duplicating_the_latest_read():
    raw = history() + exchange([("read_file", {"path": "calc.py"}, read("1: def calc(a, b):"))], "old")
    raw += exchange([("read_file", {"path": "calc.py", "start_line": 2},
                      read("2:     return a - b", start=2))])
    prepared = LeanContext(ContextConfig()).prepare(raw, definitions=[])
    assert prepared.safe
    assert "Observed source calc.py lines 1-1:\n1: def calc(a, b):" in text(prepared)
    assert "Observed source calc.py lines 2-2" not in text(prepared)
    assert text(prepared).count("return a - b") == 1


def test_native_multicall_errors_and_failures_are_not_duplicated_in_working_set():
    records = [
        ("run_tests", {"path": "tests/test_a.py"}, {
            "success": False, "status": "failed", "exit_code": 1, "stderr": "first assertion trace",
        }),
        ("read_file", {"path": "missing.py"}, {"success": False, "error": {"message": "missing source detail"}}),
        ("run_tests", {"path": "tests/test_b.py"}, {
            "success": False, "status": "failed", "exit_code": 1, "stderr": "second assertion trace",
        }),
    ]
    prepared = LeanContext(ContextConfig()).prepare(history() + exchange(records), definitions=[])
    assert prepared.safe
    for phrase in ("first assertion trace", "missing source detail", "second assertion trace"):
        assert text(prepared).count(phrase) == 1
    assert "latest_errors" not in facts(prepared)
    assert "outstanding_test_failures" not in facts(prepared)


def test_older_outstanding_test_failure_remains_exact_next_to_latest_native_result():
    raw = history() + exchange([("run_tests", {"path": "tests/test_bad.py"}, {
        "success": False, "status": "failed", "exit_code": 1, "stderr": "older unresolved assertion",
    })], "old")
    raw += exchange([("run_tests", {"path": "tests/test_unrelated.py"}, {
        "success": True, "status": "passed", "exit_code": 0,
    })])
    prepared = LeanContext(ContextConfig()).prepare(raw, definitions=[], state_summary={
        "outstanding_tests": ["tests/test_bad.py"],
    })
    assert prepared.safe
    assert facts(prepared)["outstanding_test_failures"][0]["result"]["stderr"] == "older unresolved assertion"
    assert text(prepared).count("older unresolved assertion") == 1


def test_native_exchange_and_thinking_are_fully_charged_to_budget():
    definitions = [{"type": "function", "function": {"name": "read_file", "description": "Inspect code."}}]
    raw = history() + exchange([("read_file", {"path": "calc.py"}, read("1: value = 1"))])
    raw[-2]["thinking"] = "reasoning context " * 40
    lean = LeanContext(ContextConfig())
    lean.correction_tokens = 17
    prepared = lean.prepare(raw, definitions=definitions)
    assert prepared.safe
    assert prepared.estimated_tokens == content_tokens(prepared.messages, definitions) + 17
    without_thinking = deepcopy(prepared.messages)
    next(message for message in without_thinking if message.get("tool_calls")).pop("thinking")
    assert prepared.estimated_tokens > content_tokens(without_thinking, definitions) + 17 + 200


def test_oversized_native_exchange_fails_without_clipping_or_dropping_any_result():
    latest = exchange([("run_tests", {}, {
        "success": False, "status": "failed", "exit_code": 1, "stderr": "precise failure detail " * 1000,
    })])
    raw = history() + latest
    prepared = LeanContext(ContextConfig(context_size=2048, trigger=.5)).prepare(raw, definitions=[])
    assert not prepared.safe and prepared.reason == "context_limit"
    assert latest[-1] in prepared.messages
    assert prepared.active_messages == raw
    assert len(next(group for group in _groups(prepared.messages) if group[0].get("tool_calls"))) == 2


def test_schema_working_set_does_not_replay_native_messages():
    raw = history() + exchange([("read_file", {"path": "calc.py"}, read("1: value = 1"))])
    prepared = LeanContext(ContextConfig()).prepare(raw, definitions=[], response_schema={})
    assert prepared.safe
    assert not any(message.get("tool_calls") or message["role"] == "tool" for message in prepared.messages)
    assert "Observed source calc.py lines 1-1" in text(prepared)
    assert "result_ref" not in facts(prepared)["latest_action"]


def test_external_change_does_not_reintroduce_an_old_native_read_as_current_source():
    raw = history() + exchange([("read_file", {"path": "calc.py"}, read("1: stale_value = 1"))])
    lean = LeanContext(ContextConfig())
    lean.invalidate_paths(["calc.py"], history_length=len(raw))
    prepared = lean.prepare(raw, definitions=[])
    assert prepared.safe and "stale_value" not in text(prepared)
    assert not any(message.get("tool_calls") for message in prepared.messages)
    assert facts(prepared)["stale_source_paths"] == ["calc.py"]
    raw += exchange([("read_file", {"path": "calc.py"}, read("1: fresh_value = 2"))], "fresh")
    prepared = lean.prepare(raw, definitions=[])
    assert prepared.safe and "fresh_value" in text(prepared)
    assert any(message.get("tool_calls") for message in prepared.messages)
    assert "stale_source_paths" not in facts(prepared)
