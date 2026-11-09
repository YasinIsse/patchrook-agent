"""Decision rankings affect optional evidence without weakening exact evidence."""

from copy import deepcopy
import json

from patchrook_agent.coding import CODING_SYSTEM_PROMPT
from patchrook_agent.context.budget import ContextConfig
from patchrook_agent.context.lean import LeanContext, WORKING_SET_PREFIX


def history():
    return [{"role": "system", "content": CODING_SYSTEM_PROMPT},
            {"role": "user", "content": "Repair the calculation using observed source and test failures."}]


def exchange(identity, *records):
    calls = [{"id": f"{identity}_{index}", "type": "function",
              "function": {"name": name, "arguments": arguments}}
             for index, (name, arguments, _) in enumerate(records)]
    return [{"role": "assistant", "content": "", "tool_calls": calls},
            *[{"role": "tool", "tool_name": name, "tool_call_id": calls[index]["id"],
               "content": json.dumps(result)}
              for index, (name, _, result) in enumerate(records)]]


def source(path, index, count=10):
    lines = [f"value_{index}_{number} = '{'exact source evidence ' * 3}'" for number in range(count)]
    return ("read_file", {"path": path}, {
        "success": True, "path": path, "start_line": 1, "end_line": len(lines),
        "total_lines": len(lines), "truncated": False,
        "content": "\n".join(f"{number}: {line}" for number, line in enumerate(lines, 1)),
    })


def working(prepared):
    return next(message["content"] for message in prepared.messages
                if message.get("content", "").startswith(WORKING_SET_PREFIX))


def prepare(raw, priority, *, config=None, outstanding=()):
    return LeanContext(config or ContextConfig(context_size=4096, trigger=.5, target=.3)).prepare(
        raw, definitions=[], compact=True, system_prompt=CODING_SYSTEM_PROMPT,
        source_priority=priority, state_summary={"outstanding_tests": list(outstanding)})


def optional_history():
    raw = history()
    for index in range(8):
        raw += exchange(f"read_{index}", source(f"module_{index}.py", index))
    return raw


def test_ranked_optional_source_wins_tight_budget_instead_of_recent_unrelated_source():
    raw = optional_history() + exchange("test", ("run_tests", {"path": "."}, {
        "success": False, "status": "failed", "exit_code": 1,
        "stdout": "AssertionError: calculation should equal 41", "stderr": "",
    }))
    original = deepcopy(raw)
    default = prepare(raw, [])
    ranked = prepare(raw, ["module_0.py"])
    assert default.safe and ranked.safe
    assert default.tokens_before > default.tokens_after
    assert ranked.tokens_before > ranked.tokens_after
    assert "Observed source module_0.py" not in working(default)
    assert "Observed source module_7.py" in working(default)
    assert "Observed source module_0.py" in working(ranked)
    assert "Some older source is outside this bounded view" in working(ranked)
    assert ranked.messages[-2:] == default.messages[-2:] == raw[-2:]
    assert ranked.active_messages == default.active_messages == original == raw


def test_priority_controls_render_order_without_inventing_or_altering_source():
    raw = history() + exchange("a", source("a.py", 1, count=2))
    raw += exchange("z", source("z.py", 2, count=2))
    raw += exchange("tests", ("run_tests", {"path": "."}, {
        "success": True, "status": "passed", "exit_code": 0,
    }))
    normal = prepare(raw, [])
    ranked = prepare(raw, ["invented.py", "z.py", "a.py"])
    assert normal.safe and ranked.safe
    assert working(normal).index("Observed source a.py") < working(normal).index("Observed source z.py")
    assert working(ranked).index("Observed source z.py") < working(ranked).index("Observed source a.py")
    assert "Observed source invented.py" not in working(ranked)
    for marker in ("value_1_0", "value_1_1", "value_2_0", "value_2_1"):
        assert working(ranked).count(marker) == 1
        assert working(normal).count(marker) == 1


def test_ranking_keeps_recent_native_source_and_old_outstanding_failure_exact():
    old_failure = {"success": False, "status": "failed", "exit_code": 1,
                   "stdout": "tests/test_bad.py:4: assert total(3) == 41\nE assert 42 == 41\n",
                   "stderr": "exact older failure: café → calculation\n"}
    raw = history() + exchange("old_failure", ("run_tests", {"path": "tests/test_bad.py"}, old_failure))
    raw += optional_history()[2:]
    latest = exchange("latest",
                      source("current.py", 99, count=2),
                      ("run_tests", {"path": "tests/test_unrelated.py"}, {
                          "success": True, "status": "passed", "exit_code": 0,
                      }))
    raw += latest
    original = deepcopy(raw)
    ranked = prepare(raw, ["module_0.py", "module_1.py"], outstanding=["tests/test_bad.py"])
    assert ranked.safe and ranked.tokens_before > ranked.tokens_after
    assert ranked.messages[-3:] == latest
    assert "Observed source current.py" not in working(ranked)
    assert json.loads(ranked.messages[-2]["content"]) == json.loads(latest[-2]["content"])
    assert "value_99_0" in ranked.messages[-2]["content"]
    facts = json.loads(working(ranked).split("\nObserved state (data):\n", 1)[1].split("\n", 1)[0])
    assert facts["outstanding_test_failures"][0]["result"] == old_failure
    assert ranked.active_messages == original == raw


def test_oversized_protected_failure_stops_instead_of_dropping_error_for_ranked_source():
    failure = {"success": False, "status": "failed", "exit_code": 1,
               "stderr": "exact mandatory test failure\n" * 1200}
    latest = exchange("latest", source("current.py", 99, count=2),
                      ("run_tests", {"path": "."}, failure))
    raw = optional_history() + latest
    result = prepare(raw, ["module_0.py"])
    assert not result.safe and result.reason == "context_limit"
    assert result.messages[-3:] == latest
    assert json.loads(result.messages[-1]["content"]) == failure
    assert result.active_messages == raw
