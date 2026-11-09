"""Coding prompt reductions preserve exact source, native feedback, and failures."""

from copy import deepcopy
from dataclasses import replace
import json
from os.path import commonprefix
from types import SimpleNamespace

from patchrook_agent.agent import AgentState
from patchrook_agent.coding import CODING_SYSTEM_PROMPT
from patchrook_agent.config import HarnessConfig
from patchrook_agent.context.budget import ContextConfig
from patchrook_agent.context.lean import WORKING_SET_PREFIX, LeanContext, content_tokens
from patchrook_agent.inspection import InspectionLedger
from patchrook_agent.session import Session
from patchrook_agent.tools import ToolConfig, ToolRegistry, Workspace


def history():
    return [{"role": "system", "content": CODING_SYSTEM_PROMPT},
            {"role": "user", "content": "Fix the observed addition bug and preserve unrelated behavior."}]


def exchange(name, arguments, result):
    return [{"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": arguments}}]},
            {"role": "tool", "tool_name": name, "content": json.dumps(result)}]


def read(path, source, *, truncated=False):
    lines = source.splitlines()
    record = {"success": True, "path": path, "start_line": 1, "end_line": len(lines),
              "content": "\n".join(f"{number}: {line}" for number, line in enumerate(lines, 1)), "truncated": truncated}
    if truncated:
        record["selected_end_line"] = len(lines) + 5
    return exchange("read_file", {"path": path, "start_line": 1, "end_line": record.get("selected_end_line", len(lines))}, record)


def rendered(prepared):
    return "\n".join(message.get("content", "") for message in prepared.messages)


def working(prepared):
    return next(message["content"] for message in prepared.messages if message["content"].startswith(WORKING_SET_PREFIX))


def compact_facts(prepared):
    return json.loads(working(prepared).split("\nObserved state (data):\n", 1)[1].split("\n", 1)[0])


def coding_session(tmp_path, raw):
    config = replace(HarnessConfig.preset("C"), workflow_mode="coding", persist=False,
                     checkpoint_enabled=False, logging_enabled=False)
    workspace = Workspace(tmp_path)
    session = Session(config, workspace, SimpleNamespace(context_size=8192, max_output_tokens=512),
                      tmp_path / "runs", raw[1]["content"])
    session.memory.inspection = InspectionLedger.from_messages(raw, workspace)
    state = AgentState(messages=deepcopy(raw), active_messages=deepcopy(raw))
    state.completion = {"progress": "Verification: not_run; mutation 0.",
                        "missing": ["make the requested code change", "observe passing tests after the last change"],
                        "outstanding_tests": []}
    registry = ToolRegistry(workspace)
    registry.enable_lean_tools()
    definitions = registry.definitions(enabled_tools={"read_code", "read_file", "edit_file", "write_file"})
    return session, state, definitions


def test_sparse_coding_context_is_smaller_without_removing_any_observed_source_or_native_feedback(tmp_path):
    raw = history()
    paths = [f"module_{index:02d}.py" for index in range(24)]
    for index, path in enumerate(paths):
        raw += read(path, f"def calculate_{index}(value):\n    return value + {987000 + index}")
    original = deepcopy(raw)
    session, state, definitions = coding_session(tmp_path, raw)
    guidance = "The requested change and verification are still required."
    current = session.prepare(state, definitions, protocol_instructions=guidance)
    previous = LeanContext(session.context.config, workspace=session.workspace).prepare(
        raw, definitions=definitions, system_prompt=CODING_SYSTEM_PROMPT,
        protocol_instructions=guidance, state_summary={
            "modified_files": [], "verification": "not_run", "available_source_paths": paths,
            "current": state.completion["progress"], "missing": state.completion["missing"], "outstanding_tests": [],
        })
    assert current.safe and previous.safe
    assert current.estimated_tokens + 100 < previous.estimated_tokens
    assert state.messages == original == current.active_messages
    assert current.messages[-3:-1] == raw[-2:]
    assert current.messages[-1]["content"] == guidance
    assert not compact_facts(current)
    for index, path in enumerate(paths):
        assert path in rendered(current)
        assert rendered(current).count(f"return value + {987000 + index}") == 1


def test_source_before_changing_state_preserves_a_larger_identical_request_prefix(tmp_path):
    raw = history() + read("a.py", "def add(a, b):\n    return a + b")
    raw += read("z_test.py", "def test_add():\n    assert 1 + 1 == 2")
    session, state, definitions = coding_session(tmp_path, raw)
    first = session.prepare(state, definitions)
    old_first = LeanContext(session.context.config).prepare(raw, definitions=definitions,
                                                          state_summary={"verification": "not_run"})
    state.messages += exchange("run_tests", {"path": "."}, {
        "success": True, "status": "passed", "exit_code": 0, "stdout": "1 passed",
    })
    state.verification = "passed"
    second = session.prepare(state, definitions)
    old_second = LeanContext(session.context.config).prepare(state.messages, definitions=definitions,
                                                           state_summary={"verification": "passed"})
    prefix = commonprefix([working(first), working(second)])
    assert "Observed source a.py" in prefix and "return a + b" in prefix
    assert len(prefix) > len(commonprefix([working(old_first), working(old_second)]))
    assert working(second).index("Observed source a.py") < working(second).index("Observed state (data)")


def test_path_hints_only_name_unrepresented_or_partial_source():
    source = "\n".join(f"line_{index} = '{'evidence' * 16}'" for index in range(120))
    raw = history() + read("large.py", source) + read("latest.py", "result = 1", truncated=True)
    context = LeanContext(ContextConfig(context_size=4096, trigger=.5, target=.3))
    result = context.prepare(raw, definitions=[], compact=True,
                             source_paths=["large.py", "latest.py", "not_yet_read.py"], incomplete_paths=["latest.py"])
    assert result.safe
    facts = compact_facts(result)
    assert facts["partial_source_paths"] == ["latest.py"]
    assert "not_yet_read.py" in facts["missing_source_paths"]
    assert "latest.py" not in facts["missing_source_paths"]
    assert "large.py" in facts["missing_source_paths"]
    assert result.messages[-2:] == raw[-2:]


def test_compact_context_retains_exact_unresolved_failure_but_honors_empty_controller_debt():
    failure = {"success": False, "status": "failed", "exit_code": 1, "stderr": "exact older assertion failure"}
    raw = history() + exchange("run_tests", {"path": "tests/test_bad.py"}, failure)
    raw += exchange("run_tests", {"path": "."}, {"success": True, "status": "passed", "exit_code": 0})
    context = LeanContext(ContextConfig())
    outstanding = context.prepare(raw, definitions=[], compact=True,
                                  state_summary={"outstanding_tests": ["tests/test_bad.py"]})
    assert outstanding.safe and "exact older assertion failure" in rendered(outstanding)
    resolved = context.prepare(raw, definitions=[], compact=True, state_summary={"outstanding_tests": []})
    assert resolved.safe and "exact older assertion failure" not in rendered(resolved)
    assert "progress" not in compact_facts(resolved)
    assert resolved.messages[-2:] == raw[-2:]


def test_skills_and_current_native_error_remain_protected_without_borrowing_more_context():
    skill = {"role": "system", "content": "Trusted skill document: debugging/SKILL.md\n" + "precise instruction " * 40}
    error = {"success": False, "status": "failed", "exit_code": 1, "stderr": "exact error trace " * 1000}
    raw = history() + exchange("run_tests", {"path": "."}, error)
    result = LeanContext(ContextConfig(context_size=2048, trigger=.5)).prepare(
        raw, definitions=[], compact=True, skills=[skill])
    assert not result.safe and result.reason == "context_limit"
    assert skill in result.messages and result.messages[-2:] == raw[-2:]
    assert result.active_messages == raw


def test_coding_context_never_advertises_an_unavailable_skill_tool(tmp_path):
    raw = history()
    raw[1]["content"] = "Use a relevant skill to fix the observed source."
    session, state, definitions = coding_session(tmp_path, raw)
    session.skills.index_text = lambda: "Available trusted skills; load instructions with read_skill(name):\n- debugging"
    result = session.prepare(state, definitions)
    assert result.safe and "read_skill(name)" not in rendered(result)


def test_coding_reads_omit_optional_header_tails_but_preserve_operations_and_small_module_imports(tmp_path):
    originals = {}
    for index in range(2):
        path = f"stage_{index}.py"
        originals[path] = (f"# Stage {index} documentation: preserve unrelated behavior and comments.\n" * 12
                           + f"\ndef transform(value):\n    return value + {index + 1}\n")
    originals["runner.py"] = (
        "from stage_0 import transform as first\n"
        "from stage_1 import transform as second\n\n"
        "def run(value):\n    return second(first(value))\n")
    for path, source in originals.items():
        (tmp_path / path).write_text(source)
    registry = ToolRegistry(Workspace(tmp_path, ToolConfig(max_output_chars=8000)))
    registry.enable_lean_tools()
    arguments = {"paths": list(originals)}
    expanded = registry.dispatch("read_code", arguments)
    registry.code_context = False
    compact = registry.dispatch("read_code", arguments)
    assert expanded["success"] and compact["success"] and not compact["truncated"]
    records = {record["path"]: record for record in compact["files"]}
    for index in range(2):
        record = records[f"stage_{index}.py"]
        assert f"14: def transform(value):\n15:     return value + {index + 1}" == record["content"]
        assert record["omitted_ranges"] == [[1, 13]]
    assert "1: from stage_0 import transform as first\n2: from stage_1 import transform as second" in records["runner.py"]["content"]
    assert "5:     return second(first(value))" in records["runner.py"]["content"]
    assert "documentation" not in json.dumps(compact) and "documentation" in json.dumps(expanded)
    definitions = registry.definitions()
    assert content_tokens(history() + exchange("read_code", arguments, compact), definitions) < content_tokens(
        history() + exchange("read_code", arguments, expanded), definitions)
    assert all((tmp_path / path).read_text() == source for path, source in originals.items())
