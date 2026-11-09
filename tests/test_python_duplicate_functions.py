"""Optional adjacent-clone validation is atomic and preserves repair accounting."""

from dataclasses import replace
import time

import pytest

from patchrook_agent.python_duplicates import new_duplicate_function
from patchrook_agent.tools.base import ToolError
from test_evidence_python_syntax import tools, present, apply, create, edit
from test_model_routing_workflow import Coder, LARGE, SMALL, Router, action, route, setup, tool_results


SOURCE = 'class Subject:\n    def recover(self, data):\n        """Incomplete body."""\n        # original comment\n'
CLONED = SOURCE.replace("class Subject:\n", 'class Subject:\n    def recover(self, data):\n        """Incomplete body."""\n')


@pytest.fixture(autouse=True)
def no_real_model_or_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Duplicate validation tests must not call models or network services")
    monkeypatch.setattr("httpx.Client", forbidden)
    monkeypatch.setattr("httpx.get", forbidden)
    monkeypatch.setattr("httpx.post", forbidden)


def duplicate(before, after):
    return new_duplicate_function(before.encode(), after.encode(), deadline=time.monotonic() + 10)


def test_new_adjacent_same_scope_docstring_clone_is_rejected_atomically(tmp_path):
    target = tmp_path / "subject.py"
    target.write_text(SOURCE)
    editor = tools(tmp_path)
    insertion = 'class Subject:\n    def recover(self, data):\n        """Incomplete body."""\n'
    result = apply(editor, [edit("subject.py", "class Subject:\n", insertion),
        create("new/deep/unrelated.py", "VALUE = 1\n")], [present(editor, "subject.py")])
    assert result["error"]["code"] == "python_duplicate_function"
    assert "recover" in result["error"]["message"] and "No files were written" in result["error"]["message"]
    assert target.read_text() == SOURCE and not (tmp_path / "new").exists()


def test_optional_validation_off_keeps_legacy_behavior(tmp_path):
    (tmp_path / "subject.py").write_text(SOURCE)
    editor = tools(tmp_path, enabled=False)
    result = apply(editor, [edit("subject.py", SOURCE, CLONED)], [present(editor, "subject.py")])
    assert result["success"] and (tmp_path / "subject.py").read_text() == CLONED


@pytest.mark.parametrize("after", [CLONED + "\nUNCHANGED_DUPLICATES = True\n",
                                  CLONED.replace('"""Incomplete body."""', "return data")])
def test_existing_duplicate_counts_remain_editable(after):
    assert duplicate(CLONED, after) is None


def test_a_further_clone_above_existing_duplicates_is_new():
    third = CLONED.replace("class Subject:\n", 'class Subject:\n    def recover(self, data):\n        """Incomplete body."""\n')
    assert duplicate(CLONED, third)["name"] == "recover"


def test_redefined_outer_scopes_cannot_make_an_old_inner_clone_look_new():
    before = "def outer():\n    def f():\n        pass\n    def f():\n        pass\ndef outer():\n    pass\n"
    after = before[:-len("    pass\n")] + "    def f():\n        pass\n"
    assert duplicate(before, after) is None
    assert duplicate(before, before[:before.rindex("def outer():")]) is None


@pytest.mark.parametrize("before,after", [
    ("", "@overload\ndef f(x):\n    pass\n@overload\ndef f(x):\n    pass\n"),
    ("", "@dispatch.register(int)\ndef f(x):\n    pass\n@dispatch.register(str)\ndef f(x):\n    pass\n"),
    ("", "def f(x=allocate()):\n    return x\ndef f(x=allocate()):\n    return x\n"),
    ("", "def f(x: Annotation):\n    return x\ndef f(x: Annotation):\n    return x\n"),
    ("", "if FLAG:\n    def f():\n        pass\nelse:\n    def f():\n        pass\n"),
    ("", "def f():\n    pass\nsaved = f\ndef f():\n    pass\n"),
    ("", "class A:\n    def f(self):\n        pass\nclass B:\n    def f(self):\n        pass\n"),
    ("", "class A(Base):\n    def f(self):\n        pass\n    def f(self):\n        pass\n"),
    ("", "@register\nclass A:\n    def f(self):\n        pass\n    def f(self):\n        pass\n"),
    ("def f(x):\n    return 1\n", "def f(x):\n    return 1\ndef f(x, y):\n    return 2\n"),
])
def test_legitimate_or_ambiguous_patterns_are_deliberately_outside_guard(before, after):
    assert duplicate(before, after) is None


@pytest.mark.parametrize("prefix", ["def", "async def"])
def test_plain_nested_same_scope_clones_are_detected(prefix):
    source = f"def outer():\n    {prefix} recover(x):\n        return x\n    {prefix} recover(x):\n        return x\n"
    assert duplicate("", source)["scope"] == "function:outer"


def test_no_repository_code_executes_during_inspection(tmp_path):
    marker = tmp_path / "must-not-exist"
    source = f"raise RuntimeError('never execute')\nopen({str(marker)!r},'w').write('bad')\n" + CLONED
    assert duplicate("", source)["name"] == "recover"
    assert not marker.exists()


def test_deadline_is_enforced_without_inspection_or_write():
    with pytest.raises(ToolError) as raised:
        new_duplicate_function(b"", CLONED.encode(), deadline=time.monotonic() - 1)
    assert raised.value.code == "deadline_exceeded"


@pytest.mark.parametrize("repair_succeeds", [True, False])
def test_rejected_clone_consumes_the_used_14b_call_without_refund(tmp_path, monkeypatch, repair_succeeds):
    clone = action("apply_edits", edits=[{"path": "calc.py", "old_text": "def add(a, b):",
        "new_text": "def add(a, b):\n    return a + b\n\ndef add(a, b):"}])
    small = Coder(SMALL, action("finish"))
    corrected = action("apply_edits", edits=[{"path": "calc.py", "old_text": "a - b", "new_text": "a + b"}])
    large = Coder(LARGE, clone, corrected if repair_succeeds else clone)
    agent, _, _ = setup(tmp_path, monkeypatch, small, large, router=Router("large"),
        config_overrides={"evidence_initial_checks": True, "evidence_seed_symbols": True})
    agent.registry.workspace.config = replace(agent.registry.workspace.config, validate_python_edits=True)
    state = agent.run("Fix `add`.")
    rejected = [result for result in tool_results(state, "apply_edits") if not result["success"]]
    assert rejected[0]["error"]["code"] == "python_duplicate_function"
    assert route(state)["repair"]["calls_used"] == len(large.requests) == 2
    assert '"python_duplicate_function"' in str(large.requests[1]["messages"])
    if repair_succeeds:
        assert state.status == "completed" and state.verification == "passed"
    else:
        assert state.stop_reason == "repair_call_limit" and not state.modified_files
        assert len(rejected) == 2 and not small.requests
