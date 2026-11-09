"""Optional syntax regression prevention with real atomic edits, no inference."""

from dataclasses import asdict, replace
import hashlib
from pathlib import Path
import time

import pytest

from patchrook_agent.config import HarnessConfig
from patchrook_agent.evidence_tools import EvidenceTools
from patchrook_agent.model_routing import RoutingProgress, infrastructure_failure
from patchrook_agent.tools.base import ToolConfig, Workspace


@pytest.fixture(autouse=True)
def no_model_or_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Syntax validation must not run models or network requests")
    monkeypatch.setattr("httpx.Client", forbidden)
    monkeypatch.setattr("httpx.get", forbidden)
    monkeypatch.setattr("httpx.post", forbidden)
    monkeypatch.setattr("patchrook_agent.models.ollama.OllamaClient.chat", forbidden)


def tools(root, enabled=True):
    return EvidenceTools(Workspace(root, ToolConfig(validate_python_edits=enabled)))


def present(editor, path):
    data = (editor.workspace.root / path).read_bytes()
    return {"path": path, "content": data.decode(), "start_line": 1,
            "end_line": len(data.decode().splitlines()), "source_sha256": hashlib.sha256(data).hexdigest()}


def apply(editor, edits, sources=None):
    return editor.execute("apply_edits", {"edits": edits}, deadline=time.monotonic() + 10,
                          visible_sources=sources)


def create(path, source):
    return {"path": path, "content": source}


def edit(path, old, new):
    return {"path": path, "old_text": old, "new_text": new}


@pytest.mark.parametrize("candidate", ["def __init__(self):\n", "return 3\n", "if True print('x')\n", "x = '\0'\n"])
def test_invalid_new_python_rejects_before_any_directory_or_file_write(tmp_path, candidate):
    editor = tools(tmp_path)
    result = apply(editor, [create("new/deep/subject.py", candidate)])
    assert result["error"]["code"] == "python_syntax_error"
    assert "new/deep/subject.py" in result["error"]["message"]
    assert "line " in result["error"]["message"] and "No files were written" in result["error"]["message"]
    assert not result.get("modified_files")
    assert list(tmp_path.iterdir()) == []


def test_invalid_last_member_rejects_entire_valid_edit_and_new_file_batch(tmp_path):
    source = tmp_path / "subject.py"
    source.write_text("VALUE = 1\n")
    before = source.stat()
    editor = tools(tmp_path)
    result = apply(editor, [edit("subject.py", "VALUE = 1", "VALUE = 2"),
        create("valid_new.py", "VALID = True\n"), create("deep/invalid.py", "def broken():\n")],
        [present(editor, "subject.py")])
    assert result["error"]["code"] == "python_syntax_error"
    assert source.read_text() == "VALUE = 1\n"
    assert source.stat().st_ino == before.st_ino and source.stat().st_mtime_ns == before.st_mtime_ns
    assert sorted(path.name for path in tmp_path.iterdir()) == ["subject.py"]
    # Rejection keeps a fresh source snapshot useful for the next repair call.
    fixed = apply(editor, [edit("subject.py", "VALUE = 1", "VALUE = 3")])
    assert fixed["success"] and source.read_text() == "VALUE = 3\n"


def test_valid_python_becoming_invalid_is_rejected_without_changing_original(tmp_path):
    source = tmp_path / "subject.py"
    source.write_text("class Subject:\n    def __init__(self):\n        pass\n")
    editor = tools(tmp_path)
    result = apply(editor, [edit("subject.py", "        pass\n", "")], [present(editor, "subject.py")])
    assert result["error"]["code"] == "python_syntax_error"
    assert "pass" in source.read_text()


def test_valid_disjoint_edits_compile_as_one_candidate_not_individually(tmp_path):
    source = tmp_path / "subject.py"
    source.write_text("def value():\n    return 1\n")
    editor = tools(tmp_path)
    result = apply(editor, [edit("subject.py", "def value():", "def value(number):"),
                           edit("subject.py", "    return 1", "    return number + 1")],
                   [present(editor, "subject.py")])
    assert result["success"]
    assert result["python_syntax_validation"] == {"mode": "reject_new_errors", "checked_paths": ["subject.py"],
                                                  "preexisting_invalid_paths": []}
    compile(source.read_bytes(), "subject.py", "exec")


def test_existing_invalid_python_remains_editable_and_is_explicitly_reported(tmp_path):
    source = tmp_path / "subject.py"
    source.write_text("# pending repair\ndef broken():\n")
    editor = tools(tmp_path)
    result = apply(editor, [edit("subject.py", "# pending repair", "# still pending repair")],
                   [present(editor, "subject.py")])
    assert result["success"]
    assert result["python_syntax_validation"]["checked_paths"] == []
    assert result["python_syntax_validation"]["preexisting_invalid_paths"] == ["subject.py"]
    assert source.read_text() == "# still pending repair\ndef broken():\n"


def test_existing_syntax_error_can_be_fixed_and_other_broken_files_do_not_block(tmp_path):
    source = tmp_path / "subject.py"
    source.write_text("def broken():\n")
    (tmp_path / "untouched.py").write_text("other preexisting invalid Python !\n")
    editor = tools(tmp_path)
    result = apply(editor, [edit("subject.py", "def broken():\n", "def broken():\n    return 1\n")],
                   [present(editor, "subject.py")])
    assert result["success"]
    compile(source.read_bytes(), "subject.py", "exec")
    assert (tmp_path / "untouched.py").read_text() == "other preexisting invalid Python !\n"


@pytest.mark.parametrize("path", ["README.md", "Makefile", "source.js", "template.py.j2"])
def test_non_python_sources_are_not_compiled(tmp_path, path):
    editor = tools(tmp_path)
    result = apply(editor, [create(path, "this is deliberately not Python <<<\n")])
    assert result["success"]
    assert result["python_syntax_validation"]["checked_paths"] == []


def test_compile_neither_executes_imports_or_statements_nor_writes_pycache(tmp_path):
    marker = tmp_path / "must-not-exist"
    content = ("from package_that_does_not_exist import unknown\n"
               "raise RuntimeError('never executed')\n"
               f"open({str(marker)!r}, 'w').write('forbidden')\n")
    result = apply(tools(tmp_path), [create("subject.py", content)])
    assert result["success"]
    assert not marker.exists()
    assert sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*")) == ["subject.py"]
    # Syntax validity cannot establish whether runtime imports are correct.
    assert infrastructure_failure({"exit_code": 2, "stdout": "ERROR collecting test_x.py\nImportError: missing name"})


def test_default_disabled_preserves_current_edit_behavior(tmp_path):
    editor = EvidenceTools(Workspace(tmp_path))
    result = apply(editor, [create("subject.py", "def broken():\n")])
    assert result["success"] and "python_syntax_validation" not in result
    assert ToolConfig().validate_python_edits is False


@pytest.mark.parametrize("value", ["true", "false", 1, 0, None, [], {}])
def test_tool_config_opt_in_is_strict_boolean(value):
    with pytest.raises(ValueError, match="validate_python_edits must be a boolean"):
        ToolConfig(validate_python_edits=value)


def test_tool_config_round_trip_keeps_opt_in():
    config = ToolConfig(validate_python_edits=True)
    assert ToolConfig(**asdict(config)) == config


def test_source_authority_staleness_and_overlap_guards_precede_syntax_checks(tmp_path):
    source = tmp_path / "subject.py"
    source.write_text("VALUE = 1\n")
    editor = tools(tmp_path)
    malformed = edit("subject.py", "VALUE = 1", "def broken():")
    result = apply(editor, [malformed])
    assert result["error"]["code"] == "source_not_presented"
    view = present(editor, "subject.py")
    source.write_text("VALUE = 2\n")
    result = apply(editor, [malformed], [view])
    assert result["error"]["code"] == "source_not_presented"
    source.write_text("VALUE = 1\n")
    result = apply(editor, [malformed, malformed], [present(editor, "subject.py")])
    assert result["error"]["code"] == "overlapping_edits"
    assert source.read_text() == "VALUE = 1\n"


def test_all_noop_invalid_source_keeps_existing_no_change_semantics(tmp_path):
    source = tmp_path / "subject.py"
    source.write_text("def broken():\n")
    editor = tools(tmp_path)
    result = apply(editor, [edit("subject.py", "def broken():", "def broken():")], [present(editor, "subject.py")])
    assert result["error"]["code"] == "no_change"


def test_rejected_syntax_batch_counts_toward_invalid_response_routing_without_verification(tmp_path):
    config = replace(HarnessConfig(), workflow_mode="evidence", decision_model="on", model_routing="tev")
    progress = RoutingProgress()
    editor = tools(tmp_path)
    result = apply(editor, [create("subject.py", "def broken():\n")])
    for index in range(config.routing_max_invalid_responses):
        signal = progress.observe("apply_edits", {"edits": []}, result, 0, config)
    assert signal == "invalid_responses"
    assert progress.invalid_responses == config.routing_max_invalid_responses
    assert progress.failed_repairs == []
    assert not (tmp_path / "subject.py").exists()


@pytest.mark.parametrize("exception", [RecursionError, MemoryError])
def test_compiler_limits_return_bounded_rejection_without_writing(tmp_path, monkeypatch, exception):
    import patchrook_agent.evidence_tools as module
    def unavailable(*args, **kwargs):
        raise exception("compiler limit")
    monkeypatch.setattr(module, "compile", unavailable, raising=False)
    editor = tools(tmp_path)
    result = apply(editor, [create("subject.py", "VALUE = 1\n")])
    assert result["error"]["code"] == "python_syntax_validation_limit"
    assert list(tmp_path.iterdir()) == []
    config = replace(HarnessConfig(), workflow_mode="evidence", decision_model="on", model_routing="tev")
    progress = RoutingProgress()
    for index in range(config.routing_max_invalid_responses):
        signal = progress.observe("apply_edits", {"edits": []}, result, 0, config)
    assert signal == "invalid_responses"










def test_14b_gets_its_remaining_repair_call_after_syntax_rejection(tmp_path, monkeypatch):
    import runpy
    from patchrook_agent.completion import TaskRequirements
    helpers = runpy.run_path(str(Path(__file__).with_name("test_model_routing_workflow.py")))
    small = helpers["Coder"]("qwen3:8b", helpers["read"](), helpers["check"](), helpers["action"]("finish"))
    large = helpers["Coder"]("qwen3:14b", helpers["edit"](new="("), helpers["edit"]())
    agent, loads, unloads = helpers["setup"](tmp_path, monkeypatch, small, large, router=helpers["Router"]("large"))
    agent.registry.workspace.config = replace(agent.registry.workspace.config, validate_python_edits=True)
    state = agent.run("Fix the addition fixture and preserve comments.", requirements=TaskRequirements.fix())
    assert state.status == "completed"
    patches = helpers["tool_results"](state, "apply_edits")
    assert patches[0]["error"]["code"] == "python_syntax_error"
    assert patches[1]["success"]
    assert len(large.requests) == 2
    checks = helpers["tool_results"](state, "run_check")
    assert [check["exit_code"] for check in checks] == [1, 0]
    route = helpers["route"](state)
    assert route["repair"]["status"] == "verified" and route["repair"]["calls_used"] == 2
    assert route["switches"][-1]["reason"] == "repair_verified"
    assert "a + b" in (tmp_path / "repo/calc.py").read_text()
