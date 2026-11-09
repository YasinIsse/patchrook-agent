"""Independent mutation and process-boundary checks for the evidence workflow."""

import hashlib
import os
from pathlib import Path
import sys
import time

import pytest

from patchrook_agent.evidence_tools import EvidenceTools
from patchrook_agent.tools.base import ToolConfig, Workspace


def make_tools(tmp_path, *, allow_tests=False, commands=None, **limits):
    return EvidenceTools(Workspace(tmp_path, ToolConfig(allow_tests=allow_tests, **limits)), commands=commands)


def present(tools, path, start=1, end=None):
    target = tools.workspace.root / path
    data = target.read_bytes()
    lines = data.decode().splitlines(keepends=True)
    last = len(lines) if end is None else end
    source = {"path": path, "start_line": start, "end_line": last,
              "content": "".join(lines[start - 1:last]), "source_sha256": hashlib.sha256(data).hexdigest()}
    return source


def invoke(tools, name, arguments, *, seconds=5):
    return tools.execute(name, arguments, deadline=time.monotonic() + seconds)


def patch(tools, edits):
    return invoke(tools, "apply_edits", {"edits": edits})


def replace(path="calc.py", old="return a - b", new="return a + b"):
    return {"path": path, "old_text": old, "new_text": new}


def test_patch_uses_only_last_presented_exact_source_and_invalidates_authority(tmp_path):
    source = tmp_path / "calc.py"
    source.write_text("# preserve this comment\ndef add(a, b):\n    return a - b\n")
    tools = make_tools(tmp_path)
    assert not patch(tools, [replace()])["success"]
    assert tools.register_presented([present(tools, "calc.py", 3, 3)])["accepted"] == 1
    result = patch(tools, [replace()])
    assert result["modified_files"] == ["calc.py"]
    assert source.read_text() == "# preserve this comment\ndef add(a, b):\n    return a + b\n"
    assert result["changes"][0]["after_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    repeated = patch(tools, [replace(old="return a + b", new="return b + a")])
    assert repeated["error"]["code"] == "source_not_presented"


def test_previous_read_does_not_authorize_omitted_last_request_text(tmp_path):
    (tmp_path / "calc.py").write_text("x = 1\ny = 2\n")
    tools = make_tools(tmp_path)
    tools.register_presented([present(tools, "calc.py")])
    tools.register_presented([present(tools, "calc.py", 1, 1)])
    result = patch(tools, [replace(old="y = 2", new="y = 3")])
    assert result["error"]["code"] == "source_not_presented"
    assert (tmp_path / "calc.py").read_text() == "x = 1\ny = 2\n"


def test_old_content_cannot_gain_fresh_hash_authority_on_registration(tmp_path):
    source = tmp_path / "calc.py"
    source.write_text("x = 1\n# old external state\n")
    tools = make_tools(tmp_path)
    card = present(tools, "calc.py", 1, 1)
    source.write_text("x = 1\n# new external state\n")
    status = tools.register_presented([card])
    assert status["rejected"][0]["code"] == "stale_source"
    assert patch(tools, [replace(old="x = 1", new="x = 2")])["error"]["code"] == "source_not_presented"


@pytest.mark.parametrize("alteration", ["forged_content", "forged_range", "partial_line"])
def test_presented_cards_must_match_actual_complete_lines(tmp_path, alteration):
    (tmp_path / "calc.py").write_text("x = 1\ny = 2\n")
    tools = make_tools(tmp_path)
    card = present(tools, "calc.py", 1, 1)
    if alteration == "forged_content":
        card["content"] = "x = 99\n"
    elif alteration == "forged_range":
        card["end_line"] = 2
    else:
        card["partial_line"] = True
    assert tools.register_presented([card])["accepted"] == 0
    assert patch(tools, [replace(old="x = 1", new="x = 2")])["error"]["code"] == "source_not_presented"


def test_numbered_reads_can_be_presented_but_never_automatically_authorize_edits(tmp_path):
    (tmp_path / "calc.py").write_text("x = 1\ny = 2\n")
    tools = make_tools(tmp_path)
    result = invoke(tools, "read_file", {"path": "calc.py", "start_line": 1, "end_line": 1})
    assert result["content"] == "1: x = 1"
    assert result["source_sha256"] == hashlib.sha256((tmp_path / "calc.py").read_bytes()).hexdigest()
    assert patch(tools, [replace(old="x = 1", new="x = 2")])["error"]["code"] == "source_not_presented"
    assert tools.register_presented([result])["accepted"] == 1
    assert patch(tools, [replace(old="x = 1", new="x = 2")])["success"]


def test_external_edit_after_presentation_rejects_entire_batch(tmp_path):
    for name in ["calc.py", "other.py"]:
        (tmp_path / name).write_text("return a - b\n")
    tools = make_tools(tmp_path)
    tools.register_presented([present(tools, name) for name in ["calc.py", "other.py"]])
    (tmp_path / "other.py").write_text("return a - b\n# external edit\n")
    result = patch(tools, [replace(), replace("other.py")])
    assert result["error"]["code"] == "stale_source"
    assert (tmp_path / "calc.py").read_text() == "return a - b\n"
    assert (tmp_path / "other.py").read_text().endswith("# external edit\n")


def test_invalid_second_edit_cannot_partially_apply_first_or_create_parent(tmp_path):
    (tmp_path / "calc.py").write_text("return a - b\n")
    tools = make_tools(tmp_path)
    tools.register_presented([present(tools, "calc.py")])
    result = patch(tools, [replace(), {"path": "new/deeper/file.py", "content": "new"},
                           replace("missing.py")])
    assert not result["success"]
    assert (tmp_path / "calc.py").read_text() == "return a - b\n"
    assert not (tmp_path / "new").exists()


def test_disjoint_same_file_changes_are_matched_against_original_snapshot(tmp_path):
    (tmp_path / "calc.py").write_text("x = 1\ny = 2\n")
    tools = make_tools(tmp_path)
    tools.register_presented([present(tools, "calc.py")])
    result = patch(tools, [replace(old="x = 1", new="x = 100"), replace(old="y = 2", new="y = 200")])
    assert result["success"]
    assert (tmp_path / "calc.py").read_text() == "x = 100\ny = 200\n"


@pytest.mark.parametrize("second", [replace(old="return a - b", new="return 0"),
                                    replace(old="a - b", new="a + b")])
def test_duplicate_and_overlapping_edits_are_rejected(tmp_path, second):
    (tmp_path / "calc.py").write_text("return a - b\n")
    tools = make_tools(tmp_path)
    tools.register_presented([present(tools, "calc.py")])
    result = patch(tools, [replace(), second])
    assert result["error"]["code"] == "overlapping_edits"
    assert (tmp_path / "calc.py").read_text() == "return a - b\n"


def test_disjoint_presented_ranges_do_not_authorize_hidden_middle(tmp_path):
    (tmp_path / "calc.py").write_text("x = 1\nhidden = 2\ny = 3\n")
    tools = make_tools(tmp_path)
    tools.register_presented([present(tools, "calc.py", 1, 1), present(tools, "calc.py", 3, 3)])
    result = patch(tools, [replace(old="x = 1\nhidden = 2\ny = 3", new="x = 0")])
    assert result["error"]["code"] == "source_not_presented"


def test_new_file_creation_is_exclusive_and_includes_canonical_hash(tmp_path):
    tools = make_tools(tmp_path)
    result = patch(tools, [{"path": "new/package.py", "content": "VALUE = 42\n"}])
    assert result["success"]
    assert result["changes"][0]["created"]
    assert result["changes"][0]["before_sha256"] is None
    again = patch(tools, [{"path": "new/package.py", "content": "VALUE = 0\n"}])
    assert again["error"]["code"] == "already_exists"
    assert (tmp_path / "new/package.py").read_text() == "VALUE = 42\n"


@pytest.mark.parametrize("path", ["../escape.py", ".git/config", "src/.venv/code.py", ".", "/tmp/escape.py"])
def test_protected_paths_reject_new_files(tmp_path, path):
    tools = make_tools(tmp_path)
    result = patch(tools, [{"path": path, "content": "no"}])
    assert not result["success"]
    assert list(tmp_path.iterdir()) == []


def test_internal_and_external_symlink_write_paths_are_rejected(tmp_path):
    inside = tmp_path / "inside"
    inside.mkdir()
    external = tmp_path.parent / (tmp_path.name + "-external")
    external.mkdir()
    for name, target in [("alias", inside), ("outside", external)]:
        (tmp_path / name).symlink_to(target, target_is_directory=True)
    tools = make_tools(tmp_path)
    for alias in ["alias", "outside"]:
        result = patch(tools, [{"path": alias + "/new.py", "content": "no"}])
        assert result["error"]["code"] == "symlink_path"
    assert not list(inside.iterdir())
    assert not list(external.iterdir())


def test_ambiguous_and_noop_changes_do_not_write(tmp_path):
    source = tmp_path / "calc.py"
    source.write_text("x = 1\nx = 1\n")
    tools = make_tools(tmp_path)
    tools.register_presented([present(tools, "calc.py")])
    assert patch(tools, [replace(old="x = 1", new="x = 2")])["error"]["code"] == "ambiguous_match"
    assert patch(tools, [replace(old=source.read_text(), new=source.read_text())])["error"]["code"] == "no_change"


def test_crlf_and_executable_permissions_are_preserved(tmp_path):
    source = tmp_path / "calc.py"
    source.write_bytes(b"def add(a, b):\r\n    return a - b\r\n")
    source.chmod(0o755)
    tools = make_tools(tmp_path)
    tools.register_presented([present(tools, "calc.py")])
    assert patch(tools, [replace(old="def add(a, b):\n    return a - b", new="def add(a, b):\n    return a + b")])["success"]
    assert source.read_bytes() == b"def add(a, b):\r\n    return a + b\r\n"
    assert source.stat().st_mode & 0o777 == 0o755


def test_failed_second_commit_restores_first_file_and_removes_staged_outputs(tmp_path, monkeypatch):
    for name in ["calc.py", "other.py"]:
        (tmp_path / name).write_text("return a - b\n")
    tools = make_tools(tmp_path)
    tools.register_presented([present(tools, name) for name in ["calc.py", "other.py"]])
    original_replace = os.replace

    def fail_second(src, dest):
        if Path(dest).name == "other.py" and Path(src).name.startswith(".patchrook-evidence-"):
            raise OSError("simulated commit failure")
        return original_replace(src, dest)

    monkeypatch.setattr("patchrook_agent.evidence_tools.os.replace", fail_second)
    result = patch(tools, [replace(), replace("other.py")])
    assert result["error"]["code"] == "io_error"
    assert (tmp_path / "calc.py").read_text() == "return a - b\n"
    assert (tmp_path / "other.py").read_text() == "return a - b\n"
    assert not list(tmp_path.glob(".patchrook*"))


def test_racing_source_change_after_staging_is_not_overwritten(tmp_path, monkeypatch):
    source = tmp_path / "calc.py"
    source.write_text("return a - b\n")
    tools = make_tools(tmp_path)
    tools.register_presented([present(tools, "calc.py")])
    import patchrook_agent.evidence_tools as module
    original = module.tempfile.mkstemp

    def racing_mkstemp(*args, **kwargs):
        result = original(*args, **kwargs)
        source.write_text("external change\n")
        return result

    monkeypatch.setattr(module.tempfile, "mkstemp", racing_mkstemp)
    result = patch(tools, [replace()])
    assert result["error"]["code"] == "stale_source"
    assert source.read_text() == "external change\n"
    assert not list(tmp_path.glob(".patchrook*"))


def test_external_change_between_file_commits_causes_rollback_without_clobbering_it(tmp_path, monkeypatch):
    for name in ["calc.py", "other.py"]:
        (tmp_path / name).write_text("return a - b\n")
    tools = make_tools(tmp_path)
    tools.register_presented([present(tools, name) for name in ["calc.py", "other.py"]])
    original_replace = os.replace

    def race_after_first(src, dest):
        result = original_replace(src, dest)
        if Path(dest).name == "calc.py" and Path(src).name.startswith(".patchrook-evidence-"):
            (tmp_path / "other.py").write_text("external change\n")
        return result

    monkeypatch.setattr("patchrook_agent.evidence_tools.os.replace", race_after_first)
    result = patch(tools, [replace(), replace("other.py")])
    assert result["error"]["code"] == "stale_source"
    assert (tmp_path / "calc.py").read_text() == "return a - b\n"
    assert (tmp_path / "other.py").read_text() == "external change\n"


def test_external_change_to_committed_file_blocks_destructive_rollback_and_reports_mutation(tmp_path, monkeypatch):
    for name in ["calc.py", "other.py"]:
        (tmp_path / name).write_text("return a - b\n")
    tools = make_tools(tmp_path)
    tools.register_presented([present(tools, name) for name in ["calc.py", "other.py"]])
    original_replace = os.replace

    def fail_with_external_edit(src, dest):
        if Path(dest).name == "other.py" and Path(src).name.startswith(".patchrook-evidence-"):
            (tmp_path / "calc.py").write_text("external concurrent content\n")
            raise OSError("simulated second-write failure")
        return original_replace(src, dest)

    monkeypatch.setattr("patchrook_agent.evidence_tools.os.replace", fail_with_external_edit)
    result = patch(tools, [replace(), replace("other.py")])
    assert result["error"]["code"] == "rollback_failed"
    assert result["modified"] and result["modified_files"] == ["calc.py"]
    assert (tmp_path / "calc.py").read_text() == "external concurrent content\n"
    assert (tmp_path / "other.py").read_text() == "return a - b\n"


def test_list_and_search_paginate_explicitly(tmp_path):
    for number in range(5):
        (tmp_path / f"file{number}.py").write_text("needle\n")
    tools = make_tools(tmp_path, max_list_entries=2, max_search_results=2)
    listed = invoke(tools, "list_files", {"path": ".", "recursive": True})
    assert listed["truncated"] and listed["next_offset"] == 2
    found = invoke(tools, "search_code", {"pattern": "needle"})
    assert found["truncated"] and found["next_offset"] == 2


def test_commands_need_explicit_authorization_and_cannot_be_injected(tmp_path):
    command = [sys.executable, "-c", "from pathlib import Path; Path('ran').touch()"]
    tools = make_tools(tmp_path, commands={"check_1": command})
    assert "run_check" not in [item["function"]["name"] for item in tools.definitions()]
    assert invoke(tools, "run_check", {"name": "check_1"})["error"]["code"] == "permission_denied"
    enabled = make_tools(tmp_path, allow_tests=True, commands={"check_1": command})
    assert invoke(enabled, "run_check", {"name": "check_1", "argv": ["touch", "ran"]})["error"]["code"] == "invalid_arguments"
    assert invoke(enabled, "run_check", {"name": "check_2"})["error"]["code"] == "unknown_check"
    assert not (tmp_path / "ran").exists()


def test_successful_command_is_reported_as_command_exit_not_pytest_verification(tmp_path):
    command = [sys.executable, "-c", "print('999 passed in 0.01s')"]
    tools = make_tools(tmp_path, allow_tests=True, commands={"check_1": command})
    result = invoke(tools, "run_check", {"name": "check_1"})
    assert result["success"] and result["exit_code"] == 0
    assert result["argv"] == command
    assert result["kind"] == "command_check"
    assert result["verification_kind"] == "configured_command"
    assert "test_counts" not in result
    assert "pytest" not in result


def test_command_failure_and_full_capture_survive_observation_truncation(tmp_path):
    command = [sys.executable, "-c", "import sys; print('x' * 10000); print('failure', file=sys.stderr); sys.exit(3)"]
    tools = make_tools(tmp_path, allow_tests=True, commands={"check_1": command}, max_output_chars=1024)
    result = invoke(tools, "run_check", {"name": "check_1"})
    assert not result["success"] and result["exit_code"] == 3
    assert result["truncated"]
    assert len(tools.last_execution["available_output"]["stdout"]) == 10001
    assert tools.last_execution["raw_result"]["stderr"] == "failure\n"


def test_check_timeout_kills_descendants_and_respects_configured_limit(tmp_path):
    script = ("import subprocess, sys, time; from pathlib import Path; "
              "subprocess.Popen([sys.executable, '-c', \"import time; from pathlib import Path; time.sleep(2); Path('leaked').touch()\"]); "
              "Path('started').touch(); time.sleep(10)")
    # Leave enough startup time for real interpreter imports under CPU load;
    # the delayed child still outlives the configured parent deadline.
    tools = make_tools(tmp_path, allow_tests=True, test_timeout_seconds=1.0,
                       commands={"check_1": [sys.executable, "-c", script]})
    result = invoke(tools, "run_check", {"name": "check_1"})
    assert result["timed_out"] and not result["success"]
    assert (tmp_path / "started").exists()
    time.sleep(1.2)
    assert not (tmp_path / "leaked").exists()


def test_expired_deadline_executes_nothing(tmp_path):
    tools = make_tools(tmp_path, allow_tests=True,
                       commands={"check_1": [sys.executable, "-c", "from pathlib import Path; Path('ran').touch()"]})
    result = tools.execute("run_check", {"name": "check_1"}, deadline=time.monotonic() - 1)
    assert result["error"]["code"] == "deadline_exceeded"
    assert not (tmp_path / "ran").exists()


@pytest.mark.parametrize("commands", [{"check_1": "pytest"}, {"bad name": ["pytest"]},
                                     {"check_1": []}, {"check_1": ["echo", "\0"]}])
def test_malformed_command_configuration_is_rejected(tmp_path, commands):
    with pytest.raises(ValueError):
        make_tools(tmp_path, commands=commands)
