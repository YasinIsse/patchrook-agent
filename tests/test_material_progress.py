"""Coding progress follows implementation gains rather than range expansion."""

from copy import deepcopy
import json

import pytest

from patchrook_agent.lean_progress import LeanProgress
from patchrook_agent.tools import Workspace


@pytest.fixture
def workspace(tmp_path):
    for name in ("module.py", "adjustments.py", "taxes.py", "shipping.py"):
        (tmp_path / name).write_text("raise RuntimeError('hidden source must never be opened')\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_total.py").write_text("def test_total():\n    assert False\n")
    return Workspace(tmp_path)


def record(path="module.py", text="value = 1", start=1, **extra):
    lines = text.split("\n")
    return {"success": True, "path": path, "start_line": start,
            "end_line": start + len(lines) - 1,
            "content": "\n".join(f"{number}: {source}" for number, source in enumerate(lines, start)),
            "truncated": False, **extra}


def read(progress, text="value = 1", *, start=1, path="module.py", **extra):
    return progress.observe("read_file", {"path": path}, record(path, text, start, **extra))


def recovered(progress):
    read(progress)
    read(progress)
    assert read(progress) == "recover"
    progress.acknowledge_recovery()
    assert progress.stall_count == 2


@pytest.mark.parametrize("text", [
    "", "    ", "# A newly exposed header comment", "def unused(value):", "class Settings:",
    "@cached", "def unused(\n        value=1,\n        flag=True,\n):",
    '"""A standalone description."""',
    '    """First docstring line.\n    Some extra explanation of the same helper.\n    """',
])
def test_navigation_only_source_does_not_reset_stall(workspace, text):
    progress = LeanProgress(workspace, material_source_progress=True)
    recovered(progress)
    assert read(progress, text, start=20) is None
    assert progress.stall_count == 3 and progress.recovery_delivered
    assert read(progress, text, start=20) == "stop"


@pytest.mark.parametrize("text", [
    "    return value + 1", "import helper", "from helper import total",
    "SUPPORTED = {'USD', 'EUR'}", "def one(): return 1", "if value < 0:",
    "    call_helper(value)", "    pass", "class Settings:\n    currency = 'USD'",
    'MESSAGE = """First line\nsecond line"""',
])
def test_actual_implementation_and_configuration_are_progress(workspace, text):
    progress = LeanProgress(workspace, material_source_progress=True)
    recovered(progress)
    assert read(progress, text, start=20) is None
    assert progress.stall_count == 0 and not progress.recovery_delivered


def test_comment_and_whitespace_changes_do_not_invent_new_implementation(workspace):
    progress = LeanProgress(workspace, material_source_progress=True)
    read(progress, "value = 1  # existing note")
    read(progress, "value = 1  # a changed note")
    assert read(progress, "value  =  1") == "recover"
    progress.acknowledge_recovery()
    assert read(progress, "value = 2") is None and progress.stall_count == 0


def test_header_continuation_cannot_be_reclassified_as_new_code(workspace):
    progress = LeanProgress(workspace, material_source_progress=True)
    source = "def calculate(\n    value=1,\n):\n    return value + 1"
    read(progress, source)
    read(progress, source)
    assert read(progress, source) == "recover"
    progress.acknowledge_recovery()
    assert read(progress, "    value=1,", start=2) is None
    assert progress.stall_count == 3
    assert read(progress, source) == "stop"


def test_changed_multiline_configuration_value_is_material(workspace):
    progress = LeanProgress(workspace, material_source_progress=True)
    source = 'MESSAGE = """First line\noriginal second line"""'
    read(progress, source)
    read(progress, source)
    assert read(progress, source) == "recover"
    progress.acknowledge_recovery()
    assert read(progress, source.replace("original second", "changed second")) is None
    assert progress.stall_count == 0


def test_read_search_and_batch_share_material_source_identity(workspace):
    progress = LeanProgress(workspace, material_source_progress=True)
    initial = record(text="def calculate(value):\n    return value + 1")
    assert progress.observe("read_code", {}, {"success": True, "files": [initial]}) is None
    assert progress.observe("search_code", {}, {"success": True, "matches": [
        {"path": "module.py", "line_number": 2, "text": "    return value + 1"}]}) is None
    assert progress.observe("read_files", {}, {"success": True, "files": [initial]}) == "recover"
    progress.acknowledge_recovery()
    assert progress.observe("search_code", {}, {"success": True, "matches": [
        {"path": "module.py", "line_number": 20, "text": "# unrelated comment"}]}) is None
    assert progress.stall_count == 3
    assert progress.observe("search_code", {}, {"success": True, "matches": [
        {"path": "module.py", "line_number": 21, "text": "    return value - 1"}]}) is None
    assert progress.stall_count == 0


def test_only_observed_complete_lines_may_advance_progress(workspace, monkeypatch):
    progress = LeanProgress(workspace, material_source_progress=True)
    def forbidden(*args, **kwargs):
        pytest.fail("Progress must not open unobserved filesystem source")
    monkeypatch.setattr("pathlib.Path.open", forbidden)
    assert read(progress, "return value + 1", partial_line=True, truncated=True) is None
    assert progress.stall_count == 1
    assert progress.observe("read_artifact", {}, {"success": True, "content": "return value + 2"}) == "recover"
    progress.acknowledge_recovery()
    assert read(progress, "return value + 1", start=2) is None and progress.stall_count == 0


def test_material_mode_persists_both_source_identity_and_recovery(workspace):
    progress = LeanProgress(workspace, material_source_progress=True)
    recovered(progress)
    stored = json.loads(json.dumps(progress.to_dict()))
    assert stored["material_source_progress"] is True
    restored = LeanProgress.from_dict(stored, workspace)
    assert restored.to_dict() == stored and restored.material_source_progress
    assert read(restored, "# new header", start=20) is None
    assert read(restored, "def label():", start=21) == "stop"
    # Real changes still invalidate test passes and begin a fresh generation.
    restored.observe("run_tests", {"path": "."}, {"success": True, "status": "passed", "exit_code": 0})
    restored.observe("edit_file", {}, {"success": True, "modified": True})
    assert restored.generation == 1 and restored.verification_status == "not_run"
    assert read(restored) is None and restored.stall_count == 0


def test_old_progress_checkpoints_retain_legacy_mode(workspace):
    legacy = LeanProgress(workspace)
    assert "material_source_progress" not in legacy.to_dict()
    restored = LeanProgress.from_dict(legacy.to_dict(), workspace)
    assert not restored.material_source_progress
    read(restored)
    read(restored)
    restored.acknowledge_recovery()
    assert read(restored, "# legacy treats a new visible line as evidence", start=20) is None
    assert restored.stall_count == 0
    with pytest.raises(ValueError):
        LeanProgress.from_dict({**legacy.to_dict(), "material_source_progress": "true"}, workspace)


def test_captured_read_loop_stops_despite_blank_and_signature_expansion(workspace):
    progress = LeanProgress(workspace, material_source_progress=True)
    bodies = {
        "adjustments.py": "def adjust(value):\n    return value + 1",
        "taxes.py": "def tax(value):\n    return value // 10",
        "shipping.py": "def shipping(value):\n    return value + 500",
    }
    baseline = {"success": True, "files": [record(path, text, 2) for path, text in bodies.items()]}
    untouched = deepcopy(baseline)
    progress.observe("read_code", {}, baseline)
    failed = {"success": False, "status": "failed", "exit_code": 1,
              "stdout": "E assert total == expected", "stderr": ""}
    progress.observe("run_tests", {"path": "."}, failed)
    assert read(progress, bodies["adjustments.py"], start=2, path="adjustments.py") is None
    assert read(progress, bodies["taxes.py"], start=2, path="taxes.py") == "recover"
    progress.acknowledge_recovery()
    assert read(progress, bodies["shipping.py"] + "\n\ndef shipping_label(value):", start=2, path="shipping.py") is None
    assert progress.stall_count == 3
    assert read(progress, bodies["taxes.py"], start=2, path="taxes.py") == "stop"
    assert progress.verification_status == "failed" and progress.generation == 0
    assert baseline == untouched and failed["stdout"] == "E assert total == expected"
