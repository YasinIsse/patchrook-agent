"""Real edit-guard regressions for redundant replacements in mixed batches."""

import hashlib
import os
from pathlib import Path
import sys
import time

import pytest

from patchrook_agent.evidence_tools import EvidenceTools
from patchrook_agent.tools import ToolConfig, Workspace


def tools_for(root, *, commands=None):
    return EvidenceTools(Workspace(root, ToolConfig(
        allow_tests=bool(commands), max_output_chars=12000)), commands=commands)


def source_view(root, name, start=1, end=None):
    raw = (root / name).read_bytes()
    lines = raw.decode().splitlines(keepends=True)
    end = len(lines) if end is None else end
    return {"path": name, "start_line": start, "end_line": end,
            "content": "".join(lines[start - 1:end]),
            "source_sha256": hashlib.sha256(raw).hexdigest()}


def authorize(tools, *names):
    root = tools.workspace.root
    result = tools.register_presented([source_view(root, name) for name in names])
    assert not result["rejected"]


def edit(name, old, new=None):
    return {"path": name, "old_text": old, "new_text": old if new is None else new}


def apply(tools, edits):
    return tools.execute("apply_edits", {"edits": edits}, deadline=time.monotonic() + 10)


def identity(path):
    info = path.stat()
    return path.read_bytes(), info.st_ino, info.st_mtime_ns, info.st_mode


@pytest.mark.parametrize("noop_first", [False, True])
def test_mixed_batch_only_commits_and_reports_changed_files(tmp_path, noop_first):
    changed, unchanged = tmp_path / "calc.py", tmp_path / "other.py"
    changed.write_text("x = 1\n")
    unchanged.write_text("KEEP = 2\n")
    unchanged.chmod(0o755)
    tools = tools_for(tmp_path)
    authorize(tools, "calc.py", "other.py")
    original = identity(unchanged)
    real, noop = edit("calc.py", "x = 1", "x = 2"), edit("other.py", "KEEP = 2")
    result = apply(tools, [noop, real] if noop_first else [real, noop])
    assert result["success"], result
    assert result["modified_files"] == ["calc.py"]
    assert [row["path"] for row in result["changes"]] == ["calc.py"]
    assert changed.read_text() == "x = 2\n"
    assert identity(unchanged) == original
    # A successful mutation invalidates every source view, including the noop.
    rejected = apply(tools, [edit("other.py", "KEEP = 2", "KEEP = 3")])
    assert rejected["error"]["code"] == "source_not_presented"


@pytest.mark.parametrize("noop_first", [False, True])
def test_noop_spans_do_not_conflict_with_real_same_file_replacement(tmp_path, noop_first):
    source = tmp_path / "calc.py"
    source.write_text("def add(a, b):\n    return a - b\n")
    tools = tools_for(tmp_path)
    authorize(tools, "calc.py")
    real = edit("calc.py", "a - b", "a + b")
    noop = edit("calc.py", source.read_text())
    result = apply(tools, [noop, real] if noop_first else [real, noop])
    assert result["success"], result
    assert result["modified_files"] == ["calc.py"]
    assert source.read_text() == "def add(a, b):\n    return a + b\n"


def test_real_overlaps_remain_rejected_after_a_noop(tmp_path):
    source = tmp_path / "calc.py"
    source.write_text("return a - b\n")
    tools = tools_for(tmp_path)
    authorize(tools, "calc.py")
    before = identity(source)
    result = apply(tools, [edit("calc.py", "return a - b"),
                           edit("calc.py", "a - b", "a + b"),
                           edit("calc.py", "return a - b", "return 0")])
    assert result["error"]["code"] == "overlapping_edits"
    assert identity(source) == before


def test_all_noops_reject_without_writes_and_retain_source_authority(tmp_path):
    source = tmp_path / "calc.py"
    source.write_text("x = 1\n")
    tools = tools_for(tmp_path)
    authorize(tools, "calc.py")
    before = identity(source)
    result = apply(tools, [edit("calc.py", "x = 1"), edit("calc.py", "x = 1")])
    assert result["error"]["code"] == "no_change"
    assert identity(source) == before
    assert apply(tools, [edit("calc.py", "x = 1", "x = 2")])["success"]


@pytest.mark.parametrize("with_real_creation", [False, True])
def test_net_unchanged_file_is_not_committed_or_reported(tmp_path, with_real_creation):
    source = tmp_path / "calc.py"
    source.write_text("ab\n")
    tools = tools_for(tmp_path)
    authorize(tools, "calc.py")
    before = identity(source)
    # Two disjoint real replacements compose to the original bytes.
    edits = [edit("calc.py", "a", ""), edit("calc.py", "b", "ab")]
    if with_real_creation:
        edits.append({"path": "new.py", "content": "VALUE = 1\n"})
    result = apply(tools, edits)
    if with_real_creation:
        assert result["success"], result
        assert result["modified_files"] == ["new.py"]
        assert [row["path"] for row in result["changes"]] == ["new.py"]
    else:
        assert result["error"]["code"] == "no_change"
    assert identity(source) == before


@pytest.mark.parametrize("kind,code", [
    ("unpresented", "source_not_presented"),
    ("omitted_range", "source_not_presented"),
    ("stale", "stale_source"),
    ("ambiguous", "ambiguous_match"),
    ("missing_anchor", "no_match"),
    ("invalid_shape", "invalid_arguments"),
    ("empty_anchor", "invalid_arguments"),
    ("protected_path", "protected_path"),
    ("binary_source", "binary_file"),
])
def test_invalid_noop_rejects_real_edit_and_creation_atomically(tmp_path, kind, code):
    real = tmp_path / "calc.py"
    noop = tmp_path / "other.py"
    real.write_text("x = 1\n")
    noop.write_text("KEEP = 2\nOTHER = 3\n")
    tools = tools_for(tmp_path)
    cards = [source_view(tmp_path, "calc.py")]
    if kind not in {"unpresented", "binary_source"}:
        cards.append(source_view(tmp_path, "other.py", 1, 1 if kind == "omitted_range" else 2))
    tools.register_presented(cards)
    unchanged_edit = edit("other.py", "KEEP = 2")
    if kind == "omitted_range":
        unchanged_edit = edit("other.py", "OTHER = 3")
    elif kind == "stale":
        noop.write_text("KEEP = 2\n# externally changed\n")
    elif kind == "ambiguous":
        noop.write_text("KEEP = 2\nKEEP = 2\n")
        authorize(tools, "calc.py", "other.py")
    elif kind == "missing_anchor":
        unchanged_edit = edit("other.py", "never existed")
    elif kind == "invalid_shape":
        unchanged_edit["extra"] = True
    elif kind == "empty_anchor":
        unchanged_edit = edit("other.py", "")
    elif kind == "protected_path":
        unchanged_edit = edit("../escape.py", "KEEP = 2")
    elif kind == "binary_source":
        noop.write_bytes(b"KEEP = 2\x00\n")
    real_before, noop_before = identity(real), identity(noop)
    result = apply(tools, [edit("calc.py", "x = 1", "x = 2"),
                           {"path": "new/deep.py", "content": "new"}, unchanged_edit])
    assert result["error"]["code"] == code, result
    assert identity(real) == real_before
    assert identity(noop) == noop_before
    assert not (tmp_path / "new").exists()


def test_crlf_equivalent_replacement_is_a_noop_after_normalization(tmp_path):
    changed, unchanged = tmp_path / "calc.py", tmp_path / "other.py"
    changed.write_text("x = 1\n")
    unchanged.write_bytes(b"KEEP = 2\r\nOTHER = 3\r\n")
    tools = tools_for(tmp_path)
    authorize(tools, "calc.py", "other.py")
    before = identity(unchanged)
    result = apply(tools, [edit("other.py", "KEEP = 2\nOTHER = 3", "KEEP = 2\r\nOTHER = 3"),
                           edit("calc.py", "x = 1", "x = 2")])
    assert result["success"], result
    assert result["modified_files"] == ["calc.py"]
    assert identity(unchanged) == before


def test_invalid_real_edit_after_noop_still_writes_nothing(tmp_path):
    for name in ("calc.py", "other.py"):
        (tmp_path / name).write_text("x = 1\n")
    tools = tools_for(tmp_path)
    authorize(tools, "calc.py", "other.py")
    before = {name: identity(tmp_path / name) for name in ("calc.py", "other.py")}
    result = apply(tools, [edit("other.py", "x = 1"), edit("calc.py", "x = 1", "x = 2"),
                           edit("calc.py", "missing", "x = 3")])
    assert result["error"]["code"] == "no_match"
    assert {name: identity(tmp_path / name) for name in before} == before


def test_noop_dependency_change_during_staging_rejects_before_commit(tmp_path, monkeypatch):
    real, noop = tmp_path / "calc.py", tmp_path / "other.py"
    real.write_text("x = 1\n")
    noop.write_text("KEEP = 2\n")
    tools = tools_for(tmp_path)
    authorize(tools, "calc.py", "other.py")
    import patchrook_agent.evidence_tools as module
    original = module.tempfile.mkstemp

    def race(*args, **kwargs):
        result = original(*args, **kwargs)
        noop.write_text("# external change\nKEEP = 2\n")
        return result

    monkeypatch.setattr(module.tempfile, "mkstemp", race)
    result = apply(tools, [edit("calc.py", "x = 1", "x = 2"), edit("other.py", "KEEP = 2")])
    assert result["error"]["code"] == "stale_source", result
    assert real.read_text() == "x = 1\n"
    assert noop.read_text() == "# external change\nKEEP = 2\n"
    assert not list(tmp_path.glob(".patchrook*"))


@pytest.mark.parametrize("changed_count", [1, 2])
def test_noop_dependency_race_after_real_commit_rolls_back_changes(tmp_path, monkeypatch, changed_count):
    for name in ("calc.py", "second.py"):
        (tmp_path / name).write_text("x = 1\n")
    noop = tmp_path / "other.py"
    noop.write_text("KEEP = 2\n")
    tools = tools_for(tmp_path)
    authorize(tools, "calc.py", "second.py", "other.py")
    original_replace = os.replace

    def race(src, dest):
        result = original_replace(src, dest)
        if Path(dest).name == "calc.py" and Path(src).name.startswith(".patchrook-evidence-"):
            noop.write_text("KEEP = 2\n# external change\n")
        return result

    monkeypatch.setattr("patchrook_agent.evidence_tools.os.replace", race)
    edits = [edit("calc.py", "x = 1", "x = 2")]
    if changed_count == 2:
        edits.append(edit("second.py", "x = 1", "x = 2"))
    result = apply(tools, [*edits, edit("other.py", "KEEP = 2")])
    assert result["error"]["code"] == "stale_source", result
    assert (tmp_path / "calc.py").read_text() == "x = 1\n"
    assert (tmp_path / "second.py").read_text() == "x = 1\n"
    assert noop.read_text() == "KEEP = 2\n# external change\n"
    assert not list(tmp_path.glob(".patchrook*"))


def test_recorded_14b_discount_fix_plus_shipping_noop_passes_real_checks(tmp_path):
    from coding_fixtures import verify_changes
    from coding_fixtures import create_fixture

    task = create_fixture(tmp_path, "multifile-fix")
    tools = tools_for(tmp_path, commands={"check_1": [sys.executable, "-m", "pytest", "-q"]})
    authorize(tools, "money.py", "orders.py", "settings.py")
    shipping = "shipping = SHIPPING_CENTS if merchandise < FREE_SHIPPING_CENTS else 0"
    assert apply(tools, [edit("orders.py",
        "shipping = 0 if merchandise <= FREE_SHIPPING_CENTS else SHIPPING_CENTS", shipping)])["success"]
    failed = tools.execute("run_check", {"name": "check_1"}, deadline=time.monotonic() + 30)
    assert failed["success"] is False and "4900 == 4100" in failed["stdout"]
    authorize(tools, "money.py", "orders.py", "settings.py")
    orders_before = identity(tmp_path / "orders.py")
    # Exact semantic proposal from the recorded 14B rescue, embedded so the
    # regression does not depend on an untracked local evaluation report.
    result = apply(tools, [edit("money.py", "return subtotal + subtotal * discount_percent // 100",
                               "return subtotal - subtotal * discount_percent // 100"),
                           edit("orders.py", shipping)])
    assert result["success"], result
    assert result["modified_files"] == ["money.py"]
    assert identity(tmp_path / "orders.py") == orders_before
    passed = tools.execute("run_check", {"name": "check_1"}, deadline=time.monotonic() + 30)
    assert passed["success"] is True and passed["exit_code"] == 0
    changed = {name: (tmp_path / name).read_text() for name, original in task.files.items()
               if (tmp_path / name).read_text() != original}
    independent = verify_changes(task, changed)
    assert independent["passed"], independent
    assert independent["checks"]["unchanged:tests/test_orders.py"]
    assert independent["checks"]["unchanged:settings.py"]
