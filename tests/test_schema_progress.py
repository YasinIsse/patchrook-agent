"""Hints use only visible import lines and confined file identities."""

import json

from patchrook_agent.progress import pending_local_imports
from patchrook_agent.tools import Workspace


def read(path, content):
    return {"role": "tool", "tool_name": "read_file", "content": json.dumps({
        "success": True, "path": path, "content": content})}


def test_visible_imports_become_pending_paths_without_reading_contents(tmp_path):
    (tmp_path / "helpers.py").write_text("raise RuntimeError('never import this module')\n")
    messages = [read("entry.py", "1: from helpers import operation")]
    assert pending_local_imports(messages, Workspace(tmp_path)) == ["helpers.py"]
    messages.append(read("helpers.py", "1: def operation():"))
    assert pending_local_imports(messages, Workspace(tmp_path)) == []


def test_external_and_hidden_imports_provide_no_paths(tmp_path):
    (tmp_path / "local.py").write_text("value = 1\n")
    outside = tmp_path.parent / "outside.py"
    outside.write_text("value = 1\n")
    (tmp_path / "alias.py").symlink_to(outside)
    messages = [read("entry.py", "1: import os\n2: import alias"),
                {"role": "tool", "tool_name": "read_file", "content": json.dumps({"success": True, "path": "other.py", "truncated": True})}]
    assert pending_local_imports(messages, Workspace(tmp_path)) == []


def test_relative_imports_partial_batches_and_limit(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    for name in ("a", "b", "c"):
        (package / f"{name}.py").write_text("value = 1\n")
    message = {"role": "tool", "tool_name": "read_files", "content": json.dumps({"success": False, "files": [
        {"success": True, "path": "package/entry.py", "content": "1: from . import a\n2: from . import b\n3: from . import c"},
        {"success": False, "path": "package/c.py"}]})}
    assert pending_local_imports([message], Workspace(tmp_path), limit=2) == ["package/a.py", "package/b.py"]


def test_arbitrary_source_names_work_without_pipeline_rules(tmp_path):
    (tmp_path / "garden").mkdir()
    (tmp_path / "garden/watering.py").write_text("def water():\n    return 1\n")
    message = read("schedule.py", "5: from garden.watering import water as apply_water")
    assert pending_local_imports([message], Workspace(tmp_path)) == ["garden/watering.py"]


def test_existing_task_path_bootstraps_read_without_counting_it_as_inspected(tmp_path):
    (tmp_path / "calc.py").write_text("value = 1\n")
    workspace = Workspace(tmp_path)
    assert pending_local_imports([], workspace, task="Inspect calc.py and missing.py") == ["calc.py"]
    assert pending_local_imports([read("calc.py", "1: value = 1")], workspace, task="Inspect calc.py") == []
