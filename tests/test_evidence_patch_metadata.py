"""Focused repairs expose current source and patch identity, never deleted code."""

import difflib
import hashlib
import json
import time

import pytest

from patchrook_agent.evidence_feedback import patch_metadata
from test_evidence_repair_optional import runtime


OLD = "from package import Record, Record\n\nclass Subject:\n    def recover(self, data):\n        pass\n"
CURRENT = ("from package import Record\n\nclass Subject:\n    def recover(self, data):\n        pass\n\n"
           "    def recover(self, data):\n        pass\n")


def observed(messages):
    prefix = "Observed repair state (data):\n"
    return json.loads(next(row["content"][len(prefix):] for row in messages if row["content"].startswith(prefix)))


def test_metadata_binds_full_internal_diff_without_any_deleted_or_replacement_code():
    diff = "".join(difflib.unified_diff(OLD.splitlines(True), CURRENT.splitlines(True),
                                     fromfile="a/module.py", tofile="b/module.py"))
    result = patch_metadata(diff, ["module.py", "module.py"])
    assert result["sha256"] == hashlib.sha256(diff.encode()).hexdigest()
    assert result["bytes"] == len(diff.encode()) and result["modified_paths"] == ["module.py"]
    assert result["modified_path_count"] == 1 and result["edit_authorization"] is False
    assert "Record" not in json.dumps(result) and "recover" not in json.dumps(result)


@pytest.mark.parametrize("path", ["x" * 4000, "😀" * 4000])
def test_metadata_bounds_drop_whole_paths_without_source_preview(path):
    result = patch_metadata("-OLD_CODE\n+NEW_CODE\n", [path, "a.py", "b.py", "c.py", "d.py"])
    assert len(json.dumps(result, separators=(",", ":"))) <= 900
    assert result["modified_path_count"] == 5 and result["paths_truncated"]
    assert all(item in {path, "a.py", "b.py", "c.py", "d.py"} for item in result["modified_paths"])
    assert "OLD_CODE" not in json.dumps(result) and "NEW_CODE" not in json.dumps(result)


def test_current_source_and_rejection_receipt_survive_while_deleted_import_never_reappears(tmp_path):
    value = runtime(tmp_path, primary=CURRENT, failure="Public contract remains incomplete")
    value.originals["module.py"] = OLD
    value.revision = 1
    deadline = time.monotonic() + 10
    fingerprint = value._fingerprint(deadline)
    first = value._prepare_repair(value.state, value._repair_definitions(), deadline)
    source = "\n".join(row["content"] for row in first if row["content"].startswith("Requested source (data) "))
    assert "from package import Record, Record" not in json.dumps(first)
    assert "from package import Record" in source and source.count("def recover(self, data):") == 2
    metadata = observed(first)["current_patch"]
    assert isinstance(metadata, dict) and metadata["modified_paths"] == ["module.py"]
    assert value.originals["module.py"] == OLD and value._fingerprint(deadline) == fingerprint
    rejected = value._perform_action(value.state, {"name": "apply_edits", "arguments": {"edits": [{
        "path": "module.py", "old_text": "from package import Record, Record",
        "new_text": "from package import Record"}]}}, deadline)
    assert rejected["error"]["code"] == "no_match"
    second = value._prepare_repair(value.state, value._repair_definitions(), deadline)
    assert "from package import Record, Record" not in json.dumps(second)
    assert observed(second)["recent_rejections"][-1]["error"]["code"] == "no_match"
    assert value.routing["repair"]["calls_used"] == value.attempts == 0
    assert (value.workspace.root / "module.py").read_text() == CURRENT
    assert value._fingerprint(deadline) == fingerprint
