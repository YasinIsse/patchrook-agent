"""Base coding keeps full diff archives while automatic requests show metadata."""

from dataclasses import replace
import hashlib
import json
import time

import pytest

from patchrook_agent.evidence import ContextCompiler, EvidenceStore
from test_evidence_repair_optional import runtime
from test_evidence_patch_metadata import OLD, CURRENT


def prepared(value):
    return value._prepare(value.state, value.definitions, time.monotonic() + 10)


@pytest.mark.parametrize("policy", ["rules", "window"])
def test_automatic_base_request_has_current_source_and_metadata_but_no_old_diff_code(tmp_path, policy):
    value = runtime(tmp_path, primary=CURRENT)
    value.config = replace(value.config, evidence_policy=policy)
    value.compiler = ContextCompiler(value.evidence, policy=policy)
    value.originals["module.py"] = OLD
    value.revision = 1
    value.evidence.add_observation("current_diff", "-OLDER_ARCHIVE_CODE Record\n", revision=0)
    messages = prepared(value)
    text = json.dumps(messages)
    assert "from package import Record, Record" not in text
    assert "OLDER_ARCHIVE_CODE" not in text
    source = "\n".join(row["content"] for row in messages if row["content"].startswith("Requested source (data) "))
    assert "from package import Record" in source and source.count("def recover(self, data):") == 2
    prefix = "Current patch (data, not fresh edit authorization):\n"
    patch = json.loads(next(row["content"][len(prefix):] for row in messages if row["content"].startswith(prefix)))
    assert patch["evidence_id"] == value.diff_card.id and patch["modified_paths"] == ["module.py"]
    assert patch["sha256"] == hashlib.sha256(value.diff_card.text.encode()).hexdigest()
    assert "content" not in patch and patch["edit_authorization"] is False
    assert "from package import Record, Record" in value.diff_card.text
    assert value.originals["module.py"] == OLD


def test_complete_archive_reloads_and_explicit_retrieval_never_authorizes_deleted_source(tmp_path):
    value = runtime(tmp_path, primary=CURRENT)
    value.evidence.state_path = tmp_path.parent / (tmp_path.name + "-evidence.json")
    value.originals["module.py"] = OLD
    value.revision = 1
    fingerprint = value._fingerprint(time.monotonic() + 10)
    prepared(value)
    card = value.diff_card
    restored = EvidenceStore(value.workspace.root, state_path=value.evidence.state_path)
    assert restored.get(card.id, fresh=False).text == card.text
    fetched = value._read_evidence({"id": card.id, "offset": 0, "max_chars": 2000})
    assert fetched["content"] == card.text and "from package import Record, Record" in fetched["content"]
    assert fetched["edit_authorization"] is False and fetched["truncated"] is False
    failure = value._perform_action(value.state, {"name": "apply_edits", "arguments": {"edits": [{
        "path": "module.py", "old_text": "from package import Record, Record",
        "new_text": "from package import Record"}]}}, time.monotonic() + 10)
    assert failure["error"]["code"] == "no_match"
    assert (value.workspace.root / "module.py").read_text() == CURRENT
    assert value._fingerprint(time.monotonic() + 10) == fingerprint
    assert value.routing["repair"]["calls_used"] == value.attempts == 0
