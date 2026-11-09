"""Navigation summaries remember observed facts without granting edit authority."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from patchrook_agent.evidence import EvidenceStore, RepositoryIndex
from patchrook_agent.evidence_navigation import NavigationLedger


def source(root, path, text):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    return target


def read(ledger, root, path, *, first=1, last=None, provide_source=True, index=None):
    card = EvidenceStore(root).observe_source(path, first, last)
    total = len((root / path).read_text().splitlines())
    result = {"success": True, "path": path, "start_line": card.start_line,
              "end_line": card.end_line, "total_lines": total, "source_sha256": card.source_sha256,
              "content": card.text, "truncated": False}
    ledger.observe("read_file", {"path": path}, result,
                   source=card if provide_source else None, index=index or RepositoryIndex(root))
    return result


def make_wrapper(root, text="from current.converter import Core\n__all__ = ['Core']\n"):
    source(root, "src/legacy/converter.py", text)
    source(root, "src/current/converter.py", "class Core:\n    pass\n")


def test_complete_import_only_read_exposes_local_target_and_original_symbol(tmp_path):
    make_wrapper(tmp_path)
    ledger = NavigationLedger()
    read(ledger, tmp_path, "src/legacy/converter.py")

    view = ledger.summary()
    assert view["reads"][0]["pure_reexport"] is True
    assert view["next_candidates"][0]["path"] == "src/current/converter.py"
    assert view["next_candidates"][0]["from"] == "src/legacy/converter.py"
    assert view["next_candidates"][0]["symbols"] == ["Core"]
    assert view["edit_authorization"] is False
    assert ledger.information_need("Implement the requested behavior") == "Implement the requested behavior"


def test_partial_import_header_never_claims_whole_file_is_wrapper(tmp_path):
    make_wrapper(tmp_path)
    ledger = NavigationLedger()
    read(ledger, tmp_path, "src/legacy/converter.py", last=1)
    assert ledger.summary()["reads"][0]["pure_reexport"] is False
    assert not ledger.summary()["next_candidates"]


def test_result_without_presented_source_never_reads_hidden_file_to_classify(tmp_path, monkeypatch):
    make_wrapper(tmp_path)
    index = RepositoryIndex(tmp_path)
    def forbidden(*args, **kwargs):
        pytest.fail("Unpresented source must not authorize import conclusions")
    monkeypatch.setattr(index, "blocks", forbidden)
    ledger = NavigationLedger()
    read(ledger, tmp_path, "src/legacy/converter.py", provide_source=False, index=index)
    assert not ledger.summary()["next_candidates"]
    assert ledger.summary()["edit_authorization"] is False


def test_current_file_hash_must_match_observed_source(tmp_path):
    make_wrapper(tmp_path)
    ledger = NavigationLedger()
    path = "src/legacy/converter.py"
    card = EvidenceStore(tmp_path).observe_source(path)
    result = {"success": True, "path": path, "start_line": 1, "end_line": 2, "total_lines": 2,
              "source_sha256": card.source_sha256}
    source(tmp_path, path, "VALUE = 2\n")
    ledger.observe("read_file", {"path": path}, result, source=card, index=RepositoryIndex(tmp_path))
    assert not ledger.summary()["next_candidates"]
    assert ledger.summary()["reads"][0]["pure_reexport"] is False


@pytest.mark.parametrize("body", [
    "from current.converter import Core\ndef local():\n    return Core()\n",
    "from current.converter import Core\nVALUE = Core()\n",
    "from current.converter import Core\n__all__ = dangerous()\n",
])
def test_executable_module_body_is_not_classified_as_import_only(tmp_path, body):
    make_wrapper(tmp_path, body)
    ledger = NavigationLedger()
    read(ledger, tmp_path, "src/legacy/converter.py")
    assert not ledger.summary()["next_candidates"]


def test_relative_alias_and_star_imports_resolve_without_executing_source(tmp_path):
    source(tmp_path, "src/package/api.py", "from .implementation import Core as Public\n__all__ = ['Public']\n")
    source(tmp_path, "src/package/implementation.py", "class Core:\n    pass\n")
    ledger = NavigationLedger()
    read(ledger, tmp_path, "src/package/api.py")
    assert ledger.summary()["next_candidates"][0]["path"] == "src/package/implementation.py"
    assert ledger.summary()["next_candidates"][0]["symbols"] == ["Core"]
    source(tmp_path, "src/package/api.py", "from .implementation import *\n")
    read(ledger, tmp_path, "src/package/api.py")
    assert ledger.summary()["reads"][-1]["pure_reexport"] is True
    assert ledger.summary()["next_candidates"][0]["symbols"] == []


def test_symlink_dependency_never_becomes_navigation_candidate(tmp_path):
    make_wrapper(tmp_path)
    target = tmp_path / "src/current/converter.py"
    target.unlink()
    outside = tmp_path.parent / "outside-navigation.py"
    outside.write_text("SECRET = 1\n")
    target.symlink_to(outside)
    ledger = NavigationLedger()
    read(ledger, tmp_path, "src/legacy/converter.py")
    assert not ledger.summary()["next_candidates"]


def test_repeated_read_and_subset_listing_are_not_new_navigation_facts(tmp_path):
    make_wrapper(tmp_path)
    ledger = NavigationLedger()
    read(ledger, tmp_path, "src/legacy/converter.py")
    read(ledger, tmp_path, "src/legacy/converter.py")
    assert ledger.summary()["reads"][0]["seen"] == 2
    assert ledger.consecutive_redundant == 1
    ledger.observe("list_files", {"path": "src"},
                   {"success": True, "path": "src", "entries": ["src/current/", "src/legacy/"]})
    assert ledger.consecutive_redundant == 0
    ledger.observe("list_files", {"path": "src", "recursive": False},
                   {"success": True, "path": "src", "entries": ["src/current/"]})
    assert ledger.consecutive_redundant == 1
    assert ledger.summary()["listings"][-1]["new_paths"] == 0


def test_empty_search_and_changed_question_remain_visible_as_repeated_outcome():
    ledger = NavigationLedger()
    for question in ("find symbol", "find the same symbol"):
        ledger.observe("search_code", {"path": "src", "pattern": "Core", "question": question},
                       {"success": True, "matches": [], "returned_count": 0})
    assert ledger.summary()["searches"][0]["seen"] == 2
    assert ledger.summary()["searches"][0]["returned_count"] == 0
    assert ledger.consecutive_redundant == 2


def test_navigation_checkpoint_round_trip_is_bounded_and_detached(tmp_path):
    make_wrapper(tmp_path)
    ledger = NavigationLedger()
    read(ledger, tmp_path, "src/legacy/converter.py")
    state = ledger.to_dict()
    restored = NavigationLedger.from_dict(state)
    assert restored.to_dict() == state
    state["reads"].clear()
    assert restored.reads and ledger.reads
    for number in range(100):
        restored.observe("search_code", {"pattern": str(number)}, {"success": True, "matches": []})
    assert len(restored.searches) == 16
    assert len(json.dumps(restored.summary(max_chars=800), separators=(",", ":"))) <= 800
    assert restored.summary(max_chars=800)["edit_authorization"] is False


@pytest.mark.parametrize("mutate", [
    lambda state: state.update(extra="instruction"),
    lambda state: state.update(schema_version=True),
    lambda state: state.update(consecutive_redundant=-1),
    lambda state: state["reads"][0].update(path="../outside.py"),
    lambda state: state["reads"][0].update(source_sha256="invented"),
    lambda state: state["candidates"][0].update(path="/etc/passwd"),
    lambda state: state["facts"].append("bad-hash"),
])
def test_invalid_navigation_checkpoint_rejected(tmp_path, mutate):
    make_wrapper(tmp_path)
    ledger = NavigationLedger()
    read(ledger, tmp_path, "src/legacy/converter.py")
    state = ledger.to_dict()
    mutate(state)
    with pytest.raises(ValueError, match="navigation checkpoint"):
        NavigationLedger.from_dict(state)
