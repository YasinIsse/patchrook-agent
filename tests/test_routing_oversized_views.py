"""A large observation must not repeatedly evict and reacquire its own facts."""

import json

import pytest

from patchrook_agent.config import HarnessConfig
from patchrook_agent.model_routing import RoutingProgress


def observation(kind, count=200, *, path="module.py", digest="a" * 64):
    if kind == "read_file":
        return {"success": True, "path": path, "source_sha256": digest, "start_line": 1,
                "content": "\n".join(f"{number}: line_{number}" for number in range(1, count + 1))}
    if kind == "search_code":
        return {"success": True, "matches": [{"path": path, "line_number": number,
                    "text": f"line_{number}"} for number in range(1, count + 1)]}
    return {"success": True, "path": ".", "entries": [f"src/file_{number}.py" for number in range(count)]}


@pytest.mark.parametrize("kind", ["read_file", "search_code", "list_files"])
@pytest.mark.parametrize("count", [129, 200])
def test_repeated_large_fact_set_reaches_stall_threshold(kind, count):
    progress = RoutingProgress()
    config = HarnessConfig()
    result = observation(kind, count)
    reasons = []
    for number in range(4):
        reasons.append(progress.observe(kind, {"path": ".", "question": f"wording_{number}",
                                               "max_entries": 200 + number}, result, 0, config))
    assert reasons == [None, None, None, "repeated_actions"]
    assert progress.repeated_actions == 3
    assert len(progress.observations) == 1


def test_alternating_large_views_do_not_flush_each_other():
    progress = RoutingProgress()
    config = HarnessConfig()
    first = observation("read_file", path="first.py")
    second = observation("read_file", path="second.py")
    assert progress.observe("read_file", {}, first, 0, config) is None
    assert progress.observe("read_file", {}, second, 0, config) is None
    assert progress.observe("read_file", {}, first, 0, config) is None
    assert progress.observe("read_file", {}, second, 0, config) is None
    assert progress.observe("read_file", {}, first, 0, config) == "repeated_actions"
    assert len(progress.observations) == 2


def test_order_and_paging_wording_do_not_change_an_oversized_observed_fact_set():
    progress = RoutingProgress()
    config = HarnessConfig()
    result = observation("search_code")
    progress.observe("search_code", {"pattern": "line", "offset": 0}, result, 0, config)
    result["matches"].reverse()
    progress.observe("search_code", {"pattern": "line_", "max_results": 500, "question": "new words"}, result, 0, config)
    assert progress.repeated_actions == 1


def test_changed_source_content_and_new_paths_reset_stall_counter():
    progress = RoutingProgress()
    config = HarnessConfig()
    old = observation("read_file")
    progress.observe("read_file", {}, old, 0, config)
    progress.observe("read_file", {}, old, 0, config)
    assert progress.repeated_actions == 1
    changed = observation("read_file", digest="b" * 64)
    assert progress.observe("read_file", {}, changed, 1, config) is None
    assert progress.repeated_actions == 0
    assert progress.observe("read_file", {}, observation("read_file", path="other.py"), 1, config) is None
    assert progress.repeated_actions == 0


def test_small_overlapping_reads_keep_existing_per_line_semantics():
    progress = RoutingProgress()
    config = HarnessConfig()
    def read(first, last):
        result = {"success": True, "path": "module.py", "source_sha256": "a" * 64,
                  "start_line": first,
                  "content": "\n".join(f"{n}: line_{n}" for n in range(first, last + 1))}
        return progress.observe("read_file", {}, result, 0, config)
    read(1, 6)
    read(2, 5)
    assert progress.repeated_actions == 1
    read(5, 7)
    assert progress.repeated_actions == 0
    read(1, 7)
    assert progress.repeated_actions == 1


def test_existing_checkpoint_shape_round_trips_large_view_streak():
    progress = RoutingProgress()
    config = HarnessConfig()
    result = observation("read_file")
    progress.observe("read_file", {}, result, 0, config)
    progress.observe("read_file", {}, result, 0, config)
    state = json.loads(json.dumps(progress.to_dict()))
    assert set(state) == {"invalid_responses", "repeated_actions", "observations", "failed_repairs"}
    restored = RoutingProgress.from_dict(state)
    assert restored.observe("read_file", {}, result, 0, config) is None
    assert restored.observe("read_file", {}, result, 0, config) == "repeated_actions"


def test_legacy_opaque_hash_checkpoint_remains_valid():
    config = HarnessConfig()
    old_state = {"invalid_responses": 0, "repeated_actions": 0,
                 "observations": [f"{number:064x}" for number in range(128)], "failed_repairs": []}
    restored = RoutingProgress.from_dict(old_state)
    result = observation("read_file")
    for _ in range(4):
        reason = restored.observe("read_file", {}, result, 0, config)
    assert reason == "repeated_actions"
    assert len(restored.observations) <= 128
