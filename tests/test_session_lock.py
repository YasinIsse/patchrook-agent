"""A session has one runtime writer, while readers can inspect checkpoints."""

import selectors
import subprocess
import sys

import pytest

from patchrook_agent.memory import SessionStore, WorkingMemory


def test_only_one_store_instance_can_claim_a_session(tmp_path):
    first = SessionStore(tmp_path)
    second = SessionStore(tmp_path, first.session_id)
    first.acquire()
    try:
        with pytest.raises(ValueError, match="Session is already running"):
            second.acquire()
        with pytest.raises(ValueError, match="Session is already running"):
            first.acquire()
        assert (first.directory / ".lock").stat().st_mode & 0o777 == 0o600
    finally:
        first.release()
    first.release()  # Repeated cleanup is safe.
    second.acquire()
    second.release()


def test_loading_and_inspecting_checkpoint_does_not_require_runtime_lock(tmp_path):
    owner = SessionStore(tmp_path)
    memory = WorkingMemory(session_id=owner.session_id, objective="Inspect safely")
    owner.write_checkpoint({"state": {"steps": 1}}, memory)
    owner.acquire()
    try:
        reader = SessionStore(tmp_path, owner.session_id)
        assert reader.load_checkpoint()["state"]["steps"] == 1
    finally:
        owner.release()


def test_runtime_lock_refuses_symlink_target(tmp_path):
    store = SessionStore(tmp_path)
    private = tmp_path / "private"
    private.write_text("unchanged")
    (store.directory / ".lock").symlink_to(private)
    with pytest.raises((OSError, ValueError)):
        store.acquire()
    store.release()
    assert private.read_text() == "unchanged"


def test_os_releases_session_lock_after_runtime_process_is_killed(tmp_path):
    store = SessionStore(tmp_path)
    script = (
        "import sys, time\n"
        "from patchrook_agent.memory import SessionStore\n"
        "store = SessionStore(sys.argv[1], sys.argv[2])\n"
        "store.acquire()\n"
        "print('locked', flush=True)\n"
        "time.sleep(60)\n"
    )
    child = subprocess.Popen(
        [sys.executable, "-u", "-c", script, str(tmp_path), store.session_id],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(child.stdout, selectors.EVENT_READ)
            assert selector.select(timeout=5), "Child did not claim the session lock"
        assert child.stdout.readline().strip() == "locked"
        with pytest.raises(ValueError, match="Session is already running"):
            store.acquire()
        child.kill()
        child.wait(timeout=5)
        store.acquire()
        store.release()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)
        child.stdout.close()
        child.stderr.close()
        store.release()
