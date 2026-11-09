"""Private, atomic session checkpoints and bounded workspace fingerprints."""

import hashlib
import fcntl
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any
from uuid import uuid4

from patchrook_agent.tools.base import ToolError, Workspace

from .progress import render_progress
from .state import WorkingMemory


SCHEMA_VERSION = 1
_SESSION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}\Z")


def atomic_write(path: Path, content: str) -> None:
    """Replace one file without exposing a partially serialized checkpoint."""
    path = Path(path)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".checkpoint-", delete=False) as stream:
            temporary = stream.name
            os.chmod(temporary, 0o600)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


class SessionStore:
    def __init__(self, runs_dir: str | Path, session_id: str | None = None):
        self.runs_dir = Path(runs_dir).expanduser().resolve()
        self.runs_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        if session_id is not None and not _SESSION_ID.fullmatch(session_id):
            raise ValueError("Invalid session ID")
        self.session_id = session_id or uuid4().hex
        self.directory = self.runs_dir / self.session_id
        if self.directory.is_symlink():
            raise ValueError("Session directories must not be symlinks")
        if session_id is None:
            self.directory.mkdir(mode=0o700)
        elif not self.directory.is_dir():
            raise ValueError("Session directory does not exist")
        if not self.directory.resolve().is_relative_to(self.runs_dir):
            raise ValueError("Session directory escapes the configured runs directory")
        self._lock_descriptor: int | None = None

    def acquire(self) -> None:
        """Claim this session for one runtime; inspections remain unlocked."""
        if self._lock_descriptor is not None:
            raise ValueError("Session is already running")
        descriptor = os.open(self.directory / ".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError("Session lock must be a regular file")
            os.fchmod(descriptor, 0o600)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError("Session is already running") from exc
            self._lock_descriptor = descriptor
        except BaseException:
            os.close(descriptor)
            raise

    def release(self) -> None:
        """Release ownership without deleting the shared lock-file inode."""
        descriptor = self._lock_descriptor
        if descriptor is None:
            return
        self._lock_descriptor = None
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def write_checkpoint(self, payload: dict[str, Any], memory: WorkingMemory) -> None:
        if memory.session_id != self.session_id:
            raise ValueError("Working memory belongs to a different session")
        checkpoint = {**payload, "schema_version": SCHEMA_VERSION,
                      "session_id": self.session_id, "memory": memory.to_dict()}
        # The checkpoint is committed last. It remains the authoritative snapshot
        # if the process stops between updates to the readable projections.
        atomic_write_json(self.directory / "state.json", memory.to_dict())
        atomic_write(self.directory / "TASKS.md", render_progress(memory))
        atomic_write_json(self.directory / "checkpoint.json", checkpoint)

    def load_checkpoint(self) -> dict[str, Any]:
        path = self.directory / "checkpoint.json"
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 67_108_864:
            raise ValueError("Checkpoint must be a regular file under 64 MiB")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            raise ValueError("Checkpoint is not valid UTF-8 JSON") from exc
        if not isinstance(payload, dict) or type(payload.get("schema_version")) is not int or payload["schema_version"] != SCHEMA_VERSION:
            raise ValueError("Unsupported checkpoint schema version")
        if payload.get("session_id") != self.session_id:
            raise ValueError("Checkpoint belongs to a different session")
        memory = WorkingMemory.from_dict(payload.get("memory"))
        if memory.session_id != self.session_id:
            raise ValueError("Checkpoint memory belongs to a different session")
        return payload


def snapshot_files(workspace: Workspace, paths: list[str]) -> dict[str, dict[str, Any]]:
    snapshots: dict[str, dict[str, Any]] = {}
    for name in dict.fromkeys(paths):
        try:
            path = workspace.resolve(name)
            if not path.exists():
                snapshots[name] = {"kind": "missing"}
                continue
            if not path.is_file():
                snapshots[name] = {"kind": "not_file"}
                continue
            stat = path.stat()
            metadata = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                        "resolved_path": workspace.relative(path)}
            if stat.st_size > workspace.config.max_file_bytes:
                snapshots[name] = {"kind": "unverified_large", **metadata}
                continue
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                remaining = workspace.config.max_file_bytes + 1
                while remaining:
                    block = stream.read(min(65_536, remaining))
                    if not block:
                        break
                    digest.update(block)
                    remaining -= len(block)
            if remaining == 0:
                snapshots[name] = {"kind": "unverified_large", **metadata}
            else:
                snapshots[name] = {"kind": "file", "sha256": digest.hexdigest(), **metadata}
        except (OSError, ToolError, ValueError) as exc:
            snapshots[name] = {"kind": "unavailable", "reason": getattr(exc, "code", type(exc).__name__)}
    return snapshots


def compare_snapshots(workspace: Workspace, snapshot: dict[str, dict[str, Any]]) -> list[str]:
    if not isinstance(snapshot, dict):
        raise ValueError("File snapshots must be an object")
    current = snapshot_files(workspace, list(snapshot))
    return [path for path, previous in snapshot.items()
            if not isinstance(previous, dict) or previous != current[path]
            or current[path]["kind"] in {"unverified_large", "unavailable"}]
