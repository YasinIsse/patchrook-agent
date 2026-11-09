"""Bounded, session-scoped expansion of omitted tool evidence."""

import json
import os
from pathlib import Path
import re
import stat
from uuid import uuid4

from patchrook_agent.memory.checkpoint import atomic_write_json
from patchrook_agent.observability.events import redact_data
from patchrook_agent.tools.base import ToolError


_ARTIFACT_ID = re.compile(r"tool-[0-9a-f]{24}\Z")
MAX_ARTIFACT_BYTES = 16 * 1024 * 1024


class ArtifactStore:
    def __init__(self, directory: Path, *, max_output_chars: int = 2000, redact: bool = True):
        self.directory = Path(directory).resolve(strict=True)
        self.artifacts = self.directory / "artifacts"
        self.max_output_chars = max_output_chars
        self.redact = redact
        if self.artifacts.is_symlink():
            raise ToolError("invalid_artifact", "Artifact directory must not be a symlink")
        self.artifacts.mkdir(mode=0o700, exist_ok=True)

    def save(self, payload: dict) -> str:
        if (self.directory.resolve() != self.directory or self.artifacts.is_symlink()
                or self.artifacts.resolve().parent != self.directory):
            raise ToolError("invalid_artifact", "Artifact directory no longer belongs to this session")
        artifact_id = "tool-" + uuid4().hex[:24]
        data = redact_data(payload) if self.redact else payload
        encoded = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        if len(encoded) + 1 > MAX_ARTIFACT_BYTES:
            raise ToolError("artifact_limit", "Result exceeds the 16 MiB artifact limit; narrow the original tool request")
        atomic_write_json(self.artifacts / f"{artifact_id}.json", data)
        return artifact_id

    def available_ids(self, candidates: list[str]) -> tuple[str, ...]:
        """Return known IDs that still refer to readable session artifacts."""
        if (self.directory.resolve() != self.directory or self.artifacts.is_symlink()
                or self.artifacts.resolve().parent != self.directory):
            return ()
        available = []
        for artifact_id in dict.fromkeys(candidates):
            if not isinstance(artifact_id, str) or not _ARTIFACT_ID.fullmatch(artifact_id):
                continue
            path = self.artifacts / f"{artifact_id}.json"
            if path.is_symlink() or path.resolve().parent != self.artifacts:
                continue
            try:
                descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            except OSError:
                continue
            try:
                metadata = os.fstat(descriptor)
                if stat.S_ISREG(metadata.st_mode) and metadata.st_size <= MAX_ARTIFACT_BYTES:
                    available.append(artifact_id)
            finally:
                os.close(descriptor)
        return tuple(available)

    def read(self, artifact_id: str, offset: int = 0, max_chars: int = 1000) -> dict:
        if artifact_id == "memory":
            path = self.directory / "state.json"
        elif isinstance(artifact_id, str) and _ARTIFACT_ID.fullmatch(artifact_id):
            path = self.artifacts / f"{artifact_id}.json"
        else:
            raise ToolError("invalid_artifact", "Use an artifact_id returned by a tool or 'memory'")
        if type(offset) is not int or offset < 0 or type(max_chars) is not int or max_chars <= 0:
            raise ToolError("invalid_arguments", "Use a nonnegative character offset and positive max_chars")
        if path.is_symlink() or not path.resolve().is_relative_to(self.directory):
            raise ToolError("invalid_artifact", "Artifact path escapes its session")
        try:
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except FileNotFoundError as exc:
            raise ToolError("artifact_not_found", "Artifact is not available in this session") from exc
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_ARTIFACT_BYTES:
                raise ToolError("artifact_limit", "Artifact must be a regular file no larger than 16 MiB")
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                raw = stream.read(MAX_ARTIFACT_BYTES + 1)
            if len(raw) > MAX_ARTIFACT_BYTES:
                raise ToolError("artifact_limit", "Artifact exceeds its byte limit")
        finally:
            os.close(descriptor)
        try:
            text = raw.decode("utf-8")
        except UnicodeError as exc:
            raise ToolError("invalid_artifact", "Artifact is not valid UTF-8 text") from exc
        # Working memory is full-fidelity private recovery state. Apply the same
        # configured log redaction before exposing it as an expandable artifact.
        if self.redact and artifact_id == "memory":
            try:
                text = json.dumps(redact_data(json.loads(text)), ensure_ascii=False, indent=2)
            except ValueError as exc:
                raise ToolError("invalid_artifact", "Working memory is not valid JSON") from exc
        if offset > len(text):
            raise ToolError("invalid_range", "Artifact offset exceeds the available character count")
        content = text[offset:offset + min(max_chars, self.max_output_chars)]
        result = {"success": True, "artifact_id": artifact_id, "offset": offset,
                  "total_chars": len(text), "content": content,
                  "next_offset": None, "truncated": False}
        while True:
            result["next_offset"] = offset + len(result["content"]) if offset + len(result["content"]) < len(text) else None
            result["truncated"] = result["next_offset"] is not None
            excess = len(json.dumps(result, ensure_ascii=False, separators=(",", ":"))) - self.max_output_chars
            if excess <= 0:
                return result
            result["content"] = result["content"][:max(0, len(result["content"]) - max(1, excess))]
            if not result["content"] and offset < len(text):
                raise ToolError("output_limit", "Output limit cannot hold an artifact excerpt")
