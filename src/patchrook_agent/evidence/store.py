"""Immutable exact observations, with revision-aware source freshness."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

from patchrook_agent.tools.base import EXCLUDED_DIRS


MAX_STATE_BYTES = 64 * 1024 * 1024


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def checked_path(root: Path, path: str) -> Path:
    if not isinstance(path, str) or not path or "\0" in path:
        raise ValueError("Evidence path must be a nonempty repository-relative path")
    relative = Path(path)
    if relative.is_absolute() or ".." in relative.parts or any(part in EXCLUDED_DIRS for part in relative.parts):
        raise ValueError("Evidence path must stay inside the eligible repository")
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("Symlink paths are not eligible repository evidence")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Evidence path resolves outside the repository")
    if any(part in EXCLUDED_DIRS for part in resolved.relative_to(root).parts):
        raise ValueError("Evidence path resolves into an excluded repository directory")
    return resolved


@dataclass(frozen=True)
class EvidenceCard:
    id: str
    kind: str
    text: str
    path: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    source_sha256: str | None = None
    revision: int = 0
    sequence: int = 0
    metadata_json: str = "{}"

    @property
    def metadata(self) -> dict:
        # Return a new object; callers cannot mutate the stored observation.
        return json.loads(self.metadata_json)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "EvidenceCard":
        if not isinstance(value, dict):
            raise ValueError("Invalid evidence card")
        try:
            card = cls(**value)
        except TypeError as error:
            raise ValueError("Invalid evidence card fields") from error
        if (not isinstance(card.id, str) or not isinstance(card.kind, str)
                or not isinstance(card.text, str) or type(card.revision) is not int or card.revision < 0
                or type(card.sequence) is not int or card.sequence < 1
                or not isinstance(card.metadata, dict)):
            raise ValueError("Invalid evidence card fields")
        if card.kind == "source" and (not isinstance(card.path, str)
                or type(card.start_line) is not int or type(card.end_line) is not int
                or not 1 <= card.start_line <= card.end_line
                or not isinstance(card.source_sha256, str) or len(card.source_sha256) != 64):
            raise ValueError("Invalid source evidence provenance")
        return card


class EvidenceStore:
    def __init__(self, root, state_path=None, *, max_file_bytes=1_048_576, max_cards=4096):
        self.root = Path(root).resolve(strict=True)
        if not self.root.is_dir() or type(max_file_bytes) is not int or max_file_bytes <= 0:
            raise ValueError("Evidence store requires a directory and a positive file limit")
        if type(max_cards) is not int or max_cards < 1:
            raise ValueError("Evidence card limit must be positive")
        self.state_path = Path(state_path) if state_path is not None else None
        self.max_file_bytes, self.max_cards = max_file_bytes, max_cards
        self._cards: dict[str, EvidenceCard] = {}
        self.policy_cache: dict[str, dict] = {}
        self.policy_state: dict = {}
        self.sequence = 0
        self.truncated = False
        if self.state_path is not None and self.state_path.exists():
            self.load()

    def _read(self, path, deadline=None):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Evidence read deadline exhausted")
        resolved = checked_path(self.root, path)
        with resolved.open("rb") as handle:
            raw = handle.read(self.max_file_bytes + 1)
        if len(raw) > self.max_file_bytes:
            raise ValueError("Source exceeds evidence file byte limit")
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Evidence read deadline exhausted")
        return raw, raw.decode("utf-8")

    def _add(self, kind, text, *, revision, path=None, start_line=None, end_line=None,
             source_sha256=None, metadata=None):
        if not isinstance(kind, str) or not kind or not isinstance(text, str):
            raise ValueError("Evidence kind and exact text are required")
        if type(revision) is not int or revision < 0:
            raise ValueError("Evidence revision must be nonnegative")
        if path is not None:
            checked_path(self.root, path)
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("Evidence metadata must be a dictionary")
        metadata_json = json.dumps(metadata or {}, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        if len(metadata_json.encode()) > 65536:
            raise ValueError("Evidence metadata exceeds byte limit")
        identity = json.dumps([kind, path, start_line, end_line, source_sha256, revision, text,
                               metadata_json], ensure_ascii=False, separators=(",", ":"))
        identifier = "ev_" + digest(identity.encode())[:24]
        if identifier in self._cards:
            return self._cards[identifier]
        if len(self._cards) >= self.max_cards:
            self.truncated = True
            raise ValueError("Evidence store card limit reached; observations were not silently evicted")
        self.sequence += 1
        card = EvidenceCard(identifier, kind, text, path, start_line, end_line, source_sha256,
                            revision, self.sequence, metadata_json)
        self._cards[identifier] = card
        self.save()
        return card

    def observe_source(self, path, start_line=1, end_line=None, *, revision=0, origin="read",
                       deadline=None, metadata=None):
        raw, text = self._read(path, deadline)
        lines = text.splitlines(keepends=True)
        if not lines:
            # Empty files have no source line that could authorize a replacement.
            return self.add_observation("empty_file", "", revision=revision, path=path,
                                        metadata={"source_sha256": digest(raw), "origin": origin})
        last = len(lines) if end_line is None else end_line
        if type(start_line) is not int or type(last) is not int or not 1 <= start_line <= last <= len(lines):
            raise ValueError("Evidence source range must exist exactly in the current file")
        return self._add("source", "".join(lines[start_line - 1:last]), revision=revision,
                         path=Path(path).as_posix(), start_line=start_line, end_line=last,
                         source_sha256=digest(raw), metadata={**(metadata or {}), "origin": origin})

    def add_observation(self, kind, text, *, revision=0, path=None, metadata=None):
        if kind == "source":
            raise ValueError("Source cards must be acquired through observe_source")
        return self._add(kind, text, revision=revision, path=path, metadata=metadata)

    def is_fresh(self, card, *, deadline=None):
        if card.kind not in {"source", "empty_file"}:
            return True
        try:
            raw, text = self._read(card.path, deadline)
            if card.kind == "empty_file":
                return not raw and digest(raw) == card.metadata.get("source_sha256")
            return (digest(raw) == card.source_sha256 and
                    "".join(text.splitlines(keepends=True)[card.start_line - 1:card.end_line]) == card.text)
        except (OSError, ValueError, UnicodeError, TimeoutError):
            return False

    def get(self, identifier, *, fresh=True, deadline=None):
        card = self._cards.get(identifier)
        return card if card is not None and (not fresh or self.is_fresh(card, deadline=deadline)) else None

    def cards(self, *, fresh=True, deadline=None):
        return [card for card in self._cards.values()
                if not fresh or self.is_fresh(card, deadline=deadline)]

    def save(self):
        if self.state_path is None:
            return
        payload = json.dumps({"schema_version": 1, "root": str(self.root), "sequence": self.sequence,
                              "cards": [card.to_dict() for card in self._cards.values()],
                              "policy_cache": self.policy_cache, "policy_state": self.policy_state,
                              "truncated": self.truncated}, ensure_ascii=False, separators=(",", ":")).encode()
        if len(payload) > MAX_STATE_BYTES:
            raise ValueError("Evidence state exceeds persistence byte limit")
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        name = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.state_path.parent, prefix=".evidence-", delete=False) as handle:
                name = handle.name
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(name, self.state_path)
        finally:
            if name is not None and os.path.exists(name):
                os.unlink(name)

    def load(self):
        with self.state_path.open("rb") as handle:
            raw = handle.read(MAX_STATE_BYTES + 1)
        if len(raw) > MAX_STATE_BYTES:
            raise ValueError("Evidence state exceeds persistence byte limit")
        state = json.loads(raw)
        if state.get("schema_version") != 1 or state.get("root") != str(self.root):
            raise ValueError("Evidence state belongs to a different schema or repository")
        values = state.get("cards")
        if not isinstance(values, list) or len(values) > self.max_cards:
            raise ValueError("Invalid persisted evidence cardinality")
        cards = [EvidenceCard.from_dict(value) for value in values]
        for card in cards:
            if card.path is not None:
                checked_path(self.root, card.path)
        if len({card.id for card in cards}) != len(cards):
            raise ValueError("Duplicate persisted evidence identities")
        self._cards = {card.id: card for card in cards}
        self.sequence = max((card.sequence for card in cards), default=0)
        cache, policy = state.get("policy_cache", {}), state.get("policy_state", {})
        if not isinstance(cache, dict) or len(cache) > 128 or not isinstance(policy, dict):
            raise ValueError("Invalid persisted evidence policy")
        self.policy_cache, self.policy_state = cache, policy
        self.truncated = bool(state.get("truncated", False))
