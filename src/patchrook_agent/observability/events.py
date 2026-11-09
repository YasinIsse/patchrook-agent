"""Incremental JSONL events with configurable content logging and redaction."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any


_REDACTED = "[REDACTED]"
_SENSITIVE_KEYS = {"api_key", "apikey", "authorization", "password", "passwd", "secret",
                   "credential", "credentials", "access_token", "refresh_token", "token", "private_key"}
_NO_CONTENT = {"messages", "content", "thinking", "stdout", "stderr", "old_text", "new_text",
               "objective", "task", "arguments", "body", "source", "resulting_source", "available_output"}
_PRIVATE_KEY = re.compile(r"-----BEGIN [^-]*PRIVATE KEY-----.*?-----END [^-]*PRIVATE KEY-----", re.DOTALL)
_BEARER = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]+", re.IGNORECASE)
_KNOWN_KEY = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{12,}|gh[pousr]_[A-Za-z0-9_]{12,}|github_pat_[A-Za-z0-9_]{12,}|AKIA[A-Z0-9]{16})\b")
_ASSIGNMENT = re.compile(
    r"(?i)(\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|password|passwd|secret|credential|token)\b\s*[:=]\s*)"
    r"(?:\"[^\"\r\n]*\"|'[^'\r\n]*'|[^\s,;}]+)"
)
_URL_CREDENTIAL = re.compile(r"(?<=://)[^/\s:@]+:[^@\s/]+@")


def redact_data(value: Any) -> Any:
    """Best-effort secret removal without modifying the live execution state."""
    if isinstance(value, dict):
        return {key: (_REDACTED if str(key).lower().replace("-", "_") in _SENSITIVE_KEYS
                      else redact_data(item)) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_data(item) for item in value]
    if isinstance(value, str):
        # Tool observations are often JSON encoded inside message content.
        if value.lstrip().startswith(("{", "[")):
            try:
                decoded = json.loads(value)
            except ValueError:
                pass
            else:
                if isinstance(decoded, (dict, list)):
                    return json.dumps(redact_data(decoded), ensure_ascii=False, separators=(",", ":"))
        text = _PRIVATE_KEY.sub(_REDACTED, value)
        text = _BEARER.sub("Bearer " + _REDACTED, text)
        text = _KNOWN_KEY.sub(_REDACTED, text)
        text = _ASSIGNMENT.sub(lambda match: match[1] + _REDACTED, text)
        return _URL_CREDENTIAL.sub(_REDACTED + "@", text)
    return value


def _without_content(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: ("[content logging disabled]" if key in _NO_CONTENT else _without_content(item))
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_without_content(item) for item in value]
    return value


class EventLogger:
    def __init__(self, directory: str | Path, session_id: str, enabled: bool = True,
                 redact: bool = True, log_content: bool = True):
        self.directory = Path(directory).resolve(strict=True)
        self.path = self.directory / "events.jsonl"
        self.session_id = session_id
        self.enabled = enabled
        self.redact = redact
        self.log_content = log_content

    def emit(self, event: str, step: int = 0, **data: Any) -> None:
        if not self.enabled:
            return
        if not isinstance(event, str) or not event or type(step) is not int or step < 0:
            raise ValueError("Events require a name and nonnegative integer step")
        record = {**data, "schema_version": 1, "session_id": self.session_id,
                  "timestamp_utc": datetime.now(timezone.utc).isoformat(), "event": event, "step": step}
        if not self.log_content:
            record = _without_content(record)
        if self.redact:
            record = redact_data(record)
        encoded = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        descriptor = os.open(self.path, os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            os.fchmod(descriptor, 0o600)
            size = os.fstat(descriptor).st_size
            if size and os.pread(descriptor, 1, size - 1) != b"\n":
                # A killed writer may leave a partial record. Preserve those
                # bytes, but keep the next event on its own parseable line.
                encoded = b"\n" + encoded
            written = 0
            while written < len(encoded):
                written += os.write(descriptor, encoded[written:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
