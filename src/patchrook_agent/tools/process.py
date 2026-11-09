"""Bounded subprocess capture for the two approved external tools."""

from dataclasses import dataclass
import os
from pathlib import Path
import selectors
import signal
import subprocess
import time

from .base import TOOL_CAPTURE, record_tool_output


@dataclass
class ProcessResult:
    stdout: str
    stderr: str
    exit_code: int
    duration_seconds: float
    timed_out: bool
    truncated: bool


class _Capture:
    """Keep the beginning and end of a stream without unbounded buffering."""

    def __init__(self, limit: int, preserve_tail: bool = True):
        self.limit = limit
        self.preserve_tail = preserve_tail
        self.head = bytearray()
        self.tail = bytearray()
        self.total = 0

    def append(self, chunk: bytes) -> None:
        self.total += len(chunk)
        if not self.preserve_tail:
            self.head.extend(chunk[:max(0, self.limit - len(self.head))])
            return
        head_limit = (self.limit + 1) // 2
        needed = head_limit - len(self.head)
        self.head.extend(chunk[:needed])
        self.tail.extend(chunk[max(needed, 0):])
        tail_limit = self.limit - head_limit
        if len(self.tail) > tail_limit:
            del self.tail[:len(self.tail) - tail_limit]

    def text(self) -> str:
        if not self.preserve_tail:
            return bytes(self.head).decode("utf-8", "replace")
        marker = b"\n... output truncated ...\n" if self.total > self.limit else b""
        return (bytes(self.head) + marker + bytes(self.tail)).decode("utf-8", "replace")


def run_bounded(
    command: list[str],
    *,
    cwd: Path,
    timeout_seconds: float,
    max_output_chars: int,
    env: dict[str, str] | None = None,
    stop_on_output_limit: bool = False,
    preserve_tail: bool = True,
) -> ProcessResult:
    """Drain both pipes concurrently, bounding memory and killing descendants."""
    started = time.monotonic()
    captures = {name: _Capture(max_output_chars, preserve_tail) for name in ("stdout", "stderr")}
    capture_policy = TOOL_CAPTURE.get()
    raw_captures = ({name: _Capture(capture_policy["max_bytes"], preserve_tail=False)
                     for name in captures} if capture_policy is not None else {})
    timed_out = False
    stopped_for_output = False
    killed_at = None
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )

    def kill_group() -> None:
        nonlocal killed_at
        if killed_at is None:
            killed_at = time.monotonic()
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    try:
        with selectors.DefaultSelector() as selector:
            for name in captures:
                pipe = getattr(process, name)
                os.set_blocking(pipe.fileno(), False)
                selector.register(pipe, selectors.EVENT_READ, name)
            while selector.get_map():
                # A descendant can detach from the group and retain a pipe.
                # Never wait indefinitely for that pipe after killing the job.
                if killed_at is not None and time.monotonic() - killed_at > 0.5:
                    break
                remaining = timeout_seconds - (time.monotonic() - started)
                if remaining <= 0 and not timed_out:
                    timed_out = True
                    kill_group()
                wait = 0.05 if killed_at is not None else max(0.0, min(0.05, remaining))
                for key, _ in selector.select(timeout=wait):
                    chunk = os.read(key.fileobj.fileno(), 8192)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                        continue
                    capture = captures[key.data]
                    capture.append(chunk)
                    if raw_captures:
                        raw_captures[key.data].append(chunk)
                    if stop_on_output_limit and capture.total > max_output_chars:
                        stopped_for_output = True
                        kill_group()
            try:
                process.wait(timeout=max(0.0, timeout_seconds - (time.monotonic() - started)))
            except subprocess.TimeoutExpired:
                timed_out = True
                kill_group()
                process.kill()
                process.wait()
    finally:
        kill_group()
        process.wait()
        for name in captures:
            getattr(process, name).close()
    if raw_captures:
        record_tool_output(**{name: capture.text() for name, capture in raw_captures.items()},
                           raw_output_bytes={name: capture.total for name, capture in raw_captures.items()},
                           raw_output_truncated=any(c.total > c.limit for c in raw_captures.values()))
    return ProcessResult(
        stdout=captures["stdout"].text(),
        stderr=captures["stderr"].text(),
        exit_code=process.returncode,
        duration_seconds=round(time.monotonic() - started, 4),
        timed_out=timed_out,
        truncated=stopped_for_output or any(c.total > c.limit for c in captures.values()),
    )
