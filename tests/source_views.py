"""Inspect source actually presented to a model without reading its workspace."""

import hashlib
import json
import re

from patchrook_agent.context.lean import WORKING_SET_PREFIX
from patchrook_agent.inspection import visible_lines

_SOURCE_HEADER = re.compile(r"^Observed source (.+) lines ([0-9]+)-([0-9]+):$")
_NUMBERED_LINE = re.compile(r"^[0-9]+: ")
_READERS = {"read_file", "read_files", "read_code"}


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def _ranges(numbers):
    result = []
    for number in sorted(numbers):
        if result and result[-1][1] + 1 == number:
            result[-1][1] = number
        else:
            result.append([number, number])
    return result


def _working_records(content):
    """Read only the harness's exact-source sections, not navigation prose."""
    current = None
    numbered = []
    for line in content.splitlines():
        match = _SOURCE_HEADER.fullmatch(line)
        if match:
            if current is not None:
                yield {**current, "content": "\n".join(numbered)}
            current = {"success": True, "path": match[1],
                       "start_line": int(match[2]), "end_line": int(match[3])}
            numbered = []
        elif current is not None and _NUMBERED_LINE.match(line):
            numbered.append(line)
        elif current is not None:
            yield {**current, "content": "\n".join(numbered)}
            current, numbered = None, []
    if current is not None:
        yield {**current, "content": "\n".join(numbered)}


def request_evidence(messages, candidate_paths=(), *, expected_sources=None):
    """Describe exact numbered source actually supplied in a prepared request.

    This audit does not read the workspace or infer visibility from filenames,
    candidate descriptions, assistant claims, or missing-source metadata. The
    expected-source option is for fixture validation; it never affects prompts.
    """
    candidate_paths = tuple(dict.fromkeys(candidate_paths))
    order, sources, conflicts = [], {}, {}

    def observe(record):
        path = record.get("path")
        lines = visible_lines(record)
        if not isinstance(path, str) or not lines:
            return
        if path not in sources:
            sources[path], conflicts[path] = {}, set()
            order.append(path)
        for number, text in lines.items():
            if number in sources[path] and sources[path][number] != text:
                conflicts[path].add(number)
            sources[path][number] = text

    for message in messages:
        content = message.get("content", "")
        if not isinstance(content, str):
            continue
        if message.get("role") == "user" and content.startswith(WORKING_SET_PREFIX):
            for record in _working_records(content):
                observe(record)
        elif message.get("role") == "tool" and message.get("tool_name") in _READERS:
            try:
                result = json.loads(content)
            except (ValueError, TypeError):
                continue
            if not isinstance(result, dict):
                continue
            records = ([result] if message["tool_name"] == "read_file" else result.get("files", []))
            if isinstance(records, list):
                for record in records:
                    if isinstance(record, dict):
                        observe(record)
    views = {}
    for path in order:
        lines = sources[path]
        expected = expected_sources.get(path) if expected_sources is not None else None
        full = expected.splitlines() if isinstance(expected, str) else None
        complete = (not conflicts[path] and lines == dict(enumerate(full, 1))) if full is not None else None
        views[path] = {"visible_ranges": _ranges(lines), "visible_line_count": len(lines),
                       "content_sha256": _digest(sorted(lines.items())),
                       "complete": complete, "conflicting_lines": sorted(conflicts[path])}
    return {"presented_paths": order,
            "candidate_order": [path for path in order if path in candidate_paths],
            "omitted_candidates": [path for path in candidate_paths if path not in sources],
            "source_views": views, "source_order_digest": _digest(
                [[path, views[path]["content_sha256"]] for path in order])}
