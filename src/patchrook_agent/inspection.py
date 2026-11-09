"""Coverage of source actually shown to the model, without hidden file reads."""

import json
import posixpath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from patchrook_agent.tools.base import ToolError


def _arguments(value: Any) -> dict:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return {}
    return value if isinstance(value, dict) else {}


def canonical_path(value: Any, workspace=None) -> str | None:
    if not isinstance(value, str) or not value or len(value) > 4096 or "\x00" in value:
        return None
    if workspace is not None:
        try:
            return workspace.relative(workspace.resolve(value))
        except (ValueError, OSError, TypeError, ToolError):
            # Workspace tool errors must not turn denied paths into suggestions.
            return None
    return posixpath.normpath(value)


def _source_boundaries(value: dict | None, workspace=None) -> dict[str, int]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("Source boundaries must be an object")
    boundaries = {}
    for path, index in value.items():
        canonical = canonical_path(path, workspace)
        if canonical is None or type(index) is not int or index < 0:
            raise ValueError("Invalid source boundary")
        boundaries[canonical] = max(index, boundaries.get(canonical, 0))
    return boundaries


def source_read_boundaries(messages: list[dict], workspace, *, external: dict | None = None) -> dict[str, int]:
    """Invalidate old source after observed mutations, without reading the files."""
    boundaries = _source_boundaries(external, workspace)
    pending = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant" and isinstance(message.get("tool_calls"), list):
            pending = [call for call in message["tool_calls"] if isinstance(call, dict)]
        elif message.get("role") == "tool" and message.get("tool_name") in {"write_file", "edit_file"}:
            call = next((call for call in pending if call.get("id") == message.get("tool_call_id")
                         and message.get("tool_call_id") is not None), None)
            if call is None:
                call = next((call for call in pending if isinstance(call.get("function"), dict)
                             and call["function"].get("name") == message["tool_name"]), {})
            if call in pending:
                pending.remove(call)
            try:
                result = json.loads(message.get("content", ""))
            except (TypeError, ValueError):
                continue
            if not isinstance(result, dict) or result.get("success") is not True or result.get("modified") is not True:
                continue
            arguments = _arguments(call.get("function", {}).get("arguments", {}))
            path = canonical_path(result.get("path") or arguments.get("path"), workspace)
            if path is not None:
                boundaries[path] = max(index + 1, boundaries.get(path, 0))
    return boundaries


def _merge(ranges: list[list[int]]) -> list[list[int]]:
    merged = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1] + 1:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return merged


def _ranges(numbers) -> list[list[int]]:
    return _merge([[number, number] for number in numbers])


def _missing(start: int, end: int | None, observed: list[list[int]]) -> list[list[int | None]]:
    result = []
    cursor = start
    for first, last in observed:
        if last < cursor:
            continue
        if end is not None and first > end:
            break
        if first > cursor:
            result.append([cursor, min(first - 1, end) if end is not None else first - 1])
        cursor = max(cursor, last + 1)
        if end is not None and cursor > end:
            return result
    if end is None or cursor <= end:
        result.append([cursor, end])
    return result


def visible_lines(record: dict) -> dict[int, str]:
    """Only complete numbered lines count; metadata and partial prefixes do not."""
    if record.get("success") is not True or not isinstance(record.get("content"), str):
        return {}
    start, end = record.get("start_line"), record.get("end_line")
    if start is not None and (type(start) is not int or start < 1):
        return {}
    if end is not None and (type(end) is not int or end < 0):
        return {}
    result = {}
    for line in record["content"].splitlines():
        number, separator, source = line.partition(": ")
        if not separator or not number.isascii() or not number.isdecimal() or len(number) > 10:
            continue
        number = int(number)
        if number < 1 or (start is not None and number < start) or (end is not None and number > end):
            continue
        if record.get("partial_line") and number == (end if end is not None else start):
            continue
        if number in result and result[number] != source:
            return {}
        result[number] = source
    return result


def _code_like(text: str) -> bool:
    """A conservative label for guidance, not a language parser or correctness claim."""
    text = text.strip()
    return bool(text) and not text.startswith(("#", "//", "/*", "*/", "*", '"""', "'''"))


class FileInspection(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, validate_assignment=True)
    visible_ranges: list[list[int]] = Field(default_factory=list)
    code_like_ranges: list[list[int]] = Field(default_factory=list)
    complete_selections: list[list[int]] = Field(default_factory=list)
    pending_ranges: list[list[int | None]] = Field(default_factory=list)
    pending_requests: list[list[int | None]] = Field(default_factory=list)
    total_lines: int | None = Field(default=None, ge=0)
    empty_observed: bool = False
    last_step: int = Field(default=0, ge=0)

    @field_validator("visible_ranges", "code_like_ranges", "complete_selections", "pending_ranges", "pending_requests")
    @classmethod
    def valid_ranges(cls, values):
        for pair in values:
            if (len(pair) != 2 or type(pair[0]) is not int or pair[0] < 1
                    or pair[1] is not None and (type(pair[1]) is not int or pair[1] < pair[0])):
                raise ValueError("Invalid inspection range")
        return values

    @property
    def inspected(self) -> bool:
        return self.empty_observed or bool(self.complete_selections)


class InspectionLedger(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, validate_assignment=True)
    schema_version: Literal[1] = 1
    files: dict[str, FileInspection] = Field(default_factory=dict)

    @field_validator("files")
    @classmethod
    def valid_paths(cls, files):
        if any(not isinstance(path, str) or not path or len(path) > 4096 or "\x00" in path for path in files):
            raise ValueError("Invalid inspection path")
        return files

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict):
        return cls.model_validate(data)

    def _selection(self, arguments: dict, record: dict, *, omitted=False, step=0, workspace=None):
        path = canonical_path(record.get("path") or arguments.get("path"), workspace)
        if path is None or (not omitted and record.get("success") is not True):
            return
        item = self.files.setdefault(path, FileInspection())
        item.last_step = max(item.last_step, step)
        total = record.get("total_lines")
        if type(total) is int and total >= 0:
            item.total_lines = total
        lines = visible_lines(record) if not omitted else {}
        item.visible_ranges = _merge(item.visible_ranges + _ranges(lines))
        item.code_like_ranges = _merge(item.code_like_ranges + _ranges(
            number for number, source in lines.items() if _code_like(source)))
        start = arguments.get("start_line", record.get("start_line", min(lines) if lines else 1))
        if type(start) is not int or start < 1:
            start = 1
        end = arguments.get("end_line")
        if type(end) is not int or end < start:
            end = item.total_lines
        if item.total_lines is not None and end is not None:
            end = min(end, item.total_lines)
        if end is None and lines and not record.get("truncated") and not record.get("partial_line"):
            end = max(lines)
        if not omitted and total == 0 and record.get("content") == "" and not record.get("truncated"):
            item.empty_observed = True
            item.pending_ranges = []
            item.pending_requests = []
            return
        if end is not None and end < start:
            return
        missing = _missing(start, end, item.visible_ranges)
        complete = (not omitted and not record.get("truncated") and not record.get("partial_line")
                    and bool(lines) and end is not None and not missing)
        # Keep the original requested bounds, not just their remaining holes.
        # Otherwise a later narrow read cannot distinguish its containing broad
        # request from a disjoint omitted selection in the same file.
        requests = [list(pair) for pair in (item.pending_requests or item.pending_ranges)]
        requests.append([start, end])
        if complete:
            item.complete_selections = _merge(item.complete_selections + [[start, end]])
            # A deliberate narrow code selection can replace an earlier broad
            # request containing it. Disjoint requested reads remain pending.
            if any(_code_like(source) for source in lines.values()):
                requests = [pair for pair in requests
                            if not (pair[0] <= start and (pair[1] is None or pair[1] >= end))]
        remaining = []
        pending_requests = []
        for first, last in requests:
            if item.total_lines is not None:
                last = min(last, item.total_lines) if last is not None else item.total_lines
            if last is None or first <= last:
                gaps = _missing(first, last, item.visible_ranges)
                if gaps:
                    pending_requests.append([first, last])
                    remaining.extend(gaps)
        item.pending_requests = [list(pair) for pair in dict.fromkeys(tuple(pair) for pair in pending_requests)]
        item.pending_ranges = [list(pair) for pair in dict.fromkeys(tuple(pair) for pair in remaining)]

    def observe(self, name: str, arguments: Any, result: dict, *, step: int = 0, workspace=None):
        arguments = _arguments(arguments)
        if name == "read_file":
            self._selection(arguments, result, step=step, workspace=workspace)
        elif name in {"read_files", "read_code"}:
            records = result.get("files", [])
            selections = arguments.get("files", [])
            if name == "read_code" and isinstance(arguments.get("paths"), list):
                selections = [{"path": path} for path in arguments["paths"] if isinstance(path, str)]
            records = records if isinstance(records, list) else []
            selections = selections if isinstance(selections, list) else []
            for index, record in enumerate(records):
                if not isinstance(record, dict):
                    continue
                selection = selections[index] if index < len(selections) and isinstance(selections[index], dict) else {}
                if name == "read_code":
                    # The tool deliberately selects code, rather than implying
                    # a whole-file read. A clipped record retains its chosen
                    # endpoint separately; omitted regions are not evidence.
                    selection = {**selection, "path": record.get("path", selection.get("path")),
                                 "start_line": record.get("start_line", selection.get("start_line", 1)),
                                 "end_line": record.get("selected_end_line", record.get("end_line", selection.get("end_line")))}
                self._selection(selection, record, step=step, workspace=workspace)
            if result.get("omitted_file_count") or result.get("next_file_index") is not None:
                for selection in selections[len(records):]:
                    if isinstance(selection, dict):
                        self._selection(selection, {}, omitted=True, step=step, workspace=workspace)

    @classmethod
    def from_messages(cls, messages: list[dict], workspace=None, *, source_boundaries: dict | None = None):
        ledger = cls()
        boundaries = _source_boundaries(source_boundaries, workspace)
        pending = []
        for step, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            if message.get("role") == "assistant" and isinstance(message.get("tool_calls"), list):
                pending = [call for call in message["tool_calls"] if isinstance(call, dict)]
            elif message.get("role") == "tool" and message.get("tool_name") in {"read_file", "read_files", "read_code"}:
                call = next((call for call in pending if call.get("id") == message.get("tool_call_id")
                             and message.get("tool_call_id") is not None), None)
                if call is None:
                    call = next((call for call in pending if isinstance(call.get("function"), dict)
                                 and call["function"].get("name") == message["tool_name"]), {})
                if call in pending:
                    pending.remove(call)
                try:
                    result = json.loads(message.get("content", ""))
                except (TypeError, ValueError):
                    continue
                if isinstance(result, dict):
                    ledger.observe(message["tool_name"], call.get("function", {}).get("arguments", {}),
                                   result, step=step, workspace=workspace)
                    # Applying the final per-path boundary after each observed
                    # batch removes old lines and old omitted selection debt,
                    # while retaining current records for unaffected peers.
                    ledger.files = {path: item for path, item in ledger.files.items()
                                    if item.last_step >= boundaries.get(path, 0)}
        return ledger

    def pending_selections(self, *, limit=8) -> list[dict]:
        result = []
        for path, item in self.files.items():
            for start, end in item.pending_ranges:
                selection = {"path": path, "start_line": start}
                if end is not None:
                    selection["end_line"] = min(end, start + 39)
                result.append(selection)
                if len(result) >= limit:
                    return result
        return result
