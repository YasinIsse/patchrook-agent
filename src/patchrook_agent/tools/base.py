"""Shared workspace checks and configurable limits for the tool layer."""

from dataclasses import dataclass
from contextvars import ContextVar
import json
from pathlib import Path


EXCLUDED_DIRS = frozenset({".git", ".venv", "node_modules", "__pycache__", ".pytest_cache"})

# One sequential execution's available output, kept separate from the bounded
# model observation. Process captures have their own explicit byte ceiling.
TOOL_CAPTURE: ContextVar[dict | None] = ContextVar("tool_capture", default=None)


def record_tool_output(**values) -> None:
    capture = TOOL_CAPTURE.get()
    if capture is not None:
        capture.update(values)


class ToolError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class ToolConfig:
    max_read_lines: int = 200
    max_list_entries: int = 200
    max_search_results: int = 100
    max_output_chars: int = 2000
    max_file_bytes: int = 1_048_576
    search_timeout_seconds: float = 10.0
    test_timeout_seconds: float = 60.0
    allow_tests: bool = False
    validate_python_edits: bool = False

    def __post_init__(self):
        for name in ("max_read_lines", "max_list_entries", "max_search_results", "max_file_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if type(self.max_output_chars) is not int or self.max_output_chars < 512:
            raise ValueError("max_output_chars must be at least 512")
        for name in ("allow_tests", "validate_python_edits"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        for name in ("search_timeout_seconds", "test_timeout_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < float("inf"):
                raise ValueError(f"{name} must be positive and finite")


class Workspace:
    def __init__(self, root: str | Path, config: ToolConfig | None = None):
        try:
            self.root = Path(root).expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError("Workspace must be an existing directory") from exc
        if not self.root.is_dir():
            raise ValueError("Workspace must be an existing directory")
        self.config = config or ToolConfig()

    def resolve(self, path: str) -> Path:
        if not isinstance(path, str) or not path or "\x00" in path:
            raise ToolError("invalid_path", "Path must be a nonempty string without NUL characters")
        try:
            resolved = (self.root / path).resolve()
        except (OSError, RuntimeError, ValueError) as exc:
            raise ToolError("invalid_path", "Path could not be resolved") from exc
        if not resolved.is_relative_to(self.root):
            raise ToolError("outside_workspace", "Path resolves outside the configured workspace")
        return resolved

    def relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def require_file(self, path: str) -> Path:
        resolved = self.resolve(path)
        if not resolved.exists():
            raise ToolError("not_found", f"File does not exist: {path}")
        if not resolved.is_file():
            raise ToolError("not_file", f"Path is not a regular file: {path}")
        return resolved


def clip_text(text: str, limit: int) -> tuple[str, bool]:
    return text[:limit], len(text) > limit


def observation_chars(result: dict) -> int:
    """Use the same JSON representation as the model observation boundary."""
    return len(json.dumps(result, ensure_ascii=False, separators=(",", ":")))


def with_read_guidance(result: dict, limit: int, *, enabled_tools: set[str] | None = None) -> dict:
    """Suggest narrow reads of shown evidence without displacing that evidence."""
    result = dict(result)
    result.pop("read_hint", None)
    result.pop("suggested_read", None)
    if not result.get("success"):
        return result
    batch_enabled = enabled_tools is None or "read_files" in enabled_tools
    single_enabled = enabled_tools is None or "read_file" in enabled_tools
    code_enabled = enabled_tools is not None and "read_code" in enabled_tools
    search_enabled = enabled_tools is None or "search_code" in enabled_tools
    selections = []
    seen = set()
    for match in result.get("matches", []):
        if not isinstance(match, dict):
            continue
        path, line = match.get("path"), match.get("line_number")
        if not isinstance(path, str) or not path or type(line) is not int or line < 1 or path in seen:
            continue
        seen.add(path)
        selections.append({"path": path, "start_line": line, "end_line": line + 5})
    if selections:
        if not batch_enabled and not single_enabled and not code_enabled:
            return result
        multiple = len(selections) > 1
        if code_enabled:
            result["read_hint"] = ("Inspect known paths with read_code; use read_file for exact ranges."
                                   if single_enabled else "Inspect known paths together with read_code.")
            if observation_chars(result) > limit:
                result.pop("read_hint")
                return result
            for count in range(min(10, len(selections)), 0, -1):
                result["suggested_read"] = {"name": "read_code", "arguments": {
                    "paths": [selection["path"] for selection in selections[:count]]}}
                if observation_chars(result) <= limit:
                    return result
            result.pop("suggested_read", None)
            return result
        use_batch = batch_enabled and (multiple or not single_enabled)
        result["read_hint"] = ("Read known files together with read_files; use narrow ranges." if use_batch
                               else "Use read_file around the matching line.")
        if observation_chars(result) > limit:
            result.pop("read_hint")
            return result
        if use_batch:
            minimum = 2 if multiple else 1
            for count in range(min(10, len(selections)), minimum - 1, -1):
                result["suggested_read"] = {"name": "read_files", "arguments": {"files": selections[:count]}}
                if observation_chars(result) <= limit:
                    return result
        else:
            result["suggested_read"] = {"name": "read_file", "arguments": selections[0]}
            if observation_chars(result) <= limit:
                return result
        result.pop("suggested_read", None)
    elif sum(isinstance(entry, str) and not entry.endswith("/") for entry in result.get("entries", [])) > 1:
        if code_enabled:
            result["read_hint"] = ("Use read_code for known paths; read_file accepts exact line ranges."
                                   if single_enabled else "Use read_code for known paths.")
        elif search_enabled and batch_enabled:
            result["read_hint"] = "Locate relevant lines with search_code; use read_files for known ranges."
        elif search_enabled and single_enabled:
            result["read_hint"] = "Locate relevant lines with search_code; use read_file for narrow ranges."
        elif search_enabled:
            result["read_hint"] = "Locate relevant lines with search_code."
        elif batch_enabled:
            result["read_hint"] = "Use read_files for narrow ranges in known files."
        elif single_enabled:
            result["read_hint"] = "Use read_file for narrow ranges in known files."
        if observation_chars(result) > limit:
            result.pop("read_hint", None)
    return result


def bound_read_result(result: dict, limit: int) -> dict:
    """Trim numbered source only at line boundaries, marking a partial long line.

    The full selected range is captured by read_file before this projection.
    Continuations refer to the first omitted line, including a partially shown
    line, so a shorter observation never silently advances past source text.
    """
    result = dict(result)
    if observation_chars(result) <= limit:
        return result
    result["truncated"] = True
    lines = result["content"].split("\n") if result["content"] else []
    while len(lines) > 1 and observation_chars(result) > limit:
        lines.pop()
        result["content"] = "\n".join(lines)
        result["end_line"] = result["start_line"] + len(lines) - 1
        result["next_start_line"] = result["end_line"] + 1
        result.pop("partial_line", None)
        result.pop("line_chars_returned", None)
        result.pop("follow_up", None)
    if observation_chars(result) <= limit:
        return result
    if lines:
        result["partial_line"] = True
        result["next_start_line"] = result["start_line"]
        result["end_line"] = result["start_line"]
        result["follow_up"] = "read_artifact for the full captured line"
        source = lines[0]
        prefix = f"{result['start_line']}: "
        low, high = 0, len(source)
        while low < high:
            middle = (low + high + 1) // 2
            result["content"] = source[:middle]
            result["line_chars_returned"] = max(0, middle - len(prefix))
            if observation_chars(result) <= limit:
                low = middle
            else:
                high = middle - 1
        result["content"] = source[:low]
        result["line_chars_returned"] = max(0, low - len(prefix))
    return result


def bound_read_files_result(result: dict, limit: int) -> dict:
    """Share one observation budget across exact, expandable file excerpts.

    Complete earlier requested selections before clipping at most one trailing
    record. Every shown nonempty source file includes source text; records that
    cannot fit are explicitly omitted rather than presented as empty reads.
    """
    if observation_chars(result) <= limit:
        return result
    if result.get("selection") == "code" and "outline" in result:
        result = {key: value for key, value in result.items() if key != "outline"}
        if observation_chars(result) <= limit:
            return result
    original_files = result["files"]
    already_omitted = result.get("omitted_file_count", 0)
    for count in range(len(original_files), -1, -1):
        projected = {**result, "files": [], "returned_count": count, "truncated": True}
        omitted = already_omitted + len(original_files) - count
        if omitted:
            reader = "read_code" if result.get("selection") == "code" else "read_files"
            projected.update(omitted_file_count=omitted, next_file_index=count,
                             follow_up=f"Use {reader} for omitted selections or read_artifact")
        if not count:
            return projected
        # Keep prior records intact. Reserve the envelope and their exact JSON
        # size before fitting only the next contiguous excerpt. An empty object
        # is a two-character placeholder for that trailing record.
        projected["files"] = original_files[:count - 1] + [{}]
        item_limit = limit - observation_chars(projected) + 2
        if item_limit <= 0:
            continue
        item = original_files[count - 1]
        if "content" in item and "start_line" in item and "end_line" in item:
            if result.get("selection") == "code" and observation_chars(item) > item_limit:
                item = {**item, "selected_end_line": item.get("selected_end_line", item["end_line"])}
            item = bound_read_result(item, item_limit)
            nonempty = item.get("total_lines") or (result.get("selection") == "code"
                                                  and item["end_line"] >= item["start_line"])
            if nonempty and (not item["content"] or
                             (item.get("partial_line") and not item.get("line_chars_returned"))):
                continue
        elif observation_chars(item) > item_limit:
            item = {**item, "error": dict(item["error"]), "truncated": True}
            item["error"]["message_truncated"] = True
            while item["error"]["message"] and observation_chars(item) > item_limit:
                excess = observation_chars(item) - item_limit
                item["error"]["message"] = item["error"]["message"][:-max(1, excess)]
        if observation_chars(item) > item_limit:
            continue
        projected["files"][-1] = item
        if observation_chars(projected) <= limit:
            return projected
    return result
