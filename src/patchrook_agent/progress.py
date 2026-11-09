"""A narrow, persisted guard for repeated searches that found no matches."""

from copy import deepcopy
import ast
import json
import posixpath
import re
from typing import Any, Literal

from patchrook_agent.tools.base import ToolError
from patchrook_agent.inspection import (
    InspectionLedger, _source_boundaries, source_read_boundaries, visible_lines,
)


SearchAction = Literal["recover", "stop"]


def _search(arguments: Any) -> dict[str, Any] | None:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (TypeError, ValueError):
            return None
    if not isinstance(arguments, dict):
        return None
    pattern, path, offset = arguments.get("pattern"), arguments.get("path", "."), arguments.get("offset", 0)
    if (not isinstance(pattern, str) or not pattern or not isinstance(path, str) or not path
            or "\x00" in path or type(offset) is not int or offset < 0):
        return None
    # A result limit cannot change a genuinely empty result at the same offset.
    return {"pattern": pattern, "path": posixpath.normpath(path), "offset": offset}


def _new_evidence(name: str, result: dict[str, Any]) -> bool:
    if name in {"read_file", "read_artifact", "list_files"} and result.get("success") is True:
        return True
    if name in {"read_files", "read_code"} and isinstance(result.get("files"), list):
        return any(isinstance(item, dict) and item.get("success") is True for item in result["files"])
    if name in {"write_file", "edit_file"}:
        return result.get("success") is True and result.get("modified") is True
    # An observed failing test is useful baseline evidence, unlike a setup error.
    return (name == "run_tests" and result.get("status") in ("passed", "failed", "timed_out")
            and type(result.get("exit_code")) is int)


class SearchProgress:
    def __init__(self):
        self._last_search: dict[str, Any] | None = None
        self.empty_count = 0

    @property
    def action(self) -> SearchAction | None:
        if self.empty_count >= 3:
            return "stop"
        return "recover" if self.empty_count == 2 else None

    def reset(self) -> None:
        self._last_search = None
        self.empty_count = 0

    def observe(self, name: str, arguments: Any, result: dict[str, Any]) -> SearchAction | None:
        if not isinstance(name, str) or not isinstance(result, dict):
            return None
        if _new_evidence(name, result):
            self.reset()
            return None
        if name != "search_code" or result.get("success") is not True:
            return None
        matches = result.get("matches")
        if not isinstance(matches, list):
            return None
        if matches:
            self.reset()
            return None
        if result.get("truncated") or result.get("metadata_omitted"):
            return None
        query = _search(arguments)
        if query is None:
            return None
        self.empty_count = min(3, self.empty_count + 1) if query == self._last_search else 1
        self._last_search = query
        return self.action

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, "last_search": deepcopy(self._last_search), "empty_count": self.empty_count}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SearchProgress":
        if (not isinstance(value, dict) or set(value) != {"schema_version", "last_search", "empty_count"}
                or type(value.get("schema_version")) is not int or value["schema_version"] != 1):
            raise ValueError("Unsupported search-progress state")
        count, query = value["empty_count"], value["last_search"]
        if type(count) is not int or not 0 <= count <= 3:
            raise ValueError("Invalid empty-search count")
        if query is None:
            if count:
                raise ValueError("Empty-search count requires a previous search")
        elif (not count or not isinstance(query, dict) or set(query) != {"pattern", "path", "offset"}
              or _search(query) != query):
            raise ValueError("Invalid previous empty search")
        progress = cls()
        progress._last_search, progress.empty_count = deepcopy(query), count
        return progress


_TASK_PATH = re.compile(
    r"(?<![A-Za-z0-9_./:\\-])([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*\.[A-Za-z0-9_+-]{1,16})"
    r"(?![A-Za-z0-9_/:\\-])"
)


def empty_search_guidance(task: str, *, stopping: bool = False,
                          enabled_tools: set[str] | None = None) -> str:
    """Suggest validated next actions using task text alone, without reading files."""
    paths = []
    for match in _TASK_PATH.finditer(task[:16000] if isinstance(task, str) else ""):
        path = match.group(1).rstrip(".")
        if ".." in path.split("/") or len(path) > 160:
            continue
        path = posixpath.normpath(path)
        if path not in paths:
            paths.append(path)
        if len(paths) == 4:
            break
    available = ({"read_file", "read_files", "list_files", "search_code"}
                 if enabled_tools is None else enabled_tools)
    prefix = ("Three identical searches returned no matches. The task remains incomplete. " if stopping else
              "The same search returned no matches twice. Choose a different available action. ")
    actions = []
    if paths:
        prefix += "Known paths from the task: " + json.dumps(paths) + ". "
    if "read_files" in available and (len(paths) > 1 or "read_file" not in available):
        actions.append("Read known paths together with read_files." if paths else
                       "Use read_files for paths established by the task or observed evidence.")
    elif "read_file" in available:
        actions.append("Read a known path directly with read_file." if paths else
                       "Use read_file once a relevant path is known.")
    if "list_files" in available and (not paths or not actions):
        actions.append("Use list_files to locate relevant paths.")
    if "search_code" in available:
        actions.append("Revise the search pattern or scope with search_code.")
    return prefix + (" ".join(actions) if actions else
                     "No recovery tool is enabled; report the remaining work as incomplete.")


def pending_local_imports(messages: list[dict], workspace, *, limit: int = 10,
                          task: str = "", source_boundaries: dict | None = None) -> list[str]:
    """Discover optional next reads from visible import lines, without executing code.

    These are suggestions, not inspection evidence or mandatory task scope. Only
    existing workspace files are suggested; hidden captures and package imports
    outside the workspace supply no source information.
    """
    reads = []
    boundaries = _source_boundaries(source_boundaries, workspace)
    ledger = InspectionLedger.from_messages(messages, workspace, source_boundaries=boundaries)
    seen = {path for path, item in ledger.files.items() if item.inspected and not item.pending_ranges}
    for index, message in enumerate(messages):
        if message.get("role") != "tool" or message.get("tool_name") not in {"read_file", "read_files", "read_code"}:
            continue
        try:
            result = json.loads(message.get("content", ""))
        except (TypeError, ValueError):
            continue
        if not isinstance(result, dict):
            continue
        records = [result] if message["tool_name"] == "read_file" else result.get("files", [])
        for record in records if isinstance(records, list) else []:
            if (not isinstance(record, dict) or record.get("success") is not True
                    or not isinstance(record.get("content"), str) or not record["content"]):
                continue
            try:
                path = workspace.relative(workspace.resolve(record["path"]))
            except (KeyError, ValueError, TypeError, ToolError):
                continue
            if index < boundaries.get(path, 0):
                continue
            lines = visible_lines(record)
            reads.append((path, "\n".join(f"{number}: {source}" for number, source in lines.items())))
    pending = []
    # Task paths are explicit user input; checking existence does not inspect
    # source or credit a read. This also gives the first action concrete scope.
    for match in _TASK_PATH.finditer(task[:16000]):
        candidate = match.group(1).rstrip(".")
        try:
            resolved = workspace.resolve(candidate)
            canonical = workspace.relative(resolved)
            if resolved.is_file() and canonical not in seen and canonical not in pending:
                pending.append(canonical)
        except (OSError, ValueError, ToolError):
            continue
        if len(pending) >= limit:
            return pending
    for path, content in reads:
        parent = path.split("/")[:-1]
        for line in content.splitlines():
            number, separator, text = line.partition(": ")
            if not separator or not number.isdigit() or not text.lstrip().startswith(("import ", "from ")):
                continue
            try:
                node = ast.parse(text.lstrip()).body[0]
            except (SyntaxError, IndexError):
                continue
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name.split(".") for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level > len(parent) + 1:
                    continue
                base = parent[:len(parent) - node.level + 1] if node.level else []
                module = base + (node.module.split(".") if node.module else [])
                modules = [module] if node.module else []
                modules.extend(module + alias.name.split(".") for alias in node.names if alias.name != "*")
            for module in modules:
                if not module:
                    continue
                stem = "/".join(module)
                for candidate in (stem + ".py", stem + "/__init__.py"):
                    try:
                        resolved = workspace.resolve(candidate)
                        canonical = workspace.relative(resolved)
                        if resolved.is_file() and canonical not in seen and canonical not in pending:
                            pending.append(canonical)
                    except (OSError, ToolError, ValueError):
                        continue
                    if len(pending) >= limit:
                        return pending
    return pending


def pending_read_selections(messages: list[dict], workspace, *, limit: int = 8,
                            source_boundaries: dict | None = None) -> list[dict]:
    """Bounded continuations for visibly incomplete selections, without hidden reads."""
    if type(limit) is not int or limit < 1:
        raise ValueError("limit must be a positive integer")
    selections = InspectionLedger.from_messages(messages, workspace,
                                                source_boundaries=source_boundaries).pending_selections(limit=limit)
    result = []
    for selection in selections:
        try:
            if workspace.resolve(selection["path"]).is_file():
                result.append(selection)
        except (OSError, ValueError, ToolError):
            continue
    return result
