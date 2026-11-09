"""Useful exact code selections, parsed locally without executing source."""

import ast
from pathlib import Path

from .base import ToolError, Workspace, bound_read_files_result, observation_chars, record_tool_output
from .files import _text


def _symbols(tree: ast.AST) -> list[tuple[str, ast.AST]]:
    found = []

    def visit(body, parent=""):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{parent}.{node.name}" if parent else node.name
                found.append((name, node))
                visit(node.body, name)
    visit(tree.body)
    return found


def _context_start(lines: list[str], start: int) -> int:
    """Nearby comments/imports are optional context, never rewritten source."""
    comment = start - 1
    while comment > 0 and lines[comment - 1].lstrip().startswith("#"):
        comment -= 1
    if start - 1 - comment > 3:
        # A long module-header block is outside the chosen definition. Keep its
        # exact range in omissions instead of spending batch space on its tail.
        return start
    first = start
    for number in range(start - 1, max(0, start - 5), -1):
        text = lines[number - 1].strip()
        if not text or text.startswith(("#", "import ", "from ")):
            first = number
        else:
            break
    return first


def _selection(path: Path, text: str, request: dict) -> tuple[int, int, int, list[dict]]:
    lines = text.splitlines()
    if "symbol" in request and ("start_line" in request or "end_line" in request):
        raise ToolError("invalid_selection", "Choose a symbol or a line range, not both.")
    if "start_line" in request or "end_line" in request:
        start = request.get("start_line", 1)
        end = request.get("end_line", start + 39)
        if start < 1 or end < start or (lines and start > len(lines)) or (not lines and start != 1):
            raise ToolError("invalid_range", "Use an existing positive line range with end_line >= start_line.")
        return start, min(end, len(lines)), start, []
    symbols, tree = [], None
    if path.suffix == ".py":
        try:
            tree = ast.parse(text)
            symbols = _symbols(tree)
        except (SyntaxError, ValueError, RecursionError):
            pass
    if request.get("symbol"):
        matches = [(name, node) for name, node in symbols
                   if name == request["symbol"] or name.rsplit(".", 1)[-1] == request["symbol"]]
        if not matches:
            raise ToolError("symbol_not_found", "Function or class was not found; use an explicit line range.")
        if len(matches) != 1:
            raise ToolError("ambiguous_symbol", "Symbol matches more than one definition; use its qualified name.")
        node = matches[0][1]
    elif symbols:
        node = symbols[0][1]
    else:
        # For other formats and incomplete Python, show a small exact range
        # starting after header comments where possible, without execution.
        start = next((number for number, line in enumerate(lines, 1)
                      if line.strip() and not line.lstrip().startswith(("#", "//"))), 1)
        return start, min(len(lines), start + 19), _context_start(lines, start), []
    start = min([node.lineno] + [item.lineno for item in getattr(node, "decorator_list", [])])
    end = node.end_lineno or node.lineno
    outline = [{"name": name, "start_line": item.lineno, "end_line": item.end_lineno or item.lineno}
               for name, item in symbols[:12]]
    return start, end, _context_start(lines, start), outline


def _record(path: str, lines: list[str], start: int, end: int, *, selected_end: int | None = None) -> dict:
    record = {"success": True, "path": path, "start_line": start, "end_line": end,
              "content": "\n".join(f"{number}: {lines[number - 1]}" for number in range(start, end + 1)),
              "truncated": selected_end is not None and end < selected_end}
    omitted = []
    if start > 1:
        omitted.append([1, start - 1])
    requested_end = selected_end if selected_end is not None else end
    if requested_end < len(lines):
        omitted.append([requested_end + 1, len(lines)])
    if omitted:
        record["omitted_ranges"] = omitted
    if record["truncated"]:
        record["selected_end_line"] = selected_end
        record["next_start_line"] = end + 1
    return record


def read_code(workspace: Workspace, files: list[dict], *, observation_limit: int | None = None,
              include_context: bool = True) -> dict:
    """Read selected source from 1–10 files, completing code before context."""
    if not isinstance(files, list) or not 1 <= len(files) <= 10:
        raise ToolError("invalid_arguments", "files must contain between 1 and 10 selections.")
    if any(not isinstance(request, dict) or "path" not in request for request in files):
        raise ToolError("invalid_arguments", "Each selection needs a path.")
    limit = min(workspace.config.max_output_chars, observation_limit or workspace.config.max_output_chars)
    results, captures, expansions, outlines = [], [], {}, {}
    for request in files:
        path = request["path"]
        try:
            target = workspace.require_file(path)
            path = workspace.relative(target)
            text = _text(workspace, target)
            lines = text.splitlines()
            start, selected_end, context_start, outline = _selection(target, text, request)
            end = min(selected_end, start + workspace.config.max_read_lines - 1)
            result = _record(path, lines, start, end, selected_end=selected_end)
            if not lines:
                result["total_lines"] = 0
            if include_context and context_start < start and selected_end - context_start < workspace.config.max_read_lines:
                expansions[len(results)] = _record(path, lines, context_start, selected_end)
            if (target.suffix == ".py" and not (request.keys() & {"symbol", "start_line", "end_line"})
                    and len(lines) <= min(30, workspace.config.max_read_lines)
                    and sum(bool(line.strip()) and not line.lstrip().startswith("#") for line in lines)
                    >= max(1, len(lines) // 2)):
                expansions[len(results)] = _record(path, lines, 1, len(lines))
            if len(outline) > 1:
                outlines[path] = outline
            captures.append({"path": path, "result": result,
                             "available_output": {"source": text, "total_lines": len(lines),
                                                  "selected_start_line": start, "selected_end_line": selected_end,
                                                  "symbols": outline}})
        except (ToolError, OSError) as exc:
            try:
                path = workspace.relative(workspace.resolve(path))
            except ToolError:
                pass
            result = {"success": False, "path": path, "error": {
                "code": exc.code if isinstance(exc, ToolError) else "io_error", "message": str(exc)}}
            if isinstance(exc, ToolError) and exc.code == "not_found":
                result["error"]["message"] += ' Locate the exact repository path, then call read_code with {"paths":["relative/file.py"]}.'
            captures.append({"path": path, "result": result, "available_output": {}})
        results.append(result)
    result = {"success": all(item["success"] for item in results), "selection": "code", "files": results,
              "requested_count": len(files), "returned_count": len(results),
              "truncated": any(item.get("truncated") for item in results)}
    failed = sum(not item["success"] for item in results)
    if failed:
        result["failed_count"] = failed
    # Complete small code selections across files before spending spare space
    # on comments/imports immediately preceding them or additional symbols.
    if observation_chars(result) <= limit:
        for index, expanded in expansions.items():
            candidate = {**result, "files": list(result["files"])}
            candidate["files"][index] = expanded
            if observation_chars(candidate) <= limit:
                result = candidate
                captures[index]["result"] = expanded
                captures[index]["available_output"]["selected_start_line"] = expanded["start_line"]
                captures[index]["available_output"]["selected_end_line"] = expanded["end_line"]
        if outlines and observation_chars({**result, "outline": outlines}) <= limit:
            result = {**result, "outline": outlines}
    record_tool_output(files=captures)
    return bound_read_files_result(result, limit)
