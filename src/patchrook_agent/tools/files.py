"""Small, workspace-confined file operations used by the tool dispatcher."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from patchrook_agent.tools.base import (
    EXCLUDED_DIRS, TOOL_CAPTURE, ToolError, Workspace, bound_read_files_result,
    bound_read_result, record_tool_output, with_read_guidance,
)


def _text(workspace: Workspace, path: Path) -> str:
    with path.open("rb") as stream:
        data = stream.read(workspace.config.max_file_bytes + 1)
    if len(data) > workspace.config.max_file_bytes:
        raise ToolError("file_too_large", "File exceeds the configured byte limit.")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ToolError("binary_file", "Only UTF-8 text files are supported.") from exc
    if any(ord(char) < 32 and char not in "\t\r\n\f" for char in text):
        raise ToolError("binary_file", "Binary files are not supported.")
    return text


def _content_bytes(workspace: Workspace, content: str) -> bytes:
    data = content.encode("utf-8")
    if len(data) > workspace.config.max_file_bytes:
        raise ToolError("file_too_large", "Content exceeds the configured byte limit.")
    return data


def list_files(
    workspace: Workspace,
    path: str = ".",
    recursive: bool = False,
    max_entries: int | None = None,
    offset: int = 0,
) -> dict:
    """List relative paths without following directory symlinks."""
    directory = workspace.resolve(path)
    if not directory.exists():
        raise ToolError("not_found", f"Directory does not exist: {path}")
    if not directory.is_dir():
        raise ToolError("not_directory", f"Expected a directory: {path}")
    if max_entries is not None and max_entries < 1:
        raise ToolError("invalid_limit", "max_entries must be positive.")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ToolError("invalid_arguments", "offset must be a nonnegative integer.")
    limit = min(max_entries or workspace.config.max_list_entries,
                workspace.config.max_list_entries)
    entries: list[str] = []
    truncated = False
    pending = [directory]
    seen = 0
    while pending and not truncated:
        current = pending.pop()
        subdirectories = []
        for entry in sorted(current.iterdir(), key=lambda item: item.name):
            if entry.name in EXCLUDED_DIRS:
                continue
            try:
                workspace.resolve(str(entry))
            except ToolError:
                # Do not reveal or traverse a symlink pointing outside the workspace.
                continue
            is_directory = entry.is_dir()
            if not is_directory and not entry.is_file():
                continue
            if seen >= offset:
                if len(entries) >= limit:
                    truncated = True
                    break
                relative = entry.relative_to(workspace.root).as_posix()
                entries.append(relative + ("/" if is_directory else ""))
            seen += 1
            if recursive and is_directory and not entry.is_symlink():
                subdirectories.append(entry)
        pending.extend(reversed(subdirectories))
    result = {"success": True, "path": workspace.relative(directory),
              "entries": entries, "offset": offset,
              "returned_count": len(entries), "truncated": truncated}
    if truncated:
        result["next_offset"] = offset + len(entries)
        result["follow_up"] = "list_files with next_offset as offset"
    return with_read_guidance(result, workspace.config.max_output_chars)


def read_file(
    workspace: Workspace,
    path: str,
    start_line: int = 1,
    end_line: int | None = None,
) -> dict:
    file_path = workspace.require_file(path)
    if start_line < 1 or (end_line is not None and end_line < start_line):
        raise ToolError("invalid_range", "Use a positive start_line and end_line >= start_line.")
    lines = _text(workspace, file_path).splitlines()
    if lines and start_line > len(lines):
        raise ToolError("invalid_range", f"start_line exceeds the file's {len(lines)} lines.")
    if not lines and start_line != 1:
        raise ToolError("invalid_range", "The file is empty; start_line must be 1.")
    requested_end = end_line if end_line is not None else len(lines)
    selected_end = min(requested_end, start_line + workspace.config.max_read_lines - 1,
                       len(lines))
    numbered = "\n".join(f"{index + 1}: {lines[index]}"
                         for index in range(start_line - 1, selected_end))
    record_tool_output(content=numbered, start_line=start_line, end_line=selected_end)
    result = {"success": True, "path": workspace.relative(file_path),
              "start_line": start_line, "end_line": selected_end,
              "total_lines": len(lines), "content": numbered,
              "truncated": selected_end < min(requested_end, len(lines))}
    if result["truncated"]:
        result["next_start_line"] = selected_end + 1
    return bound_read_result(result, workspace.config.max_output_chars)


def read_files(workspace: Workspace, files: list[dict]) -> dict:
    """Read up to ten targeted selections with isolated, full per-file capture."""
    if not isinstance(files, list) or not 1 <= len(files) <= 10:
        raise ToolError("invalid_arguments", "files must contain between 1 and 10 file selections.")
    if any(not isinstance(selection, dict) or "path" not in selection
           or selection.keys() - {"path", "start_line", "end_line"} for selection in files):
        raise ToolError("invalid_arguments", "Each selection needs path and optional start_line/end_line.")
    results = []
    available = []
    parent_capture = TOOL_CAPTURE.get()
    for selection in files:
        captured = ({"max_bytes": parent_capture.get("max_bytes", workspace.config.max_file_bytes)}
                    if parent_capture is not None else None)
        token = TOOL_CAPTURE.set(captured)
        try:
            try:
                result = read_file(workspace, **selection)
            except ToolError as exc:
                result = {"success": False, "path": selection["path"],
                          "error": {"code": exc.code, "message": exc.message}}
            except OSError as exc:
                result = {"success": False, "path": selection["path"],
                          "error": {"code": "io_error", "message": str(exc)}}
            if not result["success"]:
                try:
                    result["path"] = workspace.relative(workspace.resolve(selection["path"]))
                except ToolError:
                    # Rejected outside paths stay untrusted request identities.
                    pass
            results.append(result)
            if captured is not None:
                available.append({"path": result["path"], "result": result,
                                  "available_output": {key: value for key, value in captured.items()
                                                       if key != "max_bytes"}})
        finally:
            TOOL_CAPTURE.reset(token)
    record_tool_output(files=available)
    result = {"success": all(item["success"] for item in results), "files": results,
              "requested_count": len(files), "returned_count": len(results),
              "truncated": any(item.get("truncated") for item in results)}
    failures = sum(not item["success"] for item in results)
    if failures:
        result["failed_count"] = failures
    return bound_read_files_result(result, workspace.config.max_output_chars)


def _replace_bytes(path: Path, data: bytes) -> None:
    """Replace one existing regular file atomically, keeping its permissions."""
    mode = path.stat().st_mode & 0o777
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".patchrook-", delete=False) as stream:
            temporary = stream.name
            stream.write(data)
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def write_file(
    workspace: Workspace,
    path: str,
    content: str,
    overwrite: bool = False,
) -> dict:
    file_path = workspace.resolve(path)
    data = _content_bytes(workspace, content)
    if file_path.exists():
        if not overwrite:
            raise ToolError("already_exists", "File exists; use edit_file for targeted changes.")
        file_path = workspace.require_file(path)
        _replace_bytes(file_path, data)
    else:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        # Recheck after creating parents before opening the destination.
        file_path = workspace.resolve(path)
        try:
            with file_path.open("xb") as stream:
                stream.write(data)
        except FileExistsError as exc:
            raise ToolError("already_exists", "File already exists; it was not overwritten.") from exc
    return {"success": True, "path": workspace.relative(file_path),
            "bytes_written": len(data), "modified": True}


def edit_file(
    workspace: Workspace,
    path: str,
    old_text: str,
    new_text: str,
) -> dict:
    if not old_text:
        raise ToolError("invalid_match", "old_text must not be empty.")
    file_path = workspace.require_file(path)
    original = _text(workspace, file_path)
    first_match = original.find(old_text)
    if first_match < 0:
        raise ToolError("no_match", "old_text does not occur exactly in the file; read it again.")
    if original.find(old_text, first_match + 1) >= 0:
        raise ToolError("ambiguous_match", "old_text occurs more than once; include more context.")
    # JSON tool arguments normally use LF. Keep CRLF files consistently CRLF.
    if "\r\n" in original and original.count("\n") == original.count("\r\n"):
        new_text = new_text.replace("\r\n", "\n").replace("\n", "\r\n")
    updated = original.replace(old_text, new_text, 1)
    if updated == original:
        raise ToolError(
            "no_change",
            "The replacement leaves the file unchanged. Use different new_text to make the requested correction.",
        )
    _replace_bytes(file_path, _content_bytes(workspace, updated))
    return {"success": True, "path": workspace.relative(file_path),
            "replacements": 1, "modified": True,
            "description": "Replaced one exact occurrence."}
