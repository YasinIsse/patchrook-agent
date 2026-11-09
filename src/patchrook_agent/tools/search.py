"""Repository content searches using ripgrep without a shell."""

import base64
import json
import math
import shutil

from patchrook_agent.tools.base import EXCLUDED_DIRS, ToolError, Workspace, clip_text, with_read_guidance
from patchrook_agent.tools.process import run_bounded


def _json_text(value: dict) -> str:
    if "text" in value:
        return value["text"]
    return base64.b64decode(value["bytes"]).decode("utf-8", "replace")


def _excerpt(data: dict, width: int = 200) -> tuple[str, int]:
    """Return exact source around the first match, not a fabricated preview."""
    text = _json_text(data["lines"]).removesuffix("\n").removesuffix("\r")
    if len(text) <= width:
        return text, 0
    byte_offset = data["submatches"][0]["start"] if data["submatches"] else 0
    match_start = len(text.encode("utf-8")[:byte_offset].decode("utf-8", "ignore"))
    start = max(0, min(match_start - 60, len(text) - width))
    return text[start:start + width], start


def search_code(
    workspace: Workspace,
    pattern: str,
    path: str = ".",
    max_results: int | None = None,
    timeout_seconds: float | None = None,
    offset: int = 0,
    mode: str = "literal",
) -> dict:
    if not isinstance(mode, str) or mode not in {"literal", "regex"}:
        raise ToolError("invalid_arguments", "mode must be 'literal' (plain text, the default) or 'regex'.")
    target = workspace.resolve(path)
    if not target.exists():
        raise ToolError("not_found", f"Search path does not exist: {path}")
    if not target.is_file() and not target.is_dir():
        raise ToolError("invalid_path", "Search path must be a regular file or directory.")
    executable = shutil.which("rg")
    if executable is None:
        raise ToolError("unavailable", "ripgrep (rg) is required for search_code; install it with `sudo apt install ripgrep` on Ubuntu/Debian.")
    limit = workspace.config.max_search_results if max_results is None else max_results
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ToolError("invalid_arguments", "max_results must be a positive integer.")
    limit = min(limit, workspace.config.max_search_results)
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ToolError("invalid_arguments", "offset must be a nonnegative integer.")
    timeout = workspace.config.search_timeout_seconds
    if timeout_seconds is not None:
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ToolError("invalid_arguments", "timeout_seconds must be positive.")
        timeout = min(timeout, timeout_seconds)
    command = [
        executable, "--no-config", "--no-require-git", "--json", "--sort", "path",
        "--color", "never", "--max-count", str(offset + limit + 1),
    ]
    if mode == "literal":
        command.append("--fixed-strings")
    for excluded in sorted(EXCLUDED_DIRS):
        command.extend(["--glob", f"!{excluded}/**"])
    command.extend(["-e", pattern, "--", str(target)])
    result = run_bounded(
        command, cwd=workspace.root, timeout_seconds=timeout,
        max_output_chars=workspace.config.max_file_bytes,
        stop_on_output_limit=True,
        preserve_tail=False,
    )
    if result.timed_out:
        raise ToolError("timeout", f"Search exceeded {timeout:g} seconds.")
    if result.exit_code not in (0, 1) and (not result.truncated or not result.stdout):
        message, _ = clip_text(result.stderr.strip(), workspace.config.max_output_chars // 2)
        if mode == "regex" and "regex parse error" in result.stderr.lower():
            raise ToolError("invalid_regex", 'Invalid regex. For ordinary source text, retry search_code '
                            'with mode="literal" and the unchanged pattern. Ripgrep: ' + message)
        raise ToolError("search_failed", message or "ripgrep failed.")
    matches = []
    seen = 0
    malformed = False
    excerpt_truncated = False
    # The final JSON record may be incomplete at the explicit process capture
    # ceiling. Never turn that prefix into a made-up filename or source line.
    for line in result.stdout.splitlines():
        try:
            event = json.loads(line)
        except (ValueError, TypeError):
            malformed = True
            continue
        if event.get("type") != "match":
            continue
        data = event["data"]
        try:
            resolved = workspace.resolve(_json_text(data["path"]))
        except (ToolError, ValueError, KeyError):
            malformed = True
            continue
        if not resolved.is_file():
            malformed = True
            continue
        seen += 1
        if seen <= offset:
            continue
        text, column = _excerpt(data)
        match = {"path": workspace.relative(resolved), "line_number": data["line_number"], "text": text}
        full_line = _json_text(data["lines"]).removesuffix("\n").removesuffix("\r")
        if len(full_line) > len(text):
            match.update(column_start=column + 1, text_truncated=True)
            excerpt_truncated = True
        matches.append(match)
        if len(matches) > limit:
            break
    has_more = len(matches) > limit
    result_dict = {
        "success": True,
        "matches": matches[:limit],
        "offset": offset,
        "returned_count": min(len(matches), limit),
        "truncated": result.truncated or malformed or has_more or excerpt_truncated,
        "duration_seconds": result.duration_seconds,
    }
    if has_more or ((result.truncated or malformed) and matches):
        result_dict["next_offset"] = offset + min(len(matches), limit)
    if result.truncated or malformed:
        result_dict["capture_limited"] = True
        result_dict["capture_limit_bytes"] = workspace.config.max_file_bytes
    if result_dict["truncated"]:
        result_dict["follow_up"] = (
            "Use next_offset as offset. If capture_limited, narrow path or pattern."
        )
    return with_read_guidance(result_dict, workspace.config.max_output_chars)
