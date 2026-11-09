"""The fixed, explicitly enabled pytest runner. This is not a sandbox."""

import math
import os
import re
import sys
from tempfile import TemporaryDirectory

from patchrook_agent.tools.base import TOOL_CAPTURE, ToolError, Workspace
from patchrook_agent.tools.process import run_bounded


_COUNTS = re.compile(r"\b(\d+) (passed|failed|skipped|xfailed|xpassed|deselected|errors?|warnings?)\b")


def _test_counts(text: str) -> dict[str, int]:
    """Parse pytest's final aggregate line, without double-counting sections."""
    for line in reversed(text.splitlines()):
        if " in " not in line and not line.startswith("no tests ran"):
            continue
        matches = _COUNTS.findall(line)
        if matches:
            return {status.removesuffix("s") if status in {"errors", "warnings"} else status: int(count)
                    for count, status in matches}
    return {}


def _concise_output(text: str, limit: int) -> tuple[str, bool]:
    """Keep exact failures and frames; omit captured prints deterministically."""
    lines = text.splitlines()
    useful = []
    captured = False
    omitted = False
    for line in lines:
        if re.match(r"^-+ Captured (?:stdout|stderr|log)", line):
            captured = True
            omitted = True
            continue
        if captured and (line.startswith("===") or re.match(r"^_+ .* _+$", line)):
            captured = False
        if captured:
            continue
        if line == "... output truncated ...":
            omitted = True
            continue
        if line and (set(line) <= {"=", " "} or line.startswith("=== ")
                     or re.match(r"^_+ .* _+$", line)):
            continue
        if line:
            useful.append(line)
    joined = "\n".join(useful)
    if len(joined) <= limit:
        return joined, omitted
    # Prefer final counts, exact exception text, and traceback frames over
    # progress bars and arbitrary captured stdout. Preserve original ordering.
    def priority(item):
        index, line = item
        if _COUNTS.search(line) and " in " in line:
            return 0, -index
        if line.startswith("E ") or line.startswith("E\t"):
            return 1, index
        if re.search(r"\.py:\d+", line):
            return 2, index
        if line.startswith("FAILED ") or line.startswith("ERROR "):
            return 3, index
        return 4, index

    selected = {}
    remaining = limit
    for index, line in sorted(enumerate(useful), key=priority):
        if len(line) + bool(selected) <= remaining:
            selected[index] = line
            remaining -= len(line) + (len(selected) > 1)
        elif priority((index, line))[0] == 1 and remaining > 192:
            # A very long exception still deserves an exact, explicitly partial
            # prefix. Leave room for a source frame rather than consuming the
            # entire observation with one exception string.
            excerpt_size = remaining - 128 - bool(selected)
            selected[index] = line[:excerpt_size]
            remaining -= excerpt_size + (len(selected) > 1)
    if not selected and useful:
        # An individual error can itself exceed the budget. Its exact prefix is
        # marked partial and remains expandable in the full captured artifact.
        selected[0] = useful[0][:limit]
    return "\n".join(selected[index] for index in sorted(selected)), True


def run_tests(
    workspace: Workspace,
    path: str = ".",
    timeout_seconds: float | None = None,
) -> dict:
    """Run pytest at the workspace root when path is omitted or '.'."""
    if not workspace.config.allow_tests:
        raise ToolError("permission_denied", "Test execution requires allow_tests=True for a trusted workspace.")
    if not isinstance(path, str) or not path or "\0" in path:
        raise ToolError("invalid_path", "Provide an existing test file or directory, optionally with a pytest node ID.")
    parts = path.split("::")
    if any(part.startswith("-") or not part for part in parts):
        raise ToolError("invalid_arguments", "pytest options and command strings are not accepted; provide a test path.")
    target = workspace.resolve(parts[0])
    if not target.exists():
        raise ToolError("not_found", f"Test path does not exist: {parts[0]}. No tests ran. "
                        "Omit path or use '.' to run pytest at workspace root, or select an existing test path.")
    if not target.is_file() and not target.is_dir():
        raise ToolError("invalid_path", "Tests must be a regular file or directory.")
    if len(parts) > 1 and not target.is_file():
        raise ToolError("invalid_arguments", "A pytest node ID must select a file.")
    timeout = workspace.config.test_timeout_seconds
    if timeout_seconds is not None:
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ToolError("invalid_arguments", "timeout_seconds must be a positive finite number.")
        timeout = min(timeout, timeout_seconds)
    selection = str(target) + ("::" + "::".join(parts[1:]) if len(parts) > 1 else "")
    env = os.environ.copy()
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("PYTEST_PLUGINS", None)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    # A rapid, equal-length edit can have the same timestamp/size as a cached
    # module. Give each run an empty import-cache location so verification sees
    # current source, without deleting the user's __pycache__ directories.
    with TemporaryDirectory(prefix="patchrook-pycache-") as cache_directory:
        env["PYTHONPYCACHEPREFIX"] = cache_directory
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        result = run_bounded(
            [sys.executable, "-m", "pytest", "-q", "--color=no", "--tb=short", "--", selection],
            cwd=workspace.root, timeout_seconds=timeout,
            max_output_chars=max(128, workspace.config.max_output_chars // 2 - 128), env=env,
        )
    capture = TOOL_CAPTURE.get() or {}
    raw_stdout = capture.get("stdout", result.stdout)
    raw_stderr = capture.get("stderr", result.stderr)
    stream_limit = max(128, workspace.config.max_output_chars // 2 - 128)
    # The bounded process tail retains final counts even if the larger raw
    # prefix hit its capture ceiling. Avoid repeating it when capture is whole.
    stdout_source = raw_stdout
    if capture.get("raw_output_truncated") and result.stdout not in raw_stdout:
        stdout_source += "\n" + result.stdout
    stderr_source = raw_stderr
    if capture.get("raw_output_truncated") and result.stderr not in raw_stderr:
        stderr_source += "\n" + result.stderr
    stdout, stdout_omitted = _concise_output(stdout_source, stream_limit)
    stderr, stderr_omitted = _concise_output(stderr_source, stream_limit)
    counts = _test_counts(result.stdout + "\n" + result.stderr)
    result_dict = {
        "success": not result.timed_out and result.exit_code == 0,
        "status": "timed_out" if result.timed_out else ("passed" if result.exit_code == 0 else "failed"),
        "exit_code": result.exit_code,
        "stdout": stdout,
        "stderr": stderr,
        "duration_seconds": result.duration_seconds,
        "truncated": result.truncated or stdout_omitted or stderr_omitted,
    }
    if counts:
        result_dict["test_counts"] = counts
    if result_dict["truncated"]:
        result_dict["follow_up"] = "read_artifact for full available pytest output"
    return result_dict
