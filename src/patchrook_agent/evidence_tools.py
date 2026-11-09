"""Guarded tools for evidence episodes; executable checks remain explicitly trusted.

The presentation boundary is deliberate: a previous read is not permission to
edit text omitted from the latest model request. Checks run configured argv
vectors without a shell. They execute with the caller's permissions and are
not a sandbox or an independent correctness oracle.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import time

from patchrook_agent.python_anchor import recover_function_anchor
from patchrook_agent.tools.base import EXCLUDED_DIRS, TOOL_CAPTURE, ToolError, Workspace
from patchrook_agent.tools.files import _replace_bytes
from patchrook_agent.tools.process import run_bounded
from patchrook_agent.tools.registry import ToolRegistry, bounded_result


_READ_TOOLS = frozenset({"list_files", "read_file", "search_code"})
_CHECK_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
_MAX_BATCH = 16
_CAPTURE_BYTES = 1_048_576


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class EvidenceTools:
    def __init__(self, workspace: Workspace, *, commands: dict[str, list[str]] | None = None,
                 emit=None, observation_limit: int | None = None, recover_python_function_anchors: bool = False):
        self.workspace = workspace
        if type(recover_python_function_anchors) is not bool:
            raise ValueError("recover_python_function_anchors must be a boolean")
        self.recover_python_function_anchors = recover_python_function_anchors
        self._anchor_recoveries: list[dict] = []
        self.observation_limit = (workspace.config.max_output_chars if observation_limit is None
                                  else observation_limit)
        if type(self.observation_limit) is not int or self.observation_limit < 512:
            raise ValueError("observation_limit must be an integer of at least 512")
        if commands is not None and not isinstance(commands, dict):
            raise ValueError("commands must map trusted names to argv lists")
        self.commands = {}
        for name, argv in (commands or {}).items():
            if (not isinstance(name, str) or not _CHECK_NAME.fullmatch(name)
                    or not isinstance(argv, (list, tuple)) or not argv
                    or any(not isinstance(value, str) or not value or "\0" in value for value in argv)):
                raise ValueError("Each trusted command needs a valid name and nonempty argv strings")
            self.commands[name] = tuple(argv)
        self.emit = emit
        self.registry = ToolRegistry(workspace)
        self.registry.capture_enabled = True
        self.registry.guidance_tools = set(_READ_TOOLS)
        self._presented: dict[str, dict] = {}
        self.last_execution: dict | None = None

    def _path(self, value: str, *, root_allowed: bool = False) -> Path:
        if not isinstance(value, str) or not value or "\0" in value:
            raise ToolError("invalid_path", "Use a nonempty relative repository path")
        lexical = PurePosixPath(value)
        if (lexical.is_absolute() or ".." in lexical.parts
                or any(part in EXCLUDED_DIRS for part in lexical.parts)):
            raise ToolError("protected_path", "Use a repository-relative path; use '.' for the repository root. Absolute paths and '..' are rejected")
        if "\\" in value:
            raise ToolError("invalid_path", "Use forward slashes in repository paths")
        current = self.workspace.root
        for part in lexical.parts:
            current = current / part
            if current.is_symlink():
                raise ToolError("symlink_path", "Symlink paths are not available to evidence tools")
        resolved = self.workspace.resolve(value)
        if resolved == self.workspace.root and not root_allowed:
            raise ToolError("invalid_path", "A file path is required")
        return resolved

    def _bytes(self, path: Path) -> bytes:
        if not path.is_file():
            raise ToolError("not_file", "Expected an existing regular file")
        with path.open("rb") as stream:
            data = stream.read(self.workspace.config.max_file_bytes + 1)
        if len(data) > self.workspace.config.max_file_bytes:
            raise ToolError("file_too_large", "File exceeds the configured byte limit")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ToolError("binary_file", "Only UTF-8 source files can be edited") from exc
        if any(ord(char) < 32 and char not in "\t\r\n\f" for char in text):
            raise ToolError("binary_file", "Only UTF-8 source files can be edited")
        return data

    def register_presented(self, sources: list[dict]) -> dict:
        """Replace edit authority with exact, hash-matching last-request views.

        Content can be a raw excerpt or an existing numbered read result. A
        caller-supplied full-file hash is checked, never silently refreshed.
        Rejected cards do not authorize any bytes; the caller can reread them.
        """
        self._presented = {}
        rejected = []
        accepted = 0
        if not isinstance(sources, list):
            raise ValueError("Presented sources must be a list")
        for source in sources:
            label = source.get("path", "") if isinstance(source, dict) else ""
            try:
                if not isinstance(source, dict):
                    raise ToolError("invalid_source", "Source views must be dictionaries")
                path = self._path(label)
                data = self._bytes(path)
                digest = _digest(data)
                expected = source.get("source_sha256", source.get("sha256"))
                if expected is not None and expected != digest:
                    raise ToolError("stale_source", "Presented source hash no longer matches the file")
                start, end = source.get("start_line", 1), source.get("end_line")
                content = source.get("content", source.get("text"))
                if (type(start) is not int or start < 1 or type(end) is not int or end < start
                        or not isinstance(content, str) or source.get("partial_line")):
                    raise ToolError("invalid_source", "Present exact, complete source lines and their range")
                lines = data.decode("utf-8").splitlines(keepends=True)
                if end > len(lines):
                    raise ToolError("invalid_source", "Presented range exceeds the actual file")
                shown = content.splitlines()
                expected_lines = [line.rstrip("\r\n") for line in lines[start - 1:end]]
                numbered = [f"{index}: {line}" for index, line in zip(range(start, end + 1), expected_lines)]
                if shown != expected_lines and shown != numbered:
                    raise ToolError("source_mismatch", "Presented text does not match the actual source range")
                key = self.workspace.relative(path)
                entry = self._presented.setdefault(key, {"sha256": digest, "ranges": []})
                entry["ranges"].append((sum(map(len, lines[:start - 1])), sum(map(len, lines[:end]))))
                accepted += 1
            except (ToolError, OSError) as exc:
                rejected.append({"path": label, "code": getattr(exc, "code", "io_error")})
        return {"accepted": accepted, "rejected": rejected}

    def definitions(self) -> list[dict]:
        definitions = self.registry.definitions(enabled_tools=set(_READ_TOOLS), lean=False)
        for item in definitions:
            item["function"]["parameters"]["properties"]["path"]["description"] = (
                "Repository-relative path. Use '.' for the repository root when listing/searching. Absolute paths are rejected.")
            if item["function"]["name"] == "list_files":
                item["function"]["description"] = (
                    "List an existing directory, never a file. Use read_file to inspect file contents; "
                    "next_offset pages directory entries.")
                item["function"]["parameters"]["properties"]["path"]["description"] = (
                    "Repository-relative directory, such as '.' or 'src'. For a file path, use read_file.")
            if item["function"]["name"] in {"read_file", "search_code"}:
                item["function"]["parameters"]["properties"]["question"] = {
                    "type": "string", "maxLength": 500,
                    "description": "Optional current information need, used to select connected evidence."}
        definitions.append({"type": "function", "function": {
            "name": "apply_edits", "description": (
                "Apply a validated patch batch. For existing files, copy old_text exactly from fresh Requested source "
                "and implement the correction in new_text. Each item is {path,old_text,new_text} for replacement "
                "or {path,content} for a new file; do not mix these shapes. "
                "All edits are validated together before writing."),
            "parameters": {"type": "object", "additionalProperties": False, "required": ["edits"],
                "properties": {"edits": {"type": "array", "minItems": 1, "maxItems": _MAX_BATCH,
                    "items": {"oneOf": [
                        {"type": "object", "additionalProperties": False,
                         "required": ["path", "old_text", "new_text"],
                         "properties": {"path": {"type": "string"},
                                        "old_text": {"type": "string", "minLength": 1},
                                        "new_text": {"type": "string"}}},
                        {"type": "object", "additionalProperties": False,
                         "required": ["path", "content"],
                         "properties": {"path": {"type": "string"}, "content": {"type": "string"}}}]}}}}}})
        if self.commands and self.workspace.config.allow_tests:
            definitions.append({"type": "function", "function": {
                "name": "run_check", "description": "Run one configured trusted check by name; no command arguments are accepted.",
                "parameters": {"type": "object", "additionalProperties": False, "required": ["name"],
                    "properties": {"name": {"type": "string", "enum": sorted(self.commands)}}}}})
        return definitions

    def execute(self, name: str, arguments, *, deadline: float, visible_sources: list[dict] | None = None) -> dict:
        captured = {"max_bytes": _CAPTURE_BYTES}
        self._anchor_recoveries = []
        token = TOOL_CAPTURE.set(captured)
        try:
            try:
                if type(deadline) not in (int, float) or not math.isfinite(deadline):
                    raise ToolError("invalid_arguments", "A finite monotonic task deadline is required")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ToolError("deadline_exceeded", "Task deadline exceeded before tool execution")
                if visible_sources is not None:
                    self.register_presented(visible_sources)
                if isinstance(arguments, str):
                    if len(arguments) > _MAX_BATCH * self.workspace.config.max_file_bytes * 2:
                        raise ToolError("invalid_arguments", "Tool arguments exceed their input limit")
                    arguments = json.loads(arguments)
                if not isinstance(arguments, dict):
                    raise ToolError("invalid_arguments", "Tool arguments must be an object")
                if name in _READ_TOOLS:
                    arguments = dict(arguments)
                    question = arguments.pop("question", None)
                    if question is not None and (name not in {"read_file", "search_code"}
                                                  or not isinstance(question, str) or len(question) > 500):
                        raise ToolError("invalid_arguments", "A read/search question must be a string of at most 500 characters")
                    target = self._path(arguments.get("path", "."), root_allowed=name != "read_file")
                    before = _digest(self._bytes(target)) if name == "read_file" else None
                    result = self.registry.dispatch(name, arguments, remaining_seconds=deadline - time.monotonic())
                    captured.update((self.registry.last_execution or {}).get("available_output", {}))
                    if name == "read_file" and result.get("success"):
                        after = _digest(self._bytes(self._path(arguments["path"])))
                        if before != after:
                            raise ToolError("stale_source", "File changed while it was being read; read it again")
                        result["source_sha256"] = after
                elif name == "apply_edits":
                    if arguments.keys() != {"edits"}:
                        raise ToolError("invalid_arguments", "apply_edits accepts only edits")
                    result = self._apply_edits(arguments["edits"], deadline=deadline)
                elif name == "run_check":
                    if arguments.keys() != {"name"}:
                        raise ToolError("invalid_arguments", "run_check accepts only a configured name")
                    result = self._run_check(arguments["name"], remaining=deadline - time.monotonic())
                else:
                    raise ToolError("unknown_tool", "Use one of the available evidence tools")
            except ToolError as exc:
                result = {"success": False, "error": {"code": exc.code, "message": exc.message}}
                if getattr(exc, "modified_files", None):
                    result.update(modified=True, modified_files=list(exc.modified_files))
            except (ValueError, TypeError, UnicodeError) as exc:
                result = {"success": False, "error": {"code": "invalid_arguments", "message": str(exc)}}
            except OSError as exc:
                result = {"success": False, "error": {"code": "io_error", "message": str(exc)}}
            if name == "apply_edits" and self._anchor_recoveries:
                try:
                    original_sha = _digest(json.dumps(arguments, sort_keys=True, ensure_ascii=False,
                        separators=(",", ":"), allow_nan=False).encode())
                except (ValueError, TypeError, RecursionError):
                    original_sha = None
                result["anchor_recovery"] = {"kind": "python_function_formatting",
                    "original_proposal_sha256": original_sha, "applied": result.get("success") is True,
                    "requires_fresh_presented_function": True, "recoveries": deepcopy(self._anchor_recoveries)}
            available = {key: value for key, value in captured.items() if key != "max_bytes"}
            self.last_execution = {"name": name, "arguments": deepcopy(arguments), "result": deepcopy(result),
                                   "raw_result": deepcopy(result), "available_output": available, "capture": available}
            if self.emit is not None:
                self.emit("evidence_tool", name=name, success=result.get("success", False))
            return bounded_result(result, self.observation_limit, enabled_tools=set(_READ_TOOLS))
        finally:
            TOOL_CAPTURE.reset(token)

    def _apply_edits(self, edits, *, deadline: float) -> dict:
        self._anchor_recoveries = []
        if not isinstance(edits, list) or not 1 <= len(edits) <= _MAX_BATCH:
            raise ToolError("invalid_arguments", f"edits must contain 1-{_MAX_BATCH} changes")
        files: dict[str, dict] = {}
        for edit in edits:
            if not isinstance(edit, dict) or not isinstance(edit.get("path"), str):
                raise ToolError("invalid_arguments", "Every edit needs a relative path")
            path = self._path(edit["path"])
            key = self.workspace.relative(path)
            creating = set(edit) == {"path", "content"}
            if creating:
                if not isinstance(edit["content"], str):
                    raise ToolError("invalid_arguments", "New-file content must be text")
                if path.exists() or key in files:
                    raise ToolError("already_exists", "Create-file edits cannot overwrite an existing or repeated path")
                files[key] = {"path": path, "before": None, "after": edit["content"].encode("utf-8"), "spans": []}
            else:
                if (set(edit) != {"path", "old_text", "new_text"}
                        or not isinstance(edit["old_text"], str) or not edit["old_text"]
                        or not isinstance(edit["new_text"], str)):
                    raise ToolError("invalid_arguments", "Existing-file edits need nonempty old_text and text new_text")
                if key in files and files[key]["before"] is None:
                    raise ToolError("duplicate_edit", "A file cannot be created and edited in the same batch")
                entry = files.setdefault(key, {"path": path, "before": self._bytes(path), "spans": []})
                original = entry["before"].decode("utf-8")
                authority = self._presented.get(key)
                if authority is None:
                    raise ToolError("source_not_presented", "Read and present the exact old source before editing")
                if authority["sha256"] != _digest(entry["before"]):
                    raise ToolError("stale_source", "File changed after presentation; read it again")
                old, new = edit["old_text"], edit["new_text"]
                if "\r\n" in original and original.count("\n") == original.count("\r\n"):
                    old = old.replace("\r\n", "\n").replace("\n", "\r\n")
                    new = new.replace("\r\n", "\n").replace("\n", "\r\n")
                start = original.find(old)
                if start < 0 and self.recover_python_function_anchors and path.suffix == ".py":
                    recovered = recover_function_anchor(original, old, new, authority["ranges"])
                    if recovered is not None:
                        self._anchor_recoveries.append({"path": key, "function": recovered.function,
                            "start_line": recovered.first_line, "end_line": recovered.last_line,
                            "old_anchor_format_only": True, "proposed_old_sha256": _digest(old.encode()),
                            "canonical_old_sha256": _digest(recovered.old_text.encode()),
                            "proposed_new_sha256": _digest(new.encode()),
                            "normalized_new_sha256": _digest(recovered.new_text.encode())})
                        old, new = recovered.old_text, recovered.new_text
                        start = original.find(old)
                if start < 0:
                    raise ToolError("no_match", "old_text does not match the current file")
                if original.find(old, start + 1) >= 0:
                    raise ToolError("ambiguous_match", "old_text occurs more than once; include more context")
                end = start + len(old)
                cursor = start
                for first, last in sorted(authority["ranges"]):
                    if first <= cursor < last:
                        cursor = max(cursor, last)
                if cursor < end:
                    raise ToolError("source_not_presented", "old_text includes source omitted from the latest request")
                # A model may repeat an already-correct line alongside a real
                # fix. Validate its source normally, but reserve no mutation
                # span for an identical replacement.
                if old != new:
                    if any(start < last and end > first for first, last, _ in entry["spans"]):
                        raise ToolError("overlapping_edits", "Batch replacements must not overlap or duplicate one another")
                    entry["spans"].append((start, end, new))
            if time.monotonic() >= deadline:
                raise ToolError("deadline_exceeded", "Task deadline exceeded during patch validation")
        for entry in files.values():
            if entry["before"] is not None:
                updated = entry["before"].decode("utf-8")
                for start, end, replacement in sorted(entry["spans"], reverse=True):
                    updated = updated[:start] + replacement + updated[end:]
                entry["after"] = updated.encode("utf-8")
            if len(entry["after"]) > self.workspace.config.max_file_bytes:
                raise ToolError("file_too_large", "Patched file exceeds the configured byte limit")
        changed = {key: entry for key, entry in files.items()
                   if entry["before"] is None or entry["after"] != entry["before"]}
        if not changed:
            raise ToolError("no_change", "Patch batch must change the source")
        syntax_validation = (self._validate_python_edits(changed, deadline=deadline)
                             if self.workspace.config.validate_python_edits or self._anchor_recoveries else None)
        self._commit(files, deadline=deadline)
        self._presented = {}
        paths = list(changed)
        return {"success": True, "modified": True, "modified_files": paths,
                "changes": [{"path": key, "created": entry["before"] is None,
                             "before_sha256": _digest(entry["before"]) if entry["before"] is not None else None,
                             "after_sha256": _digest(entry["after"])} for key, entry in changed.items()],
                **({"python_syntax_validation": syntax_validation} if syntax_validation is not None else {})}

    def _validate_python_edits(self, changed: dict[str, dict], *, deadline: float) -> dict:
        """Reject new syntax errors and adjacent shadowing defs without executing code.

        Only changed .py paths participate. A file that already fails to compile
        remains editable: the guard reports that limitation instead of claiming
        the candidate is valid or requiring an unrelated preexisting repair.
        Source authority, exact matching and batch overlap checks happen first.
        """
        checked, skipped = [], []
        for key, entry in changed.items():
            if time.monotonic() >= deadline:
                raise ToolError("deadline_exceeded", "Task deadline exceeded during Python syntax validation")
            if entry["path"].suffix != ".py":
                continue
            before = entry["before"]
            if before is not None:
                try:
                    compile(before, key, "exec", dont_inherit=True)
                except SyntaxError:
                    skipped.append(key)
                    continue
                except (RecursionError, MemoryError) as error:
                    raise ToolError("python_syntax_validation_limit",
                        f"Python compiler limit while checking the original {key}; no files were written.") from error
            try:
                # compile only constructs a code object. It neither evaluates
                # top-level statements nor creates .pyc files or import caches.
                compile(entry["after"], key, "exec", dont_inherit=True)
            except SyntaxError as error:
                location = f"line {error.lineno or 1}, column {error.offset or 1}"
                message = str(error.msg)[:240]
                raise ToolError("python_syntax_error",
                    f"Python syntax regression in {key} at {location}: {message}. "
                    "No files were written; correct the batch before retrying.") from error
            except (RecursionError, MemoryError) as error:
                raise ToolError("python_syntax_validation_limit",
                    f"Python compiler limit while checking candidate {key}; no files were written.") from error
            from patchrook_agent.python_duplicates import new_duplicate_function
            duplicate = new_duplicate_function(before, entry["after"], deadline=deadline)
            if duplicate is not None:
                raise ToolError("python_duplicate_function",
                    f"Rejected candidate would leave {duplicate['name']} shadowed by a later same-argument "
                    f"definition in {key} (candidate lines {duplicate['first_line']} and {duplicate['duplicate_line']}). "
                    "No files were written. Replace the existing definition instead of adding another.")
            checked.append(key)
        return {"mode": "reject_new_errors", "checked_paths": checked,
                "preexisting_invalid_paths": skipped}

    def _commit(self, files: dict[str, dict], *, deadline: float) -> None:
        changed = {key: entry for key, entry in files.items()
                   if entry["before"] is None or entry["after"] != entry["before"]}
        unchanged = {key: entry for key, entry in files.items() if key not in changed}
        staged: dict[str, Path] = {}
        committed: list[str] = []
        created_dirs: list[Path] = []

        def check_unchanged():
            # Unchanged edits still assert a current source snapshot. Keep
            # these dependencies fresh without replacing their files.
            for key, entry in unchanged.items():
                if self._bytes(self._path(key)) != entry["before"]:
                    raise ToolError("stale_source", "Unchanged source changed during patch commit")

        try:
            # Prepare all output before replacing any source file.
            for key, entry in changed.items():
                path = entry["path"]
                missing = []
                parent = path.parent
                while parent != self.workspace.root and not parent.exists():
                    missing.append(parent)
                    parent = parent.parent
                for directory in reversed(missing):
                    directory.mkdir()
                    created_dirs.append(directory)
                self._path(key)
                fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".patchrook-evidence-")
                staged[key] = Path(temporary)
                with os.fdopen(fd, "wb") as stream:
                    stream.write(entry["after"])
                mode = (path.stat().st_mode & 0o777) if entry["before"] is not None else 0o644
                staged[key].chmod(mode)
            if time.monotonic() >= deadline:
                raise ToolError("deadline_exceeded", "Task deadline exceeded before patch commit")
            for key, entry in files.items():
                path = self._path(key)
                if entry["before"] is None:
                    if path.exists():
                        raise ToolError("stale_source", "A new-file destination appeared during validation")
                elif self._bytes(path) != entry["before"]:
                    raise ToolError("stale_source", "Source changed during patch validation")
            for key, entry in changed.items():
                # Recheck each destination at its commit boundary as well: an
                # external edit between two replacements must trigger rollback.
                check_unchanged()
                current = self._path(key)
                if entry["before"] is not None and self._bytes(current) != entry["before"]:
                    raise ToolError("stale_source", "Source changed before its patch was committed")
                if entry["before"] is None:
                    # link is an exclusive create, so a racing new file is never overwritten.
                    os.link(staged[key], entry["path"])
                    committed.append(key)
                    staged[key].unlink()
                else:
                    os.replace(staged[key], entry["path"])
                    committed.append(key)
            check_unchanged()
        except (OSError, ToolError):
            failed_rollbacks = []
            for key in reversed(committed):
                entry = files[key]
                try:
                    current = self._path(key)
                    if self._bytes(current) != entry["after"]:
                        raise ToolError("stale_source", "An external edit prevents safe source restoration")
                    if entry["before"] is None:
                        current.unlink()
                    else:
                        _replace_bytes(current, entry["before"])
                except (OSError, ToolError):
                    failed_rollbacks.append(key)
            if failed_rollbacks:
                self._presented = {}
                error = ToolError("rollback_failed", "Patch failed and source restoration failed for: " + ", ".join(failed_rollbacks))
                error.modified_files = failed_rollbacks
                raise error
            raise
        finally:
            for path in staged.values():
                path.unlink(missing_ok=True)
            for directory in reversed(created_dirs):
                try:
                    directory.rmdir()
                except OSError:
                    pass

    def _run_check(self, name, *, remaining: float) -> dict:
        if remaining <= 0:
            raise ToolError("deadline_exceeded", "Task deadline exceeded before check execution")
        if not self.workspace.config.allow_tests:
            raise ToolError("permission_denied", "Check execution requires allow_tests=True")
        if not isinstance(name, str) or name not in self.commands:
            raise ToolError("unknown_check", "Choose an explicitly configured check name")
        argv = list(self.commands[name])
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env.pop("PYTEST_ADDOPTS", None)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        # Existing run_bounded drains both pipes with fixed captures, starts a
        # process group, and kills descendants on timeout or completion.
        with tempfile.TemporaryDirectory(prefix="patchrook-check-pycache-") as pycache:
            # Disabling writes alone still permits reading a pre-existing .pyc
            # after a same-size, same-second edit. Use an empty external cache.
            env["PYTHONPYCACHEPREFIX"] = pycache
            result = run_bounded(argv, cwd=self.workspace.root,
                                 timeout_seconds=min(remaining, self.workspace.config.test_timeout_seconds),
                                 max_output_chars=_CAPTURE_BYTES, env=env)
        return {"success": result.exit_code == 0 and not result.timed_out,
                "kind": "command_check", "verification_kind": "configured_command", "name": name,
                "argv": argv, "status": "timed_out" if result.timed_out else "exited",
                "exit_code": result.exit_code, "timed_out": result.timed_out,
                "stdout": result.stdout, "stderr": result.stderr,
                "duration_seconds": result.duration_seconds, "truncated": result.truncated}
