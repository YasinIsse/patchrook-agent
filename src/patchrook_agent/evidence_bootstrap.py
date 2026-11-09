"""Run configured baseline checks before spending a coding generation."""

import keyword
from pathlib import PurePosixPath
import re
import time

from patchrook_agent.model_routing import infrastructure_failure
from patchrook_agent.tools.base import ToolError


def small_task_class_ranges(task, structure, *, maximum=40):
    """Locate one complete small class explicitly named in the public task.

    This is source discovery in an already observed implementation file, not
    authority to edit it. Large declarations remain available to read_file.
    """
    names = {name for name in re.findall(r"(?<!`)`([^`\r\n]{1,128})`(?!`)", task)
             if name.isidentifier() and not keyword.iskeyword(name)}
    for item in structure.get("selectors", []):
        if item.get("kind") == "class_header" and item.get("symbol") in names:
            first, last = item["start_line"], item.get("scope_end_line", item["end_line"])
            if 1 <= last - first + 1 <= maximum:
                return [(first, last)]
    return []


def _seed_nearby_task_class(runtime, observed, deadline):
    path = observed["path"]
    first, last = observed["start_line"], observed["end_line"]
    maximum = min(40, runtime.workspace.config.max_read_lines)
    structure = runtime.index.blocks(path, deadline=deadline, line_ranges=((first, last),))
    named = {name for name in re.findall(r"(?<!`)`([^`\r\n]{1,128})`(?!`)", runtime.task)
             if name.isidentifier()}
    if any(item.get("kind") != "class_header" and item.get("symbol") in named
           and first <= item["start_line"] <= last < item["end_line"]
           for item in structure.get("selectors", [])):
        # A known whole method cannot be silently promoted from a prefix.
        # Preserve the old bounded navigation behavior and label it explicitly.
        runtime.correction = (f"Automatic source in {path}:{first}-{last} is only a declaration prefix. "
                              "Use read_file to acquire the remaining current body before editing omitted source.")
        runtime.session.emit("task_context_bootstrap_prefix", runtime.state.steps,
                             path=path, reason="complete_named_function_exceeds_automatic_limit",
                             edit_authorization=False)
        return False
    for start, end in small_task_class_ranges(runtime.task, structure, maximum=maximum):
        combined = min(first, start), max(last, end)
        if first <= start <= end <= last or combined[1] - combined[0] + 1 > maximum:
            continue
        # Retain the primary declaration in the same latest direct view. This
        # bounded reread cannot lose it when repair selects the latest view.
        result = runtime._perform_action(runtime.state, {"name": "read_file", "arguments": {
            "path": path, "start_line": combined[0], "end_line": combined[1]}}, deadline, automatic=True)
        runtime.session.emit("task_context_bootstrap", runtime.state.steps, path=path,
                             success=bool(result.get("success")), edit_authorization=False)
        if not result.get("success"):
            runtime.state.stop_reason = ("deadline" if result.get("error", {}).get("code") == "deadline_exceeded"
                                         else "initial_symbol_environment_failure")
            return True
    return False


def run_initial_checks(runtime, deadline):
    """Return True when an environment failure prevents safe coding startup.

    Ordinary assertion failures are fresh task evidence, not failed repair
    attempts. The normal tool boundary retains their output and revision-bound
    check records; initial checks never authorize source edits.
    """
    if not runtime.config.evidence_initial_checks or not runtime.tools.commands:
        return False
    state = runtime.state
    if "run_check" not in runtime.enabled:
        state.stop_reason = "initial_check_environment_failure"
        state.recovery_notes.append("Initial checks require the enabled run_check tool.")
        runtime.session.emit("initial_checks_stop", state.steps, reason=state.stop_reason)
        return True
    runtime.session.emit("initial_checks_start", state.steps, checks=list(runtime.tools.commands))
    outcomes = []
    environment_failure = False
    for name in runtime.tools.commands:
        if time.monotonic() >= deadline:
            state.stop_reason = "deadline"
            runtime.session.emit("initial_checks_stop", state.steps, reason=state.stop_reason)
            return True
        result = runtime._perform_action(state, {"name": "run_check", "arguments": {"name": name}},
                                         deadline, automatic=True)
        outcomes.append({"name": name, "passed": bool(result.get("success")),
                         "exit_code": result.get("exit_code")})
        if result.get("error", {}).get("code") == "deadline_exceeded":
            state.stop_reason = "deadline"
            runtime.session.emit("initial_checks_stop", state.steps, reason=state.stop_reason)
            return True
        environment_failure = environment_failure or (
            not result.get("success") and infrastructure_failure(result))
    runtime._update_state(state)
    if environment_failure:
        state.stop_reason = "initial_check_environment_failure"
        runtime.session.emit("initial_checks_stop", state.steps, reason=state.stop_reason, checks=outcomes)
        return True
    # Baseline failures describe requested work; they must not consume the
    # coder's consecutive action-failure allowance before its first action.
    runtime.failures = 0
    runtime.session.emit("initial_checks_complete", state.steps, checks=outcomes)
    return False


def _seed_public_api(runtime, declaration, deadline):
    """Read an existing package API only for an explicit export request."""
    if not re.search(r"\b(?:exports?|exported|exporting)\b|\btop[ -]+level\b", runtime.task, re.IGNORECASE):
        return False
    try:
        source = runtime.tools._path(declaration.get("path"))
        if not source.is_file():
            return False
        relative = runtime.workspace.relative(source)
        for parent in PurePosixPath(relative).parents:
            candidate = (parent / "__init__.py").as_posix()
            if candidate == relative:
                return False
            api = runtime.tools._path(candidate)
            if not api.is_file():
                continue
            if time.monotonic() >= deadline:
                runtime.state.stop_reason = "deadline"
                runtime.session.emit("task_public_api_bootstrap_stop", runtime.state.steps, reason="deadline")
                return True
            action = {"name": "read_file", "arguments": {
                "path": candidate, "start_line": 1, "end_line": runtime.workspace.config.max_read_lines}}
            result = runtime._perform_action(runtime.state, action, deadline, automatic=True)
            runtime.session.emit("task_public_api_bootstrap", runtime.state.steps, path=candidate,
                                 success=bool(result.get("success")), edit_authorization=False)
            if not result.get("success"):
                runtime.state.stop_reason = ("deadline" if result.get("error", {}).get("code") == "deadline_exceeded"
                                             else "initial_symbol_environment_failure")
                return True
            return False
    except (OSError, ToolError):
        # Missing or guarded paths cannot create an API read or authority.
        return False
    return False


def seed_task_symbols(runtime, deadline):
    """Present an existing Python declaration explicitly named by the task.

    Searches and reads use the ordinary tools. Symbol mentions never authorize
    edits, and no repository code is executed to resolve a declaration.
    """
    if not getattr(runtime.config, "evidence_seed_symbols", False):
        return False
    state = runtime.state
    # Fresh-run bootstrap only. Resuming must invalidate and re-observe source
    # through the normal resume path, rather than replaying automatic actions.
    if (state.steps or runtime.attempts or runtime.wall_base
            or not {"search_code", "read_file"}.issubset(runtime.enabled)):
        return False
    python_paths = [path for path in runtime.index.paths if path.endswith((".py", ".pyi"))]
    if not python_paths:
        return False
    symbols = []
    for quoted in re.findall(r"(?<!`)`([^`\r\n]{1,128})`(?!`)", runtime.task):
        if quoted.isidentifier() and not keyword.iskeyword(quoted) and quoted not in symbols:
            symbols.append(quoted)
            if len(symbols) == 6:
                break
    if not symbols:
        return False
    scope = "."
    if any(path.startswith("src/") for path in python_paths):
        try:
            if runtime.tools._path("src", root_allowed=True).is_dir():
                scope = "src"
        except (OSError, ToolError):
            return False
    runtime.session.emit("task_symbol_bootstrap_start", state.steps, symbols=symbols,
                         scope=scope, edit_authorization=False)
    for symbol in symbols:
        if time.monotonic() >= deadline:
            state.stop_reason = "deadline"
            runtime.session.emit("task_symbol_bootstrap_stop", state.steps, reason=state.stop_reason)
            return True
        proposed = {"name": "search_code", "arguments": {
            "path": scope, "pattern": symbol, "mode": "literal", "max_results": 1}}
        result = runtime._perform_action(state, proposed, deadline, automatic=True)
        if not result.get("success"):
            state.stop_reason = "initial_symbol_environment_failure"
            runtime.session.emit("task_symbol_bootstrap_stop", state.steps, reason=state.stop_reason,
                                 symbol=symbol, error=result.get("error"))
            return True
        failures_before_recovery = state.tool_failures
        recovered, observed = runtime._recover_symbol_search(state, proposed, result, deadline)
        if state.tool_failures > failures_before_recovery:
            state.stop_reason = "initial_symbol_environment_failure"
            runtime.session.emit("task_symbol_bootstrap_stop", state.steps, reason=state.stop_reason,
                                 symbol=symbol, error={"code": "declaration_lookup_failed"})
            return True
        if recovered["name"] != "read_file":
            declaration = r"^\s*(?:class|(?:async\s+)?def)\s+" + re.escape(symbol) + r"\b"
            definitions = [match for match in observed.get("matches", [])
                           if match.get("path", "").endswith((".py", ".pyi"))
                           and re.search(declaration, match.get("text", ""))]
            if not definitions:
                continue
            match = definitions[0]
            recovered = runtime._declaration_read(match, deadline)
            observed = runtime._perform_action(state, recovered, deadline, automatic=True)
        if not observed.get("success"):
            state.stop_reason = "initial_symbol_environment_failure"
            runtime.session.emit("task_symbol_bootstrap_stop", state.steps, reason=state.stop_reason,
                                 symbol=symbol, error=observed.get("error"))
            return True
        runtime.session.emit("task_symbol_bootstrap_complete", state.steps, symbol=symbol,
                             path=observed.get("path"), start_line=observed.get("start_line"),
                             end_line=observed.get("end_line"), edit_authorization=False)
        if _seed_nearby_task_class(runtime, observed, deadline):
            return True
        return _seed_public_api(runtime, observed, deadline)
    runtime.session.emit("task_symbol_bootstrap_complete", state.steps, symbol=None,
                         reason="no_existing_python_declaration", edit_authorization=False)
    return False
