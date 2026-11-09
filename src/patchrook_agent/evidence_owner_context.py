"""Small optional implementation examples for an already observed stub method.

All discovery is static and repository-local. This module neither imports task
code nor manufactures edit authority; ordinary request registration still owns
source freshness and edit permission.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
from pathlib import PurePosixPath
import re
import time

from patchrook_agent.evidence import EvidenceCard
from patchrook_agent.evidence.compiler import estimate_tokens
from patchrook_agent.tools.base import ToolError


_OPTIONAL_ROLES = {"optional_dependency", "optional_owner_context"}
_MAX_NEW_TOKENS = 1536


@dataclass(frozen=True)
class OwnerContext:
    key: str
    path: str
    first: int
    last: int
    text: str
    sha256: str
    role: str
    depends_on: str | None = None

    def preview(self):
        return EvidenceCard("ev_" + "0" * 24, "source", self.text, self.path,
                            self.first, self.last, self.sha256)


def _local_nodes(node):
    pending = list(reversed(node.body))
    while pending:
        current = pending.pop()
        yield current
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        pending.extend(reversed(list(ast.iter_child_nodes(current))))


def _stub(node):
    body = node.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        body = body[1:]
    return not body or all(isinstance(item, ast.Pass)
        or isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant) and item.value.value is Ellipsis
        or isinstance(item, ast.Return) and (item.value is None or isinstance(item.value, ast.Constant) and item.value.value is None)
        or isinstance(item, ast.Raise) and (
            isinstance(item.exc, ast.Name) and item.exc.id == "NotImplementedError"
            or isinstance(item.exc, ast.Call) and isinstance(item.exc.func, ast.Name) and item.exc.func.id == "NotImplementedError")
        for item in body)


def _implementation_path(path):
    value = PurePosixPath(path)
    return (value.suffix in {".py", ".pyi"} and not any(part in {"test", "tests"} for part in value.parts)
            and value.name != "conftest.py" and not value.name.startswith("test_") and not value.name.endswith("_test.py"))


def _read(runtime, path, deadline):
    if time.monotonic() >= deadline or not _implementation_path(path):
        return None
    try:
        raw = runtime.tools._bytes(runtime.tools._path(path))
        text = raw.decode("utf-8")
        tree = ast.parse(text)
        for count, _ in enumerate(ast.walk(tree)):
            if count >= 50000 or time.monotonic() >= deadline:
                return None
        return text, tree, hashlib.sha256(raw).hexdigest()
    except (OSError, ValueError, UnicodeError, RecursionError, SyntaxError, ToolError):
        return None


def _scope(path, source, node, maximum, key, role, depends_on=None):
    first = min([node.lineno, *[item.lineno for item in node.decorator_list]])
    last = node.end_lineno
    if not 1 <= last - first + 1 <= maximum:
        return None
    text, _, sha256 = source
    body = "".join(text.splitlines(keepends=True)[first - 1:last])
    return OwnerContext(key, path, first, last, body, sha256, role, depends_on)


def _imported_callee(runtime, path, source, sibling, deadline):
    """Resolve one plain imported call, without executing imports/re-exports."""
    locals_ = {arg.arg for arg in [*sibling.args.posonlyargs, *sibling.args.args, *sibling.args.kwonlyargs]}
    locals_.update(arg.arg for arg in (sibling.args.vararg, sibling.args.kwarg) if arg is not None)
    nodes = list(_local_nodes(sibling))
    for node in nodes:
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            locals_.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            locals_.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            locals_.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            locals_.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            locals_.update(node.names)
    module_nodes = list(_local_nodes(source[1]))
    if any(isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names)
           for node in module_nodes):
        return None
    calls = sorted((node for node in nodes if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)),
                   key=lambda node: (node.lineno, node.col_offset))
    for call in calls:
        name = call.func.id
        if name in locals_:
            continue
        bindings = [(node, alias.name) for node in module_nodes if isinstance(node, ast.ImportFrom)
                    for alias in node.names if (alias.asname or alias.name) == name]
        if len(bindings) != 1:
            continue
        shadowed = any((isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)) and node.id == name)
            or isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name
            or isinstance(node, ast.Import) and any((alias.asname or alias.name.split(".")[0]) == name for alias in node.names)
            for node in module_nodes)
        if shadowed:
            continue
        imported, target_name = bindings[0]
        parent = PurePosixPath(path).parent
        if imported.level:
            for _ in range(imported.level - 1):
                parent = parent.parent
            roots = [parent]
        else:
            roots = [PurePosixPath("."), PurePosixPath("src")]
        matches = []
        for root in dict.fromkeys(roots):
            target = root / (imported.module or "").replace(".", "/")
            candidates = ([str(target) + ".py", (target / "__init__.py").as_posix()]
                          if imported.module else [(target / "__init__.py").as_posix()])
            for candidate in dict.fromkeys(value.removeprefix("./") for value in candidates):
                if candidate == path:
                    continue
                value = _read(runtime, candidate, deadline)
                if value is None:
                    continue
                definitions = [item for item in value[1].body
                               if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == target_name]
                if len(definitions) != 1 or _stub(definitions[0]):
                    continue
                plan = _scope(candidate, value, definitions[0], min(20, runtime.workspace.config.max_read_lines),
                              "callee", "optional_dependency", "sibling")
                if plan:
                    matches.append(plan)
        unique = {plan.path: plan for plan in matches}
        if len(unique) == 1:
            return next(iter(unique.values()))
    return None


def owner_context_candidates(runtime, cards, deadline, render):
    """Find at most two complete owner methods and one local imported callee."""
    names = list(dict.fromkeys(name for name in re.findall(r"(?<!`)`([^`\r\n]{1,128})`(?!`)", runtime.task)
                              if name.isidentifier() and not name.startswith("_")))
    if not names:
        return []
    required = [card for card in cards if card.kind == "source" and card.metadata.get("repair_role") not in _OPTIONAL_ROLES]
    paths = list(dict.fromkeys(card.path for card in required))
    matches = []
    for path in paths[:4]:
        source = _read(runtime, path, deadline)
        if source is None:
            continue
        views = [card for card in required if card.path == path]
        if any(card.source_sha256 != source[2] for card in views):
            raise ToolError("stale_source", "Required source changed while selecting optional owner context")
        # A nested owner can close over function-local names; optional evidence
        # does not infer those dynamic bindings from a module import.
        for owner in (node for node in source[1].body if isinstance(node, ast.ClassDef)):
            methods = [node for node in owner.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
            for focus in methods:
                if (focus.name in names and _stub(focus) and any(
                        card.start_line <= focus.end_lineno and focus.lineno <= card.end_line for card in views)):
                    matches.append((names.index(focus.name), path, source, owner, focus, methods))
    if not matches:
        return []
    first_name = min(item[0] for item in matches)
    matches = [item for item in matches if item[0] == first_name]
    if len({(path, owner.lineno) for _, path, _, owner, _, _ in matches}) != 1:
        return []
    _, path, source, _, focus, methods = matches[0]
    by_name = {}
    for method in methods:
        by_name.setdefault(method.name, []).append(method)
    maximum = min(40, runtime.workspace.config.max_read_lines)
    base = None
    for index in range(1, len(focus.name.split("_"))):
        stem = "_".join(focus.name.split("_")[index:])
        values = by_name.get(stem, [])
        if len(values) == 1 and not stem.startswith("_") and not _stub(values[0]):
            base = _scope(path, source, values[0], maximum, "base", "optional_owner_context")
            break
    if base is None:
        return []
    words = set(re.findall(r"[a-z]{4,}", runtime.task.lower())) - {stem}
    siblings = []
    for name, values in by_name.items():
        if len(values) != 1 or not name.startswith(stem + "_") or _stub(values[0]):
            continue
        plan = _scope(path, source, values[0], maximum, "sibling", "optional_owner_context", "base")
        if plan:
            score = sum(word in name.lower() for word in words)
            cost = estimate_tokens([render(plan.preview())])
            siblings.append((-score, cost, name, values[0], plan))
    result = [base]
    if siblings:
        _, _, _, sibling, plan = min(siblings, key=lambda item: item[:3])
        result.append(plan)
        imported = _imported_callee(runtime, path, source, sibling, deadline)
        if imported:
            result.append(imported)
    return result


def enrich_owner_context(runtime, cards, messages, deadline, *, render, input_limit,
                         native_cost=0, correction=0, optional_limit=None):
    """Admit whole fresh optional ranges against the actual complete packet."""
    required = [card for card in cards if card.metadata.get("repair_role") not in _OPTIONAL_ROLES]
    required_messages = [*messages[:2], *[render(card) for card in required], messages[-1]]
    required_cost = estimate_tokens(required_messages) + native_cost + correction
    original_cost = estimate_tokens(messages) + native_cost + correction
    required_paths = {card.path for card in required}
    admitted, additions = set(), []
    for plan in owner_context_candidates(runtime, cards, deadline, render):
        if time.monotonic() >= deadline:
            break
        if plan.depends_on and plan.depends_on not in admitted:
            continue
        if any(card.path == plan.path and card.source_sha256 == plan.sha256
               and card.start_line <= plan.first <= plan.last <= card.end_line for card in cards):
            admitted.add(plan.key)
            continue
        proposed = [*messages[:-1], render(plan.preview()), messages[-1]]
        cost = estimate_tokens(proposed) + native_cost + correction
        if (cost > input_limit or cost - original_cost > _MAX_NEW_TOKENS
                or optional_limit is not None and cost - required_cost > optional_limit):
            continue
        try:
            # Reuse the store's guarded byte-read boundary, checking the plan
            # before recording anything. Unavailable optional imports cannot
            # leave a partial source card or block an otherwise valid repair.
            raw, text = runtime.evidence._read(plan.path, deadline)
            body = "".join(text.splitlines(keepends=True)[plan.first - 1:plan.last])
            if hashlib.sha256(raw).hexdigest() != plan.sha256 or body != plan.text:
                raise ToolError("stale_source", "Optional owner implementation changed before presentation")
        except (OSError, ValueError, UnicodeError, TimeoutError, ToolError) as error:
            if plan.path in required_paths:
                raise ToolError("stale_source", "Required source changed while admitting owner context") from error
            runtime.session.emit("repair_dependency_skipped", getattr(runtime.state, "steps", 0),
                                 path=plan.path, reason="optional_owner_context_unavailable")
            continue
        card = runtime.evidence._add("source", body, revision=runtime.revision, path=plan.path,
            start_line=plan.first, end_line=plan.last, source_sha256=plan.sha256,
            metadata={"origin": "repair_packet", "repair_role": plan.role,
                      "context_reason": "owner_implementation", "context_key": plan.key})
        messages = [*messages[:-1], render(card), messages[-1]]
        additions.append(card)
        admitted.add(plan.key)
    return [*cards, *additions], messages
