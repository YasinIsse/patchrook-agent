"""Conservative detection of newly inserted adjacent shadowing Python defs."""

import ast
from collections import Counter
import time

from patchrook_agent.tools.base import ToolError


FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


def _plain_function(node):
    return (isinstance(node, FUNCTIONS) and not node.decorator_list
            and not node.args.defaults and not any(value is not None for value in node.args.kw_defaults)
            and node.returns is None and getattr(node, "type_comment", None) is None
            and not getattr(node, "type_params", ())
            and all(item.annotation is None for item in (
                *node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs,
                *([node.args.vararg] if node.args.vararg else []),
                *([node.args.kwarg] if node.args.kwarg else []))))


def _inspect(tree, deadline):
    for count, _ in enumerate(ast.walk(tree)):
        if time.monotonic() >= deadline:
            raise ToolError("deadline_exceeded", "Task deadline exceeded during Python duplicate validation")
        if count >= 50000:
            raise ToolError("python_syntax_validation_limit", "Python duplicate validation node limit; no files were written.")
    counts, pairs, ambiguous = Counter(), [], set()
    pending = [(tree.body, ())]
    while pending:
        body, scope = pending.pop()
        owners = Counter(node.name for node in body if isinstance(node, (*FUNCTIONS, ast.ClassDef)))
        for node in body:
            if isinstance(node, FUNCTIONS):
                counts[(scope, node.name)] += 1
                if owners[node.name] > 1:
                    ambiguous.add((*scope, "function:" + node.name))
                if _plain_function(node):
                    pending.append((node.body, (*scope, "function:" + node.name)))
            elif isinstance(node, ast.ClassDef) and not node.decorator_list and not node.bases and not node.keywords:
                if owners[node.name] > 1:
                    ambiguous.add((*scope, "class:" + node.name))
                pending.append((node.body, (*scope, "class:" + node.name)))
        for first, second in zip(body, body[1:]):
            if (_plain_function(first) and _plain_function(second)
                    and type(first) is type(second) and first.name == second.name):
                pairs.append(((scope, first.name), first, second))
    return counts, pairs, ambiguous


def new_duplicate_function(before, after, *, deadline):
    """Return a new adjacent same-argument binding; preserve existing counts.

    Decorators, inheritance/metaclasses, definition-time expressions and
    control-flow branches are deliberately outside this opt-in heuristic.
    """
    try:
        old, _, old_ambiguous = _inspect(ast.parse(before or b"", type_comments=True), deadline)
        current, pairs, current_ambiguous = _inspect(ast.parse(after, type_comments=True), deadline)
        for key, first, second in pairs:
            if time.monotonic() >= deadline:
                raise ToolError("deadline_exceeded", "Task deadline exceeded during Python duplicate validation")
            scope, name = key
            if any(scope[:depth] in old_ambiguous | current_ambiguous for depth in range(1, len(scope) + 1)):
                continue
            if current[key] > old[key] and ast.dump(first.args, include_attributes=False) == ast.dump(second.args, include_attributes=False):
                return {"name": name, "scope": ".".join(scope) or "module",
                        "first_line": first.lineno, "duplicate_line": second.lineno}
    except SyntaxError:
        # Syntax regression prevention owns parse failures and its diagnostics.
        return None
    except (RecursionError, MemoryError) as error:
        raise ToolError("python_syntax_validation_limit", "Python duplicate validation compiler limit; no files were written.") from error
    return None
