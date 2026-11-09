"""Bounded, static evidence descriptions for optional decision assistance.

This index supplies navigation hints only. Source still has to pass through the
normal read tools before it can authorize an edit. No repository code executes.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
from itertools import islice
from pathlib import Path
import re

from patchrook_agent.coding import _TASK_PATH, _candidate, _inventory, _is_test
from patchrook_agent.tools.base import ToolError, Workspace


MAX_INDEX_FILES = 128
MAX_FILE_BYTES = 65536
MAX_INDEX_BYTES = 524288
MAX_AST_NODES = 4096
_CONFIG_SUFFIXES = {".toml", ".ini", ".cfg", ".yaml", ".yml", ".json"}
_PATH_LITERAL = re.compile(r"^[\w./-]+\.(?:py|sql|toml|ini|cfg|yaml|yml|json)$")
_SQL_REFERENCE = re.compile(r"\bREFERENCES\s+([\w.]+)", re.IGNORECASE)


@dataclass(frozen=True)
class EvidencePlan:
    initial_paths: tuple[str, ...]
    related_paths: tuple[str, ...]
    descriptions: dict[str, str]
    relationships: tuple[str, ...]
    fingerprint: str

    def ordered_paths(self, initial_ranking=(), related_ranking=()) -> list[str]:
        """Rank only known locations, retaining all candidates and their tails."""
        def ordered(paths, ranking):
            result = []
            for path in (*ranking, *paths):
                if path in paths and path not in result:
                    result.append(path)
            return result
        return ordered(self.initial_paths, initial_ranking) + ordered(self.related_paths, related_ranking)


@dataclass
class _Record:
    path: str
    text: str
    tree: ast.AST | None
    nodes: list[ast.AST]
    imports: list[tuple[str, int]]
    symbols: list[str]


def _resolve_imports(path: str, nodes: list[ast.AST], known: set[str]) -> list[tuple[str, int]]:
    parent = Path(path).parent
    roots = (Path("."), Path("src"), parent)
    result = []
    for node in nodes:
        bases = []
        if isinstance(node, ast.Import):
            for alias in node.names[:16]:
                bases.extend(root / alias.name.replace(".", "/") for root in roots)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                root = parent
                for _ in range(min(node.level - 1, len(parent.parts) + 1)):
                    root = root.parent
                module_bases = [root / (node.module or "").replace(".", "/")]
            else:
                module_bases = [root / (node.module or "").replace(".", "/") for root in roots]
            for base in module_bases:
                bases.append(base)
                bases.extend(base / alias.name for alias in node.names[:16] if alias.name != "*")
        for base in bases:
            targets = ([base.with_suffix(".py")] if base.name else []) + [base / "__init__.py"]
            for target in targets:
                relative = target.as_posix()
                pair = (relative, node.lineno)
                if relative in known and relative != path and pair not in result:
                    result.append(pair)
                    if len(result) >= 32:
                        return result
    return result


def _read_record(workspace: Workspace, path: str, remaining_bytes: int) -> _Record | None:
    try:
        resolved = workspace.require_file(path)
        size = resolved.stat().st_size
        if size > min(MAX_FILE_BYTES, remaining_bytes, workspace.config.max_file_bytes):
            return None
        with resolved.open("rb") as handle:
            data = handle.read(min(MAX_FILE_BYTES, remaining_bytes) + 1)
        if len(data) > min(MAX_FILE_BYTES, remaining_bytes):
            return None
        text = data.decode("utf-8")
    except (OSError, UnicodeError, ToolError, ValueError):
        return None
    tree = None
    nodes = []
    if path.endswith((".py", ".pyi")):
        try:
            tree = ast.parse(text)
            nodes = list(islice(ast.walk(tree), MAX_AST_NODES))
        except (SyntaxError, ValueError, RecursionError):
            pass
    symbols = [f"{node.name}:{node.lineno}" for node in nodes
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))][:8]
    return _Record(path, text, tree, nodes, [], symbols)


def _description(record: _Record | None, path: str, reason: str = "") -> str:
    kind = "tests" if _is_test(path) else "configuration" if Path(path).suffix in _CONFIG_SUFFIXES else "source"
    parts = [path, kind]
    if reason:
        parts.append(reason)
    if record and record.symbols:
        parts.append("symbols " + ", ".join(record.symbols))
    if record and record.imports:
        parts.append("imports " + ", ".join(target for target, _ in record.imports[:4]))
    return "; ".join(parts)[:320]


def build_evidence_plan(workspace: Workspace, task: str, source_paths: list[str], *, limit: int = 24) -> EvidencePlan:
    """Build one reusable static snapshot; preserve every existing discovery path.

    At most 128 small files and 512 KiB are parsed, with bounded AST walks. The
    original source list is never filtered by relevance scores. Additional
    callers, test configuration and referenced helpers only fill spare slots.
    Decisions reorder the resulting two groups of at most twelve locations.
    """
    if type(limit) is not int or not 1 <= limit <= 24:
        raise ValueError("evidence limit must be between 1 and 24")
    if not isinstance(task, str) or not isinstance(source_paths, list) or any(not isinstance(path, str) for path in source_paths):
        raise ValueError("task and source paths must be strings")
    base = list(dict.fromkeys(source_paths))
    if len(base) > limit:
        raise ValueError("evidence limit cannot discard existing source paths")
    # Validate paths, but fail rather than silently dropping required evidence.
    if any(_candidate(workspace, path) != path for path in base):
        raise ValueError("source paths must refer to existing workspace files")
    inventory = _inventory(workspace, limit)
    words = set(re.findall(r"[a-z][a-z0-9_]{2,}", task[:16000].lower()))
    def relevant(path):
        return len(words & set(re.findall(r"[a-z][a-z0-9_]{2,}", path.lower())))
    inventory = sorted(inventory, key=lambda path: (-relevant(path), _is_test(path), path))
    # Root test configuration belongs to the existing pytest verification path.
    configs = []
    if any(_is_test(path) for path in base):
        for path in ("pytest.ini", "pyproject.toml", "setup.cfg"):
            if _candidate(workspace, path) == path:
                configs.append(path)
    paths = list(dict.fromkeys([*base, *configs, *inventory]))[:MAX_INDEX_FILES]
    known = set(inventory) | set(base) | set(configs)
    records = {}
    used = 0
    for path in paths:
        record = _read_record(workspace, path, MAX_INDEX_BYTES - used)
        if record is not None:
            records[path] = record
            used += len(record.text.encode("utf-8"))
    for record in records.values():
        record.imports = _resolve_imports(record.path, record.nodes, known)

    # Investigate explicit source locations or entry modules first. Imported
    # helpers and tests form the second, related-evidence decision boundary.
    explicit = set()
    for match in _TASK_PATH.finditer(task[:16000]):
        raw = match.group(1).rstrip(".")
        matches = [path for path in base if path == raw or ("/" not in raw and Path(path).name == raw)]
        if len(matches) == 1:
            explicit.add(matches[0])
    # Test modules point into the production graph to exercise its entry
    # functions. Those edges do not turn the entry into a production helper.
    # In particular, pytest's conventional conftest module is test setup even
    # when it lives at the repository root rather than inside tests/.
    production = {path for path in base if not _is_test(path)
                  and Path(path).name != "conftest.py"
                  and Path(path).suffix not in _CONFIG_SUFFIXES}
    imported = {target for path in production if path in records
                for target, _ in records[path].imports}
    focused = [path for path in base if not _is_test(path)
               and Path(path).suffix not in _CONFIG_SUFFIXES
               and (path in explicit if explicit else path in production and path not in imported)]
    if not focused and base:
        focused = [next((path for path in base if path in production), base[0])]
    initial = focused[:12]
    remaining = [path for path in base if path not in initial]
    # A full existing selection still needs two bounded groups. Promote only
    # enough required evidence to fit; no original location is discarded.
    while len(remaining) > 12:
        initial.append(remaining.pop(0))
    capacity = min(limit, len(initial) + 12)

    reasons = {}
    edges = []
    related = []
    def add(path, reason):
        if path not in base and path not in related and len(base) + len(related) < capacity:
            related.append(path)
            reasons[path] = reason
    def edge(value):
        if len(edges) < 16 and value not in edges:
            edges.append(value[:180])
    base_set = set(base)
    for path in base:
        record = records.get(path)
        if not record:
            continue
        local_helpers = {node.name for node in record.nodes
                         if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        for target, line in record.imports:
            add(target, f"dependency of {path}")
            edge(f"{path}:{line} imports {target}")
        # Paths come from literal references in real source, resolved only to
        # existing workspace files. This also covers external schema fixtures.
        for node in record.nodes:
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and len(node.value) <= 200 and _PATH_LITERAL.fullmatch(node.value):
                for raw in (str(Path(path).parent / node.value), node.value):
                    target = _candidate(workspace, raw)
                    if target is not None and target != path:
                        add(target, f"referenced by {path}:{node.lineno}")
                        edge(f"{path}:{node.lineno} references {target}")
                        break
            if isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
                if isinstance(node.func, ast.Name) and name in local_helpers:
                    edge(f"{path}:{node.lineno} calls local helper {name}")
                if name in {"ForeignKey", "ForeignKeyConstraint", "relationship"} and node.args:
                    value = node.args[0]
                    if isinstance(value, ast.Constant) and isinstance(value.value, str):
                        edge(f"{path}:{node.lineno} {name}({value.value[:80]!r})")
        if path.endswith(".sql"):
            for match in islice(_SQL_REFERENCE.finditer(record.text), 8):
                line = record.text.count("\n", 0, match.start()) + 1
                edge(f"{path}:{line} REFERENCES {match.group(1)}")

    # Reverse dependencies identify callers without executing imports or
    # guessing names from model output. Imported modules remain retrievable.
    for record in records.values():
        if record.path in base_set:
            continue
        for target, line in record.imports:
            if target in base_set:
                add(record.path, f"imports {target}; possible caller")
                if record.path in related:
                    edge(f"{record.path}:{line} imports {target}")
                break
    for path in configs:
        add(path, "pytest configuration")
    selected = [*base, *related]
    descriptions = {path: _description(records.get(path), path, reasons.get(path, "existing source discovery"))
                    for path in selected}
    digest = hashlib.sha256()
    digest.update(task[:16000].encode("utf-8"))
    for path in sorted(records):
        digest.update(path.encode("utf-8") + b"\0" + records[path].text.encode("utf-8") + b"\0")
    for path in selected:
        digest.update(path.encode("utf-8") + b"\0")
        try:
            stat = workspace.require_file(path).stat()
            digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode("ascii"))
        except (OSError, ToolError):
            digest.update(b"unavailable")
    return EvidencePlan(tuple(initial), tuple([*remaining, *related]), descriptions, tuple(edges), digest.hexdigest())
