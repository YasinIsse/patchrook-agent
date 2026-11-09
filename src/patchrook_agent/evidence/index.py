"""Whole-checkout metadata inventory and bounded, lazy source navigation."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import re
import selectors
import shutil
import subprocess
import time

from patchrook_agent.tools.base import EXCLUDED_DIRS
from .store import checked_path, digest


def _words(value):
    return set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", value.lower()))


class RepositoryIndex:
    """Index paths, not all repository contents; report every bounded omission.

    Ripgrep is run with hidden and ignored files included, then explicit excluded
    directories and out-of-repository symlinks are removed. Search therefore
    does not inherit a hidden first-N-file limit from the old repair workflow.
    """

    def __init__(self, root, *, max_file_bytes=1_048_576, time_budget=10.0,
                 inventory_byte_limit=16 * 1024 * 1024):
        self.root = Path(root).resolve(strict=True)
        if not self.root.is_dir() or max_file_bytes <= 0 or time_budget <= 0 or inventory_byte_limit <= 0:
            raise ValueError("Repository index limits must be positive")
        self.max_file_bytes = max_file_bytes
        self.time_budget = time_budget
        self.inventory_byte_limit = inventory_byte_limit
        self.paths: tuple[str, ...] = ()
        self.inventory = {"paths": [], "count": 0, "truncated": False, "reasons": []}
        self._loaded = False
        self._ast_cache: dict[tuple[str, str], dict] = {}

    def _deadline(self, deadline):
        own = time.monotonic() + self.time_budget
        return min(own, deadline) if deadline is not None else own

    def _eligible(self, path):
        try:
            resolved = checked_path(self.root, path)
            return resolved.is_file()
        except (OSError, ValueError, RuntimeError):
            return False

    @staticmethod
    def _globs():
        return [value for directory in sorted(EXCLUDED_DIRS)
                for value in ("-g", "!" + directory + "/**", "-g", "!**/" + directory + "/**")]

    def _run(self, arguments, deadline, byte_limit):
        if time.monotonic() >= deadline:
            return b"", ["deadline"], None
        reasons, output = [], bytearray()
        process = subprocess.Popen(arguments, cwd=self.root, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        selector.register(process.stderr, selectors.EVENT_READ, "stderr")
        try:
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    reasons.append("deadline")
                    break
                for key, _ in selector.select(min(remaining, .1)):
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    if key.data == "stdout":
                        room = byte_limit - len(output)
                        output.extend(chunk[:max(0, room)])
                        if len(chunk) > room:
                            reasons.append("output_byte_limit")
                            break
                if reasons:
                    break
            if reasons:
                process.kill()
            process.wait(timeout=max(.1, min(1, deadline - time.monotonic())))
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            selector.close()
            process.stdout.close()
            process.stderr.close()
        return bytes(output), reasons, process.returncode

    def refresh(self, deadline=None):
        end = self._deadline(deadline)
        reasons, paths = [], []
        if shutil.which("rg"):
            raw, reasons, code = self._run(["rg", "--files", "--hidden", "--no-ignore", "-0",
                                             *self._globs(), "."], end, self.inventory_byte_limit)
            # A partial final record is never interpreted as a real path.
            records = raw.split(b"\0")[:-1]
            if code not in (0, 1) and not reasons:
                reasons.append("inventory_process_failed")
            for record in records:
                if time.monotonic() >= end:
                    reasons.append("deadline")
                    break
                try:
                    path = record.decode("utf-8").removeprefix("./")
                except UnicodeDecodeError:
                    reasons.append("non_utf8_path")
                    continue
                if self._eligible(path):
                    paths.append(path)
        else:
            used = 0
            for parent, directories, files in os.walk(self.root, followlinks=False):
                directories[:] = sorted(name for name in directories if name not in EXCLUDED_DIRS
                                        and not (Path(parent) / name).is_symlink())
                for name in sorted(files):
                    if time.monotonic() >= end:
                        reasons.append("deadline")
                        break
                    path = (Path(parent) / name).relative_to(self.root).as_posix()
                    used += len(path.encode()) + 1
                    if used > self.inventory_byte_limit:
                        reasons.append("output_byte_limit")
                        break
                    if self._eligible(path):
                        paths.append(path)
                if reasons:
                    break
        self.paths = tuple(sorted(set(paths)))
        self._loaded = True
        self.inventory = {"paths": list(self.paths), "count": len(self.paths),
                          "truncated": bool(reasons), "reasons": sorted(set(reasons)),
                          "includes_hidden": True, "includes_ignored": True,
                          "excluded_directories": sorted(EXCLUDED_DIRS)}
        return dict(self.inventory)

    def search(self, query, limit=30, deadline=None):
        if not isinstance(query, str) or not query.strip() or type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("Search requires a query and a result limit between one and 1000")
        end = self._deadline(deadline)
        if not self._loaded:
            self.refresh(deadline=end)
        terms = sorted(_words(query[:512]), key=lambda word: (-len(word), word))[:8]
        if not terms:
            terms = [query[:128]]
        matches, reasons = [], list(self.inventory["reasons"])
        if shutil.which("rg"):
            raw, extra, code = self._run(["rg", "--json", "--hidden", "--no-ignore", "-i",
                                           "--max-filesize", str(self.max_file_bytes), *self._globs(),
                                           "-e", "|".join(re.escape(term) for term in terms), "."],
                                          end, min(self.inventory_byte_limit, 1024 * 1024))
            reasons.extend(extra)
            if code not in (0, 1) and not extra:
                reasons.append("search_process_failed")
            for line in raw.splitlines():
                try:
                    event = json.loads(line)
                    if event.get("type") != "match":
                        continue
                    value = event["data"]
                    path = value["path"]["text"].removeprefix("./")
                    text = value["lines"]["text"]
                    # A partial/stale metadata inventory must not restrict
                    # ripgrep's whole-checkout search to its recorded prefix.
                    if not self._eligible(path):
                        continue
                    matches.append({"path": path, "line_number": value["line_number"], "text": text})
                    if len(matches) > limit:
                        reasons.append("result_limit")
                        break
                except (KeyError, ValueError, TypeError):
                    reasons.append("unreadable_search_record")
        else:
            pattern = re.compile("|".join(re.escape(term) for term in terms), re.I)
            for path in self.paths:
                if time.monotonic() >= end:
                    reasons.append("deadline")
                    break
                try:
                    with checked_path(self.root, path).open("rb") as handle:
                        raw = handle.read(self.max_file_bytes + 1)
                    if len(raw) > self.max_file_bytes or b"\0" in raw:
                        continue
                    for number, line in enumerate(raw.decode().splitlines(keepends=True), 1):
                        if pattern.search(line):
                            matches.append({"path": path, "line_number": number, "text": line})
                            if len(matches) > limit:
                                reasons.append("result_limit")
                                break
                except (OSError, ValueError, UnicodeError):
                    continue
                if len(matches) > limit:
                    break
        return {"matches": matches[:limit], "truncated": bool(reasons),
                "reasons": sorted(set(reasons)), "inventory_count": len(self.paths),
                "max_file_bytes": self.max_file_bytes, "query_terms": terms}

    def blocks(self, path, query="", deadline=None, *, line_ranges=()):
        """Return bounded complete blocks, prioritizing explicitly observed ranges.

        A global block limit may omit unrelated definitions after a successful
        AST walk. A focused block limit explicitly reports when it also omits
        a definition intersecting a requested range or its structural context.
        """
        line_ranges = tuple(line_ranges)
        if any(type(first) is not int or type(last) is not int or not 1 <= first <= last
               for first, last in line_ranges):
            raise ValueError("Block focus requires valid complete line ranges")
        end = self._deadline(deadline)
        if time.monotonic() >= end:
            return {"selectors": [], "dependencies": [], "truncated": True, "reasons": ["deadline"]}
        if not self._loaded:
            self.refresh(deadline=end)
        resolved = checked_path(self.root, path)
        with resolved.open("rb") as handle:
            raw = handle.read(self.max_file_bytes + 1)
        if len(raw) > self.max_file_bytes:
            return {"selectors": [], "dependencies": [], "truncated": True, "reasons": ["file_byte_limit"]}
        if time.monotonic() >= end:
            return {"selectors": [], "dependencies": [], "truncated": True, "reasons": ["deadline"]}
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return {"selectors": [], "dependencies": [], "truncated": True, "reasons": ["not_utf8"]}
        key = path, digest(raw)
        if key not in self._ast_cache:
            ranges, imports, reasons = [], [], []
            source_lines = text.splitlines()
            def source_end(node):
                """Include a block's trailing indented comments, never next code."""
                last = node.end_lineno
                declaration = source_lines[node.lineno - 1]
                indentation = len(declaration[:len(declaration) - len(declaration.lstrip(" \t"))].expandtabs(8))
                for number in range(last + 1, len(source_lines) + 1):
                    if time.monotonic() >= end:
                        reasons.append("deadline")
                        break
                    line = source_lines[number - 1]
                    stripped = line.lstrip(" \t")
                    if not stripped:
                        # Keep blank lines between body comments, without
                        # swallowing unrelated separator/tail whitespace.
                        continue
                    elif stripped.startswith("#"):
                        width = len(line[:len(line) - len(stripped)].expandtabs(8))
                        if width <= indentation:
                            break
                        last = number
                    else:
                        break
                return last
            try:
                tree = ast.parse(text) if path.endswith((".py", ".pyi")) else None
                for count, node in enumerate(ast.walk(tree) if tree is not None else []):
                    if count >= 50000 or time.monotonic() >= end:
                        reasons.append("ast_node_limit" if count >= 50000 else "deadline")
                        break
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        first = min([node.lineno, *[deco.lineno for deco in node.decorator_list]])
                        ranges.append({"path": path, "start_line": first, "end_line": source_end(node),
                                       "symbol": node.name})
                    elif isinstance(node, ast.ClassDef):
                        first = min([node.lineno, *[deco.lineno for deco in node.decorator_list]])
                        declarations = [min([member.lineno, *[deco.lineno for deco in member.decorator_list]])
                                        for member in node.body if isinstance(member, (
                                            ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
                        # The declaration, docstring and initial fields form a
                        # bounded structural insertion point; methods remain
                        # separate complete functions.
                        last = min(declarations) - 1 if declarations else node.end_lineno
                        ranges.append({"path": path, "start_line": first, "end_line": last,
                                       "symbol": node.name, "kind": "class_header",
                                       "scope_end_line": source_end(node)})
                    elif isinstance(node, ast.Import):
                        imports.extend((alias.name, 0, (alias.asname or alias.name.split(".")[0],))
                                       for alias in node.names)
                    elif isinstance(node, ast.ImportFrom):
                        imports.append((node.module or "", node.level,
                                        tuple(alias.asname or alias.name for alias in node.names
                                              if alias.name != "*")))
                        imports.extend(((node.module + "." if node.module else "") + alias.name,
                                        node.level, (alias.asname or alias.name,))
                                       for alias in node.names if alias.name != "*")
                if tree is not None and ranges:
                    # Keep module-level imports/constants and class context
                    # available alongside complete function bodies.
                    first_block = min(item["start_line"] for item in ranges)
                    if first_block > 1:
                        ranges.append({"path": path, "start_line": 1,
                                       "end_line": first_block - 1, "symbol": "module_header"})
            except (SyntaxError, ValueError, RecursionError):
                reasons.append("ast_unavailable")
            if not ranges and text:
                # Structural fallback preserves complete contiguous source,
                # instead of synthesizing a misleading function extraction.
                total = len(text.splitlines())
                ranges = [{"path": path, "start_line": first, "end_line": min(first + 39, total),
                           "symbol": ""} for first in range(1, total + 1, 40)]
            known, dependencies, dependency_bindings = set(self.paths), [], {}
            for module, level, bindings in imports:
                parent = Path(path).parent
                for _ in range(max(0, level - 1)):
                    parent = parent.parent
                roots = [parent] if level else [Path("."), Path("src"), Path(path).parent]
                for base in roots:
                    target = base / module.replace(".", "/")
                    for candidate in (str(target) + ".py", (target / "__init__.py").as_posix()):
                        candidate = candidate.removeprefix("./")
                        if (candidate in known or self._eligible(candidate)) and candidate != path:
                            if candidate not in dependencies:
                                dependencies.append(candidate)
                            dependency_bindings.setdefault(candidate, set()).update(bindings)
            value = {"selectors": ranges, "dependencies": dependencies,
                     "truncated": bool(reasons), "reasons": reasons,
                     "source_sha256": key[1], "_dependency_bindings": dependency_bindings}
            if "deadline" not in reasons:
                self._ast_cache[key] = value
                self._ast_cache = dict(list(self._ast_cache.items())[-128:])
        else:
            value = self._ast_cache[key]
        words = _words(query)
        lines = text.splitlines()
        def score(item):
            body = "\n".join(lines[item["start_line"] - 1:item["end_line"]])
            return len(words & _words(body)) + 4 * len(words & _words(item["symbol"]))
        def focused(item):
            if line_ranges and item["symbol"] == "module_header":
                return True
            last = item.get("scope_end_line", item["end_line"])
            return any(item["start_line"] <= end and first <= last for first, end in line_ranges)
        ordered = sorted(value["selectors"],
                         key=lambda item: (not focused(item), -score(item), item["start_line"]))
        reasons = list(value["reasons"])
        if len(ordered) > 64:
            reasons.append("block_limit")
            if any(focused(item) for item in ordered[64:]):
                reasons.append("focused_block_limit")
        if time.monotonic() >= end:
            reasons.append("deadline")
        focused_words = set()
        for item in ordered[:64]:
            if focused(item) and item["symbol"] != "module_header":
                focused_words.update(_words("\n".join(lines[item["start_line"] - 1:item["end_line"]])))
        bindings = value.get("_dependency_bindings", {})
        dependencies = sorted(value["dependencies"], key=lambda dependency:
                              -len(focused_words & _words(" ".join(bindings.get(dependency, ())))))
        public = {key: item for key, item in value.items() if key != "_dependency_bindings"}
        return {**public, "selectors": ordered[:64], "dependencies": dependencies,
                "truncated": bool(reasons), "reasons": sorted(set(reasons))}
