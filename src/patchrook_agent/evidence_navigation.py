"""Bounded navigation facts derived from observed tool results, never edit rights."""

from copy import deepcopy
import ast
import hashlib
import json
from pathlib import PurePosixPath
import re

from patchrook_agent.evidence.store import checked_path
from patchrook_agent.tools.base import EXCLUDED_DIRS


_HASH = re.compile(r"[0-9a-f]{64}\Z")
_LIMIT = 16


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _path(value):
    if not isinstance(value, str) or not value or len(value) > 4096 or "\0" in value or "\\" in value:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and ".." not in path.parts and not any(p in EXCLUDED_DIRS for p in path.parts)


def _reexport_symbols(text):
    """Recognize imports plus literal __all__, without executing repository code."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return None
    imported = []
    has_import = False
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            has_import = True
            imported.extend(alias.name for alias in node.names if alias.name != "*" and len(alias.name) <= 256)
            continue
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == "__all__":
                try:
                    names = ast.literal_eval(node.value)
                except (ValueError, TypeError, SyntaxError, RecursionError):
                    return None
                if isinstance(names, (list, tuple)) and all(isinstance(name, str) for name in names):
                    continue
        return None
    return list(dict.fromkeys(imported))[:16] if has_import else None


class NavigationLedger:
    """Remember completed inspections independently of the two-message window.

    A source card may describe an import-only file only when the *entire* file
    was read and its frozen hash still matches RepositoryIndex's current view.
    Import targets are navigation metadata, not presented source or authority.
    """

    def __init__(self):
        self.reads = []
        self.searches = []
        self.listings = []
        self.candidates = []
        self.facts = []
        self.consecutive_redundant = 0

    @staticmethod
    def _record(records, key, value):
        old = next((record for record in records if record["key"] == key), None)
        seen = min(100_000, old["seen"] + 1) if old else 1
        records[:] = [record for record in records if record["key"] != key]
        records.append({**value, "key": key, "seen": seen})
        del records[:-_LIMIT]

    def _progress(self, facts):
        keys = list(dict.fromkeys(_digest(item) for item in facts))
        new = [key for key in keys if key not in self.facts]
        self.consecutive_redundant = 0 if new else min(100_000, self.consecutive_redundant + 1)
        self.facts = list(dict.fromkeys([*self.facts, *new]))[-512:]
        return len(new)

    def observe(self, name, arguments, result, *, source=None, index=None, deadline=None):
        if name not in {"read_file", "search_code", "list_files"} or not isinstance(result, dict):
            return
        path = result.get("path", arguments.get("path", "."))
        if not _path(path):
            return
        if not result.get("success"):
            self._progress([])
            return
        if name == "list_files":
            entries = [entry for entry in result.get("entries", []) if _path(entry)]
            new = self._progress([['path', path, entry] for entry in entries])
            self._record(self.listings, _digest([path, sorted(entries)]),
                         {"path": path, "returned_count": len(entries), "new_paths": new})
        elif name == "search_code":
            matches = [match for match in result.get("matches", [])
                       if isinstance(match, dict) and _path(match.get("path"))]
            pattern = arguments.get("pattern", "")
            pattern = pattern if isinstance(pattern, str) else ""
            key = _digest([path, pattern, arguments.get("mode", "literal"), matches])
            self._progress([['match', match] for match in matches])
            self._record(self.searches, key, {"path": path, "pattern": pattern[:4096],
                "pattern_truncated": len(pattern) > 4096, "returned_count": len(matches)})
        else:
            first, last = result.get("start_line"), result.get("end_line")
            digest = result.get("source_sha256")
            if (type(first) is not int or type(last) is not int or first < 1 or last < first
                    or not isinstance(digest, str) or not _HASH.fullmatch(digest)):
                return
            self._progress([['read', path, digest, line] for line in range(first, min(last + 1, first + 10000))])
            value = {"path": path, "lines": [first, last], "source_sha256": digest,
                     "pure_reexport": False}
            complete = first == 1 and last == result.get("total_lines") and not result.get("truncated")
            if (complete and source is not None and index is not None
                    and source.path == path and source.start_line == first and source.end_line == last
                    and source.source_sha256 == digest):
                symbols = _reexport_symbols(source.text)
                if symbols is not None:
                    try:
                        blocks = index.blocks(path, deadline=deadline)
                        if blocks.get("source_sha256") == digest and not blocks.get("truncated"):
                            value["pure_reexport"] = True
                            for target in blocks.get("dependencies", [])[:8]:
                                if _path(target) and checked_path(index.root, target).is_file():
                                    candidate = {"path": target, "from": path, "symbols": symbols,
                                                 "pure_reexport": True, "source_sha256": digest}
                                    self.candidates = [item for item in self.candidates
                                                       if (item["path"], item["from"]) != (target, path)]
                                    self.candidates.append(candidate)
                            self.candidates = self.candidates[-_LIMIT:]
                    except (OSError, ValueError, RuntimeError):
                        pass
            # New source content invalidates old import conclusions for this path.
            self.candidates = [item for item in self.candidates
                               if item["from"] != path or item["source_sha256"] == digest]
            self._record(self.reads, _digest([path, first, last, digest]), value)

    def information_need(self, objective):
        """A completed read's question is not a continuing retrieval objective."""
        if not isinstance(objective, str):
            raise ValueError("Navigation objective must be text")
        return objective

    def summary(self, max_items=4, max_chars=1200):
        if type(max_items) is not int or not 1 <= max_items <= _LIMIT or type(max_chars) is not int or max_chars < 256:
            raise ValueError("Navigation summary requires bounded positive limits")
        result = {"reads": [{k: v for k, v in item.items() if k != "key"}
                            for item in self.reads[-max_items:]],
                  "searches": [{k: v for k, v in item.items() if k != "key"}
                               for item in self.searches[-max_items:]],
                  "listings": [{k: v for k, v in item.items() if k != "key"}
                               for item in self.listings[-max_items:]],
                  "next_candidates": deepcopy(self.candidates[-max_items:]),
                  "consecutive_redundant": self.consecutive_redundant,
                  "edit_authorization": False, "truncated": False}
        while len(json.dumps(result, ensure_ascii=False, separators=(",", ":"))) > max_chars:
            for key in ("listings", "searches", "reads", "next_candidates"):
                if result[key]:
                    result[key].pop(0)
                    result["truncated"] = True
                    break
            else:
                break
        return result

    def to_dict(self):
        return deepcopy({"schema_version": 1, "reads": self.reads, "searches": self.searches,
                         "listings": self.listings, "candidates": self.candidates,
                         "facts": self.facts, "consecutive_redundant": self.consecutive_redundant})

    @classmethod
    def from_dict(cls, value):
        required = {"schema_version", "reads", "searches", "listings", "candidates", "facts", "consecutive_redundant"}
        if (not isinstance(value, dict) or set(value) != required or type(value.get("schema_version")) is not int
                or value["schema_version"] != 1
                or type(value.get("consecutive_redundant")) is not int
                or not 0 <= value["consecutive_redundant"] <= 100_000):
            raise ValueError("Invalid navigation checkpoint")
        expected = {"reads": {"key", "seen", "path", "lines", "source_sha256", "pure_reexport"},
                    "searches": {"key", "seen", "path", "pattern", "pattern_truncated", "returned_count"},
                    "listings": {"key", "seen", "path", "returned_count", "new_paths"},
                    "candidates": {"path", "from", "symbols", "pure_reexport", "source_sha256"}}
        for name, fields in expected.items():
            records = value.get(name)
            if not isinstance(records, list) or len(records) > _LIMIT:
                raise ValueError("Invalid navigation checkpoint records")
            for item in records:
                if not isinstance(item, dict) or set(item) != fields or not _path(item.get("path")):
                    raise ValueError("Invalid navigation checkpoint record")
                for key in ("key", "source_sha256"):
                    if key in item and (not isinstance(item[key], str) or not _HASH.fullmatch(item[key])):
                        raise ValueError("Invalid navigation checkpoint hash")
                for key in ("seen", "returned_count", "new_paths"):
                    if key in item and (type(item[key]) is not int or not 0 <= item[key] <= 100_000):
                        raise ValueError("Invalid navigation checkpoint count")
                if "seen" in item and item["seen"] < 1:
                    raise ValueError("Invalid navigation checkpoint seen count")
                if name == "reads" and (not isinstance(item["lines"], list) or len(item["lines"]) != 2
                        or any(type(n) is not int or n < 1 for n in item["lines"])
                        or item["lines"][1] < item["lines"][0] or type(item["pure_reexport"]) is not bool):
                    raise ValueError("Invalid navigation checkpoint read")
                if name == "searches" and (not isinstance(item["pattern"], str) or len(item["pattern"]) > 4096
                                           or type(item["pattern_truncated"]) is not bool):
                    raise ValueError("Invalid navigation checkpoint search")
                if name == "candidates" and (not _path(item["from"]) or item["pure_reexport"] is not True
                        or not isinstance(item["symbols"], list) or len(item["symbols"]) > 16
                        or any(not isinstance(symbol, str) or len(symbol) > 256 for symbol in item["symbols"])):
                    raise ValueError("Invalid navigation checkpoint candidate")
        facts = value.get("facts")
        if not isinstance(facts, list) or len(facts) > 512 or any(not isinstance(f, str) or not _HASH.fullmatch(f) for f in facts):
            raise ValueError("Invalid navigation checkpoint facts")
        ledger = cls()
        for key in required - {"schema_version"}:
            setattr(ledger, key, deepcopy(value[key]))
        return ledger
