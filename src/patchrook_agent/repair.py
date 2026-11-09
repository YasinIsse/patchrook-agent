"""Failure-specific repair evidence assembled only from observed tool results.

The plan is navigation and context-selection advice. It neither determines a
patch nor grants permission to edit: the actual prepared request still controls
source-before-edit validation. No repository files are read by this module.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
from itertools import islice
import json
from pathlib import PurePosixPath
import re
from textwrap import dedent

from patchrook_agent.context.compaction import _arguments, _groups
from patchrook_agent.inspection import canonical_path, visible_lines


MAX_REPAIR_ITEMS = 8
MAX_SOURCE_FILES = 64
MAX_SOURCE_BYTES = 524288
MAX_FILE_BYTES = 65536
MAX_AST_NODES = 4096
_READERS = {"read_file", "read_files", "read_code"}
_MUTATIONS = {"edit_file", "write_file"}
_LOCATION = re.compile(r"(?m)^\s*([^\s:]+\.(?:py|pyi|js|ts|go|rs|java|sql)):(\d+)(?::|\b)")
_FAILED = re.compile(r"(?m)^FAILED\s+([^\s]+)")
_WORD = re.compile(r"[a-zA-Z][a-zA-Z0-9_]{2,}")
_COMMON = {"test", "tests", "def", "return", "assert", "self", "none", "true", "false",
           "the", "and", "for", "with", "this", "that", "from", "must", "before", "after"}


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _path(value) -> str | None:
    result = canonical_path(value)
    if (result is None or result in {".", ".."} or result.startswith(("/", "../"))
            or len(result) > 1024):
        return None
    return result


def _is_test(path: str) -> bool:
    name = PurePosixPath(path).name
    return "tests" in PurePosixPath(path).parts or name.startswith("test_") or name.endswith("_test.py")


def _words(value: str) -> set[str]:
    words = {part.lower() for word in _WORD.findall(value[:16000])
             for part in word.split("_") if len(part) > 2 and part.lower() not in _COMMON}
    # A small spelling normalization makes "discounts" and "discount" useful
    # lexical evidence without domain-specific synonyms or solution labels.
    return words | {word[:-1] for word in words if len(word) > 4 and word.endswith("s")}


def _ranges(path: str, numbers) -> tuple[tuple[str, int, int], ...]:
    result = []
    for number in sorted(set(numbers)):
        if result and result[-1][2] + 1 == number:
            result[-1] = (path, result[-1][1], number)
        else:
            result.append((path, number, number))
    return tuple(result)


def _unique_ranges(ranges, *, line_limit=72):
    """Retain order and exact line identities, within an explicit source budget."""
    selected = []
    seen = set()
    for path, start, end in ranges:
        numbers = []
        for number in range(start, min(end, start + line_limit) + 1):
            if (path, number) not in seen and len(seen) < line_limit:
                seen.add((path, number))
                numbers.append(number)
        selected.extend(_ranges(path, numbers))
    return tuple(selected)


@dataclass(frozen=True)
class RepairItem:
    id: str
    path: str
    symbol: str
    description: str
    ranges: tuple[tuple[str, int, int], ...]
    failure_ids: tuple[str, ...]
    narrow_ranges: tuple[tuple[str, int, int], ...] = ()
    evidence_range: tuple[int, int] | None = None
    evidence_card: str = ""


@dataclass(frozen=True)
class RepairPlan:
    fingerprint: str
    failure_excerpt: str
    items: tuple[RepairItem, ...]

    def decision_context(self, task: str, items=None) -> str:
        """Small factual cards for a choice, rather than repeated pytest logs.

        The fingerprint is a cache key, not useful semantic input. Exact raw
        failures and source remain in canonical history and the coder packet.
        """
        if not isinstance(task, str):
            raise ValueError("Decision task must be text")
        requested = self.items[:4] if items is None else tuple(items)
        selected = []
        for value in requested:
            identity = value.id if isinstance(value, RepairItem) else value
            item = next((candidate for candidate in self.items if candidate.id == identity), None)
            if item is not None and item.id not in {candidate.id for candidate in selected}:
                selected.append(item)
            if len(selected) >= 4:
                break
        brief = re.split(r"\bpreserve\b", task, maxsplit=1, flags=re.IGNORECASE)[0].strip()[:260]
        salient = [line.strip() for line in self.failure_excerpt.splitlines()
                   if ("assert " in line or re.match(r"E\s+", line.strip())
                       or any(word in line for word in ("Error:", "Exception:")))]
        failure = "\n".join(dict.fromkeys(salient))[:380]
        if not failure:
            failure = " ".join(self.failure_excerpt.split())[:260]
        prefix = "Task (data):\n" + brief + "\nObserved failure (data):\n" + failure
        if not selected:
            return prefix[:2400]
        allowance = min(700, (2400 - len(prefix) - 2 * len(selected)) // len(selected))
        cards = []
        for item in selected:
            card = item.evidence_card or f"{item.id} {item.description}\nNo implementation excerpt supplied."
            # Cut only between complete source lines. A very long individual
            # line is omitted, never supplied as misleading partial source.
            lines = card.splitlines()
            support = [line for line in lines if line.startswith(("Test ", "Caller "))]
            body = [line for line in lines if not line.startswith(("Test ", "Caller "))]
            kept = []
            used = 0
            # Preserve the binding and assertion while admitting the highest
            # priority implementation lines. Merely clipping a 700-char card
            # would repeatedly lose those useful facts at the bottom.
            for line in [*body[:2], *support, *body[2:]]:
                if used + len(line) + bool(kept) <= allowance:
                    kept.append(line)
                    used += len(line) + bool(len(kept) > 1)
            cards.append("\n".join(kept))
        return (prefix + "\n\n" + "\n\n".join(cards))[:2400]

    def packet(self, item_id=None, *, narrow=False) -> dict:
        if type(narrow) is not bool:
            raise ValueError("narrow must be a boolean")
        selected = next((item for item in self.items if item.id == item_id), None) if item_id else self.items[0]
        if selected is None:
            raise ValueError("Unknown repair work item")
        ranges = selected.narrow_ranges if narrow and selected.narrow_ranges else selected.ranges
        return {"id": selected.id, "fingerprint": self.fingerprint,
                "source_ranges": [{"path": path, "start_line": start, "end_line": end}
                                  for path, start, end in ranges],
                "failure_ids": list(selected.failure_ids), "narrow": narrow}


@dataclass(frozen=True)
class _Function:
    path: str
    name: str
    start: int
    end: int
    calls: tuple[tuple[str, int], ...]
    anchors: tuple[int, ...]


def _stable_failure(result):
    """Measurement noise cannot create a new repair episode."""
    stable = {key: result[key] for key in ("success", "status", "exit_code", "stdout", "stderr", "test_counts")
              if key in result}
    for key in ("stdout", "stderr"):
        if isinstance(stable.get(key), str):
            stable[key] = re.sub(r"(?m)(\b\d+ (?:failed|passed|errors?|skipped)(?:, \d+ [a-z]+)* in )\d+(?:\.\d+)?s\b",
                                 r"\1<elapsed>s", stable[key])
            # Only ephemeral fixture/worktree prefixes are measurement noise;
            # retain the actual relative filename and all error/amount text.
            stable[key] = re.sub(r"/tmp/patchrook-[^/\s\"']+/", "<workspace>/", stable[key])
    return stable


def _contains(selection, other):
    # A broad green result resolves only the scopes that it actually includes.
    return selection == "." or selection == other or other.startswith(selection.rstrip("/") + "/") or other.startswith(selection + "::")


def _observed(messages, stale_before):
    sources, failed = {}, {}
    boundaries = {}
    for path, index in (stale_before or {}).items():
        path = _path(path)
        if path is None or type(index) is not int or index < 0:
            raise ValueError("Invalid repair source boundary")
        boundaries[path] = index
    offset = 0
    for group in _groups(messages):
        for position, (call, observation) in enumerate(zip(group[0].get("tool_calls") or [], group[1:]), 1):
            name = call["function"]["name"]
            if name not in _READERS | _MUTATIONS | {"run_tests"}:
                continue
            try:
                result = json.loads(observation.get("content", ""))
            except (ValueError, TypeError):
                continue
            if not isinstance(result, dict):
                continue
            arguments = _arguments(call)
            index = offset + position
            path = _path(result.get("path") or arguments.get("path"))
            if name in _MUTATIONS and result.get("success") is True and result.get("modified") is True and path:
                sources.pop(path, None)
            if name == "run_tests":
                selection = arguments.get("path", ".")
                if not isinstance(selection, str):
                    continue
                if result.get("status") in {"failed", "timed_out"}:
                    failed[selection] = _stable_failure(result)
                elif result.get("status") == "passed" and result.get("exit_code") == 0:
                    failed = {scope: failure for scope, failure in failed.items() if not _contains(selection, scope)}
            records = ([result] if name == "read_file" else result.get("files", []) if name in _READERS else [])
            if not isinstance(records, list):
                continue
            for record in records:
                if not isinstance(record, dict):
                    continue
                source_path = _path(record.get("path") or arguments.get("path"))
                if not source_path or index < boundaries.get(source_path, 0):
                    continue
                lines = visible_lines(record)
                if not lines:
                    continue
                previous = sources.get(source_path, {})
                if any(number in previous and previous[number] != source for number, source in lines.items()):
                    previous = {}
                total = record.get("total_lines")
                if type(total) is int:
                    previous = {number: source for number, source in previous.items() if number <= total}
                combined = {**previous, **lines}
                if sum(len(line.encode("utf-8")) for line in combined.values()) > MAX_FILE_BYTES:
                    continue
                if source_path in sources or len(sources) < MAX_SOURCE_FILES:
                    sources[source_path] = combined
        offset += len(group)
    return sources, failed


def _chunks(lines):
    chunks = []
    current = []
    for number in sorted(lines):
        if current and number != current[-1] + 1:
            chunks.append(current)
            current = []
        current.append(number)
    if current:
        chunks.append(current)
    return chunks


def _functions(sources):
    functions = []
    used = 0
    for path, lines in sorted(sources.items()):
        if not path.endswith((".py", ".pyi")):
            continue
        for chunk in _chunks(lines):
            text = "\n".join(lines[number] for number in chunk)
            used += len(text.encode("utf-8"))
            if used > MAX_SOURCE_BYTES:
                return functions
            try:
                candidate_text = dedent(text)
                try:
                    tree = ast.parse(candidate_text)
                except SyntaxError:
                    # A read may append only the next helper's header. Keep a
                    # fully observed parseable prefix without inventing that
                    # helper's missing body or losing the completed function.
                    parts = candidate_text.splitlines()
                    tree = None
                    for trim in range(1, min(8, len(parts))):
                        try:
                            tree = ast.parse("\n".join(parts[:-trim]))
                            break
                        except SyntaxError:
                            continue
                    if tree is None:
                        continue
                nodes = list(islice(ast.walk(tree), MAX_AST_NODES + 1))
            except (ValueError, SyntaxError, RecursionError):
                continue
            if len(nodes) > MAX_AST_NODES:
                continue
            for node in nodes:
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                start, end = chunk[0] + node.lineno - 1, chunk[0] + node.end_lineno - 1
                calls = []
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        if isinstance(child.func, ast.Name):
                            calls.append((child.func.id, chunk[0] + child.lineno - 1))
                        # Attribute names alone do not identify a module/function.
                # Guard-only validation is retrievable, but repeating dozens
                # of raises is poor use of a small repair context. Select exact
                # computational statements, signatures and terminal statements.
                # This is context selection, never a claim that omitted guards
                # are irrelevant or that their source is supplied to the model.
                def validation(statement):
                    return (isinstance(statement, ast.Raise)
                            or isinstance(statement, ast.If) and not statement.orelse
                            and bool(statement.body) and all(validation(part) for part in statement.body))
                computation = []
                for statement in node.body:
                    if validation(statement) or (isinstance(statement, ast.Expr)
                            and isinstance(statement.value, ast.Constant)
                            and isinstance(statement.value.value, str)):
                        continue
                    first = chunk[0] + statement.lineno - 1
                    last = chunk[0] + statement.end_lineno - 1
                    computation.extend(range(first, last + 1) if last - first < 8 else (first, last))
                signature = range(start, min(end, start + 2) + 1)
                ordered = [*signature, *computation[:8], *computation[-8:], *computation]
                anchors = tuple(sorted(list(dict.fromkeys(ordered))[:22]))
                functions.append(_Function(path, node.name, start, end, tuple(dict.fromkeys(calls)), anchors))
    return functions


def _evidence_card(identity, path, symbol, numbers, sources, test_ranges, caller_ranges):
    """Prioritized exact lines, with an assertion and observed call binding."""
    primary = sources.get(path, {})
    available = [number for number in set(numbers) if number in primary]
    def priority(number):
        line = primary[number].strip()
        if not line or line.startswith(("#", '"""', "'''")):
            return 6, number
        if re.search(r"(?:\s[+*%/-]\s|//)", line) and line.startswith("return "):
            return 0, number
        if re.search(r"(?:\s[+*%/-]\s|//)", line) and "=" in line:
            return 1, number
        if line.startswith("return "):
            return 2, number
        if "=" in line and not line.startswith(("def ", "async def ")):
            return 3, number
        if line.startswith(("if ", "for ", "while ", "try:")):
            return 4, number
        return 5, number
    available.sort(key=priority)
    affinities = _words(path + " " + symbol + " " + " ".join(primary.get(number, "") for number in available))
    assertions = [(source_path, number, sources.get(source_path, {}).get(number, ""))
                  for source_path, first, last in test_ranges for number in range(first, last + 1)
                  if "assert " in sources.get(source_path, {}).get(number, "")]
    assertions.sort(key=lambda entry: -len(_words(entry[2]) & affinities))
    callers = [(source_path, number, sources.get(source_path, {}).get(number, ""))
               for source_path, first, last in caller_ranges for number in range(first, last + 1)
               if symbol != "<observed source>" and re.search(
                   r"\b" + re.escape(symbol) + r"\s*\(", sources.get(source_path, {}).get(number, ""))]
    support = []
    if assertions:
        source_path, number, line = assertions[0]
        support.append(f"Test {source_path}:{number}: {line}")
    if callers and (not assertions or callers[0][:2] != assertions[0][:2]):
        source_path, number, line = callers[0]
        support.append(f"Caller {source_path}:{number}: {line}")
    # A single exceptional source line cannot consume the whole decision state.
    support = [line for line in support if len(line) <= 220]
    heading = f"{identity} {path}::{symbol}"[:170] + "\nImplementation (observed):"
    remaining = 700 - len(heading) - sum(len(line) + 1 for line in support)
    selected = []
    for number in available:
        line = f"{number}: {primary[number]}"
        if len(line) + 1 <= remaining:
            selected.append(line)
            remaining -= len(line) + 1
        if len(selected) == 8:
            break
    return "\n".join([heading, *selected, *support])[:700]


def build_repair_plan(messages, task, *, revision=0, stale_before=None) -> RepairPlan | None:
    """Compile at most eight observed implementation choices for failed tests.

    Conservative unqualified call matching uses same-file definitions first,
    then a globally unique observed definition. Ambiguous symbols are never
    treated as established edges. Unsupported/partial source remains a bounded
    path-level evidence choice, rather than disappearing from the plan.
    """
    if not isinstance(messages, list) or not isinstance(task, str) or type(revision) is not int or revision < 0:
        raise ValueError("Invalid repair plan inputs")
    try:
        sources, failed = _observed(messages, stale_before)
    except (ValueError, TypeError, KeyError, AttributeError, RecursionError):
        return None
    if not failed or not sources:
        return None
    evidence = "\n".join(str(value.get(key, "")) for _, value in sorted(failed.items())
                         for key in ("stdout", "stderr"))
    locations = [(path, int(number)) for raw_path, number in _LOCATION.findall(evidence)
                 if (path := _path(raw_path)) in sources]
    failure_ids = tuple(dict.fromkeys(match.rstrip(":") for match in _FAILED.findall(evidence)))[:16]
    if not failure_ids:
        failure_ids = tuple(f"{path}:{number}" for path, number in locations)[:16] or tuple(sorted(failed))[:4]
    functions = _functions(sources)
    by_name = {}
    for function in functions:
        by_name.setdefault(function.name, []).append(function)
    roots = [function for function in functions if any(
        function.path == path and function.start <= number <= function.end for path, number in locations)
        or any(identity.startswith(function.path + "::") and function.name in identity.split("::")[1:]
               for identity in failure_ids)]
    if not roots:
        failed_paths = {scope.split("::", 1)[0] for scope in failed}
        roots = [function for function in functions if _is_test(function.path)
                 and (function.path in failed_paths or any(path == function.path for path, _ in locations))]
    edges = {}
    for function in functions:
        targets = []
        for name, number in function.calls:
            candidates = by_name.get(name, [])
            local = [candidate for candidate in candidates if candidate.path == function.path]
            candidate = local[0] if len(local) == 1 else candidates[0] if not local and len(candidates) == 1 else None
            if candidate is not None and candidate != function:
                targets.append((candidate, number))
        edges[function] = targets
    reachable, depth = [], {}
    pending = [(function, 0) for function in roots]
    while pending and len(depth) < 64:
        function, level = pending.pop(0)
        if function in depth:
            continue
        depth[function] = level
        if not _is_test(function.path):
            reachable.append(function)
        if level < 3:
            pending.extend((target, level + 1) for target, _ in edges.get(function, []))
    # A path explicitly appearing in the traceback is useful even when parsing
    # or call resolution fails. Other observed implementations stay retrievable.
    traced_paths = {path for path, _ in locations if not _is_test(path)}
    candidates = reachable or [function for function in functions if function.path in traced_paths]
    if not candidates:
        candidates = [function for function in functions if not _is_test(function.path)]
    failure_words = _words("\n".join(identity.split("::")[-1] for identity in failure_ids)
                           + "\n" + "\n".join(line for line in evidence.splitlines() if "assert " in line))
    # Preservation constraints name distractors by design. They are constraints
    # on all patches, rather than evidence that a named component is faulty.
    task_words = _words(re.split(r"\bpreserve\b", task, maxsplit=1, flags=re.IGNORECASE)[0])
    candidates.sort(key=lambda function: (
        -(4 * len(_words(function.name) & failure_words)
          + 2 * len(_words(function.name) & task_words)
          + len(_words(function.path) & task_words)),
        len(edges.get(function, [])), depth.get(function, 4), function.path, function.start))
    test_ranges = []
    for function in roots:
        if _is_test(function.path):
            numbers = range(function.start, min(function.end, function.start + 7) + 1)
            test_ranges.extend(_ranges(function.path, numbers))
    test_ranges = _unique_ranges(test_ranges, line_limit=16)
    items = []
    for function in candidates[:MAX_REPAIR_ITEMS]:
        target = _ranges(function.path, function.anchors)
        callers = []
        for caller in functions:
            if caller == function or caller not in depth:
                continue
            for called, number in edges.get(caller, []):
                if called == function:
                    callers.extend(_ranges(caller.path, range(max(caller.start, number - 2), min(caller.end, number + 3) + 1)))
        callers = _unique_ranges(callers, line_limit=6)
        narrow_ranges = _unique_ranges([*target, *test_ranges, *callers], line_limit=40)
        peer_limit = 2 if len(edges.get(function, [])) > 1 else 1
        peers = [peer for peer in candidates if peer != function and peer.path != function.path][:peer_limit]
        # Allocate peer evidence fairly. Start at each terminal statement, then
        # include its bounded computational tail instead of starving the second
        # candidate with the first candidate's validation bulk.
        peer_numbers = [list(dict.fromkeys([peer.anchors[-1],
                         *[number for number in peer.anchors if number > peer.start + 2][-8:]])) for peer in peers]
        peer_ranges = []
        for offset in range(8):
            for peer, numbers in zip(peers, peer_numbers):
                if offset < len(numbers):
                    peer_ranges.extend(_ranges(peer.path, [numbers[offset]]))
        ranges = _unique_ranges([*narrow_ranges, *peer_ranges], line_limit=48)
        identity = "repair_" + _digest([function.path, function.name, function.start])[:12]
        description = f"{function.path}; {function.name}:{function.start}-{function.end}; observed implementation linked to failed tests"
        card = _evidence_card(identity, function.path, function.name, function.anchors,
                              sources, test_ranges, callers)
        items.append(RepairItem(identity, function.path, function.name, description[:240], ranges, failure_ids,
                                narrow_ranges, (function.start, function.end), card))
    represented = {function.path for function in candidates}
    for path, lines in sorted(sources.items()):
        if len(items) >= MAX_REPAIR_ITEMS:
            break
        if path in represented or _is_test(path):
            continue
        numbers = sorted(lines)
        anchors = numbers if len(numbers) <= 24 else numbers[:6] + numbers[-18:]
        target = _ranges(path, anchors)
        ranges = _unique_ranges([*target, *test_ranges], line_limit=48)
        identity = "repair_" + _digest([path, "observed"])[:12]
        card = _evidence_card(identity, path, "<observed source>", anchors, sources, test_ranges, ())
        items.append(RepairItem(identity, path, "<observed source>",
                                f"{path}; partial or unsupported observed implementation; retrieve missing evidence"[:240],
                                ranges, failure_ids, ranges, None, card))
    if not items:
        return None
    relevant_numbers = {}
    for function in candidates[:MAX_REPAIR_ITEMS]:
        relevant_numbers.setdefault(function.path, set()).update(range(function.start, function.end + 1))
    for item in items:
        if item.evidence_range is None:
            relevant_numbers.setdefault(item.path, set()).update(
                number for number, line in sources[item.path].items()
                if line.strip() and not line.lstrip().startswith(("#", "//", "--")))
    for path, first, last in test_ranges:
        relevant_numbers.setdefault(path, set()).update(range(first, last + 1))
    fingerprint = _digest({"policy_version": 2, "revision": revision, "failed": failed,
                           "source": {path: [(number, sources[path][number]) for number in sorted(numbers)
                                             if number in sources[path]]
                                      for path, numbers in sorted(relevant_numbers.items())},
                           "items": [(item.id, item.ranges, item.evidence_card) for item in items]})
    excerpt = evidence if len(evidence) <= 1400 else evidence[:700] + "\n…\n" + evidence[-697:]
    return RepairPlan(fingerprint, excerpt, tuple(items))
