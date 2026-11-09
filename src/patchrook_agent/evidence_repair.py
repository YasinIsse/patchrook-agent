"""Acquire bounded, current implementation bodies before a repair generation.

This collector observes source; it does not authorize an edit or decide which
observations fit a model request. The caller still owns context compilation,
last-request edit authority and verification.
"""

from __future__ import annotations

import hashlib
from difflib import SequenceMatcher
from pathlib import PurePosixPath
import time

from patchrook_agent.evidence import ContextBudgetError
from patchrook_agent.evidence_bootstrap import small_task_class_ranges
from patchrook_agent.tools.base import ToolError


_MAX_PATHS = 4
_MAX_DEPENDENCIES = 2
_MAX_RANGES = 16


def _check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise ToolError("deadline_exceeded", "Task deadline exceeded while preparing repair source")


def _test_path(path):
    value = PurePosixPath(path)
    return (any(part in {"test", "tests"} for part in value.parts)
            or value.name.startswith("test_") or value.name.endswith("_test.py")
            or value.name == "conftest.py")


def _implementation_path(path):
    value = PurePosixPath(path)
    return (not _test_path(path) and value.suffix.lower() not in {
        ".md", ".rst", ".txt", ".csv", ".log", ".svg", ".png", ".jpg", ".jpeg", ".gif", ".pdf",
    } and value.name.upper() not in {"README", "LICENSE", "COPYING", "NOTICE"})


def _merge(ranges):
    merged = []
    for first, last in sorted(set(ranges)):
        if merged and first <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], last))
        else:
            merged.append((first, last))
    return merged


def _join_source_gaps(ranges, lines, maximum):
    """Keep small blank/comment gaps available for unambiguous edit anchors."""
    joined = []
    for first, last in _merge(ranges):
        if joined:
            previous = joined[-1]
            gap = lines[previous[1]:first - 1]
            if (len(gap) <= 4 and last - previous[0] + 1 <= maximum
                    and all(not line.strip() or line.lstrip().startswith("#") for line in gap)):
                joined[-1] = previous[0], last
                continue
        joined.append((first, last))
    return joined


def _changed_ranges(original, current, deadline):
    """Locate all edits in current coordinates, anchoring deleted text nearby."""
    lines = current.splitlines()
    if not lines:
        return []
    if original is None:
        return [(1, len(lines))]
    _check_deadline(deadline)
    changes = []
    for tag, _, _, first, last in SequenceMatcher(None, original.splitlines(), lines).get_opcodes():
        _check_deadline(deadline)
        if tag == "equal":
            continue
        if first == last:
            changes.append((max(1, first), min(len(lines), first + 1)))
        else:
            changes.append((first + 1, last))
    return _merge(changes)


def _covers(ranges, first, last):
    cursor = first
    for start, end in sorted(ranges):
        if start > cursor:
            break
        if end >= cursor:
            cursor = end + 1
        if cursor > last:
            return True
    return False


def collect_repair_sources(workflow, deadline):
    """Return exact, refreshed source cards from observed code and static imports.

    At most four implementation files, two acquired dependencies and sixteen
    merged ranges are returned. Large functions are never clipped to force a
    packet to fit. Recorded edits determine current focus; otherwise the latest
    direct read takes precedence over automatic expansion. Active repairs also
    retain prior same-path complete selectors alongside new direct anchors.
    Previous selectors provide no edit authority and are always
    reread after mutation or resume.
    """
    _check_deadline(deadline)
    cards = [card for card in workflow.evidence.cards(deadline=deadline)
             if card.kind == "source" and card.path is not None
             and card.metadata.get("origin") != "diagnostic_oracle"]
    _check_deadline(deadline)
    cards.sort(key=lambda card: card.sequence, reverse=True)
    by_id = {card.id: card for card in cards}
    newest, direct = {}, {}
    for card in cards:
        newest.setdefault(card.path, card)
    # Identical reads may reuse an old evidence card. The actual read ledger
    # determines recency before falling back to source-card sequence numbers.
    for identifier in reversed(getattr(workflow, "direct", [])):
        card = by_id.get(identifier)
        if card is not None:
            direct.setdefault(card.path, card)
    for card in cards:
        if card.metadata.get("origin") == "direct":
            direct.setdefault(card.path, card)
    previous, changed, focus_hashes = {}, {}, {}
    candidates = []

    def seed(path, first=None, last=None, *, candidate=True):
        if not isinstance(path, str) or not path or not _implementation_path(path):
            return
        # All selectors, including checkpoint hints, cross the same workspace
        # guard before source discovery. Invalid/protected paths are not hidden.
        workflow.tools._path(path)
        if candidate and path not in candidates:
            candidates.append(path)
        if first is not None or last is not None:
            if type(first) is not int or type(last) is not int or not 1 <= first <= last:
                raise ValueError("Repair source selectors require a valid complete line range")
            previous.setdefault(path, []).append((first, last))

    state = getattr(workflow, "state", None)
    for path in getattr(state, "modified_files", []):
        seed(path)
    for path, original in getattr(workflow, "originals", {}).items():
        if not _implementation_path(path):
            continue
        target = workflow.tools._path(path)
        if target.is_file():
            raw = workflow.tools._bytes(target)
            current = raw.decode("utf-8")
            if current != original:
                seed(path)
                changed[path] = _changed_ranges(original, current, deadline)
                focus_hashes[path] = hashlib.sha256(raw).hexdigest()
    if len(changed) > _MAX_PATHS:
        raise ContextBudgetError("Changed repair source exceeds four implementation files")
    candidates.sort(key=lambda path: path not in changed)
    repair = (getattr(workflow, "routing", None) or {}).get("repair") or {}
    for selection in repair.get("source_ranges", []):
        if not isinstance(selection, dict):
            raise ValueError("Invalid repair source selector")
        path = selection.get("path")
        # Validate every persisted selector through the same path/range guard,
        # without allowing old dependency packets to displace current focus.
        seed(path, selection.get("start_line"), selection.get("end_line"), candidate=False)
    for path in direct:
        seed(path)
    if not candidates:
        # Optional enrichment from an earlier packet must not become required
        # merely because the current direct-read ledger was consumed.
        prior_roles = {}
        for card in sorted(workflow.evidence.cards(fresh=False), key=lambda item: item.sequence, reverse=True):
            if card.kind == "source" and card.metadata.get("repair_role"):
                prior_roles.setdefault(card.path, card.metadata["repair_role"])
        for path in previous or newest:
            if prior_roles.get(path) != "optional_dependency":
                seed(path)

    def hints(path):
        if path in changed:
            return changed[path]
        if path in direct:
            card = direct[path]
            if repair.get("status") == "active" and path in previous:
                # A rejected edit may pin only its class-header anchor. Keep
                # the focused episode's already acquired complete bodies as
                # selectors too; every range is freshly observed below.
                return _merge([*previous[path], (card.start_line, card.end_line)])
            return [(card.start_line, card.end_line)]
        if path in previous:
            return previous[path]
        card = newest.get(path)
        return [(card.start_line, card.end_line)] if card is not None else []

    query = getattr(workflow, "query", "") or getattr(workflow, "task", "")
    block_cache = {}

    def blocks(path):
        if path not in block_cache:
            _check_deadline(deadline)
            value = workflow.index.blocks(path, query=query, deadline=deadline,
                                          line_ranges=hints(path))
            _check_deadline(deadline)
            if path in focus_hashes and value.get("source_sha256") not in {None, focus_hashes[path]}:
                raise ToolError("stale_source", "Source changed while locating repair edits")
            if value.get("truncated") and any(reason in {"deadline", "ast_node_limit", "focused_block_limit"}
                                               for reason in value.get("reasons", [])):
                raise ContextBudgetError("Complete repair source structure exceeds the bounded acquisition limit")
            block_cache[path] = value
        return block_cache[path]

    dependencies = []

    def add_dependencies(path):
        for related in blocks(path).get("dependencies", []):
            if len(dependencies) >= _MAX_DEPENDENCIES:
                break
            if _implementation_path(related) and related not in candidates:
                seed(related)
                dependencies.append(related)

    if not candidates:
        # An observed test can identify imports, but is never silently promoted
        # to the primary editable implementation in a repair packet.
        for card in cards:
            if _test_path(card.path):
                add_dependencies(card.path)
            if candidates:
                break
    required_paths = set(candidates[:_MAX_PATHS])
    for path in list(candidates[:_MAX_PATHS]):
        add_dependencies(path)
    if not candidates:
        return []

    selections = []
    expected_hashes = {}
    maximum = workflow.workspace.config.max_read_lines

    def ranges_for(path):
        _check_deadline(deadline)
        raw = workflow.tools._bytes(workflow.tools._path(path))
        text = raw.decode("utf-8")
        total = len(text.splitlines())
        if not total:
            return []
        expected_hashes[path] = hashlib.sha256(raw).hexdigest()
        if path in focus_hashes and expected_hashes[path] != focus_hashes[path]:
            raise ToolError("stale_source", "Source changed while preparing its changed repair body")
        structure = blocks(path)
        if structure.get("source_sha256") not in {None, expected_hashes[path]}:
            raise ToolError("stale_source", "Source changed while preparing its complete repair body")
        if path not in required_paths and "ast_unavailable" in structure.get("reasons", []):
            raise ContextBudgetError("Optional dependency AST is unavailable")
        duplicate_families = []
        if total <= 80:
            ranges = [(1, total)]
        else:
            selectors = structure.get("selectors", [])
            functions = [item for item in selectors if item.get("symbol") not in {None, "", "module_header"}
                         and item.get("kind") != "class_header"]
            headers = [item for item in selectors if item.get("kind") == "class_header"]
            observed = hints(path)
            if not (functions or headers) or "ast_unavailable" in structure.get("reasons", []):
                # Non-Python and malformed Python fallback blocks are not
                # complete-function guarantees. Preserve the whole file or fail.
                ranges = [(1, total)]
            else:
                selected = [item for item in functions if any(
                    item["start_line"] <= last and first <= item["end_line"] for first, last in observed)]
                selected_headers = [item for item in headers if any(
                    item["start_line"] <= last and first <= item["end_line"] for first, last in observed)]
                if not selected and not selected_headers:
                    selected = functions[:1]
                def binding_scope(member):
                    enclosing = [item for item in [*functions, *headers] if item is not member
                                 and item["start_line"] <= member["start_line"]
                                 and member["end_line"] <= item.get("scope_end_line", item["end_line"])]
                    if not enclosing:
                        return None
                    owner = min(enclosing, key=lambda item: item.get("scope_end_line", item["end_line"])
                                - item["start_line"])
                    return owner["start_line"], owner.get("scope_end_line", owner["end_line"])
                bindings = {(item.get("symbol"), binding_scope(item)) for item in selected}
                # Repeated definitions bind the same name; the last one wins.
                # A repair needs every complete current definition of a selected
                # binding, rather than an insertion's first matching body only.
                selected = [item for item in functions
                            if (item.get("symbol"), binding_scope(item)) in bindings]
                for binding in bindings:
                    family = [item for item in selected
                              if (item.get("symbol"), binding_scope(item)) == binding]
                    if len(family) > 1:
                        family.extend(item for item in headers
                                      if (item["start_line"], item["scope_end_line"]) == binding[1])
                        duplicate_families.extend(_join_source_gaps(
                            [(item["start_line"], item["end_line"]) for item in family],
                            text.splitlines(), maximum))
                selected_headers.extend(item for item in headers if any(
                    item["start_line"] <= member["start_line"] <= item["scope_end_line"]
                    for member in selected))
                ranges = [(item["start_line"], item["end_line"]) for item in selected]
                ranges.extend((item["start_line"], item["end_line"]) for item in selected_headers)
                ranges.extend((item["start_line"], item["end_line"]) for item in selectors
                              if item.get("symbol") == "module_header")
                if path in required_paths:
                    # A partial earlier patch can change one method while its
                    # small public result/container class remains unchanged.
                    # Reacquire that explicitly named complete class as source.
                    ranges.extend(small_task_class_ranges(workflow.task, structure,
                                                         maximum=min(40, maximum)))
        ranges = _merge([*ranges, *duplicate_families])
        # Diff ranges can include separator whitespace after an inserted body.
        # Keep those exact changed blank lines as source, without extending a
        # function across the next executable declaration.
        separators = []
        source_lines = text.splitlines()
        for first, last in changed.get(path, []):
            for number in range(first, last + 1):
                _check_deadline(deadline)
                if not source_lines[number - 1].strip() and not _covers(ranges, number, number):
                    if separators and separators[-1][1] + 1 == number:
                        separators[-1] = separators[-1][0], number
                    else:
                        separators.append((number, number))
        ranges = _merge([*ranges, *separators])
        for first, last in changed.get(path, []):
            if not _covers(ranges, first, last):
                raise ContextBudgetError(
                    f"Changed repair source in {path}:{first}-{last} is missing from the complete packet")
        if any(last - first + 1 > maximum for first, last in ranges):
            raise ContextBudgetError(f"Complete repair source in {path} exceeds max_read_lines={maximum}")
        return ranges

    for path in candidates[:_MAX_PATHS]:
        try:
            ranges = ranges_for(path)
            if len(selections) + len(ranges) > _MAX_RANGES:
                raise ContextBudgetError("Complete repair source exceeds sixteen merged ranges")
        except (ContextBudgetError, ToolError) as exc:
            # An explicitly observed/changed file remains mandatory. Missing
            # time, source integrity and path guards are never optional.
            if (path in required_paths or isinstance(exc, ToolError)
                    and exc.code not in {"file_too_large", "binary_file"}):
                raise
            session = getattr(workflow, "session", None)
            if session is not None:
                session.emit("repair_dependency_skipped", getattr(state, "steps", 0), path=path,
                             reason=getattr(exc, "code", "source_limit"), detail=str(exc),
                             edit_authorization=False)
            continue
        selections.extend((path, first, last) for first, last in ranges)

    result = []
    for path, first, last in selections:
        _check_deadline(deadline)
        card = workflow.evidence.observe_source(path, first, last, revision=workflow.revision,
                                               origin="repair_packet", deadline=deadline,
                                               metadata={"repair_role": "required_source" if path in required_paths
                                                         else "optional_dependency"})
        if card.source_sha256 != expected_hashes[path]:
            raise ToolError("stale_source", "Source changed while recording complete repair evidence")
        result.append(card)
    return result
