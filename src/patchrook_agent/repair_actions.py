"""Closed, bounded repair actions compiled from observed work-item selectors.

A decision chooses an existing ID. It cannot supply a path, expand a range,
authorize an edit, or execute repository code through this manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import PurePosixPath
import re

from patchrook_agent.inspection import canonical_path
from patchrook_agent.repair import RepairItem, RepairPlan


_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,47}\Z")
_CONFIG_SUFFIXES = {".toml", ".ini", ".cfg", ".yaml", ".yml", ".json"}
_CONFIG_NAME = re.compile(r"(?:^|[_-])(?:config|settings|options)(?:[_\-.]|$)", re.IGNORECASE)
_INSPECTIONS = {"inspect_helper", "inspect_config"}


def _safe_path(path):
    canonical = canonical_path(path)
    return (isinstance(path, str) and canonical == path and path not in {".", ".."}
            and not path.startswith(("/", "../")) and len(path) <= 1024)


@dataclass(frozen=True)
class RepairAction:
    id: str
    kind: str
    item_id: str
    path: str | None
    start_line: int | None
    end_line: int | None
    description: str

    def __post_init__(self):
        if (not isinstance(self.id, str) or not _IDENTIFIER.fullmatch(self.id)
                or not isinstance(self.item_id, str) or not _IDENTIFIER.fullmatch(self.item_id)
                or self.kind not in {"prepare_patch", *_INSPECTIONS}
                or not isinstance(self.description, str) or not self.description or len(self.description) > 240):
            raise ValueError("Invalid repair action")
        if self.path is not None and not _safe_path(self.path):
            raise ValueError("Invalid repair action path")
        if self.kind in _INSPECTIONS:
            if (self.path is None or type(self.start_line) is not int or type(self.end_line) is not int
                    or self.start_line < 1 or self.end_line < self.start_line
                    or self.end_line - self.start_line >= 24):
                raise ValueError("Invalid bounded repair inspection")
        elif self.start_line is not None or self.end_line is not None:
            raise ValueError("A patch action cannot carry read arguments")

    @property
    def arguments(self) -> dict:
        """Ready read_file arguments; normal tool validation still applies."""
        if self.kind not in _INSPECTIONS:
            return {}
        return {"path": self.path, "start_line": self.start_line, "end_line": self.end_line}


def _configuration(item: RepairItem) -> bool:
    path = PurePosixPath(item.path)
    return path.suffix.lower() in _CONFIG_SUFFIXES or bool(_CONFIG_NAME.search(path.name))


def _span(item: RepairItem) -> tuple[int, int] | None:
    known = [(start, end) for path, start, end in item.ranges
             if path == item.path and type(start) is int and type(end) is int and 1 <= start <= end]
    span = item.evidence_range
    if (isinstance(span, (tuple, list)) and len(span) == 2 and type(span[0]) is int and type(span[1]) is int
            and 1 <= span[0] <= span[1]):
        first, last = span
        represented = {number for start, end in known for number in range(start, min(end, start + 80) + 1)}
        # Prefer the nearest omitted context beside the computational tail.
        # Every endpoint is derived from observed locations; the eventual read
        # supplies fresh source and is independently checked by the tool layer.
        missing = []
        cursor = first
        for number in sorted(number for number in represented if first <= number <= last):
            if number > cursor:
                missing.append((cursor, number - 1))
            cursor = max(cursor, number + 1)
        if cursor <= last:
            missing.append((cursor, last))
        if missing:
            start, end = missing[-1]
            last = min(last, end + 1)
            first = max(first, start - 1, last - 23)
        else:
            first = max(first, last - 23)
        return first, min(last, first + 23)
    if known:
        # A partial/non-Python observation still provides a concrete bounded
        # selector, without inventing a complete function or reading hidden data.
        start, end = known[-1]
        return start, min(end, start + 23)
    return None


def _inspect(item: RepairItem, *, identity=None) -> RepairAction | None:
    span = _span(item)
    if span is None:
        return None
    kind = "inspect_config" if _configuration(item) else "inspect_helper"
    identity = identity or "inspect_" + hashlib.sha256(
        f"{item.id}:{kind}:{span[0]}:{span[1]}".encode()).hexdigest()[:16]
    label = "configuration" if kind == "inspect_config" else "implementation context"
    return RepairAction(identity, kind, item.id, item.path, span[0], span[1],
                        f"{item.path}; read lines {span[0]}-{span[1]} for omitted {label} before preparing a patch."[:240])


def build_repair_actions(plan: RepairPlan, *, allow_inspection=True) -> tuple[RepairAction, ...]:
    """At most six real actions; static ordering is the rules control."""
    if not isinstance(plan, RepairPlan):
        raise ValueError("Repair actions require a repair plan")
    if type(allow_inspection) is not bool:
        raise ValueError("allow_inspection must be a boolean")
    items = [item for item in plan.items
             if isinstance(item.id, str) and _IDENTIFIER.fullmatch(item.id) and _safe_path(item.path)]
    if not items:
        return ()
    actions = [RepairAction(item.id, "prepare_patch", item.id, item.path, None, None,
                            f"{item.path}; prepare a targeted patch for {item.symbol}, using the supplied implementation and failed assertion."[:240])
               for item in items[:3]]
    if not allow_inspection:
        return tuple(actions)
    inspections = set()
    # Keep the historical abstention ID, but map it to one concrete evidence
    # action rather than a vague instruction to investigate the repository.
    primary = _inspect(items[0], identity="insufficient_evidence")
    if primary is not None:
        actions.append(primary)
        inspections.add((primary.path, primary.start_line, primary.end_line))
    # A second helper or observed configuration can be diagnosed directly.
    targets = [*items[1:2], *(item for item in items[1:] if _configuration(item))]
    for item in targets:
        action = _inspect(item)
        if action is None or (action.path, action.start_line, action.end_line) in inspections:
            continue
        inspections.add((action.path, action.start_line, action.end_line))
        actions.append(action)
        if len(actions) == 6:
            break
    return tuple(actions)


def action_for_choice(plan: RepairPlan, choice, *, allow_inspection=True) -> RepairAction | None:
    """Map a known choice to immutable Python selectors, with a safe default."""
    actions = build_repair_actions(plan, allow_inspection=allow_inspection)
    return next((action for action in actions if action.id == choice), actions[0] if actions else None)

