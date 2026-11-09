"""Bounded automatic reacquisition of small changed Python files for base coding.

Receipts are data, never edit authority. Only normally observed source cards
that actually enter the next request can be registered for an edit.
"""
import hashlib
import json
from pathlib import Path
import time

from patchrook_agent.tools.base import ToolError


KIND = "modified_source_refresh"


def refresh_modified_sources(workflow, deadline):
    if (not workflow.config.evidence_refresh_modified_python or workflow.revision < 1
            or "read_file" not in workflow.enabled
            or workflow.routing is not None
                and workflow.routing["active_model"] == workflow.config.escalation_model):
        return []
    # One bounded acquisition per revision survives checkpoints. Reusing these
    # cards still revalidates hashes; a stale receipt cannot authorize anything.
    previous = next((card for card in reversed(workflow.evidence.cards(fresh=False))
                     if card.kind == KIND and card.revision == workflow.revision), None)
    if previous is not None:
        cards = [card for identifier in previous.metadata.get("source_ids", [])
                 if (card := workflow.evidence.get(identifier, deadline=deadline)) is not None
                 and card.kind == "source"]
        workflow.session.emit(KIND + "_reuse", workflow.state.steps, revision=workflow.revision,
                              source_ids=[card.id for card in cards], tool_reads=0,
                              edit_authorization=False)
        return cards
    cards, files = [], []
    paths = sorted(path for path in workflow.state.modified_files if Path(path).suffix.lower() == ".py")
    for number, path in enumerate(paths):
        record = {"path": path}
        files.append(record)
        if number >= 2:
            record["status"] = "file_limit"
            continue
        if time.monotonic() >= deadline:
            record["status"] = "deadline"
            continue
        try:
            raw = workflow.tools._bytes(workflow.tools._path(path))
            text = raw.decode("utf-8")
            lines = text.splitlines()
            record.update(bytes=len(raw), lines=len(lines), inspected=True)
            if not lines:
                record["status"] = "empty_file"
                continue
            if len(lines) > workflow.workspace.config.max_read_lines:
                record["status"] = "line_limit"
                continue
            numbered = "\n".join(f"{index}: {line}" for index, line in enumerate(lines, 1))
            if len(numbered) + 512 > workflow.workspace.config.max_output_chars:
                record["status"] = "output_limit"
                continue
            source_sha = hashlib.sha256(raw).hexdigest()
            current = next((card for card in reversed(workflow.evidence.cards(fresh=False))
                            if card.kind == "source" and card.path == path
                            and card.source_sha256 == source_sha and card.text == text
                            and card.start_line == 1 and card.end_line == len(lines)), None)
            if current is not None:
                record.update(status="already_fresh", source_id=current.id, source_sha256=source_sha)
                cards.append(current)
                continue
            # Persist read debt before the normal tool boundary. An interrupted
            # automatic read is not replayed on resume (source can be read by
            # the coder explicitly if no complete card was recorded).
            record.update(status="read_pending", tool_read=True)
            workflow.evidence.add_observation(KIND, json.dumps({"revision": workflow.revision,
                "files": files, "edit_authorization": False}, separators=(",", ":")),
                revision=workflow.revision, metadata={"source_ids": [card.id for card in cards]})
            direct = list(workflow.direct)
            try:
                result = workflow._perform_action(workflow.state,
                    {"name": "read_file", "arguments": {"path": path, "start_line": 1}},
                    deadline, automatic=True, source_refresh=True)
                current = next((card for identifier in reversed(workflow.direct)
                                if (card := workflow.evidence.get(identifier, deadline=deadline)) is not None
                                and card.kind == "source" and card.path == path
                                and card.start_line == 1 and card.end_line == len(lines)
                                and card.source_sha256 == source_sha and card.text == text), None)
            finally:
                # Automatic refresh is admitted as bounded optional evidence,
                # not an unbounded new mandatory source pin.
                workflow.direct = direct
            record["tool_read"] = True
            if (result.get("success") and not result.get("truncated")
                    and result.get("total_lines") == len(lines) and current is not None):
                record.update(status="read", source_id=current.id, source_sha256=source_sha)
                cards.append(current)
            else:
                record.update(status="incomplete_read", error_code=result.get("error", {}).get("code"))
        except (OSError, ValueError, UnicodeError, ToolError, TimeoutError) as exc:
            record.update(status="read_error", error_type=type(exc).__name__)
    receipt = {"revision": workflow.revision, "files": files, "source_ids": [card.id for card in cards],
               "edit_authorization": False}
    workflow.evidence.add_observation(KIND, json.dumps(receipt, separators=(",", ":")),
        revision=workflow.revision, metadata={"source_ids": receipt["source_ids"]})
    workflow.session.emit(KIND, workflow.state.steps, **receipt)
    workflow._checkpoint(workflow.state)
    return cards
