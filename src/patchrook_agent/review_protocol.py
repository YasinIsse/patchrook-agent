"""Bounded, non-executing proposals for selective review of difficult repairs."""

from __future__ import annotations

import hashlib
from difflib import SequenceMatcher
import json
import posixpath
import re


PROPOSAL_TOOL = "propose_next_steps"
MAX_REVIEW_CALLS = 8
MAX_PROPOSALS = 3
_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,31}\Z")
_EVIDENCE_ID = re.compile(r"e[1-9][0-9]{0,2}\Z")
_EVIDENCE_KINDS = {"source", "failure", "verification", "new_file", "known_path"}


def _relative(value):
    return (isinstance(value, str) and 0 < len(value) <= 1024 and "\x00" not in value
            and "\\" not in value and value == posixpath.normpath(value)
            and value not in {".", ".."} and not value.startswith(("/", "../")))


def _text(value, name, limit):
    if not isinstance(value, str) or not value.strip() or len(value) > limit or "\x00" in value:
        raise ValueError(f"{name} must be nonempty text of at most {limit} characters")
    return value.strip()


def evidence_paths(evidence) -> list[str]:
    if not isinstance(evidence, dict):
        raise ValueError("Evidence must be an object")
    return list(dict.fromkeys(record["path"] for record in evidence.values()
                              if isinstance(record, dict) and _relative(record.get("path"))))


def _inputs(evidence, allowed_paths, allow_inspection):
    if type(allow_inspection) is not bool:
        raise ValueError("allow_inspection must be a boolean")
    if (not isinstance(evidence, dict) or not evidence or len(evidence) > 64
            or any(not isinstance(key, str) or not _EVIDENCE_ID.fullmatch(key)
                   or not isinstance(value, dict) for key, value in evidence.items())):
        raise ValueError("Evidence must contain known e1..eN records")
    for value in evidence.values():
        if value.get("kind") not in _EVIDENCE_KINDS or not isinstance(value.get("text"), str):
            raise ValueError("Evidence requires an observed kind and text")
        if value.get("kind") in {"source", "new_file", "known_path"} and not _relative(value.get("path")):
            raise ValueError("Source evidence requires a canonical relative path")
    if (not isinstance(allowed_paths, list) or len(allowed_paths) > 64
            or any(not _relative(path) for path in allowed_paths)):
        raise ValueError("Allowed paths must be canonical relative paths")
    return tuple(dict.fromkeys(allowed_paths))


def proposal_definition(evidence: dict[str, dict], *, allowed_paths: list[str],
                        allow_inspection: bool = True) -> dict:
    """A virtual native tool: proposals are data and never perform an edit."""
    paths = _inputs(evidence, allowed_paths, allow_inspection)
    properties = {
        "id": {"type": "string", "pattern": r"^[A-Za-z][A-Za-z0-9_-]{0,31}$", "maxLength": 32},
        "kind": {"type": "string", "enum": (["implement_change", "inspect_source"]
                                                 if allow_inspection else ["implement_change"])},
        "path": {"type": "string", "enum": list(paths)},
        "approach": {"type": "string", "minLength": 1, "maxLength": 240},
        "expected": {"type": "string", "minLength": 1, "maxLength": 160},
        "evidence_refs": {"type": "array", "minItems": 1, "maxItems": 3, "uniqueItems": True,
                          "description": "Required in every proposal: cite 1-3 supplied evidence IDs; never invent IDs.",
                          "items": {"type": "string", "enum": list(evidence)}},
    }
    required = ["id", "kind", "path", "approach", "expected", "evidence_refs"]
    if allow_inspection:
        properties.update(
            start_line={"type": "integer", "minimum": 1,
                        "description": "Required only for inspect_source; omit for implement_change."},
            end_line={"type": "integer", "minimum": 1,
                      "description": "Required only for inspect_source; inclusive span must be at most 24 lines."},
        )
    return {"type": "function", "function": {
        "name": PROPOSAL_TOOL,
        "description": ("Propose 2 distinct concise next steps for review, or 3 when necessary. "
                        "Use supplied evidence IDs and allowed paths. Each inspection spans at most 24 lines. "
                        "Keep total prose under 1000 characters. This tool proposes only; it edits nothing."),
        "parameters": {
            "type": "object", "additionalProperties": False,
            "required": ["goal", "reason", "proposals"],
            "properties": {
                "goal": {"type": "string", "minLength": 1, "maxLength": 180},
                "reason": {"type": "string", "minLength": 1, "maxLength": 200},
                "proposals": {"type": "array", "minItems": 2, "maxItems": 3,
                              "items": {"type": "object", "additionalProperties": False,
                                        "required": required, "properties": properties}},
            },
        },
    }}


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate proposal object key")
        result[key] = value
    return result


def _field_errors(value, required, allowed, label):
    errors = []
    missing = required - set(value)
    unsupported = set(value) - allowed
    def names(keys):
        return ", ".join(repr(str(key)[:40]) for key in sorted(keys, key=str)[:6])
    if missing:
        errors.append(f"{label}: missing required fields: {names(missing)}")
    if unsupported:
        errors.append(f"{label}: unsupported keys: {names(unsupported)}")
    return errors


def _invalid_proposals(errors):
    message = "Invalid proposals: " + "; ".join(error[:250] for error in errors[:16])
    if len(message) > 1400:
        message = message[:1360] + "; additional errors omitted."
    raise ValueError(message)


def parse_proposals(arguments, evidence, allowed_paths, allow_inspection=True, *, min_proposals=2) -> dict:
    """Validate a closed proposal packet without granting execution authority."""
    paths = _inputs(evidence, allowed_paths, allow_inspection)
    if type(min_proposals) is not int or min_proposals not in {1, 2}:
        raise ValueError("min_proposals must be one or two")
    if isinstance(arguments, str):
        if len(arguments) > 4096:
            raise ValueError("Proposal arguments are too large")
        try:
            arguments = json.loads(arguments, object_pairs_hook=_object)
        except (ValueError, TypeError, RecursionError) as exc:
            raise ValueError("Proposal arguments must be one valid JSON object: " + str(exc)[:160]) from exc
    if not isinstance(arguments, dict):
        raise ValueError("Proposal packet must be an object with goal, reason, and proposals")
    wrapper_keys = {"goal", "reason", "proposals"}
    errors = _field_errors(arguments, wrapper_keys, wrapper_keys, "packet")
    def text_field(value, key, limit, label):
        if key not in value:
            return None  # The structural error already names this missing key.
        try:
            return _text(value[key], label + "." + key, limit)
        except ValueError as exc:
            errors.append(str(exc))
            return None
    goal = text_field(arguments, "goal", 180, "packet")
    reason = text_field(arguments, "reason", 200, "packet")
    proposals = arguments.get("proposals")
    if not isinstance(proposals, list) or not min_proposals <= len(proposals) <= MAX_PROPOSALS:
        if "proposals" in arguments:
            errors.append(f"Provide {min_proposals} to three distinct proposals in packet.proposals")
        _invalid_proposals(errors)
    result, identities, alternatives = [], set(), set()
    prose = len(goal or "") + len(reason or "")
    for index, proposal in enumerate(proposals):
        label = f"proposals[{index}]"
        if not isinstance(proposal, dict):
            errors.append(label + ": each proposal must be an object")
            continue
        before_errors = len(errors)
        identity = proposal.get("id")
        if isinstance(identity, str) and _IDENTIFIER.fullmatch(identity):
            label += f" ({identity})"
        kind = proposal.get("kind")
        keys = {"id", "kind", "path", "approach", "expected", "evidence_refs"}
        if kind == "inspect_source":
            keys |= {"start_line", "end_line"}
        errors.extend(_field_errors(proposal, keys, keys, label))
        if "kind" in proposal and kind not in ("inspect_source", "implement_change"):
            errors.append(label + ": unsupported kind " + repr(str(kind)[:64])
                          + "; use inspect_source or implement_change")
        if "id" in proposal:
            if (not isinstance(identity, str) or not _IDENTIFIER.fullmatch(identity)
                    or identity.casefold() == "none" or identity.casefold() in identities):
                errors.append(label + ": id must be a unique safe ID other than none")
            else:
                identities.add(identity.casefold())
        path = proposal.get("path")
        path_valid = _relative(path) and path in paths
        if "path" in proposal and not path_valid:
            errors.append(label + ": path is not a known allowed path: " + repr(str(path)[:80]))
        approach = text_field(proposal, "approach", 240, label)
        expected = text_field(proposal, "expected", 160, label)
        refs = proposal.get("evidence_refs")
        refs_valid = (isinstance(refs, list) and 1 <= len(refs) <= 3
                      and all(isinstance(ref, str) and ref in evidence for ref in refs)
                      and len(set(refs)) == len(refs))
        if "evidence_refs" in proposal and not refs_valid:
            errors.append(label + ": evidence_refs must contain 1-3 distinct supplied evidence IDs")
        clean = {"id": identity, "kind": kind, "path": path, "approach": approach,
                 "expected": expected, "evidence_refs": list(refs) if refs_valid else []}
        signature = None
        if kind == "inspect_source":
            start, end = proposal.get("start_line"), proposal.get("end_line")
            if not allow_inspection:
                errors.append(label + ": inspect_source is disabled")
            for key, value in (("start_line", start), ("end_line", end)):
                if key in proposal and (type(value) is not int or value < 1):
                    errors.append(label + ": " + key + " must be a positive integer")
            if type(start) is int and type(end) is int and start >= 1 and end >= 1:
                if end < start:
                    errors.append(label + ": end_line must be at least start_line")
                elif end - start >= 24:
                    errors.append(label + f": inspect_source range {start}-{end} spans {end - start + 1} lines; maximum is 24 lines")
                else:
                    clean.update(start_line=start, end_line=end)
                    if path_valid:
                        signature = (kind, path, start, end)
        elif kind == "implement_change":
            if path_valid and refs_valid and not any(evidence[ref].get("path") == path
                       and evidence[ref].get("kind") in {"source", "new_file"}
                       and evidence[ref]["text"].strip() for ref in refs):
                errors.append(label + ": implementation must cite actual source or new-file evidence for its path")
            # Operators are meaningful evidence in coding hypotheses: a+b and
            # a-b must remain different alternatives. Normalize prose casing,
            # spacing and sentence punctuation without erasing code symbols.
            if path_valid and approach is not None:
                normalized = re.sub(r"\s+", " ", approach.casefold()).strip().rstrip(".!?")
                signature = (kind, path, normalized)
        if signature is not None:
            if signature in alternatives:
                errors.append(label + ": proposals must describe meaningfully distinct alternatives")
            alternatives.add(signature)
        prose += len(approach or "") + len(expected or "")
        if len(errors) == before_errors:
            result.append(clean)
    if prose > 1000:
        errors.append("Proposal prose exceeds the compact review budget")
    if errors:
        _invalid_proposals(errors)
    packet = {"goal": goal, "reason": reason, "proposals": result}
    if len(json.dumps(packet, ensure_ascii=False, separators=(",", ":"))) > 2000:
        raise ValueError("Proposal packet exceeds the compact output budget")
    return packet


def proposal_description(proposal: dict) -> str:
    """The option description remains a bounded interpretation of one proposal."""
    label = f"{proposal['path']}; {proposal['kind']}"
    if proposal["kind"] == "inspect_source":
        label += f" lines {proposal['start_line']}-{proposal['end_line']}"
    return (label + ": " + proposal["approach"])[:240]


def _semantic(value):
    if isinstance(value, dict):
        return {key: _semantic(item) for key, item in value.items()
                if key not in {"duration_seconds", "timestamp_utc", "elapsed_seconds", "artifact_id"}}
    if isinstance(value, list):
        return [_semantic(item) for item in value]
    if isinstance(value, str):
        return re.sub(r"(?m)(\b\d+ (?:failed|passed|errors?|skipped)(?:, \d+ [a-z]+)* in )\d+(?:\.\d+)?s\b",
                      r"\1<elapsed>s", value)
    return value


def proposal_digest(task, packet, evidence, history, *, model="tev1:4b") -> str:
    semantic = {"policy": 1, "model": model, "task": task, "packet": packet,
                "evidence": evidence, "history": history}
    return hashlib.sha256(json.dumps(_semantic(semantic), sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _evidence_excerpt(record, budget):
    lines = record["text"].splitlines()
    if record.get("kind") in {"failure", "verification"}:
        preferred = [line for line in lines if "assert " in line or re.match(r"\s*E\s+", line)
                     or "Error:" in line or "Exception:" in line]
    else:
        preferred = [line for line in lines if re.search(r"\breturn\b|//|[+*%]|\s=\s", line)]
        preferred.sort(key=lambda line: (not bool(re.search(r"\breturn\b", line)),
                                         not bool(re.search(r"//|[+*%]|\s-\s", line))))
    ordered = list(dict.fromkeys([*preferred, *lines]))
    result = []
    used = 0
    for line in ordered:
        if used + len(line) + bool(result) <= budget:
            result.append(line)
            used += len(line) + bool(len(result) > 1)
    return "\n".join(result)


def _variant_change(item, budget):
    changes = item.get("changes", [])
    if not isinstance(changes, list):
        return "", ""
    for change in reversed(changes):
        if not isinstance(change, dict):
            continue
        delta = change.get("accepted_delta", {})
        if not isinstance(delta, dict):
            continue
        new = delta.get("new_text", delta.get("content"))
        if not isinstance(new, str) or not new.strip():
            continue
        old = delta.get("old_text")
        lines = new.splitlines()
        if isinstance(old, str) and len(lines) > 1:
            before = old.splitlines()
            changed = [line for tag, _, _, first, last in SequenceMatcher(
                None, before, lines, autojunk=False).get_opcodes() if tag in {"insert", "replace"}
                       for line in lines[first:last]]
            lines = changed or lines
        text = _evidence_excerpt({"kind": "source", "text": "\n".join(line.strip() for line in lines)}, budget)
        return str(change.get("path", ""))[:50], text
    return "", ""


def _failed_history(history, budget=460):
    """Two balanced failed variants, retaining code and exact counterexamples."""
    values = history.get("failed_states") if isinstance(history, dict) else None
    if not isinstance(values, list):
        return ""
    variants = []
    seen = set()
    for item in reversed(values):
        if not isinstance(item, dict):
            continue
        path, code = _variant_change(item, 140)
        witness = str(item.get("witness", ""))
        if not witness and isinstance(item.get("outcome"), dict):
            witness = "\n".join(str(item["outcome"].get(key, "")) for key in ("stdout", "stderr"))
        identity = (path, code, witness)
        if identity in seen:
            continue
        seen.add(identity)
        variants.append((path, code, witness))
        if len(variants) == 2:
            break
    variants.reverse()
    if not variants:
        return ""
    each = (budget - 3 * (len(variants) - 1)) // len(variants)
    rows = []
    for path, code, witness in variants:
        label = f"Tried {path}: " if path else "Earlier variant: "
        code = _evidence_excerpt({"kind": "source", "text": code}, max(0, min(140, each - len(label) - 55)))
        # E-lines carry the observed unequal amounts. Keep them ahead of long
        # test names or expected-assertion labels when the witness is bounded.
        lines = witness.splitlines()
        priority = [line for line in lines if re.match(r"\s*E\s+", line)]
        priority += [line for line in lines if "assert " in line and line not in priority]
        priority += [line for line in lines if line not in priority]
        failure = _evidence_excerpt({"kind": "failure", "text": "\n".join(priority)},
                                     max(0, each - len(label) - len(code) - 10))
        row = label + code + "\nFailed: " + failure
        rows.append(row)
    return " | ".join(rows)


def _history_excerpt(history):
    failed = _failed_history(history)
    if failed:
        return failed
    if isinstance(history, str):
        return history[:180]
    if isinstance(history, list):
        return " | ".join(_history_excerpt(item) for item in history[-2:])[:180]
    if isinstance(history, dict):
        parts = []
        # Put rejected reasoning ahead of bookkeeping hashes or timestamps.
        for key in ("proposal", "approach", "expected", "path", "failure", "outcome", "reason", "status"):
            if key in history:
                parts.append(key + "=" + _history_excerpt(history[key]))
        return "; ".join(parts)[:180] if parts else json.dumps(_semantic(history), ensure_ascii=False,
                                                               separators=(",", ":"))[:180]
    return str(history)[:180]


def review_context(task, packet, evidence, history) -> str:
    """Balance all alternatives and their observed facts below 2400 characters."""
    if not isinstance(task, str) or not isinstance(packet, dict) or not isinstance(evidence, dict):
        raise ValueError("Invalid proposal review inputs")
    proposals = packet.get("proposals")
    if not isinstance(proposals, list) or not 1 <= len(proposals) <= 3:
        raise ValueError("Invalid proposal count")
    prefix = "Task (data): " + " ".join(task.split())[:400]
    # The original requirement outranks the model's explanation of its own
    # proposal. Spend the remaining space on hypotheses and observed failures.
    prefix += "\nGoal (proposal): " + packet["goal"][:60]
    rows = []
    for proposal in proposals:
        row = (f"{proposal['id']} {proposal['kind']} {proposal['path'][:80]}"
               + (f":{proposal['start_line']}-{proposal['end_line']}" if proposal["kind"] == "inspect_source" else "")
               + "\nApproach: " + proposal["approach"] + "\nExpected: " + proposal["expected"][:80]
               + "\nRefs: " + ",".join(proposal["evidence_refs"]))
        rows.append(row)
    plan_text = "\n\n".join(rows)
    # Favor one cited fact per proposal, followed by failure evidence and then
    # other references. No candidate can consume the budget before its peers.
    refs = list(dict.fromkeys([proposal["evidence_refs"][0] for proposal in proposals]
                             + [key for key, record in evidence.items()
                                if record.get("kind") in {"failure", "verification"}]
                             + [ref for proposal in proposals for ref in proposal["evidence_refs"]]))
    refs = [ref for ref in refs if ref in evidence][:4]
    history_text = ""
    if history:
        history_text = "\nPrevious attempts (data): " + _history_excerpt(history)
    base = prefix + "\n\nProposals (data):\n" + plan_text + history_text
    remaining = 2400 - len(base) - 24
    evidence_rows = []
    if refs and remaining > 0:
        each = max(0, remaining // len(refs) - 8)
        for ref in refs:
            record = evidence[ref]
            label = f"{ref} {record.get('kind', '')} {record.get('path', '')}"[:100]
            if len(label) + 1 > each:
                raise ValueError("Proposal wording leaves no room for referenced evidence")
            excerpt = _evidence_excerpt(record, max(0, each - len(label) - 1))
            if record["text"].strip() and not excerpt:
                raise ValueError("Proposal wording leaves no room for complete evidence lines")
            evidence_rows.append(label + "\n" + excerpt)
    elif refs:
        raise ValueError("Proposal wording exceeds the bounded review context")
    text = base + "\nEvidence (observed):\n" + "\n\n".join(evidence_rows)
    if len(text) > 2400:
        raise ValueError("Review context exceeds its evidence budget")
    return text

