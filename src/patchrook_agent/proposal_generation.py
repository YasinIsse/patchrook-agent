"""Minimal constrained proposal output, compiled against observed evidence.

The model supplies two approaches. Target identity and provenance belong to
the harness and never come from model-generated paths, IDs, or coordinates.
"""

from __future__ import annotations

import json
import re

from patchrook_agent.review_protocol import (
    _inputs, _relative, _text, evidence_paths, parse_proposals,
)


MAX_GENERATION_CHARACTERS = 3000


def generation_schema() -> dict:
    """Return the uniformly required, proposal-only JSON output schema."""
    return {
        "type": "object", "additionalProperties": False,
        "required": ["approaches"],
        "properties": {
            "approaches": {
                "type": "array", "minItems": 2, "maxItems": 2,
                "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["approach", "expected"],
                    "properties": {
                        "approach": {"type": "string", "minLength": 1, "maxLength": 240},
                        "expected": {"type": "string", "minLength": 1, "maxLength": 160},
                    },
                },
            },
        },
    }


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate generation object key: " + repr(key)[:60])
        result[key] = value
    return result


def _closed(value, required, label):
    if not isinstance(value, dict):
        raise ValueError(label + " must be an object")
    missing, extra = required - set(value), set(value) - required
    errors = []
    if missing:
        errors.append("missing required fields: " + ", ".join(sorted(missing)))
    if extra:
        errors.append("unsupported keys: " + ", ".join(repr(str(key)[:40]) for key in sorted(extra, key=str)[:6]))
    if errors:
        raise ValueError(label + ": " + "; ".join(errors))


def parse_generation(content) -> dict:
    """Validate exactly two distinct approaches without granting authority."""
    if isinstance(content, str):
        if len(content) > MAX_GENERATION_CHARACTERS:
            raise ValueError("Proposal generation exceeds 3000 characters")
        try:
            content = json.loads(content, object_pairs_hook=_unique_object)
        except (ValueError, TypeError, RecursionError) as exc:
            raise ValueError("Proposal generation must be one valid JSON object: " + str(exc)[:160]) from exc
    _closed(content, {"approaches"}, "generation")
    approaches = content["approaches"]
    if not isinstance(approaches, list) or len(approaches) != 2:
        raise ValueError("generation.approaches must contain exactly two distinct approaches")
    result, seen = [], set()
    for index, item in enumerate(approaches):
        label = f"approaches[{index}]"
        _closed(item, {"approach", "expected"}, label)
        approach = _text(item["approach"], label + ".approach", 240)
        expected = _text(item["expected"], label + ".expected", 160)
        # Preserve coding operators: a+b and a-b are different hypotheses.
        normalized = re.sub(r"\s+", " ", approach.casefold()).strip().rstrip(".!?")
        if normalized in seen:
            raise ValueError("generation.approaches must describe meaningfully distinct approaches")
        seen.add(normalized)
        result.append({"approach": approach, "expected": expected})
    canonical = {"approaches": result}
    if len(json.dumps(canonical, ensure_ascii=False, separators=(",", ":"))) > 2000:
        raise ValueError("Proposal generation exceeds the compact output budget")
    return canonical


def compile_generation(content, *, target: str, evidence: dict,
                       goal: str, reason: str) -> dict:
    """Bind model approaches to one harness-selected observed target.

    These citations record the supplied factual context, not a model claim
    that its approach is correct. Existing proposal validation and review
    remain responsible for rejecting unsupported approaches.
    """
    generated = parse_generation(content)
    if not _relative(target):
        raise ValueError("Proposal target must be a canonical relative path")
    try:
        paths = _inputs(evidence, evidence_paths(evidence), False)
    except (TypeError, RecursionError) as exc:
        raise ValueError("Proposal evidence must contain valid observed records") from exc
    if target not in paths:
        raise ValueError("Proposal target must have actual source or new-file evidence")
    # Numeric evidence ordering makes compilation stable when an equivalent
    # registry is reconstructed with a different dictionary insertion order.
    ordered = sorted(evidence, key=lambda key: int(key[1:]))
    primary = next((key for key in ordered if evidence[key].get("path") == target
                    and evidence[key]["kind"] in {"source", "new_file"}
                    and evidence[key]["text"].strip()), None)
    if primary is None:
        raise ValueError("Proposal target must have actual source or new-file evidence")
    failure = next((key for key in ordered if evidence[key]["kind"] == "failure"
                    and evidence[key]["text"].strip()), None)
    refs = [primary] + ([failure] if failure is not None else [])
    packet = {
        "goal": goal, "reason": reason,
        "proposals": [
            {"id": f"p{index}", "kind": "implement_change", "path": target,
             **item, "evidence_refs": list(refs)}
            for index, item in enumerate(generated["approaches"], 1)
        ],
    }
    return parse_proposals(packet, evidence, list(paths), allow_inspection=False)
