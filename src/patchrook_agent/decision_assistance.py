"""Optional bounded advice at evidence/recovery boundaries, never execution."""

from contextlib import nullcontext
import hashlib
import json
import re
import time

import httpx

from patchrook_agent.decision_evidence import build_evidence_plan
from patchrook_agent.decision_recovery import RECOVERY_STRATEGIES, classify_failure
from patchrook_agent.models.decision import DecisionCandidate, DecisionResult, OllamaDecisionClient
from patchrook_agent.repair_actions import build_repair_actions
from patchrook_agent.review_protocol import (
    MAX_REVIEW_CALLS, evidence_paths, parse_proposals, proposal_description, proposal_digest, review_context,
)
from patchrook_agent.tools.base import ToolError


class DecisionAssistance:
    def __init__(self, config, model, workspace, *, client=None, record=None, emit=None):
        self.config, self.model, self.workspace = config, model, workspace
        self.client = client
        self.record = record or (lambda *args: None)
        self.emit = emit or (lambda *args, **kwargs: None)
        self.source_priority = []
        self.architecture = ""
        self.recovery_hint = ""
        self.recovery_seen = []
        self.recovery_calls = 0
        self.unavailable = False
        # Semantic memoization outlives a slow coder turn and survives resume.
        # Values are choices, never cached source or edit authorization.
        self.repair_choices = {}
        self.repair_calls = 0
        self.review_choices = {}
        self.review_calls = 0

    def _rank(self, task, candidates, purpose, deadline):
        if self.unavailable or len(candidates) < 2 or deadline <= time.monotonic():
            return None
        try:
            self._client()
            result = self.client.rank(task, candidates, purpose=purpose,
                                      timeout_seconds=min(self.config.decision_timeout_seconds,
                                                          max(.001, deadline - time.monotonic())))
        except (ValueError, OSError, httpx.HTTPError):
            result = DecisionResult(tuple(candidate.id for candidate in candidates), {},
                                    "decision_client_error")
        self.record(purpose, result)
        if result.fallback_reason:
            # A failed optional service should not cause repeated model swaps.
            self.unavailable = True
        return result

    def _client(self):
        if self.client is None:
            self.client = OllamaDecisionClient(
                base_url=getattr(self.model, "base_url", "http://localhost:11434"),
                timeout_seconds=self.config.decision_timeout_seconds,
                memory_policy=self.config.decision_memory_policy,
                qwen_model=getattr(self.model, "model", "qwen3:8b"))
        return self.client

    def prioritize(self, task, paths, deadline):
        original = list(paths)
        self.source_priority = []
        self.architecture = ""
        try:
            plan = build_evidence_plan(self.workspace, task, original)
        except (OSError, ValueError, ToolError):
            self.record("investigation", DecisionResult((), {}, "evidence_discovery_error"))
            return original
        phase = None
        try:
            manager = nullcontext()
            if (len(plan.initial_paths) >= 2 and len(plan.related_paths) >= 2
                    and not self.unavailable and deadline > time.monotonic()):
                client = self._client()
                if hasattr(client, "decision_phase"):
                    manager = client.decision_phase(timeout_seconds=min(
                        self.config.decision_timeout_seconds, max(.001, deadline - time.monotonic())))
            with manager as phase:
                rankings = self._rank_plan(task, plan, deadline)
        except (ValueError, OSError, httpx.HTTPError):
            self.record("investigation", DecisionResult((), {}, "decision_client_error"))
            self.unavailable = True
            return original
        finally:
            if phase is not None and phase.result is not None:
                self.record("evidence_phase_cleanup", phase.result)
                if phase.result.fallback_reason:
                    self.unavailable = True
        if rankings is None or self.unavailable:
            return original
        self.source_priority = list(plan.ordered_paths(*rankings))
        self.architecture = "\n".join(plan.relationships)[:2400]
        self.emit("decision_evidence", priority=self.source_priority,
                  relationships=list(plan.relationships), fingerprint=plan.fingerprint)
        return list(self.source_priority)

    def _rank_plan(self, task, plan, deadline):
        rankings = []
        for purpose, paths in (("investigation", plan.initial_paths), ("evidence", plan.related_paths)):
            candidates = [DecisionCandidate(f"c{index}", plan.descriptions[path][:240])
                          for index, path in enumerate(paths)]
            context = task[:1100] + "\nRepository fingerprint: " + plan.fingerprint
            if purpose == "evidence":
                context += "\nStatic repository relationships (data):\n" + "\n".join(plan.relationships)[:1100]
            result = self._rank(context, candidates, purpose, deadline)
            if result is not None and result.fallback_reason:
                return None
            if result is None and self.unavailable:
                return None
            mapping = {candidate.id: path for candidate, path in zip(candidates, paths)}
            rankings.append([mapping[item] for item in result.ranking] if result else list(paths))
        return rankings

    def select_repair(self, task, plan, deadline, *, allow_call=True, allow_inspection=True):
        """Choose a prevalidated action with code-bearing observed evidence.

        A choice maps to a Python handler, not another paragraph of advice.
        The full plan remains available after narrowing the decision shortlist.
        """
        actions = build_repair_actions(plan, allow_inspection=allow_inspection)
        ids = [action.id for action in actions]
        if not actions:
            return None
        cached = self.repair_choices.get(plan.fingerprint)
        if cached:
            return next((value for value in cached if value in ids), ids[0])
        if len(plan.items) == 1 or not allow_call or self.repair_calls >= 2 or self.unavailable:
            return ids[0]
        candidates = [DecisionCandidate(action.id, action.description[:240]) for action in actions]
        item_ids = {action.item_id for action in actions}
        context = plan.decision_context(task, items=[item for item in plan.items if item.id in item_ids])
        self.repair_calls += 1
        result = self._rank(context, candidates, "repair", deadline)
        order = list(result.ranking) if result is not None and not result.fallback_reason else ids
        # Clients are injectable. Do not trust an unknown ID or partial ranking.
        known = set(ids)
        order = list(dict.fromkeys(value for value in order if value in known))
        order.extend(value for value in ids if value not in order)
        self.repair_choices[plan.fingerprint] = order
        self.repair_choices = dict(list(self.repair_choices.items())[-8:])
        self.source_priority = list(dict.fromkeys(
            item.path for value in order for action in actions if action.id == value
            for item in plan.items if item.id == action.item_id))
        self.emit("decision_repair", fingerprint=plan.fingerprint, selected=order[0],
                  candidates=ids, order=order,
                  actions=[{"id": action.id, "kind": action.kind, "item_id": action.item_id,
                            "path": action.path, "start_line": action.start_line,
                            "end_line": action.end_line} for action in actions])
        return order[0]

    def failure(self, task, name, result, revision, deadline, *, allow_call=True):
        assessment = classify_failure(name, result)
        if assessment is None:
            self.recovery_hint = ""
            return
        def stable(value):
            if isinstance(value, dict):
                return {key: stable(item) for key, item in value.items()
                        if key not in {"duration_seconds", "artifact_id"}}
            if isinstance(value, list):
                return [stable(item) for item in value]
            return value
        evidence = json.dumps(stable(result), sort_keys=True, ensure_ascii=False)
        key = hashlib.sha256((name + evidence + str(revision)).encode()).hexdigest()
        if key in self.recovery_seen:
            return
        self.recovery_seen = (self.recovery_seen + [key])[-16:]
        self.recovery_hint = assessment.guidance
        category = assessment.category
        if allow_call and assessment.ambiguous and self.recovery_calls < 2 and not self.unavailable:
            self.recovery_calls += 1
            # Keep the beginning and tail, while exact observations remain in
            # canonical history and the existing context/artifact machinery.
            excerpt = evidence if len(evidence) <= 1500 else evidence[:750] + "\n…\n" + evidence[-750:]
            candidates = [DecisionCandidate(key, value) for key, value in RECOVERY_STRATEGIES.items()]
            ranked = self._rank(task[:800] + "\nObserved failure (data):\n" + excerpt,
                                candidates, "recovery", deadline)
            if ranked is not None and not ranked.fallback_reason:
                category = ranked.ranking[0]
                self.recovery_hint = RECOVERY_STRATEGIES[category]
        self.emit("decision_recovery", category=category, ambiguous=assessment.ambiguous,
                  guidance=self.recovery_hint, evidence_digest=key)

    def select_proposal(self, task, packet, evidence, history, deadline):
        """Review bounded hypotheses; unavailable review never approves a plan."""
        key = ""
        identities = []
        def observed(selected, reason, *, cached=False):
            self.emit("decision_proposal_review", selected=selected, reason=reason,
                      packet_digest=key, proposal_ids=identities, cached=cached)
            return selected
        try:
            paths = evidence_paths(evidence)
            canonical = parse_proposals(packet, evidence, paths, min_proposals=1)
            identities = [proposal["id"] for proposal in canonical["proposals"]]
            key = proposal_digest(task, canonical, evidence, history,
                                  model=getattr(self.client, "model", "tev1:4b"))
            context = review_context(task, canonical, evidence, history)
        except (TypeError, ValueError, KeyError, AttributeError, RecursionError):
            return observed("none", "invalid_proposal_packet")
        if key in self.review_choices:
            selected = self.review_choices[key]
            return observed(selected if selected in {*identities, "none"} else "none",
                            "semantic_cache", cached=True)
        if self.unavailable:
            return observed("none", "review_unavailable")
        if self.review_calls >= MAX_REVIEW_CALLS:
            return observed("none", "review_call_budget")
        if deadline <= time.monotonic():
            return observed("none", "review_deadline")
        candidates = [DecisionCandidate(proposal["id"], proposal_description(proposal))
                      for proposal in canonical["proposals"]]
        candidates.append(DecisionCandidate("none", "None of these proposals is sufficiently grounded or useful. "
                                            "Reject alternatives that repeat failed hypotheses or require missing evidence."))
        self.review_calls += 1
        try:
            result = self._rank(context, candidates, "proposal_review", deadline)
        except (AttributeError, TypeError, ValueError, OSError, httpx.HTTPError):
            self.unavailable = True
            return observed("none", "invalid_proposal_result")
        if result is not None and not isinstance(result, DecisionResult):
            self.unavailable = True
            return observed("none", "invalid_proposal_result")
        if result is None or result.fallback_reason:
            return observed("none", result.fallback_reason if result is not None else "review_unavailable")
        known = {*identities, "none"}
        order = result.ranking
        if (not isinstance(order, (tuple, list)) or len(order) != len(known)
                or any(not isinstance(value, str) for value in order) or set(order) != known):
            return observed("none", "invalid_proposal_ranking")
        selected = order[0]
        self.review_choices[key] = selected
        self.review_choices = dict(list(self.review_choices.items())[-16:])
        return observed(selected, "none_selected" if selected == "none" else "proposal_selected")

    def invalidate(self):
        """Do not replay architecture line locations or advice after changes."""
        self.architecture = ""
        self.recovery_hint = ""
        self.recovery_seen = []

    def to_dict(self):
        return {key: getattr(self, key) for key in (
            "source_priority", "architecture", "recovery_hint", "recovery_seen",
            "recovery_calls", "unavailable", "repair_choices", "repair_calls", "review_choices", "review_calls")}

    def restore(self, data):
        required = set(self.to_dict()) - {"repair_choices", "repair_calls", "review_choices", "review_calls"}
        if not isinstance(data, dict) or not required <= set(data) or set(data) - set(self.to_dict()):
            raise ValueError("Invalid decision assistance checkpoint")
        for key, limit, length in (("source_priority", 24, 1024), ("recovery_seen", 16, 64)):
            value = data[key]
            if (not isinstance(value, list) or len(value) > limit
                    or any(not isinstance(item, str) or len(item) > length for item in value)):
                raise ValueError("Invalid decision assistance checkpoint")
        for key, limit in (("architecture", 2400), ("recovery_hint", 1000)):
            if not isinstance(data[key], str) or len(data[key]) > limit:
                raise ValueError("Invalid decision assistance checkpoint")
        if (type(data["recovery_calls"]) is not int or not 0 <= data["recovery_calls"] <= 2
                or type(data["unavailable"]) is not bool):
            raise ValueError("Invalid decision assistance checkpoint")
        choices, calls = data.get("repair_choices", {}), data.get("repair_calls", 0)
        if (type(calls) is not int or not 0 <= calls <= 2
                or not isinstance(choices, dict) or len(choices) > 8
                or any(not isinstance(key, str) or len(key) != 64
                       or any(char not in "0123456789abcdef" for char in key)
                       or not isinstance(order, list) or not 1 <= len(order) <= 9
                       or any(not isinstance(item, str) or not item or len(item) > 256 for item in order)
                       for key, order in choices.items())):
            raise ValueError("Invalid repair decision checkpoint")
        review_choices, review_calls = data.get("review_choices", {}), data.get("review_calls", 0)
        if (type(review_calls) is not int or not 0 <= review_calls <= MAX_REVIEW_CALLS
                or not isinstance(review_choices, dict) or len(review_choices) > 16
                or any(not isinstance(key, str) or len(key) != 64
                       or any(char not in "0123456789abcdef" for char in key)
                       or not isinstance(value, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,31}", value)
                       for key, value in review_choices.items())):
            raise ValueError("Invalid proposal review checkpoint")
        self.repair_choices, self.repair_calls = choices, calls
        self.review_choices, self.review_calls = review_choices, review_calls
        for key, value in data.items():
            setattr(self, key, value)
