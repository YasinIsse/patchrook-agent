"""Consistent estimated prompt budgets, separate from reported model tokens."""

from dataclasses import asdict, dataclass
import json
import math
from typing import Any


def estimate_tokens(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> int:
    """Reuse the Milestone 1 byte estimate; this is not an exact tokenizer."""
    payload = json.dumps({"messages": messages, "tools": tools},
                         ensure_ascii=False, separators=(",", ":"))
    return math.ceil(len(payload.encode("utf-8", errors="replace")) / 3) + 32 * len(messages)


@dataclass(frozen=True)
class ContextConfig:
    context_size: int = 8192
    trigger: float = 0.80
    # Retained for explicit legacy configurations; no fixed target by default.
    target: float | None = None
    hard_limit: float = 0.85
    reserved_output_tokens: int = 512
    enabled: bool = True
    keep_recent_turns: int = 2
    headroom_tokens: int = 512
    estimation_margin_tokens: int = 128

    def __post_init__(self):
        if isinstance(self.context_size, bool) or not isinstance(self.context_size, int):
            raise ValueError("context_size must be an integer")
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               for value in (self.trigger, self.hard_limit)):
            raise ValueError("Context thresholds must be numeric fractions")
        if not 0 < self.trigger < self.hard_limit <= 1:
            raise ValueError("Require 0 < trigger < hard_limit <= 1")
        if self.target is not None and (
                isinstance(self.target, bool) or not isinstance(self.target, (int, float))
                or not 0 < self.target < self.trigger):
            raise ValueError("An explicit legacy target must satisfy 0 < target < trigger")
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be a boolean")
        if not isinstance(self.reserved_output_tokens, int) or isinstance(self.reserved_output_tokens, bool):
            raise ValueError("reserved_output_tokens must be an integer")
        if not 512 <= self.reserved_output_tokens < self.context_size:
            raise ValueError("Reserve at least 512 generation tokens, below context_size")
        if isinstance(self.keep_recent_turns, bool) or not isinstance(self.keep_recent_turns, int) or self.keep_recent_turns < 0:
            raise ValueError("keep_recent_turns must be a nonnegative integer")
        for name in ("headroom_tokens", "estimation_margin_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.input_limit <= 0:
            raise ValueError("Output reservation and estimation margin leave no input capacity")
        if self.enabled and self.context_size * self.trigger >= self.input_limit:
            raise ValueError(
                "Compaction trigger must precede the effective input ceiling; lower trigger "
                "or output reservation, or select a larger supported context window")

    @property
    def input_limit(self) -> int:
        return min(math.floor(self.context_size * self.hard_limit),
                   self.context_size - self.reserved_output_tokens - self.estimation_margin_tokens)

    @property
    def compaction_goal(self) -> int:
        """Best-effort room for subsequent observations, never a safety override."""
        if self.target is not None:
            return math.floor(self.context_size * self.target)
        return max(0, min(math.floor(self.context_size * self.trigger), self.input_limit)
                   - self.headroom_tokens)

    def fits(self, estimated_tokens: int) -> bool:
        return estimated_tokens <= self.input_limit

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
