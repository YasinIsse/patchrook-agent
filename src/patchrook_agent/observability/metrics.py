"""Per-step measurements and reproducible run totals without model calls."""

from dataclasses import asdict, dataclass, field, fields
from copy import deepcopy
import math
from typing import Any


def _timing_summary(measurements: list[dict[str, Any]]) -> dict[str, Any]:
    """Known subtotals and rates over matching, measured response cohorts."""
    result: dict[str, Any] = {}
    coverage = {"recorded_calls": len(measurements),
                "prompt_token_calls": sum(item.get("prompt_tokens") is not None for item in measurements),
                "completion_token_calls": sum(item.get("completion_tokens") is not None for item in measurements)}
    for name, count_name in (
        ("prompt_eval_seconds", "prompt_eval_calls"), ("eval_seconds", "eval_calls"),
        ("load_seconds", "load_calls"), ("prompt_eval_cached_count", "cache_calls"),
        ("uncached_input_tokens", "uncached_input_calls"),
    ):
        values = [item[name] for item in measurements if item.get(name) is not None]
        result[name] = sum(values) if values else None
        coverage[count_name] = len(values)
    for name, token_key, seconds_key in (
        ("prefill", "uncached_input_tokens", "prompt_eval_seconds"),
        ("generation", "completion_tokens", "eval_seconds"),
    ):
        paired = [item for item in measurements if item.get(token_key) is not None
                  and item.get(seconds_key) is not None and item[seconds_key] > 0]
        seconds = sum(item[seconds_key] for item in paired)
        tokens = sum(item[token_key] for item in paired)
        result[f"{name}_tokens_per_second"] = tokens / seconds if seconds else None
        coverage[f"{name}_throughput_calls"] = len(paired)
    result["timing_coverage"] = coverage
    return result


@dataclass
class ExecutionMetrics:
    per_step: list[dict[str, Any]] = field(default_factory=list)
    inference_calls: int = 0
    tools_by_name: dict[str, dict[str, Any]] = field(default_factory=dict)
    context_utilizations: list[float] = field(default_factory=list)
    compactions: list[dict[str, Any]] = field(default_factory=list)
    compaction_attempts: list[dict[str, Any]] = field(default_factory=list)
    checkpoint_count: int = 0
    resume_attempts: int = 0
    resume_success: int = 0
    skills_discovered: int = 0
    skills_loaded: int = 0
    skills_activated: int = 0
    wall_seconds: float = 0.0
    compaction_seconds: float = 0.0
    context_preparation_seconds: float = 0.0
    checkpoint_seconds: float = 0.0
    decision_enabled: bool = False
    decision_calls: int = 0
    decision_failures: int = 0
    decision_fallbacks: int = 0
    decision_cache_hits: int = 0
    decision_seconds: float = 0.0
    decision_inference_seconds: float = 0.0
    decision_management_seconds: float = 0.0
    decision_input_tokens: int = 0
    decision_load_seconds: float | None = None
    decision_load_measurements: int = 0

    def record_decision(self, *, elapsed_seconds: float, management_seconds: float = 0.0,
                        inference_seconds: float = 0.0, load_seconds: float | None = None,
                        input_tokens: int | None = 0, fallback_reason: str | None = None,
                        cache_hit: bool = False, called: bool = True) -> None:
        """Keep optional decision costs separate from Qwen generation metrics."""
        self.decision_calls += int(called and not cache_hit)
        self.decision_cache_hits += int(cache_hit)
        self.decision_failures += int(bool(fallback_reason))
        self.decision_fallbacks += int(bool(fallback_reason))
        self.decision_seconds += elapsed_seconds
        self.decision_inference_seconds += inference_seconds
        self.decision_management_seconds += management_seconds
        self.decision_input_tokens += input_tokens or 0
        if load_seconds is not None:
            self.decision_load_seconds = (self.decision_load_seconds or 0.0) + load_seconds
            self.decision_load_measurements += 1

    def _step(self, step: int) -> dict[str, Any]:
        for record in self.per_step:
            if record["step"] == step:
                return record
        record = {"step": step}
        self.per_step.append(record)
        return record

    def record_context(self, step: int, estimated_tokens: int, utilization: float) -> None:
        self._step(step).update(estimated_context_tokens=estimated_tokens, context_utilization=utilization)
        self.context_utilizations.append(utilization)

    def record_model(self, step: int, prompt_tokens: int, completion_tokens: int,
                     inference_seconds: float, *, prompt_eval_duration_ns: int | None = None,
                     eval_duration_ns: int | None = None, load_duration_ns: int | None = None,
                     prompt_eval_cached_count: int | None = None, prompt_tokens_reported: bool = True,
                     completion_tokens_reported: bool = True) -> None:
        self.inference_calls += 1
        record = self._step(step)
        record["actual_input_tokens"] = (record.get("actual_input_tokens") or 0) + prompt_tokens
        record["actual_output_tokens"] = (record.get("actual_output_tokens") or 0) + completion_tokens
        record["inference_seconds"] = record.get("inference_seconds", 0.0) + inference_seconds

        def seconds(value: int | None) -> float | None:
            if type(value) is not int or value < 0:
                return None
            try:
                converted = value / 1e9
            except OverflowError:
                return None
            return converted if math.isfinite(converted) else None

        cached = prompt_eval_cached_count
        if (prompt_tokens_reported is not True or type(cached) is not int
                or type(prompt_tokens) is not int or not 0 <= cached <= prompt_tokens):
            cached = None
        measurements = record.setdefault("inference_measurements", [])
        measurements.append({"prompt_eval_seconds": seconds(prompt_eval_duration_ns),
                             "eval_seconds": seconds(eval_duration_ns),
                             "load_seconds": seconds(load_duration_ns),
                             "prompt_eval_cached_count": cached,
                             "uncached_input_tokens": prompt_tokens - cached if cached is not None else None,
                             "prompt_tokens": prompt_tokens if prompt_tokens_reported is True else None,
                             "completion_tokens": completion_tokens if completion_tokens_reported is True else None})
        record.update(_timing_summary(measurements))

    def record_tool(self, name: str, success: bool, duration_seconds: float, retry: bool = False) -> None:
        aggregate = self.tools_by_name.setdefault(name, {"calls": 0, "successes": 0, "failures": 0,
                                                        "retries": 0, "seconds": 0.0})
        aggregate["calls"] += 1
        aggregate["successes" if success else "failures"] += 1
        aggregate["retries"] += int(retry)
        aggregate["seconds"] += duration_seconds

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionMetrics":
        if not isinstance(data, dict) or set(data) - {item.name for item in fields(cls)}:
            raise ValueError("Unsupported metrics state")
        values = deepcopy(data)
        if "compaction_attempts" not in values and isinstance(values.get("compactions"), list):
            # Older checkpoints recorded reductions only. Each is a known
            # attempt; historical unsuccessful attempts remain unknown.
            values["compaction_attempts"] = [{**event, "reduced": True,
                                               "reason": "legacy_recorded_reduction",
                                               "outcome_reason": "Earlier ineffective attempts were not recorded"}
                                              for event in values["compactions"]]
        result = cls(**values)
        count_fields = ("inference_calls", "checkpoint_count", "resume_attempts", "resume_success",
                        "skills_discovered", "skills_loaded", "skills_activated", "decision_calls",
                        "decision_failures", "decision_fallbacks", "decision_cache_hits", "decision_input_tokens",
                        "decision_load_measurements")
        for name in count_fields:
            if type(getattr(result, name)) is not int or getattr(result, name) < 0:
                raise ValueError(f"Invalid metrics counter: {name}")
        if not isinstance(result.per_step, list) or not all(isinstance(row, dict) and type(row.get("step")) is int for row in result.per_step):
            raise ValueError("Invalid per-step metrics")
        if not isinstance(result.context_utilizations, list) or not all(
            isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value < float("inf")
            for value in result.context_utilizations
        ):
            raise ValueError("Invalid context utilization metrics")
        if (not isinstance(result.tools_by_name, dict) or not isinstance(result.compactions, list)
                or not isinstance(result.compaction_attempts, list)):
            raise ValueError("Invalid tools or compaction metrics")
        if type(result.decision_enabled) is not bool:
            raise ValueError("Invalid decision enabled metric")
        for name in ("wall_seconds", "compaction_seconds", "context_preparation_seconds", "checkpoint_seconds",
                     "decision_seconds", "decision_inference_seconds", "decision_management_seconds"):
            value = getattr(result, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid metrics duration: {name}")
        if (result.decision_load_seconds is not None
                and (type(result.decision_load_seconds) not in (int, float)
                     or not math.isfinite(result.decision_load_seconds) or result.decision_load_seconds < 0)):
            raise ValueError("Invalid decision load duration")
        if ((result.decision_load_seconds is None) != (result.decision_load_measurements == 0)
                or result.decision_load_measurements > result.decision_calls):
            raise ValueError("Invalid decision load measurement coverage")
        for record in result.per_step:
            if record["step"] < 0:
                raise ValueError("Invalid metrics step")
            if "response_received" in record and type(record["response_received"]) is not bool:
                raise ValueError("Invalid model response status")
            for name in ("actual_input_tokens", "actual_output_tokens", "estimated_context_tokens"):
                if (name in {"actual_input_tokens", "actual_output_tokens"}
                        and name in record and record[name] is None
                        and record.get("response_received") is False):
                    continue
                if name in record and (type(record[name]) is not int or record[name] < 0):
                    raise ValueError(f"Invalid per-step counter: {name}")
            for name in ("inference_seconds", "request_wall_seconds", "context_utilization"):
                if name in record and (isinstance(record[name], bool) or not isinstance(record[name], (int, float))
                                       or not math.isfinite(record[name]) or record[name] < 0):
                    raise ValueError(f"Invalid per-step measurement: {name}")
            measurements = record.get("inference_measurements", [])
            if not isinstance(measurements, list):
                raise ValueError("Invalid inference timing measurements")
            if measurements and record.get("response_received") is False:
                raise ValueError("Unreceived model response cannot contain inference timing")
            for measurement in measurements:
                if not isinstance(measurement, dict):
                    raise ValueError("Invalid inference timing measurement")
                for name in ("prompt_eval_seconds", "eval_seconds", "load_seconds"):
                    value = measurement.get(name)
                    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                              or not math.isfinite(value) or value < 0):
                        raise ValueError(f"Invalid inference timing: {name}")
                for name in ("prompt_tokens", "completion_tokens", "prompt_eval_cached_count", "uncached_input_tokens"):
                    value = measurement.get(name)
                    if value is not None and (type(value) is not int or value < 0):
                        raise ValueError(f"Invalid inference timing counter: {name}")
                cached, prompt = measurement.get("prompt_eval_cached_count"), measurement.get("prompt_tokens")
                uncached = measurement.get("uncached_input_tokens")
                if (cached is not None and (prompt is None or cached > prompt or uncached != prompt - cached)
                        or cached is None and uncached is not None):
                    raise ValueError("Invalid cached and uncached input counts")
            # Old checkpoints have no timing evidence; do not turn it into zero.
            record.update(_timing_summary(measurements))
        if sum(len(record.get("inference_measurements", [])) for record in result.per_step) > result.inference_calls:
            raise ValueError("Inference timing coverage exceeds completed model calls")
        for name, aggregate in result.tools_by_name.items():
            if not isinstance(name, str) or not isinstance(aggregate, dict):
                raise ValueError("Invalid tool metrics")
            for key in ("calls", "successes", "failures", "retries"):
                if type(aggregate.get(key)) is not int or aggregate[key] < 0:
                    raise ValueError("Invalid tool metrics counters")
            seconds = aggregate.get("seconds")
            if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
                raise ValueError("Invalid tool metrics duration")
        for event in result.compactions + result.compaction_attempts:
            if not isinstance(event, dict) or any(type(event.get(key)) is not int or event[key] < 0
                                                  for key in ("tokens_before", "tokens_after")):
                raise ValueError("Invalid compaction metrics")
        return result

    def summary(self) -> dict[str, Any]:
        input_tokens = sum(row.get("actual_input_tokens") or 0 for row in self.per_step)
        output_tokens = sum(row.get("actual_output_tokens") or 0 for row in self.per_step)
        inference_seconds = sum(row.get("inference_seconds", 0.0) for row in self.per_step)
        tool_seconds = sum(row.get("seconds", 0.0) for row in self.tools_by_name.values())
        timing = _timing_summary([measurement for row in self.per_step
                                  for measurement in row.get("inference_measurements", [])])
        timing["timing_coverage"]["completed_calls"] = self.inference_calls
        reductions = []
        for event in self.compactions:
            before, after = event.get("tokens_before", 0), event.get("tokens_after", 0)
            reductions.append({"tokens_before": before, "tokens_after": after,
                               "reduction_percent": 100 * (before - after) / before if before else 0.0})
        return {
            "inference_calls": self.inference_calls,
            **{name: getattr(self, name) for name in (
                "decision_enabled", "decision_calls", "decision_failures", "decision_fallbacks",
                "decision_cache_hits", "decision_seconds", "decision_inference_seconds",
                "decision_management_seconds", "decision_input_tokens", "decision_load_seconds",
                "decision_load_measurements")},
            "actual_input_tokens": input_tokens,
            "actual_output_tokens": output_tokens,
            "inference_seconds": inference_seconds,
            **timing,
            "tool_seconds": tool_seconds,
            "wall_seconds": self.wall_seconds,
            "tools_by_name": self.tools_by_name,
            "unknown_tool_calls": sum(row.get("unknown", 0) for row in self.tools_by_name.values()),
            "average_context_utilization": (sum(self.context_utilizations) / len(self.context_utilizations)
                                            if self.context_utilizations else 0.0),
            "peak_context_utilization": max(self.context_utilizations, default=0.0),
            "compaction_count": len(self.compactions),
            "compaction_attempt_count": len(self.compaction_attempts),
            "compaction_attempts": self.compaction_attempts,
            "compactions": reductions,
            "compaction_seconds": self.compaction_seconds,
            "context_preparation_seconds": self.context_preparation_seconds,
            "checkpoint_seconds": self.checkpoint_seconds,
            "checkpoint_count": self.checkpoint_count,
            "resume_attempts": self.resume_attempts,
            "resume_success": self.resume_success,
            "skills_discovered": self.skills_discovered,
            "skills_loaded": self.skills_loaded,
            "skills_activated": self.skills_activated,
        }
