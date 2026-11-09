"""Explicit ablation settings; the legacy Python API defaults to baseline A."""

from dataclasses import asdict, dataclass, field, replace
import math
from typing import Any

from patchrook_agent.context.budget import ContextConfig


def validate_sampling(temperature: float, top_p: float | None, top_k: int | None,
                      min_p: float | None) -> None:
    """Validate explicit options; None preserves the server default on old sessions."""
    if (type(temperature) not in (int, float) or not math.isfinite(temperature)
            or temperature < 0):
        raise ValueError("temperature must be finite and nonnegative")
    for name, value in (("top_p", top_p), ("min_p", min_p)):
        if value is not None and (type(value) not in (int, float)
                                  or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError(f"{name} must be finite and between 0 and 1")
    if top_k is not None and (type(top_k) is not int or top_k < 0):
        raise ValueError("top_k must be a nonnegative integer")


@dataclass(frozen=True)
class GenerationProfile:
    think: bool
    max_output_tokens: int
    temperature: float
    top_p: float | None
    top_k: int | None
    min_p: float | None

    def __post_init__(self):
        if type(self.think) is not bool or type(self.max_output_tokens) is not int or self.max_output_tokens < 512:
            raise ValueError("Generation profiles require boolean think and at least 512 output tokens")
        validate_sampling(self.temperature, self.top_p, self.top_k, self.min_p)

    def validate_settings(self, settings, think):
        validate_sampling(settings.get("temperature"), settings.get("top_p"), settings.get("top_k"), settings.get("min_p"))
        if type(settings.get("max_output_tokens")) is not int:
            raise ValueError("Generation profile output limit must be an integer")
        if type(think) is not bool or think != self.think or any(
                settings.get(key) != value for key, value in asdict(self).items() if key != "think"):
            raise ValueError("Active generation settings do not match the saved phase profile")


@dataclass(frozen=True)
class RoutingGenerationProfiles:
    base: GenerationProfile
    repair: GenerationProfile

    @classmethod
    def parse(cls, value):
        if value is None or isinstance(value, cls):
            return value
        if not isinstance(value, dict) or set(value) != {"base", "repair"}:
            raise ValueError("routing_generation_profiles requires complete base and repair profiles")
        fields = {"think", "max_output_tokens", "temperature", "top_p", "top_k", "min_p"}
        if any(not isinstance(profile, dict) or set(profile) != fields for profile in value.values()):
            raise ValueError("Generation profiles require every explicit generation field")
        return cls(**{name: GenerationProfile(**profile) for name, profile in value.items()})

    def __post_init__(self):
        if not isinstance(self.base, GenerationProfile) or not isinstance(self.repair, GenerationProfile):
            raise ValueError("Both routing phase profiles must be GenerationProfile values")


@dataclass(frozen=True)
class HarnessConfig:
    variant: str = "A"
    context: ContextConfig = field(default_factory=ContextConfig)
    compaction_enabled: bool = False
    memory_enabled: bool = False
    skills_enabled: bool = False
    checkpoint_enabled: bool = False
    persist: bool = True
    logging_enabled: bool = True
    log_content: bool = True
    redact_logs: bool = True
    trusted_skills_dir: str | None = None
    memory_projection_tokens: int = 512
    action_protocol: str = "native"
    workflow_mode: str = "standard"
    temperature: float = 0.7
    top_p: float | None = 0.8
    top_k: int | None = 20
    min_p: float | None = 0.0
    decision_model: str = "off"
    decision_timeout_seconds: float = 30.0
    decision_memory_policy: str = "serial"
    model_routing: str = "off"
    escalation_model: str = "qwen3:14b"
    routing_max_failed_repairs: int = 2
    routing_max_invalid_responses: int = 2
    routing_max_repeated_actions: int = 3
    routing_max_repair_calls: int = 2
    routing_generation_profiles: RoutingGenerationProfiles | None = None
    review_mode: str = "adaptive"
    evidence_policy: str = "rules"
    evidence_context_tokens: int = 4000
    evidence_optional_tokens: int | None = None
    evidence_max_decision_calls: int = 2
    evidence_test_commands: tuple[tuple[str, ...], ...] = ()
    evidence_initial_checks: bool = False
    evidence_auto_check_edits: bool = False
    evidence_seed_symbols: bool = False
    evidence_recover_python_function_anchors: bool = False
    evidence_refresh_modified_python: bool = False
    evidence_diagnostic_oracle: bool = False
    evidence_oracle_locations: tuple[tuple[str, int, int], ...] = ()

    def __post_init__(self):
        if not isinstance(self.variant, str) or self.variant not in {"A", "B", "C"}:
            raise ValueError("Variant must be A, B, or C")
        if not isinstance(self.action_protocol, str) or self.action_protocol not in {"native", "schema"}:
            raise ValueError("action_protocol must be native or schema")
        if not isinstance(self.workflow_mode, str) or self.workflow_mode not in {"standard", "lean", "coding", "evidence"}:
            raise ValueError("workflow_mode must be standard, lean, coding, or evidence")
        if self.workflow_mode == "coding" and self.action_protocol != "native":
            raise ValueError("The coding workflow requires action_protocol native")
        if not isinstance(self.decision_model, str) or self.decision_model not in {"on", "off"}:
            raise ValueError("decision_model must be on or off")
        if self.decision_model == "on" and self.workflow_mode not in {"coding", "evidence"}:
            raise ValueError("Decision assistance requires workflow_mode coding or evidence")
        if not isinstance(self.model_routing, str) or self.model_routing not in {"off", "tev"}:
            raise ValueError("model_routing must be off or tev")
        if (not isinstance(self.escalation_model, str) or not self.escalation_model.strip()
                or self.escalation_model != self.escalation_model.strip()
                or ":cloud" in self.escalation_model or "\x00" in self.escalation_model
                or len(self.escalation_model) > 200):
            raise ValueError("escalation_model must be an explicit local model name")
        for name, minimum, maximum in (("routing_max_failed_repairs", 1, 8),
                                       ("routing_max_invalid_responses", 1, 8),
                                       ("routing_max_repeated_actions", 2, 16),
                                       ("routing_max_repair_calls", 1, 2)):
            value = getattr(self, name)
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")
        profiles = RoutingGenerationProfiles.parse(self.routing_generation_profiles)
        object.__setattr__(self, "routing_generation_profiles", profiles)
        if profiles is not None:
            if not isinstance(self.context, ContextConfig):
                raise ValueError("Generation profiles require a ContextConfig")
            if self.model_routing != "tev":
                raise ValueError("Generation profiles require model_routing tev")
            if any(profile.max_output_tokens >= self.context.context_size for profile in (profiles.base, profiles.repair)):
                raise ValueError("Generation profile output must fit the shared context")
            base_context = replace(self.context, enabled=False,
                reserved_output_tokens=max(self.context.reserved_output_tokens, profiles.base.max_output_tokens))
            if min(self.evidence_context_tokens, base_context.input_limit) <= max(
                    0, profiles.repair.max_output_tokens - profiles.base.max_output_tokens):
                raise ValueError("Generation profiles leave no conservative repair input budget")
        if self.model_routing == "tev":
            if self.workflow_mode != "evidence":
                raise ValueError("Model routing requires workflow_mode evidence")
            if self.decision_model != "on":
                raise ValueError("Model routing requires decision_model on")
            if self.evidence_policy == "decision":
                raise ValueError("Model routing uses rules/window evidence, not decision admission")
            if self.decision_memory_policy != "serial":
                raise ValueError("Model routing requires serial decision memory")
        if self.evidence_policy not in {"window", "rules", "decision"}:
            raise ValueError("evidence_policy must be window, rules, or decision")
        if type(self.evidence_context_tokens) is not int or self.evidence_context_tokens < 512:
            raise ValueError("evidence_context_tokens must be an integer of at least 512")
        if self.evidence_optional_tokens is not None and (
                type(self.evidence_optional_tokens) is not int or self.evidence_optional_tokens < 0):
            raise ValueError("evidence_optional_tokens must be a nonnegative integer or None")
        if type(self.evidence_max_decision_calls) is not int or not 0 <= self.evidence_max_decision_calls <= 16:
            raise ValueError("evidence_max_decision_calls must be between zero and sixteen")
        evidence_uses_decision = self.evidence_policy == "decision" or self.model_routing == "tev"
        if self.workflow_mode == "evidence" and (evidence_uses_decision != (self.decision_model == "on")):
            raise ValueError("Evidence decision policy or model routing requires decision_model on; otherwise use off")
        if (not isinstance(self.evidence_test_commands, tuple) or len(self.evidence_test_commands) > 16
                or any(not isinstance(command, tuple) or not command or len(command) > 64
                       or any(not isinstance(arg, str) or not arg or "\x00" in arg or len(arg) > 4096
                              for arg in command) for command in self.evidence_test_commands)):
            raise ValueError("evidence_test_commands must contain up to sixteen nonempty argv tuples")
        if type(self.evidence_diagnostic_oracle) is not bool:
            raise ValueError("evidence_diagnostic_oracle must be a boolean")
        if (not isinstance(self.evidence_oracle_locations, tuple)
                or any(not isinstance(item, tuple) or len(item) != 3 or not isinstance(item[0], str)
                       or not item[0] or type(item[1]) is not int or type(item[2]) is not int
                       or not 1 <= item[1] <= item[2] for item in self.evidence_oracle_locations)):
            raise ValueError("evidence_oracle_locations must contain path/start/end tuples")
        if self.evidence_oracle_locations and not self.evidence_diagnostic_oracle:
            raise ValueError("Oracle locations require an explicitly labeled diagnostic run")
        if (type(self.decision_timeout_seconds) not in (int, float)
                or not math.isfinite(self.decision_timeout_seconds)
                or self.decision_timeout_seconds <= 0):
            raise ValueError("decision_timeout_seconds must be positive and finite")
        if not isinstance(self.decision_memory_policy, str) or self.decision_memory_policy not in {"serial", "resident"}:
            raise ValueError("decision_memory_policy must be serial or resident")
        if not isinstance(self.review_mode, str) or self.review_mode not in {"adaptive", "off"}:
            raise ValueError("review_mode must be adaptive or off")
        validate_sampling(self.temperature, self.top_p, self.top_k, self.min_p)
        for name in ("compaction_enabled", "memory_enabled", "skills_enabled", "checkpoint_enabled",
                     "persist", "logging_enabled", "log_content", "redact_logs", "evidence_initial_checks",
                     "evidence_auto_check_edits", "evidence_seed_symbols", "evidence_recover_python_function_anchors",
                     "evidence_refresh_modified_python"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if self.evidence_refresh_modified_python and self.workflow_mode != "evidence":
            raise ValueError("Modified Python source refresh requires evidence workflow")
        if self.evidence_recover_python_function_anchors and self.workflow_mode != "evidence":
            raise ValueError("Python function anchor recovery requires workflow_mode evidence")
        if not isinstance(self.context, ContextConfig):
            raise ValueError("context must be a ContextConfig")
        if type(self.memory_projection_tokens) is not int or self.memory_projection_tokens <= 0:
            raise ValueError("memory_projection_tokens must be a positive integer")
        if self.trusted_skills_dir is not None and not isinstance(self.trusted_skills_dir, str):
            raise ValueError("trusted_skills_dir must be a path string")
        if self.checkpoint_enabled and not self.persist:
            raise ValueError("Checkpointing requires session persistence")

    @classmethod
    def preset(cls, variant: str) -> "HarnessConfig":
        if variant == "A":
            return cls()
        if variant == "B":
            return cls(variant="B", compaction_enabled=True)
        if variant == "C":
            return cls(variant="C", compaction_enabled=True, memory_enabled=True,
                       skills_enabled=True, checkpoint_enabled=True)
        raise ValueError("Variant must be A, B, or C")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HarnessConfig":
        values = dict(data)
        # Checkpoints predating sampling controls used greedy decoding and left
        # the other sampling options unspecified in the Ollama request.
        for name, default in (("temperature", 0.0), ("top_p", None), ("top_k", None), ("min_p", None)):
            values.setdefault(name, default)
        values["context"] = ContextConfig(**values["context"])
        for name in ("evidence_test_commands", "evidence_oracle_locations"):
            if name in values:
                values[name] = tuple(tuple(item) for item in values[name])
        return cls(**values)
