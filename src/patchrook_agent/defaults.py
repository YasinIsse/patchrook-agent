"""The supported local coding profile used by the command-line harness."""

from dataclasses import replace

from patchrook_agent.config import GenerationProfile, HarnessConfig, RoutingGenerationProfiles


BASE_MODEL = "qwen3:8b"
ROUTER_MODEL = "tev1:4b"
REPAIR_MODEL = "qwen3:14b"
BASE_URL = "http://localhost:11434"
CONTEXT_SIZE = 8192
OUTPUT_TOKENS = 1024
MAX_STEPS = 12
DEADLINE_SECONDS = 3600.0
TEST_TIMEOUT_SECONDS = 120.0
DECISION_TIMEOUT_SECONDS = 90.0


def coding_config(*, checks=(), context_size=CONTEXT_SIZE, max_output_tokens=OUTPUT_TOKENS,
                  model_routing="tev", escalation_model=REPAIR_MODEL, think=False,
                  temperature=.7, top_p=.8, top_k=20, min_p=0.,
                  decision_timeout=DECISION_TIMEOUT_SECONDS, inspect_only=False):
    """Return the verified evidence workflow, without experimental runner imports."""
    preset = HarnessConfig.preset("C")
    context = replace(preset.context, context_size=context_size, enabled=False,
                      reserved_output_tokens=max(512, max_output_tokens))
    profiles = None
    if model_routing == "tev":
        profiles = RoutingGenerationProfiles(
            base=GenerationProfile(think, max_output_tokens, temperature, top_p, top_k, min_p),
            repair=GenerationProfile(think, max_output_tokens, .6, .95, 20, 0.))
    return replace(preset, context=context, compaction_enabled=False,
        action_protocol="schema", workflow_mode="evidence", review_mode="off",
        decision_model="on" if model_routing == "tev" else "off",
        decision_timeout_seconds=decision_timeout, decision_memory_policy="serial",
        model_routing=model_routing, escalation_model=escalation_model,
        routing_max_failed_repairs=1, routing_max_repair_calls=2,
        routing_generation_profiles=profiles,
        temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p,
        evidence_policy="rules", evidence_context_tokens=context.input_limit,
        evidence_optional_tokens=1536, evidence_test_commands=tuple(tuple(c) for c in checks),
        evidence_initial_checks=not inspect_only, evidence_auto_check_edits=not inspect_only,
        evidence_seed_symbols=True, evidence_recover_python_function_anchors=True,
        evidence_refresh_modified_python=True, log_content=False)
