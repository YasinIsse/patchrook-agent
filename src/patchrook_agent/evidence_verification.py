"""Optional trusted verification after accepted base-model patches."""

import time

from patchrook_agent.model_routing import infrastructure_failure


def run_base_checks(runtime, deadline):
    """Run missing current checks without spending a coding generation.

    A failed assertion is a completed observation, so it is not rerun on every
    loop iteration. Existing resume invalidation still requires fresh checks.
    Return True only when the normal coding loop should stop.
    """
    if (not runtime.config.evidence_auto_check_edits or runtime.revision < 1
            or not runtime.tools.commands
            or runtime.routing is not None and runtime.routing["active_model"] == runtime.config.escalation_model):
        return False
    state = runtime.state
    if "run_check" not in runtime.enabled:
        state.stop_reason = "base_check_environment_failure"
        runtime.session.emit("base_checks_stop", state.steps, reason="run_check_disabled")
        return True
    snapshot = runtime._fingerprint(deadline)
    missing = [name for name in runtime.tools.commands
               if runtime.checks.get(name, {}).get("revision") != runtime.revision
               or runtime.checks.get(name, {}).get("snapshot") != snapshot]
    if not missing:
        return False
    runtime.session.emit("base_checks_start", state.steps, revision=runtime.revision, checks=missing)
    environment_failure = False
    routing_reason = None
    observed_checks = []
    for name in missing:
        if time.monotonic() >= deadline:
            state.stop_reason = "deadline"
            runtime.session.emit("base_checks_stop", state.steps, reason=state.stop_reason)
            return True
        result = runtime._perform_action(state, {"name": "run_check", "arguments": {"name": name}},
                                         deadline, automatic=True)
        if result.get("error", {}).get("code") == "deadline_exceeded" or time.monotonic() >= deadline:
            state.stop_reason = "deadline"
            runtime.session.emit("base_checks_stop", state.steps, reason=state.stop_reason)
            return True
        observed_checks.append((name, result))
    # All current checks are now available for independent exception corroboration.
    for name, result in observed_checks:
        derive_context = getattr(runtime, "_code_failure_context", None)
        context = derive_context(name, result, deadline) if callable(derive_context) else None
        infrastructure = not result.get("success") and infrastructure_failure(result, code_failure_context=context)
        environment_failure = environment_failure or infrastructure
        if runtime.routing is not None and not infrastructure:
            # Automatic checks can prove a failed repair, but are not model
            # decisions and cannot spend its repetition/invalid allowances.
            progress = runtime.routing_progress
            counters = progress.invalid_responses, progress.repeated_actions
            reason = progress.observe("run_check", {"name": name}, result, runtime.revision, runtime.config,
                                      code_failure_context=context)
            progress.invalid_responses, progress.repeated_actions = counters
            if reason == "failed_repairs":
                routing_reason = reason
    runtime._update_state(state)
    runtime.session.emit("base_checks_complete", state.steps, revision=runtime.revision,
                         verification=runtime._verification())
    if environment_failure:
        state.stop_reason = "base_check_environment_failure"
        runtime.session.emit("base_checks_stop", state.steps, reason=state.stop_reason)
        return True
    if runtime.routing is not None:
        if runtime._routing_signal(routing_reason, deadline):
            return True
        if runtime._maybe_start_repair(deadline):
            return True
    return False
