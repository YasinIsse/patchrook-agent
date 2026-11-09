"""Small deterministic failure classifier; decisions only break ambiguity."""

from dataclasses import dataclass
import re


RECOVERY_STRATEGIES = {
    "implementation": "Inspect the failing implementation and its helpers; propose a targeted correction using the exact failure.",
    "dependency": "Inspect imports and dependency declarations for a missing or incompatible dependency; preserve the error.",
    "configuration": "Inspect relevant configuration and test setup before proposing a configuration correction.",
    "evidence": "Read a specific missing caller, helper, or test range before choosing a correction.",
    "test": "Inspect the failing assertion and expected behavior; preserve independent verification and the user's test constraints.",
}


@dataclass(frozen=True)
class FailureAssessment:
    category: str
    ambiguous: bool
    guidance: str


def classify_failure(name: str, result: dict) -> FailureAssessment | None:
    """Never use a model to reinterpret an explicit tool rejection or success."""
    if result.get("success") is True:
        return None
    error = result.get("error") or {}
    code = error.get("code", "") if isinstance(error, dict) else ""
    if code:
        return FailureAssessment("tool", False,
                                 "Correct the rejected tool arguments using the exact tool error; retain existing execution restrictions.")
    if name in {"read_code", "read_files"} and any(
            isinstance(item, dict) and item.get("error") for item in result.get("files", [])):
        return FailureAssessment("evidence", False,
                                 "Use the per-file read errors to correct the source selection or request a specific missing range.")
    evidence = "\n".join(str(result.get(key, "")) for key in ("stdout", "stderr"))
    if name == "run_tests":
        if result.get("status") == "timed_out":
            return FailureAssessment("timeout", False,
                                     "Inspect the timeout evidence and test for blocking behavior; do not claim verification passed.")
        if "ModuleNotFoundError" in evidence or "No module named" in evidence:
            return FailureAssessment("dependency", False, RECOVERY_STRATEGIES["dependency"])
        if any(marker in evidence for marker in ("unrecognized arguments:", "ERROR: usage:", "no tests ran")):
            return FailureAssessment("configuration", False, RECOVERY_STRATEGIES["configuration"])
        if any(marker in evidence for marker in ("AssertionError", "SyntaxError", "IndentationError")) or re.search(r"(?m)^E\s+assert\b", evidence):
            return FailureAssessment("implementation", False, RECOVERY_STRATEGIES["implementation"])
    return FailureAssessment("ambiguous", True,
                             "Investigate the exact observed error and current source before proposing a correction.")
