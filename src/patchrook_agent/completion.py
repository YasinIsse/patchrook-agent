"""Caller-selected completion evidence; task correctness needs independent checks."""

from dataclasses import asdict, dataclass
from typing import Any

from patchrook_agent.inspection import InspectionLedger


@dataclass(frozen=True)
class TaskRequirements:
    require_inspection: bool = False
    require_change: bool = False
    require_tests: bool = False

    def __post_init__(self):
        if any(type(value) is not bool for value in asdict(self).values()):
            raise ValueError("Task requirements must be booleans")

    @classmethod
    def code(cls):
        """Coding work may already be correct, with inspection and passing tests."""
        return cls(require_inspection=True, require_tests=True)

    @classmethod
    def fix(cls):
        """A caller that requires a patch as well as observed verification."""
        return cls(require_inspection=True, require_change=True, require_tests=True)

    @classmethod
    def inspect(cls):
        return cls(require_inspection=True)

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Any):
        if not isinstance(value, dict) or set(value) != set(cls().to_dict()):
            raise ValueError("Invalid checkpoint task requirements")
        return cls(**value)

    def missing(self, *, inspected: bool, changed: bool, verification: str) -> list[str]:
        missing = []
        if self.require_inspection and not inspected:
            missing.append("inspect source code with an enabled file-reading tool")
        if self.require_change and not changed:
            missing.append("make the requested code change")
        if self.require_tests and verification != "passed":
            missing.append("observe passing tests after the last change")
        return missing

    def instructions(self) -> str:
        needs = self.missing(inspected=False, changed=False, verification="not_run")
        return ("Completion requirements for this task: " + "; ".join(needs) + ". "
                "Execute the requested work with structured tools before a final response. "
                "A plan to act does not fulfill these requirements.") if needs else ""


def inspection_observed(messages: list[dict]) -> bool:
    """Count model-visible file contents, including successful members of a partial batch.

    Names from listings, hidden artifact captures, and failed reads are not source
    inspection. Raw history survives compaction, so a legitimate reread is never
    necessary merely to re-establish this evidence.
    """
    return any(item.inspected for item in InspectionLedger.from_messages(messages).files.values())
