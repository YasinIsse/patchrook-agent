"""External, exact evidence and task-conditioned context admission.

Archived observations are not edit authorization. Only source actually included
in a model request may authorize the next edit.
"""

from .store import EvidenceCard, EvidenceStore
from .index import RepositoryIndex
from .compiler import CompiledContext, ContextBudgetError, ContextCompiler

__all__ = ["EvidenceCard", "EvidenceStore", "RepositoryIndex", "CompiledContext",
           "ContextBudgetError", "ContextCompiler"]
