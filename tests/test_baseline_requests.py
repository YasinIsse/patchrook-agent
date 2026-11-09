"""Only positive reproduction requests add a controller test phase."""

import pytest

from patchrook_agent.coding import requires_baseline


@pytest.mark.parametrize("task", [
    "Run the existing test to reproduce its failure, fix the parser, and rerun tests.",
    "Run baseline tests before editing calc.py.",
    "Reproduce the parsing error and fix calc.py.",
    "Reproduce the bug in calc.py.",
    "Do not modify tests, but reproduce the failing case before repair.",
])
def test_requested_reproduction_runs_a_baseline(task):
    assert requires_baseline(task)


@pytest.mark.parametrize("task", [
    "Fix calc.py and run tests afterward.",
    "Do not reproduce the failure; fix calc.py.",
    "Don't run baseline tests; fix calc.py.",
    "Avoid reproducing the failing test; fix calc.py.",
    "Skip the baseline tests and fix calc.py.",
    "Fix calc.py without reproducing the test failure.",
    "Make the regression tests reproducible.",
])
def test_negated_or_descriptive_mentions_do_not_run_a_baseline(task):
    assert not requires_baseline(task)
