"""Production entry inference excludes incoming test/setup edges."""

import pytest

from invoice_fixtures import create_fixture
from patchrook_agent.coding import discover_source_paths
from patchrook_agent.decision_evidence import build_evidence_plan
from patchrook_agent.tools.base import Workspace


def put(root, name, text):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_controlled_fixture_keeps_checkout_as_entry_despite_test_import(tmp_path):
    task = create_fixture(tmp_path)
    workspace = Workspace(tmp_path)
    base = discover_source_paths(workspace, task.instructions)
    assert base[0] == "audit.py"  # Candidate order must not become entry evidence.
    plan = build_evidence_plan(workspace, task.instructions, base)
    assert plan.initial_paths == ("checkout.py",)
    assert plan.related_paths == tuple(path for path in base if path != "checkout.py")
    assert set(plan.ordered_paths()) == set(base)
    assert "checkout.py:7 imports audit.py" in plan.relationships
    assert "tests/test_checkout.py:1 imports checkout.py" in plan.relationships


@pytest.mark.parametrize("test_path", [
    "tests/test_entry.py", "test_entry.py", "entry_test.py", "conftest.py",
])
def test_test_and_pytest_setup_imports_do_not_disqualify_production_entry(tmp_path, test_path):
    put(tmp_path, "audit.py", "def describe(value):\n    return str(value)\n")
    put(tmp_path, "entry.py", "import audit\n\ndef run(value):\n    return audit.describe(value)\n")
    put(tmp_path, test_path, "from entry import run\n")
    base = ["audit.py", test_path, "entry.py"]
    plan = build_evidence_plan(Workspace(tmp_path), "Investigate the application behavior", base)
    assert plan.initial_paths == ("entry.py",)
    assert set(plan.ordered_paths()) == set(base)
    assert test_path in plan.related_paths


def test_production_callers_still_disqualify_helpers_from_root_group(tmp_path):
    put(tmp_path, "helper.py", "def total(value):\n    return value\n")
    put(tmp_path, "entry.py", "from helper import total\n")
    put(tmp_path, "launcher.py", "import entry\n")
    put(tmp_path, "tests/test_entry.py", "import entry\n")
    base = ["helper.py", "entry.py", "tests/test_entry.py", "launcher.py"]
    plan = build_evidence_plan(Workspace(tmp_path), "Investigate the application behavior", base)
    assert plan.initial_paths == ("launcher.py",)
    assert set(plan.ordered_paths()) == set(base)


def test_explicit_helper_keeps_requested_priority_over_inferred_entry(tmp_path):
    put(tmp_path, "helper.py", "def total(value):\n    return value\n")
    put(tmp_path, "entry.py", "from helper import total\n")
    put(tmp_path, "tests/test_entry.py", "import entry\n")
    base = ["entry.py", "helper.py", "tests/test_entry.py"]
    plan = build_evidence_plan(Workspace(tmp_path), "Inspect helper.py", base)
    assert plan.initial_paths == ("helper.py",)
    assert set(plan.ordered_paths()) == set(base)


def test_explicit_conftest_request_still_has_requested_priority(tmp_path):
    put(tmp_path, "helper.py", "def total(value):\n    return value\n")
    put(tmp_path, "entry.py", "from helper import total\n")
    put(tmp_path, "conftest.py", "import entry\n")
    base = ["helper.py", "entry.py", "conftest.py"]
    plan = build_evidence_plan(Workspace(tmp_path), "Inspect conftest.py", base)
    assert plan.initial_paths == ("conftest.py",)
    assert set(plan.ordered_paths()) == set(base)


def test_production_cycle_has_bounded_deterministic_fallback(tmp_path):
    put(tmp_path, "a.py", "import b\n")
    put(tmp_path, "b.py", "import a\n")
    put(tmp_path, "tests/test_cycle.py", "import b\n")
    base = ["b.py", "a.py", "tests/test_cycle.py"]
    plan = build_evidence_plan(Workspace(tmp_path), "Investigate cyclic application behavior", base)
    again = build_evidence_plan(Workspace(tmp_path), "Investigate cyclic application behavior", base)
    assert plan.initial_paths == again.initial_paths == ("b.py",)
    assert set(plan.ordered_paths()) == set(base)
    assert len(plan.initial_paths) <= 12 and len(plan.related_paths) <= 12
