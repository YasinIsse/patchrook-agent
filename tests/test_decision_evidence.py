from pathlib import Path

import pytest

from patchrook_agent.coding import discover_source_paths
from patchrook_agent.decision_evidence import build_evidence_plan
from patchrook_agent.tools.base import Workspace


def put(root: Path, path: str, content: str = "value = 1\n"):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)


def test_dependencies_callers_tests_and_configuration_are_retained(tmp_path):
    put(tmp_path, "orders.py", "from money import discounted_total\n\ndef order_total(value):\n    return discounted_total(value)\n")
    put(tmp_path, "money.py", "def discounted_total(value):\n    return value\n")
    put(tmp_path, "checkout.py", "from orders import order_total\n\ndef checkout(value):\n    return order_total(value)\n")
    put(tmp_path, "tests/test_orders.py", "from orders import order_total\n")
    put(tmp_path, "pytest.ini", "[pytest]\n")
    workspace = Workspace(tmp_path)
    base = discover_source_paths(workspace, "Fix orders.py")
    plan = build_evidence_plan(workspace, "Fix orders.py", base)
    assert set(base) <= set(plan.ordered_paths())
    assert plan.initial_paths == ("orders.py",)
    assert set(plan.related_paths) == {"money.py", "tests/test_orders.py", "checkout.py", "pytest.ini"}
    assert "orders.py:1 imports money.py" in plan.relationships
    assert "checkout.py:1 imports orders.py" in plan.relationships
    assert "order_total:3" in plan.descriptions["orders.py"]
    assert "possible caller" in plan.descriptions["checkout.py"]
    assert "configuration" in plan.descriptions["pytest.ini"]


def test_relative_imports_and_src_layout_are_resolved(tmp_path):
    put(tmp_path, "src/shop/orders.py", "from . import money\n")
    put(tmp_path, "src/shop/money.py", "def total(value):\n    return value\n")
    put(tmp_path, "src/shop/checkout.py", "from shop.orders import total\n")
    plan = build_evidence_plan(Workspace(tmp_path), "Fix orders", ["src/shop/orders.py"])
    assert set(plan.related_paths) == {"src/shop/money.py", "src/shop/checkout.py"}
    assert "src/shop/orders.py:1 imports src/shop/money.py" in plan.relationships


def test_rankings_only_reorder_known_candidates_and_never_drop_tails(tmp_path):
    for path in ("a.py", "b.py", "caller.py"):
        put(tmp_path, path, "import a\n")
    plan = build_evidence_plan(Workspace(tmp_path), "Fix modules", ["a.py", "b.py"])
    assert plan.ordered_paths(["invented.py", "b.py", "b.py"], ["missing.py"]) == [
        "b.py", "a.py", "caller.py"]


def test_full_existing_selection_is_preserved_with_two_bounded_groups(tmp_path):
    base = [f"part_{index}.py" for index in range(24)]
    for path in base:
        put(tmp_path, path)
    plan = build_evidence_plan(Workspace(tmp_path), "Fix part_23.py", base)
    assert len(plan.initial_paths) == 12
    assert len(plan.related_paths) == 12
    assert plan.ordered_paths()[0] == "part_23.py"
    assert set(plan.ordered_paths()) == set(base)


def test_explicit_sources_and_related_tests_form_separate_decision_groups(tmp_path):
    put(tmp_path, "orders.py", "from money import total\n")
    put(tmp_path, "money.py", "def total():\n    return 0\n")
    put(tmp_path, "tests/test_orders.py", "from orders import total\n")
    put(tmp_path, "tests/test_money.py", "from money import total\n")
    base = ["orders.py", "money.py", "tests/test_orders.py", "tests/test_money.py"]
    plan = build_evidence_plan(Workspace(tmp_path), "Fix orders.py and money.py", base)
    assert plan.initial_paths == ("orders.py", "money.py")
    assert plan.related_paths == ("tests/test_orders.py", "tests/test_money.py")
    assert plan.ordered_paths(["money.py", "orders.py"], ["tests/test_money.py", "tests/test_orders.py"]) == [
        "money.py", "orders.py", "tests/test_money.py", "tests/test_orders.py"]


def test_imported_helpers_are_related_to_entry_module_without_explicit_path(tmp_path):
    put(tmp_path, "orders.py", "from money import total\n")
    put(tmp_path, "money.py", "def total():\n    return 0\n")
    put(tmp_path, "tests/test_orders.py", "from orders import total\n")
    plan = build_evidence_plan(Workspace(tmp_path), "Fix order processing", ["orders.py", "money.py", "tests/test_orders.py"])
    assert plan.initial_paths == ("orders.py",)
    assert plan.related_paths == ("money.py", "tests/test_orders.py")


def test_optional_related_pool_is_bounded_without_removing_base(tmp_path):
    put(tmp_path, "module.py")
    for index in range(30):
        put(tmp_path, f"caller_{index:02d}.py", "import module\n")
    plan = build_evidence_plan(Workspace(tmp_path), "Fix module.py", ["module.py"])
    assert plan.initial_paths == ("module.py",)
    assert len(plan.related_paths) == 12
    assert len(plan.ordered_paths()) == 13


def test_real_database_relationships_and_referenced_schema_are_described(tmp_path):
    put(tmp_path, "models.py", "schema = 'schema.sql'\nowner = ForeignKey('users.id')\nusers = relationship('User')\n")
    put(tmp_path, "schema.sql", "CREATE TABLE orders (\n user_id INTEGER REFERENCES users(id)\n);\n")
    plan = build_evidence_plan(Workspace(tmp_path), "Inspect models.py and schema.sql", ["models.py", "schema.sql"])
    assert "models.py:1 references schema.sql" in plan.relationships
    assert "models.py:2 ForeignKey('users.id')" in plan.relationships
    assert "models.py:3 relationship('User')" in plan.relationships
    assert "schema.sql:2 REFERENCES users" in plan.relationships


def test_references_must_exist_in_workspace_and_helpers_have_real_lines(tmp_path):
    put(tmp_path, "main.py", "config = 'absent.yaml'\n\ndef helper():\n    return 1\n\ndef main():\n    return helper()\n")
    plan = build_evidence_plan(Workspace(tmp_path), "Inspect main.py", ["main.py"])
    assert plan.related_paths == ()
    assert "main.py:7 calls local helper helper" in plan.relationships
    assert "absent.yaml" not in " ".join(plan.relationships)


def test_source_never_executes_and_invalid_python_is_still_retrievable(tmp_path):
    put(tmp_path, "main.py", "raise RuntimeError('never execute')\nimport broken\n")
    put(tmp_path, "broken.py", "def broken(:\n")
    plan = build_evidence_plan(Workspace(tmp_path), "Fix main.py", ["main.py"])
    assert "broken.py" in plan.related_paths
    assert "main.py:2 imports broken.py" in plan.relationships


def test_oversized_source_retained_without_parsing(tmp_path):
    put(tmp_path, "large.py", "#" * 70000)
    plan = build_evidence_plan(Workspace(tmp_path), "Fix large.py", ["large.py"])
    assert plan.ordered_paths() == ["large.py"]
    assert plan.descriptions["large.py"].startswith("large.py; source;")


def test_hidden_generated_and_escape_files_do_not_become_related(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    put(root, "module.py")
    for path in (".hidden/caller.py", "runs/caller.py", "reports/caller.py"):
        put(root, path, "import module\n")
    put(tmp_path, "outside.py", "import module\n")
    (root / "escape.py").symlink_to(tmp_path / "outside.py")
    plan = build_evidence_plan(Workspace(root), "Fix module.py", ["module.py"])
    assert plan.ordered_paths() == ["module.py"]


def test_content_change_invalidates_fingerprint(tmp_path):
    put(tmp_path, "main.py", "import helper\n")
    put(tmp_path, "helper.py", "value = 1\n")
    workspace = Workspace(tmp_path)
    before = build_evidence_plan(workspace, "Fix main.py", ["main.py"])
    unchanged = build_evidence_plan(workspace, "Fix main.py", ["main.py"])
    put(tmp_path, "helper.py", "value = 2\n")
    after = build_evidence_plan(workspace, "Fix main.py", ["main.py"])
    assert before.fingerprint == unchanged.fingerprint
    assert before.fingerprint != after.fingerprint


def test_existing_evidence_cannot_be_silently_lost_to_limit(tmp_path):
    put(tmp_path, "main.py")
    put(tmp_path, "helper.py")
    with pytest.raises(ValueError, match="discard"):
        build_evidence_plan(Workspace(tmp_path), "Fix", ["main.py", "helper.py"], limit=1)


@pytest.mark.parametrize("paths", [["missing.py"], ["../outside.py"]])
def test_required_invalid_paths_fail_instead_of_inventing_locations(tmp_path, paths):
    with pytest.raises(ValueError, match="existing workspace"):
        build_evidence_plan(Workspace(tmp_path), "Fix", paths)
