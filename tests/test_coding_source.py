from pathlib import Path

import pytest

from patchrook_agent.coding import discover_source_paths
from patchrook_agent.tools.base import ToolConfig, Workspace


def put(root: Path, path: str, content: str = "value = 1\n"):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)


def test_pipeline_dependencies_are_gathered_before_unrelated_source(tmp_path):
    put(tmp_path, "pipeline/runner.py", "\n".join(
        f"from pipeline.stage_{i} import transform as stage_{i}" for i in range(8)))
    for i in range(8):
        put(tmp_path, f"pipeline/stage_{i}.py", f"def transform(value):\n    return value + {i + 1}\n")
    put(tmp_path, "tests/test_pipeline.py", "from pipeline.runner import process\n")
    for i in range(20):
        put(tmp_path, f"unrelated/file_{i}.py")
    paths = discover_source_paths(Workspace(tmp_path), "Trace pipeline/runner.py through its stages", limit=10)
    assert paths[0] == "pipeline/runner.py"
    assert set(paths[1:9]) == {f"pipeline/stage_{i}.py" for i in range(8)}
    assert paths[9] == "tests/test_pipeline.py"


def test_relative_package_imports_and_src_layout(tmp_path):
    put(tmp_path, "src/package/main.py", "from .parts import core\nfrom . import sibling\n")
    put(tmp_path, "src/package/parts/core.py", "from ..shared import operation\n")
    put(tmp_path, "src/package/sibling.py")
    put(tmp_path, "src/package/shared.py", "from package import other\n")
    put(tmp_path, "src/package/other.py")
    paths = discover_source_paths(Workspace(tmp_path), "Fix src/package/main.py", limit=5)
    assert paths[0] == "src/package/main.py"
    assert set(paths) == {"src/package/main.py", "src/package/parts/core.py",
                          "src/package/sibling.py", "src/package/shared.py", "src/package/other.py"}


def test_discovery_never_executes_imported_source(tmp_path):
    put(tmp_path, "main.py", "import dangerous\n")
    put(tmp_path, "dangerous.py", "raise RuntimeError('discovery must not execute code')\n")
    assert discover_source_paths(Workspace(tmp_path), "Fix main.py", limit=2) == ["main.py", "dangerous.py"]


def test_relevant_filename_and_tests_win_without_explicit_path(tmp_path):
    put(tmp_path, "app/account.py")
    put(tmp_path, "app/other.py")
    put(tmp_path, "tests/test_account.py")
    put(tmp_path, "tests/test_other.py")
    paths = discover_source_paths(Workspace(tmp_path), "Repair account balance calculation", limit=2)
    assert paths == ["app/account.py", "tests/test_account.py"]
    assert discover_source_paths(Workspace(tmp_path), "Repair account balance calculation") == paths


def test_explicit_file_does_not_pull_unrelated_source_into_working_set(tmp_path):
    put(tmp_path, "calculator.py")
    put(tmp_path, "billing.py")
    put(tmp_path, "tests/test_calculator.py")
    put(tmp_path, "tests/test_billing.py")
    assert discover_source_paths(Workspace(tmp_path), "Fix calculator.py") == [
        "calculator.py", "tests/test_calculator.py"]


def test_small_repo_fallback_and_bounded_count(tmp_path):
    put(tmp_path, "calculator.py")
    put(tmp_path, "tests/test_calculator.py")
    paths = discover_source_paths(Workspace(tmp_path), "Correct addition and run verification")
    assert set(paths) == {"calculator.py", "tests/test_calculator.py"}
    assert len(discover_source_paths(Workspace(tmp_path), "Correct addition", limit=1)) == 1


def test_discovery_confines_paths_and_skips_generated_hidden_and_oversized_files(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    put(root, "good.py")
    for path in (".git/hook.py", ".venv/site.py", "runs/history.py", "reports/capture.py", ".hidden/file.py"):
        put(root, path)
    put(root, "large.py", "x" * 101)
    put(tmp_path, "outside.py", "secret = 42\n")
    (root / "escape.py").symlink_to(tmp_path / "outside.py")
    (root / "escape_dir").symlink_to(tmp_path, target_is_directory=True)
    paths = discover_source_paths(Workspace(root, ToolConfig(max_file_bytes=100)),
                                  "Read escape.py, ../outside.py and good.py")
    assert paths == ["good.py"]


@pytest.mark.parametrize("limit", [0, -1, 101, True, 2.0])
def test_invalid_limits_are_rejected(tmp_path, limit):
    with pytest.raises(ValueError, match="limit"):
        discover_source_paths(Workspace(tmp_path), "task", limit=limit)
