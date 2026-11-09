import shutil

import pytest

from patchrook_agent.tools.base import ToolConfig, ToolError, Workspace
from patchrook_agent.tools.search import search_code


pytestmark = pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep is required")


def test_search_finds_symbol_and_line_number(tmp_path):
    (tmp_path / "maths.py").write_text("# helpers\ndef add_numbers(a, b):\n    return a + b\n")
    result = search_code(Workspace(tmp_path), "add_numbers")
    assert result["success"]
    assert result["matches"] == [{"path": "maths.py", "line_number": 2, "text": "def add_numbers(a, b):"}]


def test_search_empty_result(tmp_path):
    (tmp_path / "file.py").write_text("nothing here\n")
    assert search_code(Workspace(tmp_path), "not_present")["matches"] == []


def test_search_limits_results(tmp_path):
    (tmp_path / "many.py").write_text("needle\n" * 10)
    result = search_code(Workspace(tmp_path), "needle", max_results=2)
    assert len(result["matches"]) == 2
    assert result["truncated"]


def test_search_obeys_ignores_and_directory(tmp_path):
    (tmp_path / ".gitignore").write_text("ignored.py\n")
    (tmp_path / "ignored.py").write_text("needle\n")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "dependency.py").write_text("needle\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "allowed.py").write_text("needle\n")
    assert [m["path"] for m in search_code(Workspace(tmp_path), "needle")["matches"]] == ["src/allowed.py"]
    assert search_code(Workspace(tmp_path), "needle", path="src")["success"]


def test_search_does_not_follow_symlinks(tmp_path):
    inside = tmp_path / "workspace"
    inside.mkdir()
    outside = tmp_path / "secret.py"
    outside.write_text("needle\n")
    (inside / "escape.py").symlink_to(outside)
    assert search_code(Workspace(inside), "needle")["matches"] == []
    with pytest.raises(ToolError):
        search_code(Workspace(inside), "needle", path="escape.py")


def test_search_rejects_bad_pattern_and_missing_path(tmp_path):
    workspace = Workspace(tmp_path)
    with pytest.raises(ToolError):
        search_code(workspace, "[", mode="regex")
    with pytest.raises(ToolError):
        search_code(workspace, "needle", path="missing")


def test_search_timeout_is_structured(tmp_path):
    (tmp_path / "file.py").write_text("needle\n")
    with pytest.raises(ToolError) as error:
        search_code(Workspace(tmp_path, ToolConfig(search_timeout_seconds=0.000001)), "needle")
    assert error.value.code == "timeout"


def test_search_ignores_external_ripgrep_config(tmp_path, monkeypatch):
    config = tmp_path / "rg.conf"
    config.write_text("--glob\n!*.py\n")
    monkeypatch.setenv("RIPGREP_CONFIG_PATH", str(config))
    (tmp_path / "file.py").write_text("needle\n")
    assert search_code(Workspace(tmp_path), "needle")["matches"][0]["path"] == "file.py"


def test_search_paths_with_colons_and_newlines(tmp_path):
    (tmp_path / "odd:name\nfile.py").write_text("needle\n")
    result = search_code(Workspace(tmp_path), "needle")
    assert result["matches"][0]["path"] == "odd:name\nfile.py"


def test_search_bounds_large_output_without_fabricating_paths(tmp_path):
    (tmp_path / "many.py").write_text(("needle " + "x" * 500 + "\n") * 1000)
    result = search_code(Workspace(tmp_path, ToolConfig(max_output_chars=512)), "needle")
    assert result["success"]
    assert result["truncated"]
    assert result["matches"]
    assert all(match["path"] == "many.py" for match in result["matches"])


def test_invalid_pattern_remains_an_error_when_stderr_truncates(tmp_path):
    with pytest.raises(ToolError) as error:
        search_code(Workspace(tmp_path, ToolConfig(max_output_chars=512)), "[" * 4000, mode="regex")
    assert error.value.code == "invalid_regex"

