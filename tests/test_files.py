from pathlib import Path

import pytest

from patchrook_agent.tools.base import ToolConfig, ToolError, Workspace
from patchrook_agent.tools.files import edit_file, list_files, read_file, write_file


@pytest.fixture
def workspace(tmp_path):
    return Workspace(tmp_path)


def test_listing_recursion_exclusions_and_limit(workspace):
    root = workspace.root
    (root / "src").mkdir()
    (root / "src" / "main.py").write_text("pass\n")
    (root / "README.md").write_text("Hello\n")
    for name in (".git", ".venv", "node_modules", "__pycache__", ".pytest_cache"):
        (root / name).mkdir()
        (root / name / "ignored").write_text("ignored")
    assert list_files(workspace)["entries"] == ["README.md", "src/"]
    assert list_files(workspace, recursive=True)["entries"] == ["README.md", "src/", "src/main.py"]
    limited = list_files(workspace, recursive=True, max_entries=2)
    assert len(limited["entries"]) == 2
    assert limited["truncated"]
    with pytest.raises(ToolError):
        list_files(workspace, "missing")


def test_read_file_and_ranges(workspace):
    (workspace.root / "example.py").write_text("first\nsecond\nthird\n")
    result = read_file(workspace, "example.py")
    assert result["content"] == "1: first\n2: second\n3: third"
    assert not result["truncated"]
    assert read_file(workspace, "example.py", 2, 2)["content"] == "2: second"
    for start, end in [(0, None), (3, 2), (5, None)]:
        with pytest.raises(ToolError):
            read_file(workspace, "example.py", start, end)
    with pytest.raises(ToolError):
        read_file(workspace, "missing")


def test_read_limits_binary_empty_and_large_files(tmp_path):
    workspace = Workspace(tmp_path, ToolConfig(max_read_lines=2))
    (tmp_path / "text").write_text("one\ntwo\nthree\n")
    result = read_file(workspace, "text")
    assert result["content"] == "1: one\n2: two"
    assert result["truncated"]
    (tmp_path / "empty").write_text("")
    assert read_file(workspace, "empty")["content"] == ""
    for data in (b"\x00hello", b"\xff\xfe", b"hello\x01"):
        (tmp_path / "binary").write_bytes(data)
        with pytest.raises(ToolError):
            read_file(workspace, "binary")
    (tmp_path / "large").write_bytes(b"a" * (workspace.config.max_file_bytes + 1))
    with pytest.raises(ToolError):
        read_file(workspace, "large")


def test_create_file_parents_and_reject_overwrite(workspace):
    result = write_file(workspace, "src/new.py", "x = 1\n")
    assert result["success"]
    assert (workspace.root / "src/new.py").read_text() == "x = 1\n"
    with pytest.raises(ToolError):
        write_file(workspace, "src/new.py", "lost\n")
    assert (workspace.root / "src/new.py").read_text() == "x = 1\n"
    write_file(workspace, "src/new.py", "x = 2\n", overwrite=True)
    assert (workspace.root / "src/new.py").read_text() == "x = 2\n"


def test_precise_edit_and_reject_ambiguous_or_missing_match(workspace):
    path = workspace.root / "maths.py"
    original = "def add(a, b):\n    return a - b\n\nUNCHANGED = True\n"
    path.write_text(original)
    result = edit_file(workspace, "maths.py", "return a - b", "return a + b")
    assert result["success"] and result["modified"]
    assert result["replacements"] == 1
    assert path.read_text() == original.replace("return a - b", "return a + b")
    for old in ("missing", "", "a"):
        with pytest.raises(ToolError):
            edit_file(workspace, "maths.py", old, "bad")
    with pytest.raises(ToolError):
        edit_file(workspace, "missing", "a", "b")


def test_edits_preserve_crlf_and_permissions(workspace):
    path = workspace.root / "windows.py"
    path.write_bytes(b"def add(a, b):\r\n    return a - b\r\n# unchanged\r\n")
    path.chmod(0o750)
    edit_file(workspace, "windows.py", "return a - b", "result = a + b\n    return result")
    assert path.read_bytes() == b"def add(a, b):\r\n    result = a + b\r\n    return result\r\n# unchanged\r\n"
    assert path.stat().st_mode & 0o777 == 0o750


def test_edit_rejects_overlapping_matches(workspace):
    path = workspace.root / "overlap.txt"
    path.write_text("aaa")
    with pytest.raises(ToolError):
        edit_file(workspace, "overlap.txt", "aa", "b")
    assert path.read_text() == "aaa"


@pytest.mark.parametrize("tool,args", [
    (list_files, {}), (read_file, {}),
    (write_file, {"content": "changed"}),
    (edit_file, {"old_text": "outside", "new_text": "changed"}),
])
def test_reject_outside_paths_and_symlinks(workspace, tmp_path, tool, args):
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.mkdir()
    outside_file = outside / "secret.txt"
    outside_file.write_text("outside")
    (workspace.root / "escape").symlink_to(outside, target_is_directory=True)
    candidates = [str(outside), "../" + outside.name, "escape"]
    if tool is not list_files:
        candidates = [str(outside_file), "../" + outside.name + "/secret.txt", "escape/secret.txt"]
    for path in candidates:
        with pytest.raises(ToolError):
            tool(workspace, path, **args)
    assert outside_file.read_text() == "outside"


def test_listing_skips_escaped_links_and_does_not_follow_cycles(workspace, tmp_path):
    (workspace.root / "outside").symlink_to(tmp_path.parent, target_is_directory=True)
    (workspace.root / "loop").symlink_to(workspace.root, target_is_directory=True)
    assert list_files(workspace, recursive=True)["entries"] == ["loop/"]


def test_write_rejects_new_file_through_escaped_symlink(workspace, tmp_path):
    outside = tmp_path.parent / (tmp_path.name + "-target")
    outside.mkdir()
    (workspace.root / "escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ToolError):
        write_file(workspace, "escape/new.py", "bad")
    assert not (outside / "new.py").exists()
