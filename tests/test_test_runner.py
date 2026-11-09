import time
import os
import py_compile
import sys

import pytest

from patchrook_agent.tools.base import ToolConfig, ToolError, Workspace
from patchrook_agent.tools.tests import run_tests
from patchrook_agent.tools.process import run_bounded


def trusted_workspace(tmp_path, **kwargs):
    return Workspace(tmp_path, ToolConfig(allow_tests=True, **kwargs))


def test_runner_requires_explicit_trust(tmp_path):
    with pytest.raises(ToolError) as error:
        run_tests(Workspace(tmp_path))
    assert error.value.code == "permission_denied"


def test_runner_passes_and_executes_test(tmp_path):
    (tmp_path / "test_example.py").write_text(
        "from pathlib import Path\n"
        "def test_example():\n"
        "    Path('executed.txt').write_text('yes')\n"
        "    assert 1 + 1 == 2\n"
    )
    result = run_tests(trusted_workspace(tmp_path), path="test_example.py::test_example")
    assert result["success"]
    assert result["status"] == "passed"
    assert result["exit_code"] == 0
    assert "1 passed" in result["stdout"]
    assert (tmp_path / "executed.txt").read_text() == "yes"


def test_runner_fails_with_debugging_output(tmp_path):
    (tmp_path / "test_example.py").write_text("def test_broken():\n    assert 1 == 2\n")
    result = run_tests(trusted_workspace(tmp_path), path="test_example.py")
    assert not result["success"]
    assert result["status"] == "failed"
    assert result["exit_code"] == 1
    assert "test_broken" in result["stdout"]
    assert "assert 1 == 2" in result["stdout"]


def test_runner_times_out(tmp_path):
    (tmp_path / "test_example.py").write_text("import time\ndef test_sleep():\n    time.sleep(10)\n")
    started = time.monotonic()
    result = run_tests(trusted_workspace(tmp_path, test_timeout_seconds=0.2), "test_example.py")
    assert result["status"] == "timed_out"
    assert not result["success"]
    assert result["exit_code"] != 0
    assert time.monotonic() - started < 3


def test_runner_bounds_noisy_output(tmp_path):
    (tmp_path / "test_example.py").write_text(
        "def test_noisy():\n    print('x' * 200000)\n    assert False\n"
    )
    result = run_tests(trusted_workspace(tmp_path, max_output_chars=1000), "test_example.py")
    assert not result["success"]
    assert result["truncated"]
    assert len(result["stdout"]) < 600
    assert "1 failed" in result["stdout"]


@pytest.mark.parametrize("path", ["--help", "-k test", "pytest tests", "test_example.py --help", "../outside.py"])
def test_runner_rejects_options_commands_and_escape(tmp_path, path):
    (tmp_path / "test_example.py").write_text("def test_ok():\n    pass\n")
    with pytest.raises(ToolError):
        run_tests(trusted_workspace(tmp_path), path)


def test_runner_ignores_environment_options(tmp_path, monkeypatch):
    (tmp_path / "test_example.py").write_text("def test_ok():\n    pass\n")
    monkeypatch.setenv("PYTEST_ADDOPTS", "--invalid-option")
    monkeypatch.setenv("PYTEST_PLUGINS", "nonexistent_plugin")
    result = run_tests(trusted_workspace(tmp_path), "test_example.py")
    assert result["success"]


def test_runner_rejects_external_symlink(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "test_outside.py"
    outside.write_text("def test_ok():\n    pass\n")
    (root / "test_escape.py").symlink_to(outside)
    with pytest.raises(ToolError):
        run_tests(trusted_workspace(root), "test_escape.py")


def test_timeout_kills_descendants(tmp_path):
    script = (
        "import subprocess, sys, time\n"
        "from pathlib import Path\n"
        "subprocess.Popen([sys.executable, '-c', \"import time; from pathlib import Path; time.sleep(1); Path('leak.txt').write_text('leak')\"])\n"
        "Path('spawned.txt').write_text('yes')\n"
        "time.sleep(10)\n"
    )
    result = run_bounded(
        [sys.executable, "-c", script], cwd=tmp_path,
        timeout_seconds=0.3, max_output_chars=512,
    )
    assert result.timed_out
    assert (tmp_path / "spawned.txt").exists()
    time.sleep(1.1)
    assert not (tmp_path / "leak.txt").exists()


def test_timeout_still_applies_after_child_closes_output_pipes(tmp_path):
    started = time.monotonic()
    result = run_bounded(
        [sys.executable, "-c", "import os, time; os.close(1); os.close(2); time.sleep(10)"],
        cwd=tmp_path, timeout_seconds=0.2, max_output_chars=512,
    )
    assert result.timed_out
    assert result.exit_code != 0
    assert time.monotonic() - started < 3


def test_runner_ignores_stale_bytecode_after_equal_length_edit(tmp_path):
    source = tmp_path / "maths.py"
    source.write_text("def add_numbers(a, b):\n    return a - b\n")
    (tmp_path / "test_maths.py").write_text(
        "from maths import add_numbers\n"
        "def test_add():\n    assert add_numbers(2, 3) == 5\n"
    )
    workspace = trusted_workspace(tmp_path)
    assert run_tests(workspace, "test_maths.py")["status"] == "failed"
    py_compile.compile(str(source), doraise=True)
    original_stat = source.stat()
    source.write_text("def add_numbers(a, b):\n    return a + b\n")
    os.utime(source, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    result = run_tests(workspace, "test_maths.py")
    assert result["status"] == "passed"
    assert "1 passed" in result["stdout"]
