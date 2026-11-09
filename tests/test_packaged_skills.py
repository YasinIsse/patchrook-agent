"""Trusted skill discovery must work in wheels without trusting the task checkout."""

from pathlib import Path
import shutil

import pytest

import patchrook_agent.skills.loader as module
from patchrook_agent.skills import SkillLoader
from patchrook_agent.tools.base import ToolError


CANONICAL = Path(__file__).resolve().parents[1] / "skills"
EXPECTED = {"debugging", "feature-implementation", "investigation"}


def installed_module(tmp_path, monkeypatch):
    filename = tmp_path / "venv/lib/python3.12/site-packages/patchrook_agent/skills/loader.py"
    filename.parent.mkdir(parents=True)
    filename.touch()
    monkeypatch.setattr(module, "__file__", str(filename))
    return filename


def test_installed_loader_uses_bundled_skills_and_reference(tmp_path, monkeypatch):
    filename = installed_module(tmp_path, monkeypatch)
    bundled = filename.with_name("builtin")
    shutil.copytree(CANONICAL, bundled)
    monkeypatch.chdir(tmp_path)
    loader = SkillLoader()
    assert loader.root == bundled
    assert {item["name"] for item in loader.index()} == EXPECTED
    assert loader.load("debugging")["success"]
    assert loader.load("debugging", "references/example.md")["success"]


def test_installed_loader_never_falls_back_to_ambient_or_task_skills(tmp_path, monkeypatch):
    filename = installed_module(tmp_path, monkeypatch)
    shutil.copytree(CANONICAL, filename.resolve().parents[3] / "skills")
    task = tmp_path / "task"
    task.mkdir()
    shutil.copytree(CANONICAL, task / "skills")
    monkeypatch.chdir(task)
    with pytest.raises(ToolError, match="Trusted skill root"):
        SkillLoader()


def test_explicit_trusted_root_overrides_packaged_default(tmp_path, monkeypatch):
    filename = installed_module(tmp_path, monkeypatch)
    shutil.copytree(CANONICAL, filename.with_name("builtin"))
    explicit = tmp_path / "explicit"
    shutil.copytree(CANONICAL / "debugging", explicit / "debugging")
    loader = SkillLoader(explicit)
    assert loader.root == explicit
    assert [item["name"] for item in loader.index()] == ["debugging"]
    with pytest.raises(ToolError, match="Trusted skill root"):
        SkillLoader(tmp_path / "missing")


def test_actual_source_checkout_retains_canonical_skills():
    loader = SkillLoader()
    assert loader.root == CANONICAL
    assert {item["name"] for item in loader.index()} == EXPECTED
