"""Tool errors offer bounded corrections without executing a wider action."""

from patchrook_agent.skills.loader import SkillLoader
from patchrook_agent.tools import ToolConfig, ToolRegistry, Workspace
from patchrook_agent.tools.base import observation_chars


def registry(tmp_path, *, allow_tests=True, limit=2000, enabled=None):
    tools = ToolRegistry(Workspace(tmp_path, ToolConfig(allow_tests=allow_tests, max_output_chars=limit)))
    tools.observation_overhead_chars = 160
    tools.guidance_tools = enabled
    tools.capture_enabled = True
    return tools


def test_run_tests_without_path_runs_pytest_at_workspace_root(tmp_path):
    (tmp_path / "test_default.py").write_text(
        "from pathlib import Path\n"
        "def test_default():\n    Path('root-was-tested').write_text('yes')\n    assert True\n"
    )
    tools = registry(tmp_path)
    result = tools.dispatch("run_tests", {})
    assert result["success"] and result["status"] == "passed" and result["exit_code"] == 0
    assert result["test_counts"] == {"passed": 1}
    assert (tmp_path / "root-was-tested").read_text() == "yes"
    assert tools.last_execution["arguments"] == {}


def test_invalid_test_target_does_not_broaden_execution_and_suggests_enabled_discovery(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("An invalid test path must not launch pytest")
    monkeypatch.setattr("patchrook_agent.tools.tests.run_bounded", forbidden)
    tools = registry(tmp_path, limit=512, enabled={"run_tests", "list_files"})
    result = tools.dispatch("run_tests", {"path": "tests/missing.py"})
    assert not result["success"] and result["error"]["code"] == "not_found"
    assert "Test path does not exist: tests/missing.py" in result["error"]["message"]
    assert "No tests ran" in result["error"]["message"]
    assert result["suggested_call"] == {"name": "list_files", "arguments": {
        "path": ".", "recursive": True, "max_entries": 30}}
    assert observation_chars(result) <= 352


def test_missing_target_only_suggests_root_pytest_when_authorized_and_discovery_disabled(tmp_path):
    tools = registry(tmp_path, enabled={"run_tests"})
    result = tools.dispatch("run_tests", {"path": "missing.py"})
    assert result["error"]["code"] == "not_found"
    assert result["suggested_call"] == {"name": "run_tests", "arguments": {"path": "."}}
    denied = registry(tmp_path, allow_tests=False, enabled={"run_tests"}).dispatch("run_tests", {"path": "missing.py"})
    assert denied["error"]["code"] == "permission_denied" and "suggested_call" not in denied


def test_test_path_escape_retains_security_error_without_a_wider_test_suggestion(tmp_path):
    result = registry(tmp_path).dispatch("run_tests", {"path": "../outside.py"})
    assert result["error"]["code"] == "outside_workspace" and "suggested_call" not in result


def test_full_and_lean_descriptions_expose_the_same_root_default_compactly(tmp_path):
    tools = registry(tmp_path)
    for definitions in (tools.definitions(), tools.definitions(lean=True)):
        description = next(item["function"]["description"] for item in definitions
                           if item["function"]["name"] == "run_tests")
        assert "run_tests()" in description and "workspace root" in description
        assert len(description) < 100


def trusted_skills(tmp_path, names):
    root = tmp_path / "trusted"
    root.mkdir()
    for name in names:
        directory = root / name
        directory.mkdir()
        (directory / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: Trusted guidance.\nactivation: When selected.\n---\nUse observed evidence.\n"
        )
    return SkillLoader(root)


def test_invalid_skill_name_lists_only_actual_frozen_trusted_names(tmp_path):
    names = ["debugging", "investigation"]
    tools = registry(tmp_path, limit=512, enabled={"read_skill"})
    loader = trusted_skills(tmp_path, names)
    tools.attach_skills(loader)
    result = tools.dispatch("read_skill", {"name": "invented-skill"})
    assert not result["success"] and result["error"]["code"] == "skill_not_found"
    assert result["valid_names"] == names
    assert not loader.loaded and observation_chars(result) <= 352
    missing_name = tools.dispatch("read_skill", {})
    assert missing_name["error"]["code"] == "invalid_arguments" and missing_name["valid_names"] == names


def test_many_skill_names_are_bounded_but_full_actual_names_remain_in_capture(tmp_path):
    names = ["skill-" + str(index).zfill(2) + "-" + "a" * 45 for index in range(12)]
    tools = registry(tmp_path, limit=512, enabled={"read_skill"})
    tools.attach_skills(trusted_skills(tmp_path, names))
    result = tools.dispatch("read_skill", {"name": "not-an-indexed-name"})
    assert observation_chars(result) <= 352 and result["valid_names"]
    assert result["valid_names"] == names[:len(result["valid_names"])]
    assert len(result["valid_names"]) + result["additional_valid_names"] == len(names)
    assert tools.last_execution["available_output"]["valid_skill_names"] == names


def test_skill_reference_escape_and_disabled_skills_do_not_gain_name_recovery(tmp_path):
    tools = registry(tmp_path)
    loader = trusted_skills(tmp_path, ["debugging"])
    tools.attach_skills(loader)
    result = tools.dispatch("read_skill", {"name": "debugging", "reference": "../outside.md"})
    assert result["error"]["code"] == "outside_skills" and "valid_names" not in result
    tools.attach_skills(SkillLoader(loader.root, enabled=False))
    disabled = tools.dispatch("read_skill", {"name": "debugging"})
    assert disabled["error"]["code"] == "unknown_tool" and "valid_names" not in disabled
