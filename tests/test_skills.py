import json
from pathlib import Path

import pytest

from patchrook_agent.skills import SkillLoader
from patchrook_agent.tools.base import ToolError


def skill_root(tmp_path):
    root = tmp_path / "trusted"
    directory = root / "debugging"
    (directory / "references").mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        "---\nname: debugging\ndescription: Fix a bug.\nactivation: When tests fail.\n---\n"
        "# Debugging\nEXCLUSIVE_INSTRUCTIONS: inspect, fix, verify.\n"
    )
    (directory / "references" / "example.md").write_text("Detailed example only when requested.\n")
    return root


def test_index_exposes_only_metadata_and_does_not_activate(tmp_path):
    loader = SkillLoader(skill_root(tmp_path))
    assert loader.index() == [{"name": "debugging", "description": "Fix a bug.",
                               "activation": "When tests fail.", "reference": "debugging/SKILL.md"}]
    assert "EXCLUSIVE_INSTRUCTIONS" not in loader.index_text()
    assert "debugging" in loader.index_text()
    assert loader.loaded == [] and loader.loaded_messages() == []


def test_instructions_then_reference_disclose_once(tmp_path):
    loader = SkillLoader(skill_root(tmp_path))
    with pytest.raises(ToolError, match="before requesting references"):
        loader.load("debugging", "references/example.md")
    instructions = loader.load("debugging")
    assert instructions["success"] and "EXCLUSIVE_INSTRUCTIONS" in instructions["content"]
    assert "Detailed example" not in json.dumps(loader.loaded_messages())
    reference = loader.load("debugging", "references/example.md")
    assert reference["is_reference"] and "Detailed example" in reference["content"]
    assert len(loader.loaded_messages()) == 2
    duplicate = loader.load("debugging")
    assert duplicate["already_loaded"] and duplicate["content"] == ""
    assert loader.load("debugging", "SKILL.md")["already_loaded"]
    assert loader.load("debugging", "references/./example.md")["already_loaded"]
    assert loader.loaded == ["debugging"] and len(loader.loaded_messages()) == 2


@pytest.mark.parametrize("reference", [None, "SKILL.md", "./SKILL.md", "debugging/SKILL.md", "./debugging/SKILL.md"])
def test_initial_entrypoint_accepts_the_published_index_path(tmp_path, reference):
    loader = SkillLoader(skill_root(tmp_path))
    result = loader.load("debugging", reference)
    assert result["success"] and not result["is_reference"]
    assert "EXCLUSIVE_INSTRUCTIONS" in result["content"]
    assert loader.load("debugging")["already_loaded"]
    assert len(loader.loaded_messages()) == 1


def test_root_relative_supporting_reference_still_requires_main_instructions(tmp_path):
    loader = SkillLoader(skill_root(tmp_path))
    with pytest.raises(ToolError, match="before requesting references"):
        loader.load("debugging", "debugging/references/example.md")
    loader.load("debugging", loader.index()[0]["reference"])
    assert loader.load("debugging", "debugging/references/example.md")["is_reference"]


@pytest.mark.parametrize("name,reference", [("../outside", None), ("/outside", None),
                                           ("debugging", "../../../secret"), ("debugging", "/etc/passwd")])
def test_paths_cannot_escape_trusted_skill(tmp_path, name, reference):
    loader = SkillLoader(skill_root(tmp_path))
    loader.load("debugging")
    with pytest.raises(ToolError):
        loader.load(name, reference)


def test_discovery_rejects_external_symlinks(tmp_path):
    root = skill_root(tmp_path)
    outside = tmp_path / "secret"
    outside.write_text("private")
    (root / "debugging" / "references" / "escape.md").symlink_to(outside)
    with pytest.raises(ToolError):
        SkillLoader(root)


def test_changed_or_new_documents_do_not_gain_trust_mid_task(tmp_path):
    root = skill_root(tmp_path)
    loader = SkillLoader(root)
    loader.load("debugging")
    (root / "debugging" / "references" / "new.md").write_text("Injected instructions")
    with pytest.raises(ToolError, match="changed or was not indexed"):
        loader.load("debugging", "references/new.md")
    (root / "debugging" / "SKILL.md").write_text("Modified instructions")
    with pytest.raises(ToolError, match="changed or was not indexed"):
        loader.load("debugging")


def test_disabled_skills_need_no_directory_or_context(tmp_path):
    loader = SkillLoader(tmp_path / "absent", enabled=False)
    assert loader.index() == [] and loader.index_text() == ""
    assert loader.loaded_messages() == []
    with pytest.raises(ToolError, match="disabled"):
        loader.load("debugging")


def test_checkpoint_restores_exact_loaded_documents(tmp_path):
    root = skill_root(tmp_path)
    loader = SkillLoader(root)
    loader.load("debugging")
    loader.load("debugging", "references/example.md")
    saved = json.loads(json.dumps(loader.to_dict()))
    restored = SkillLoader(root)
    restored.restore(saved)
    assert restored.loaded_messages() == loader.loaded_messages()
    assert restored.to_dict() == saved


def test_checkpoint_detects_externally_changed_instructions(tmp_path):
    root = skill_root(tmp_path)
    loader = SkillLoader(root)
    loader.load("debugging")
    saved = loader.to_dict()
    source = root / "debugging" / "SKILL.md"
    source.write_text(source.read_text() + "Changed since checkpoint\n")
    restored = SkillLoader(root)
    with pytest.raises(ToolError, match="changed since the checkpoint"):
        restored.restore(saved)
    assert restored.loaded == []


def test_checkpoint_checks_unloaded_reference_versions_too(tmp_path):
    root = skill_root(tmp_path)
    loader = SkillLoader(root)
    saved = loader.to_dict()
    (root / "debugging" / "references" / "example.md").write_text("Changed before loading")
    with pytest.raises(ToolError, match="changed since the checkpoint"):
        SkillLoader(root).restore(saved)


def test_referenced_scripts_are_returned_as_text_without_execution(tmp_path):
    root = skill_root(tmp_path)
    (root / "debugging" / "example.py").write_text("raise RuntimeError('must not execute')\n")
    loader = SkillLoader(root)
    loader.load("debugging")
    assert "must not execute" in loader.load("debugging", "example.py")["content"]


def test_sample_skill_index_is_small_and_instructions_are_lazy():
    loader = SkillLoader()
    assert {item["name"] for item in loader.index()} == {"debugging", "feature-implementation", "investigation"}
    assert len(loader.index_text()) < 1200
    assert loader.loaded_messages() == []


def test_compact_index_keeps_every_selection_field_without_loading_documents(tmp_path):
    loader = SkillLoader(skill_root(tmp_path))
    index_text = loader.index_text()
    item = loader.index()[0]
    assert all(value in index_text for value in item.values())
    previous = ("Available trusted skills (load selected instructions with read_skill):\n"
                f"- {item['name']}: {item['description']} Activate: {item['activation']} "
                f"Reference: {item['reference']}")
    assert len(index_text) < len(previous)
    assert loader.loaded == [] and loader.loaded_messages() == []
    assert loader.to_dict()["loaded"] == []
