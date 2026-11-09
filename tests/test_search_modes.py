"""Literal searches and explicit regex errors exercise real ripgrep locally."""

import shutil
import time

import pytest

from patchrook_agent.evidence_tools import EvidenceTools
from patchrook_agent.tools import ToolConfig, ToolRegistry, Workspace
from patchrook_agent.tools.base import ToolError
from patchrook_agent.tools.search import search_code


pytestmark = pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep is required")


def test_default_literal_search_handles_interpolation_and_regex_metacharacters(tmp_path):
    target = "${NAME:-fallback}"
    (tmp_path / "source.py").write_text(target + "\nplain NAME text\n")
    result = search_code(Workspace(tmp_path), target)
    assert result["success"]
    assert result["matches"] == [{"path": "source.py", "line_number": 1, "text": target}]


def test_literal_default_does_not_interpret_a_regex(tmp_path):
    (tmp_path / "source.py").write_text("item42\nitem\\d+\n")
    workspace = Workspace(tmp_path)
    assert [item["line_number"] for item in search_code(workspace, r"item\d+")["matches"]] == [2]
    assert [item["line_number"] for item in search_code(workspace, r"item\d+", mode="regex")["matches"]] == [1]


def test_explicit_regex_uses_ripgrep_syntax_and_preserves_escaping(tmp_path):
    (tmp_path / "source.py").write_text("${NAME:-fallback}\n${OTHER:-}\n")
    result = search_code(Workspace(tmp_path), r"\$\{NAME:-fallback\}", mode="regex")
    assert result["success"] and len(result["matches"]) == 1
    assert result["matches"][0]["text"] == "${NAME:-fallback}"


def test_invalid_explicit_regex_returns_exact_literal_retry_without_executing_it(tmp_path):
    pattern = "${NAME:-fallback}"
    (tmp_path / "source.py").write_text(pattern + "\n")
    registry = ToolRegistry(Workspace(tmp_path))
    arguments = {"pattern": pattern, "path": ".", "max_results": 3, "mode": "regex"}
    result = registry.dispatch("search_code", arguments)
    assert not result["success"] and result["error"]["code"] == "invalid_regex"
    assert 'mode="literal"' in result["error"]["message"]
    assert "matches" not in result, "An explicit regex error was silently changed into a literal search"
    assert result["suggested_call"] == {"name": "search_code", "arguments": {
        "pattern": pattern, "path": ".", "max_results": 3, "mode": "literal"}}
    retry = registry.dispatch(**result["suggested_call"])
    assert retry["success"] and retry["matches"][0]["text"] == pattern


def test_evidence_search_accepts_literal_default_and_explicit_regex(tmp_path):
    (tmp_path / "source.py").write_text("${NAME:-fallback}\nitem42\n")
    tools = EvidenceTools(Workspace(tmp_path))
    literal = tools.execute("search_code", {"pattern": "${NAME:-fallback}"}, deadline=time.monotonic() + 10)
    regex = tools.execute("search_code", {"pattern": r"item\d+", "mode": "regex"}, deadline=time.monotonic() + 10)
    assert literal["success"] and literal["matches"][0]["line_number"] == 1
    assert regex["success"] and regex["matches"][0]["line_number"] == 2
    definition = next(item["function"] for item in tools.definitions()
                      if item["function"]["name"] == "search_code")
    assert "literal" in definition["description"]
    assert definition["parameters"]["properties"]["mode"]["enum"] == ["literal", "regex"]
    assert "mode" not in definition["parameters"].get("required", [])


@pytest.mark.parametrize("mode", [None, True, "automatic", [], 1])
def test_direct_search_rejects_unsupported_mode(tmp_path, mode):
    with pytest.raises(ToolError) as error:
        search_code(Workspace(tmp_path), "text", mode=mode)
    assert error.value.code == "invalid_arguments"


def test_literal_pattern_is_an_argument_without_shell_execution(tmp_path):
    marker = tmp_path / "must-not-exist"
    pattern = "$(touch " + str(marker) + ")"
    (tmp_path / "source.py").write_text(pattern + "\n")
    result = search_code(Workspace(tmp_path), pattern)
    assert result["success"] and result["matches"][0]["text"] == pattern
    assert not marker.exists()


def test_long_regex_error_remains_bounded_and_never_shortens_retry_pattern(tmp_path):
    pattern = "[" * 4000
    registry = ToolRegistry(Workspace(tmp_path, ToolConfig(max_output_chars=512)))
    result = registry.dispatch("search_code", {"pattern": pattern, "mode": "regex"})
    assert result["error"]["code"] == "invalid_regex"
    assert 'mode="literal"' in result["error"]["message"]
    assert "suggested_call" not in result

