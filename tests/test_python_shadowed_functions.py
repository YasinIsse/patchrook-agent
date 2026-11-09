"""New live bodies must replace the current binding, not precede an old stub."""

import time

import pytest

from patchrook_agent.python_duplicates import new_duplicate_function
from test_evidence_python_syntax import tools, present, apply, create, edit


@pytest.mark.parametrize("prefix", ["def", "async def"])
def test_new_different_body_is_rejected_when_old_same_arguments_definition_overrides_it(prefix):
    before = f"class Subject:\n    {prefix} recover(self, data):\n        pass\n"
    after = before.replace("class Subject:\n", f"class Subject:\n    {prefix} recover(self, data):\n        return data\n")
    found = new_duplicate_function(before.encode(), after.encode(), deadline=time.monotonic() + 10)
    assert found == {"name": "recover", "scope": "class:Subject", "first_line": 2, "duplicate_line": 4}
    assert new_duplicate_function(after.encode(), after.replace("return data", "return data + 1").encode(),
                                  deadline=time.monotonic() + 10) is None


def test_new_shadowed_body_and_unrelated_edits_are_rejected_before_any_write(tmp_path):
    source = 'class Subject:\n    def recover(self, data):\n        """Existing stub."""\n'
    (tmp_path / "subject.py").write_text(source)
    editor = tools(tmp_path)
    result = apply(editor, [edit("subject.py", "class Subject:\n",
        "class Subject:\n    def recover(self, data):\n        return data\n"),
        create("new/deep/other.py", "VALUE = 1\n")], [present(editor, "subject.py")])
    assert result["error"]["code"] == "python_duplicate_function"
    message = result["error"]["message"]
    assert "Rejected candidate" in message and "later same-argument definition" in message
    assert "No files were written" in message and "Replace the existing definition" in message
    assert (tmp_path / "subject.py").read_text() == source and not (tmp_path / "new").exists()
