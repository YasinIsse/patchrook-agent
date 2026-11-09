"""Small observed-evidence guards for the opt-in lean execution loop.

Test scope is explicit pytest selection containment, never inferred dependency
coverage. Progress means a changed observation, not a claim in model prose.
"""

from dataclasses import dataclass
import hashlib
import io
import json
import re
import tokenize
from typing import Any

from patchrook_agent.inspection import canonical_path, visible_lines


def _arguments(value: Any) -> dict:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return {}
    return value if isinstance(value, dict) else {}


@dataclass(frozen=True)
class TestSelection:
    path: str
    nodes: tuple[str, ...] = ()
    directory: bool = False

    @property
    def label(self) -> str:
        return self.path + ("::" + "::".join(self.nodes) if self.nodes else "")

    def covers(self, other: "TestSelection") -> bool:
        if self.directory:
            return self.path == "." or other.path == self.path or other.path.startswith(self.path + "/")
        if self.path != other.path or other.directory or len(self.nodes) > len(other.nodes):
            return False
        for index, node in enumerate(self.nodes):
            target = other.nodes[index]
            # Running an unparameterized pytest function includes its cases.
            if node != target and not ("[" not in node and target.startswith(node + "[")):
                return False
        return True

    def to_dict(self) -> dict:
        return {"path": self.path, "nodes": list(self.nodes), "directory": self.directory}


def _selection(value: Any, workspace, *, directory: bool | None = None) -> TestSelection | None:
    if not isinstance(value, str) or not value or len(value) > 4096 or "\x00" in value:
        return None
    parts = value.split("::")
    if any(not part or part.startswith("-") for part in parts):
        return None
    path = canonical_path(parts[0], workspace)
    if path is None:
        return None
    if directory is None:
        target = workspace.resolve(path)
        if not target.is_file() and not target.is_dir():
            return None
        directory = target.is_dir()
    if directory and len(parts) > 1:
        return None
    return TestSelection(path, tuple(parts[1:]), directory)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def _test_output(value: Any) -> str:
    # Pytest's duration is incidental; changing it must not reset a stall.
    return re.sub(r"\b\d+(?:\.\d+)?\s*(?:seconds?|ms|s)\b", "<time>", value) if isinstance(value, str) else ""


def _material_source(path: str, lines: dict[int, str]) -> dict[int, Any]:
    """Fingerprint observed implementation, excluding navigation-only text.

    Tokenization is local to contiguous visible source. It never opens a file
    or promotes hidden/truncated output to evidence. Declarations without a
    body, standalone docstrings, comments, and whitespace do not constitute a
    new implementation observation. Import and constant declarations do.
    """
    if not path.endswith((".py", ".pyi")):
        return {number: source.strip() for number, source in lines.items()
                if source.strip() and not source.lstrip().startswith(("#", "//", "/*", "*"))}
    segments: list[list[tuple[int, str]]] = []
    for number, source in sorted(lines.items()):
        if not segments or number != segments[-1][-1][0] + 1:
            segments.append([])
        segments[-1].append((number, source))
    material: dict[int, Any] = {}
    for segment in segments:
        text = "\n".join(source for _, source in segment) + "\n"
        statements: list[list[tokenize.TokenInfo]] = []
        current: list[tokenize.TokenInfo] = []
        try:
            for token in tokenize.generate_tokens(io.StringIO(text).readline):
                if token.type == tokenize.NEWLINE:
                    if current:
                        statements.append(current)
                        current = []
                elif token.type not in {tokenize.ENCODING, tokenize.INDENT, tokenize.DEDENT,
                                        tokenize.NL, tokenize.COMMENT, tokenize.ENDMARKER}:
                    current.append(token)
        except (tokenize.TokenError, IndentationError, SyntaxError):
            # An incomplete observed header/string is not proof of a body.
            # Tokens completed before the missing range remain usable.
            pass
        if current:
            statements.append(current)
        for statement in statements:
            first = statement[0].string
            declaration = first in {"def", "class"} or (
                first == "async" and len(statement) > 1 and statement[1].string == "def")
            if first == "@":
                continue
            if declaration:
                depth, body_at = 0, None
                for index, token in enumerate(statement):
                    if token.string in {"(", "[", "{"}:
                        depth += 1
                    elif token.string in {")", "]", "}"}:
                        depth -= 1
                    elif token.string == ":" and depth == 0:
                        body_at = index + 1
                        break
                statement = statement[body_at:] if body_at is not None else []
                if not statement:
                    continue
                first = statement[0].string
            if all(token.type == tokenize.STRING or token.string in {"(", ")"}
                   for token in statement):
                continue
            # A prose line inside a clipped docstring is not new executable
            # evidence merely because Python can tokenize individual words.
            operators = {"=", ":=", "+", "-", "*", "/", "//", "%", "**", "+=", "-=", "*=", "/=",
                         "==", "!=", "<", ">", "<=", ">=", "&", "|", "^", "<<", ">>", ":"}
            has_operation = any(token.type == tokenize.OP and token.string in operators for token in statement)
            has_call = any(token.string == "(" and index > 0
                           and (statement[index - 1].type == tokenize.NAME or statement[index - 1].string in {")", "]"})
                           for index, token in enumerate(statement))
            if not (has_operation or has_call or first in {
                    "return", "raise", "yield", "await", "assert", "import", "from", "pass", "break",
                    "continue", "del", "global", "nonlocal"}):
                continue
            per_line: dict[int, list[tuple[int, str]]] = {}
            for token in statement:
                if token.type == tokenize.ERRORTOKEN and token.string.isspace():
                    continue
                if token.string in {",", ")", "]", "}"}:
                    continue
                number = segment[0][0] + token.start[0] - 1
                if number in lines:
                    per_line.setdefault(number, []).append((token.type, token.string))
            for number, tokens in per_line.items():
                source = lines[number]
                indentation = len(source.expandtabs(8)) - len(source.expandtabs(8).lstrip())
                material[number] = [indentation, tokens]
    return material


class LeanProgress:
    """Persist verification debt and a bounded stall episode between tool calls."""

    def __init__(self, workspace, required_tests=(), *, recovery_after=2, stop_after=4,
                 material_source_progress=False):
        if (type(recovery_after) is not int or type(stop_after) is not int
                or not 1 <= recovery_after < stop_after <= 100):
            raise ValueError("Invalid lean progress thresholds")
        if not isinstance(required_tests, (tuple, list)):
            raise ValueError("Required tests must be a sequence of pytest selections")
        if type(material_source_progress) is not bool:
            raise ValueError("material_source_progress must be a boolean")
        self.workspace = workspace
        self.material_source_progress = material_source_progress
        self.required_tests = []
        for value in required_tests:
            selection = _selection(value, workspace)
            if selection is None:
                raise ValueError("Required tests must be confined existing pytest selections")
            if selection not in self.required_tests:
                self.required_tests.append(selection)
        self.recovery_after, self.stop_after = recovery_after, stop_after
        self.generation = 0
        self.passed: dict[str, TestSelection] = {}
        self.failed: dict[str, tuple[TestSelection, str]] = {}
        self._seen: set[str] = set()
        self._test_outcomes: dict[str, str] = {}
        self.stall_count = 0
        self.recovery_delivered = False

    @property
    def action(self) -> str | None:
        if self.stall_count >= self.stop_after:
            return "stop"
        if self.stall_count >= self.recovery_after and not self.recovery_delivered:
            return "recover"
        return None

    def acknowledge_recovery(self) -> None:
        self.recovery_delivered = True

    def _progress(self) -> None:
        self.stall_count = 0
        self.recovery_delivered = False

    def invalidate(self) -> None:
        """Invalidate passes after an actual mutation, including one seen on resume."""
        self.generation += 1
        self.passed.clear()
        self._seen.clear()
        self._test_outcomes.clear()
        self._progress()

    @property
    def verification_status(self) -> str:
        if self.failed:
            return "failed" if any(status == "failed" for _, status in self.failed.values()) else "timed_out"
        if self.required_tests and any(not any(passed.covers(required) for passed in self.passed.values())
                                       for required in self.required_tests):
            return "not_run"
        return "passed" if self.passed else "not_run"

    def missing(self, require_tests=True) -> list[str]:
        missing = [f"resolve observed {status} tests for {selection.label}"
                   for selection, status in self.failed.values()]
        if require_tests:
            for required in self.required_tests:
                if not any(passed.covers(required) for passed in self.passed.values()):
                    missing.append(f"observe passing tests after the last change for {required.label}")
            if not self.required_tests and not self.passed:
                missing.append("observe passing tests after the last change")
        return missing

    def summary(self) -> str:
        failed = list(self.failed)
        passed = list(self.passed)
        def labels(values):
            return ", ".join(value[:120] for value in values[:3]) + (f", +{len(values) - 3} more" if len(values) > 3 else "")
        parts = [f"Verification: {self.verification_status}; mutation {self.generation}."]
        if failed:
            parts.append("Outstanding failed selections: " + labels(failed) + ".")
        if passed:
            parts.append("Current passing selections: " + labels(passed) + ".")
        if self.stall_count:
            parts.append(f"Unchanged observations: {self.stall_count}/{self.stop_after}.")
        return " ".join(parts)

    def recovery_hint(self) -> str:
        needs = self.missing(require_tests=False)
        return ("Recent tools returned no new evidence. Use a new relevant source range, revise the search, "
                "or act on the evidence already observed. "
                + (("; ".join(need[:150] for need in needs[:3]) + ".") if needs else
                   "Finish if the requested work and verification are complete; otherwise report the blocker."))

    def _source_evidence(self, path: str, lines: dict[int, str]) -> list:
        if not self.material_source_progress:
            return [["source", path, line, source] for line, source in lines.items()]
        # Remember every visible raw line, including non-material headers.
        # A later partial read must not reclassify the same unchanged default
        # argument or docstring fragment as a new implementation discovery.
        fresh = {number for number, source in lines.items()
                 if _digest(["observed_source", path, number, source]) not in self._seen}
        self._seen.update(_digest(["observed_source", path, number, source])
                          for number, source in lines.items())
        evidence = []
        for number, source in _material_source(path, lines).items():
            changed_literal = (isinstance(source, list) and any(
                kind == tokenize.STRING and "\n" in text
                and any(line in fresh for line in range(number, number + text.count("\n") + 1))
                for kind, text in source[1]))
            if number in fresh or changed_literal:
                evidence.append(["source", path, number, source])
        return evidence

    def observe(self, name: str, arguments: Any, result: dict) -> str | None:
        if not isinstance(result, dict):
            return self.action
        arguments = _arguments(arguments)
        if name in {"write_file", "edit_file"} and result.get("success") is True and result.get("modified") is True:
            self.invalidate()
            return None
        evidence = []
        if name in {"read_file", "read_files", "read_code"}:
            records = [result] if name == "read_file" else result.get("files", [])
            for record in records if isinstance(records, list) else []:
                if not isinstance(record, dict):
                    continue
                path = canonical_path(record.get("path"), self.workspace)
                if path is not None:
                    evidence.extend(self._source_evidence(path, visible_lines(record)))
        elif name == "read_skill" and result.get("success") is True and not result.get("already_loaded"):
            evidence.append(["skill", result.get("path"), result.get("content")])
        elif name == "search_code" and result.get("success") is True:
            for match in result.get("matches", []) if isinstance(result.get("matches"), list) else []:
                if not isinstance(match, dict):
                    continue
                path, line, source = canonical_path(match.get("path"), self.workspace), match.get("line_number"), match.get("text")
                if path is not None and type(line) is int and line > 0 and isinstance(source, str):
                    evidence.extend(self._source_evidence(path, {line: source}))
        elif name == "list_files" and result.get("success") is True:
            for entry in result.get("entries", []) if isinstance(result.get("entries"), list) else []:
                path = canonical_path(entry, self.workspace)
                if path is not None:
                    evidence.append(["entry", path])
        elif name == "run_tests":
            status, exit_code = result.get("status"), result.get("exit_code")
            valid = ((status == "passed" and result.get("success") is True and type(exit_code) is int and exit_code == 0)
                     or (status == "failed" and result.get("success") is False and type(exit_code) is int and exit_code != 0)
                     or (status == "timed_out" and result.get("success") is False
                         and (exit_code is None or type(exit_code) is int)))
            selection = _selection(arguments.get("path", "."), self.workspace) if valid else None
            if selection is not None:
                label = selection.label
                if status == "passed":
                    self.passed[label] = selection
                    self.failed = {key: value for key, value in self.failed.items() if not selection.covers(value[0])}
                else:
                    self.failed[label] = (selection, status)
                    # A later failed run invalidates overlapping green evidence.
                    self.passed = {key: value for key, value in self.passed.items()
                                   if not value.covers(selection) and not selection.covers(value)}
                outcome = _digest([status, exit_code, _test_output(result.get("stdout")), _test_output(result.get("stderr"))])
                if self._test_outcomes.get(label) != outcome:
                    self._test_outcomes[label] = outcome
                    self._progress()
                    return None
        novel = {_digest(item) for item in evidence} - self._seen
        self._seen.update(novel)
        if novel:
            self._progress()
        else:
            self.stall_count = min(self.stop_after, self.stall_count + 1)
        return self.action

    def to_dict(self) -> dict:
        return {"schema_version": 1, "generation": self.generation,
                **({"material_source_progress": True} if self.material_source_progress else {}),
                "required_tests": [item.to_dict() for item in self.required_tests],
                "passed": [item.to_dict() for item in self.passed.values()],
                "failed": [{"selection": selection.to_dict(), "status": status} for selection, status in self.failed.values()],
                "seen": sorted(self._seen), "test_outcomes": dict(self._test_outcomes),
                "stall_count": self.stall_count, "recovery_delivered": self.recovery_delivered,
                "recovery_after": self.recovery_after, "stop_after": self.stop_after}

    @classmethod
    def from_dict(cls, data: dict, workspace, required_tests=None):
        keys = {"schema_version", "generation", "required_tests", "passed", "failed", "seen", "test_outcomes",
                "stall_count", "recovery_delivered", "recovery_after", "stop_after"}
        if (not isinstance(data, dict) or not keys <= set(data)
                or set(data) - keys - {"material_source_progress"}
                or type(data.get("schema_version")) is not int or data["schema_version"] != 1):
            raise ValueError("Unsupported lean progress state")
        progress = cls(workspace, (), recovery_after=data["recovery_after"], stop_after=data["stop_after"],
                       material_source_progress=data.get("material_source_progress", False))
        def restore(value):
            if (not isinstance(value, dict) or set(value) != {"path", "nodes", "directory"}
                    or not isinstance(value["nodes"], list) or any(not isinstance(node, str) for node in value["nodes"])
                    or type(value["directory"]) is not bool or not isinstance(value["path"], str)):
                raise ValueError("Invalid stored pytest selection")
            label = value["path"] + ("::" + "::".join(value["nodes"]) if value["nodes"] else "")
            selection = _selection(label, workspace, directory=value["directory"])
            if selection is None or selection.to_dict() != value:
                raise ValueError("Invalid stored pytest selection")
            return selection
        for key in ("required_tests", "passed", "failed", "seen"):
            if not isinstance(data[key], list):
                raise ValueError("Invalid lean progress collection")
        progress.required_tests = [restore(item) for item in data["required_tests"]]
        if required_tests is not None and cls(workspace, required_tests).required_tests != progress.required_tests:
            raise ValueError("Required tests differ from the saved lean state")
        progress.passed = {item.label: item for item in map(restore, data["passed"])}
        for record in data["failed"]:
            if (not isinstance(record, dict) or set(record) != {"selection", "status"}
                    or not isinstance(record["status"], str) or record["status"] not in {"failed", "timed_out"}):
                raise ValueError("Invalid failed pytest selection")
            selection = restore(record["selection"])
            progress.failed[selection.label] = (selection, record["status"])
        if (type(data["generation"]) is not int or data["generation"] < 0
                or type(data["stall_count"]) is not int or not 0 <= data["stall_count"] <= progress.stop_after
                or type(data["recovery_delivered"]) is not bool
                or any(not isinstance(item, str) or re.fullmatch(r"[0-9a-f]{64}", item) is None for item in data["seen"])
                or not isinstance(data["test_outcomes"], dict)):
            raise ValueError("Invalid lean progress counters or fingerprints")
        for label, digest in data["test_outcomes"].items():
            if (_selection(label, workspace, directory=False) is None or not isinstance(digest, str)
                    or re.fullmatch(r"[0-9a-f]{64}", digest) is None):
                raise ValueError("Invalid pytest outcome fingerprint")
        progress.generation, progress.stall_count = data["generation"], data["stall_count"]
        progress.recovery_delivered = data["recovery_delivered"]
        progress._seen, progress._test_outcomes = set(data["seen"]), dict(data["test_outcomes"])
        return progress
