"""Disposable coding regression fixtures and independent acceptance helpers."""

import ast
from dataclasses import dataclass
import io
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import tokenize
from typing import Any

from patchrook_agent.tools.process import run_bounded


@dataclass(frozen=True)
class BenchmarkTask:
    name: str
    instructions: str
    files: dict[str, str]
    acceptance_code: str
    expected_answer: tuple[str, ...] = ()
    unchanged_files: tuple[str, ...] = ()


_ADD_SOURCE = "def add_numbers(a, b):\n    return a - b\n\ndef multiply_numbers(a, b):\n    return a * b\n"
_ADD_TEST = "from math_utils import add_numbers\n\ndef test_add_numbers():\n    assert add_numbers(5, 3) == 8\n    assert add_numbers(-2, 7) == 5\n"
_ADD_CHECK = """import math_utils
assert math_utils.add_numbers(5, 3) == 8
assert math_utils.add_numbers(-2, 7) == 5
assert math_utils.add_numbers(0, 0) == 0
assert math_utils.multiply_numbers(5, 3) == 15
"""

_ORDER_SOURCE = """from money import discounted_total

FREE_SHIPPING_CENTS = 5000
SHIPPING_CENTS = 500

# Shipping eligibility uses the discounted merchandise amount.
def order_total(unit_cents, quantity, discount_percent=0):
    merchandise = discounted_total(unit_cents, quantity, discount_percent)
    shipping = 0 if merchandise <= FREE_SHIPPING_CENTS else SHIPPING_CENTS
    return merchandise + shipping

def order_reference(number):
    return f"ORDER-{number:04d}"
"""
_MONEY_SOURCE = """# All monetary calculations use integer cents.
def discounted_total(unit_cents, quantity, discount_percent=0):
    subtotal = unit_cents * quantity
    return subtotal + subtotal * discount_percent // 100

def cents_to_text(cents):
    return f"{cents / 100:.2f}"
"""
_ORDER_TEST = """from orders import order_total

def test_discount_and_shipping():
    assert order_total(2000, 2, 10) == 4100
    assert order_total(3000, 2, 10) == 5400
    assert order_total(2500, 2) == 5000
"""
_ORDER_CHECK = """from orders import order_total, order_reference
from money import discounted_total, cents_to_text
assert discounted_total(2000, 2, 10) == 3600
assert discounted_total(1999, 3, 15) == 5098
assert discounted_total(2500, 2, 0) == 5000
assert discounted_total(99, 1, 1) == 99
assert discounted_total(101, 1, 1) == 100
assert discounted_total(101, 3, 10) == 273
assert discounted_total(1, 1, 99) == 1
assert discounted_total(5000, 1, 100) == 0
assert discounted_total(0, 3, 50) == 0
assert order_total(2000, 2, 10) == 4100
assert order_total(3000, 2, 10) == 5400
assert order_total(2500, 2) == 5000
assert order_total(4999, 1) == 5499
assert order_total(5001, 1) == 5001
assert order_total(6248, 1, 20) == 5499
assert order_total(6250, 1, 20) == 5000
assert order_total(6251, 1, 20) == 5001
assert order_reference(7) == 'ORDER-0007'
assert cents_to_text(1234) == '12.34'
for args in ((2000, 2, 10), (101, 3, 10), (99, 1, 1), (5000, 1, 100)):
    assert type(discounted_total(*args)) is int
    assert type(order_total(*args)) is int
"""


def _long_task() -> BenchmarkTask:
    files = {}
    for stage in range(8):
        operation = "-" if stage == 7 else "+"
        notes = "\n".join(
            f"# Stage {stage} note {index}: keep the input and all other stages unchanged when tracing this pipeline."
            for index in range(12)
        )
        files[f"pipeline/stage_{stage}.py"] = (
            f"{notes}\n\ndef transform(value):\n    return value {operation} {stage + 1}\n"
        )
    files["pipeline/__init__.py"] = ""
    imports = "\n".join(f"from pipeline.stage_{stage} import transform as stage_{stage}" for stage in range(8))
    calls = "\n".join(f"    value = stage_{stage}(value)" for stage in range(8))
    files["pipeline/runner.py"] = f"{imports}\n\ndef process(value):\n{calls}\n    return value\n"
    files["tests/test_pipeline.py"] = (
        "from pipeline.runner import process\n\ndef test_pipeline():\n    assert process(10) == 46\n"
    )
    return BenchmarkTask(
        "long-context",
        "Trace pipeline/runner.py through all eight stage modules before editing. Each stage must add its "
        "1-based stage number to the input. Find the incorrect stage, fix only that operation, run the relevant "
        "test, and report the stages inspected and verification. Preserve the other stages and their comments.",
        files,
        "from pipeline.runner import process\nassert process(10) == 46\nassert process(-36) == 0\n",
        unchanged_files=tuple(f"pipeline/stage_{stage}.py" for stage in range(7)) + ("pipeline/runner.py",),
    )


TASKS = {
    "simple-bugfix": BenchmarkTask(
        "simple-bugfix",
        "Inspect this repository. Find the Python function called add_numbers. It subtracts numbers instead "
        "of adding them. Fix the implementation, run its relevant tests, and report what changed and whether "
        "the tests passed. Preserve unrelated behavior.",
        {"math_utils.py": _ADD_SOURCE, "tests/test_math_utils.py": _ADD_TEST},
        _ADD_CHECK,
    ),
    "multifile-investigation": BenchmarkTask(
        "multifile-investigation",
        "Investigate how a delivery quote is calculated. Identify the entry function, the file defining "
        "base_rate, both service rates, and how negative weights behave. Cite relevant file paths and "
        "function names. Do not modify the repository.",
        {
            "shipping.py": "from rates import base_rate\n\ndef quote(weight, service='ground'):\n    return max(weight, 0) * base_rate(service)\n",
            "rates.py": "RATES = {'ground': 4, 'express': 9}\n\ndef base_rate(service):\n    return RATES[service]\n",
            "tests/test_shipping.py": "from shipping import quote\n\ndef test_quote():\n    assert quote(2, 'ground') == 8\n    assert quote(-2, 'express') == 0\n",
        },
        "from shipping import quote\nassert quote(2, 'ground') == 8\nassert quote(2, 'express') == 18\nassert quote(-1, 'ground') == 0\n",
        expected_answer=("shipping.py", "rates.py", "quote", "base_rate", "4|four", "9|nine", "0|zero"),
        unchanged_files=("shipping.py", "rates.py", "tests/test_shipping.py"),
    ),
    "test-diagnosis": BenchmarkTask(
        "test-diagnosis",
        "Run the existing ID parser test to reproduce its failure. Diagnose why empty comma-separated "
        "fields fail, fix the parser to ignore empty or whitespace-only fields, preserve negative IDs, "
        "and rerun the relevant tests. Explain the observed failure and final verification.",
        {
            "ids.py": "def parse_ids(value):\n    return [int(part) for part in value.split(',')]\n\ndef format_id(value):\n    return str(value)\n",
            "tests/test_ids.py": "from ids import parse_ids\n\ndef test_empty_fields():\n    assert parse_ids('1, ,2,,') == [1, 2]\n    assert parse_ids('') == []\n",
        },
        "from ids import parse_ids, format_id\nassert parse_ids('1, ,2,,') == [1, 2]\nassert parse_ids('') == []\nassert parse_ids(' -3, 4 ,') == [-3, 4]\nassert format_id(7) == '7'\n",
        unchanged_files=("tests/test_ids.py",),
    ),
    "multifile-fix": BenchmarkTask(
        "multifile-fix",
        "Inspect orders.py and money.py. Fix the order-total calculation: subtract the integer-cent "
        "percentage discount, then charge 500 cents shipping only when the discounted merchandise "
        "amount is below 5000 cents. Orders at or above 5000 cents receive free shipping. Correct "
        "both defects, run the tests, and report the changed files and verification. Preserve the "
        "other helpers, comments, tests, and settings.",
        {"orders.py": _ORDER_SOURCE, "money.py": _MONEY_SOURCE,
         "tests/test_orders.py": _ORDER_TEST, "settings.py": "CURRENCY = 'USD'\n"},
        _ORDER_CHECK,
        unchanged_files=("tests/test_orders.py", "settings.py"),
    ),
    "long-context": _long_task(),
    "interrupted-resume": BenchmarkTask(
        "interrupted-resume",
        "Inspect this repository. Find the Python function called add_numbers. It subtracts numbers instead "
        "of adding them. Fix the implementation, run its relevant tests, and report what changed and whether "
        "the tests passed. Preserve unrelated behavior.",
        {"math_utils.py": _ADD_SOURCE, "tests/test_math_utils.py": _ADD_TEST},
        _ADD_CHECK,
    ),
}


def create_fixture(root: Path, task_name: str) -> BenchmarkTask:
    """Reset only a new empty directory or a directory we previously marked."""
    if task_name not in TASKS:
        raise ValueError(f"Unknown benchmark task: {task_name}")
    root = Path(root)
    if root.is_symlink():
        raise ValueError("Benchmark workspaces must not be symlinks")
    root.mkdir(parents=True, exist_ok=True)
    marker = root / ".patchrook-benchmark.json"
    if any(root.iterdir()):
        try:
            data = json.loads(marker.read_text())
        except (OSError, ValueError) as exc:
            raise ValueError("Refusing to reset a directory without a benchmark marker") from exc
        if data.get("schema_version") != 1 or data.get("fixture") not in TASKS:
            raise ValueError("Invalid benchmark workspace marker")
        for entry in root.iterdir():
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry)
            else:
                entry.unlink()
    task = TASKS[task_name]
    for name, content in task.files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    marker.write_text(json.dumps({"schema_version": 1, "fixture": task_name}) + "\n")
    return task


def _stage_inspection_before_edit(task: BenchmarkTask, state: Any) -> dict[str, bool]:
    """Check exact model-visible stage operations before the first mutation request."""
    required = {}
    observed = {}
    for index in range(8):
        path = f"pipeline/stage_{index}.py"
        lines = task.files[path].splitlines()
        required[path] = {number: line for number, line in enumerate(lines, 1)
                          if line.startswith("def transform(") or line.startswith("    return ")}
        observed[path] = set()
    for message in getattr(state, "messages", []):
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant":
            calls = message.get("tool_calls") or []
            if isinstance(calls, list) and any(
                    isinstance(call, dict) and isinstance(call.get("function"), dict)
                    and call["function"].get("name") in {"edit_file", "write_file"} for call in calls):
                break
        if message.get("role") != "tool" or message.get("tool_name") not in {"read_file", "read_files", "read_code"}:
            continue
        try:
            result = json.loads(message.get("content", ""))
        except (TypeError, ValueError):
            continue
        if not isinstance(result, dict):
            continue
        records = result.get("files", []) if message["tool_name"] in {"read_files", "read_code"} else [result]
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict) or record.get("success") is not True:
                continue
            path, content = record.get("path"), record.get("content")
            if not isinstance(path, str) or path not in required or not isinstance(content, str):
                continue
            start, end = record.get("start_line"), record.get("end_line")
            if type(start) is not int or type(end) is not int or start < 1 or end < start:
                continue
            for numbered in content.splitlines():
                number, separator, text = numbered.partition(": ")
                if (separator and number.isascii() and number.isdecimal() and len(number) <= 10
                        and start <= int(number) <= end
                        and required[path].get(int(number)) == text):
                    observed[path].add(int(number))
    return {f"inspected_before_edit:{path}": bool(lines) and set(lines) <= observed[path]
            for path, lines in required.items()}


def _operation_span(source: str, function_name: str, assignment: str | None) -> tuple[int, int]:
    """Locate the one permitted expression, including surrounding parentheses.

    AST positions use UTF-8 byte columns. The statement end includes any outer
    parentheses that its value node omits, so equivalent parenthesized or
    multiline expressions remain valid without relaxing adjacent source.
    """
    functions = [node for node in ast.parse(source).body
                 if isinstance(node, ast.FunctionDef) and node.name == function_name]
    if len(functions) != 1:
        raise ValueError("Expected one unchanged top-level function")
    if assignment is None:
        statements = [node for node in functions[0].body
                      if isinstance(node, ast.Return) and node.value is not None]
    else:
        statements = [node for node in functions[0].body
                      if isinstance(node, ast.Assign) and len(node.targets) == 1
                      and isinstance(node.targets[0], ast.Name) and node.targets[0].id == assignment]
    if len(statements) != 1:
        raise ValueError("Expected one designated operation")
    statement = statements[0]
    raw = source.encode("utf-8")
    starts = [0]
    for line in raw.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))

    def position(line, column):
        return starts[line - 1] + column

    if assignment is None:
        start = position(statement.lineno, statement.col_offset) + len(b"return")
    else:
        target = statement.targets[0]
        start = position(target.end_lineno, target.end_col_offset)
        while raw[start:start + 1] in {b" ", b"\t"}:
            start += 1
        if raw[start:start + 1] != b"=":
            raise ValueError("Expected an unchanged assignment target")
        start += 1
    while raw[start:start + 1] in {b" ", b"\t"}:
        start += 1
    end = position(statement.end_lineno, statement.end_col_offset)
    if start >= end:
        raise ValueError("Expected a nonempty expression")
    return start, end


def _operation_comments(source: str, span: tuple[int, int]) -> list[str]:
    lines = source.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line.encode("utf-8")))
    comments = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            row, column = token.start
            offset = starts[row - 1] + len(lines[row - 1][:column].encode("utf-8"))
            if span[0] <= offset < span[1]:
                comments.append(token.string)
    return comments


def _only_operation_changed(original: str, current: str, *, function_name: str,
                            assignment: str | None = None) -> bool:
    """Check edit scope and preservation; independent tests judge behavior."""
    try:
        before = _operation_span(original, function_name, assignment)
        after = _operation_span(current, function_name, assignment)
        old, new = original.encode("utf-8"), current.encode("utf-8")
        if (old[:before[0]] != new[:after[0]] or old[before[1]:] != new[after[1]:]
                or old[before[0]:before[1]] == new[after[0]:after[1]]):
            return False
        # Existing comments within a multiline expression must remain, in
        # order. Comments outside the span are already protected byte-for-byte.
        expected_comments = _operation_comments(original, before)
        preserved = 0
        for comment in _operation_comments(current, after):
            if preserved < len(expected_comments) and comment == expected_comments[preserved]:
                preserved += 1
        return preserved == len(expected_comments)
    except (SyntaxError, ValueError, TypeError, tokenize.TokenError):
        return False


def verify_fixture(root: Path, task: BenchmarkTask, state: Any | None = None) -> dict[str, Any]:
    """Run fixed acceptance code outside the model's mutable pytest files."""
    root = Path(root).resolve()
    checks: dict[str, bool] = {}
    for name in task.unchanged_files:
        try:
            checks[f"unchanged:{name}"] = (root / name).read_text() == task.files[name]
        except OSError:
            checks[f"unchanged:{name}"] = False
    if task.name in {"simple-bugfix", "interrupted-resume"}:
        try:
            checks["only_intended_source_change"] = (root / "math_utils.py").read_text() == _ADD_SOURCE.replace("a - b", "a + b")
        except OSError:
            checks["only_intended_source_change"] = False
    if task.name == "long-context":
        expected = task.files["pipeline/stage_7.py"].replace("value - 8", "value + 8")
        try:
            checks["only_intended_stage_change"] = (root / "pipeline/stage_7.py").read_text() == expected
        except OSError:
            checks["only_intended_stage_change"] = False
    if task.name == "multifile-fix":
        operations = (("orders.py", "order_total", "shipping"),
                      ("money.py", "discounted_total", None))
        for name, function_name, assignment in operations:
            try:
                current = (root / name).read_bytes().decode("utf-8")
                checks[f"only_intended_change:{name}"] = _only_operation_changed(
                    task.files[name], current, function_name=function_name, assignment=assignment)
            except (OSError, UnicodeDecodeError):
                checks[f"only_intended_change:{name}"] = False
    # Isolated interpreter ignores inherited PYTHONPATH; only this fixture is
    # explicitly placed on sys.path. A fresh cache avoids stale rapid-edit pyc.
    with TemporaryDirectory(prefix="patchrook-verifier-") as directory:
        checker = Path(directory) / "acceptance.py"
        checker.write_text("import sys\nsys.path.insert(0, " + repr(str(root)) + ")\n" + task.acceptance_code)
        result = run_bounded([sys.executable, "-I", "-B", "-X", f"pycache_prefix={directory}/cache", str(checker)],
                             cwd=root, timeout_seconds=20, max_output_chars=2000)
    checks["independent_acceptance"] = result.exit_code == 0 and not result.timed_out
    if state is not None:
        checks["agent_completed"] = state.status == "completed"
        if task.name == "long-context":
            checks.update(_stage_inspection_before_edit(task, state))
        if task.expected_answer:
            answer = state.final_response.lower()
            checks["answer_contains_required_facts"] = all(any(option.lower() in answer for option in term.split("|"))
                                                          for term in task.expected_answer)
        else:
            checks["observed_tests_passed"] = state.verification == "passed"
        if task.name == "test-diagnosis":
            # Reproducing the bug means an observed failing test before the
            # first proposed mutation, not any later failed command.
            before_edit, reproduced, passed_after_change = True, False, False
            for message in state.messages:
                calls = message.get("tool_calls")
                if message.get("role") == "assistant" and isinstance(calls, list) and any(
                        isinstance(call, dict) and isinstance(call.get("function"), dict)
                        and call["function"].get("name") in {"edit_file", "write_file"}
                        for call in calls):
                    before_edit = False
                    passed_after_change = False
                if message.get("role") != "tool" or message.get("tool_name") != "run_tests":
                    continue
                try:
                    observed = json.loads(message.get("content", ""))
                except (TypeError, ValueError):
                    continue
                if not isinstance(observed, dict):
                    continue
                if before_edit and observed.get("status") == "failed" and observed.get("exit_code") == 1:
                    reproduced = True
                if (not before_edit and observed.get("status") == "passed"
                        and observed.get("success") is True and observed.get("exit_code") == 0):
                    passed_after_change = True
            checks["failure_reproduced"] = reproduced
            checks["tests_passed_after_change"] = passed_after_change
    return {"passed": all(checks.values()), "checks": checks,
            "verifier": {"exit_code": result.exit_code, "stdout": result.stdout,
                         "stderr": result.stderr, "timed_out": result.timed_out,
                         "duration_seconds": result.duration_seconds}}


def verify_changes(task: BenchmarkTask, changed_files: dict[str, str]) -> dict[str, Any]:
    """Grade proposed source in a fresh fixture, outside its mutable tests."""
    with TemporaryDirectory(prefix="patchrook-regression-grade-") as directory:
        root = Path(directory)
        create_fixture(root, task.name)
        for name, content in changed_files.items():
            if name not in task.files:
                return {"passed": False, "error": "Unexpected file in patch"}
            (root / name).write_text(content, encoding="utf-8")
        return verify_fixture(root, task)
