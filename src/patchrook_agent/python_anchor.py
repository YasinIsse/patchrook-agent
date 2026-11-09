"""Conservative full-function formatting recovery, never semantic anchor matching."""
import ast
from dataclasses import dataclass
import io
import re
import textwrap
import tokenize


_FORMAT = {tokenize.INDENT, tokenize.DEDENT, tokenize.NL, tokenize.NEWLINE,
           tokenize.ENDMARKER, tokenize.ENCODING}


@dataclass(frozen=True)
class FunctionAnchor:
    old_text: str
    new_text: str
    first_line: int
    last_line: int
    function: str


def _tokens(text):
    values = []
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        kind = tokenize.tok_name.get(token.type, '')
        if (token.type == tokenize.COMMENT or 'STRING' in kind and token.type != tokenize.STRING
                or token.type == tokenize.STRING and (token.start[0] != token.end[0]
                    or re.match(r'(?i)^[rubf]*f[rubf]*[\'"]', token.string))):
            raise ValueError('Comments, multiline strings and interpolated strings are not recoverable')
        if token.type not in _FORMAT:
            values.append((token.type, token.string))
    return values


def _single(text):
    tree = ast.parse(text, type_comments=True)
    if (len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef)
            or tree.body[0].decorator_list or tree.body[0].col_offset):
        raise ValueError('A complete single plain function is required')
    return tree.body[0]


def _signature(node):
    dump = lambda value: ast.dump(value, include_attributes=False) if value is not None else None
    return (node.name, dump(node.args), dump(node.returns), node.type_comment,
            tuple(dump(value) for value in getattr(node, 'type_params', [])))


def _covered(ranges, first, last):
    cursor = first
    for start, end in sorted(ranges):
        if start <= cursor < end:
            cursor = max(cursor, end)
    return cursor >= last


def recover_function_anchor(original, old_text, new_text, presented_ranges):
    """Return only an exact-token/AST-equivalent, fully presented top-level scope.

    The caller must first validate file identity and its current presentation
    hash. Returned strings still go through ordinary overlap/compile/write guards.
    Failures mean keep the original no_match, never expand source authority.
    """
    try:
        old, new = textwrap.dedent(old_text), textwrap.dedent(new_text)
        before, after = _single(old), _single(new)
        old_tokens = _tokens(old)
        if _tokens(new_text) != _tokens(new) or _signature(before) != _signature(after):
            return None
        tree = ast.parse(original, type_comments=True)
        key = ast.dump(before, include_attributes=False)
        # A matching inner definition makes an indented anchor ambiguous too.
        matches = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                   and ast.dump(node, include_attributes=False) == key]
        if len(matches) != 1 or matches[0] not in tree.body or matches[0].decorator_list:
            return None
        current = matches[0]
        if sum(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == current.name
               for node in tree.body) != 1:
            return None
        lines = original.splitlines(keepends=True)
        physical = ''.join(lines[current.lineno - 1:current.end_lineno])
        _tokens(physical)  # Also reject a comment after the final AST column.
        canonical = ast.get_source_segment(original, current)
        if canonical is None or _tokens(canonical) != old_tokens or original.count(canonical) != 1:
            return None
        start = sum(map(len, lines[:current.lineno - 1]))
        # get_source_segment handles AST's UTF8 byte columns; len is now chars,
        # matching EvidenceTools' presented character ranges, even with Unicode.
        end = start + len(canonical)
        if original[start:end] != canonical or not _covered(presented_ranges, start, end):
            return None
        return FunctionAnchor(canonical, new, current.lineno, current.end_lineno, current.name)
    except (SyntaxError, ValueError, TypeError, tokenize.TokenError, RecursionError):
        return None
