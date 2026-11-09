"""Recognize explicitly requested names from an already trusted skill index."""

import re


_NAME = r"[a-z][a-z0-9-]{0,63}"
_GENERIC = frozenset({
    "a", "an", "the", "any", "all", "this", "that", "these", "those", "their",
    "appropriate", "available", "best", "existing", "needed", "relevant", "requested",
    "right", "specific", "suitable", "trusted", "your", "my", "our", "new",
    "and", "or", "to", "do", "does", "is", "are", "was", "will", "can", "should",
    "name", "names", "index", "instructions", "document", "documents", "reference", "references",
    "use", "using", "apply", "applying", "load", "loading", "consult", "consulting",
})


def _declined(prefix: str) -> bool:
    prefix = re.split(r"[.!?;\n]", prefix)[-1]
    negative = re.search(r"\b(?:do\s+not|don't|never|without|avoid)\b", prefix, re.IGNORECASE)
    if negative is None:
        return False
    # A positive instruction later in the same sentence is a separate request:
    # "Do not change tests, but use the debugging skill."
    return not re.search(r"(?:,|\band\b|\bbut\b)\s*(?:please\s+)?(?:use|using|apply|load|consult)\b",
                         prefix[negative.end():], re.IGNORECASE)


def _named_activation(prefix: str) -> bool:
    """Named prose is a request only when an activation verb introduces it."""
    return bool(re.search(
        r"\b(?:use|using|apply|applying|load|loading|consult|consulting)\s+"
        r"(?:(?:the|a|an|trusted|named|called)\s+|[`'\"])*$", prefix, re.IGNORECASE))


def _dollar_activation(prefix: str) -> bool:
    """Skill tags need an explicit invocation position, not variable prose."""
    if not prefix.strip():
        return True
    clause = re.split(r"[.!?;\n]", prefix)[-1]
    activation = re.search(r"\b(?:use|using|apply|applying|load|loading|consult|consulting)\s+",
                           clause, re.IGNORECASE)
    if activation is None:
        tail = clause
        if not tail.lstrip().startswith("$"):
            return False
    else:
        tail = clause[activation.end():]
    # Inherit an invocation through a list such as "Apply $investigation
    # before $debugging", but not through unrelated source/variable prose.
    return bool(re.fullmatch(
        rf"(?:(?:the|a|an|trusted|skills?|before|after|afterwards|then|and|or|with)\s+"
        rf"|[,`'\"()]\s*|\${_NAME}\s*)*", tail))


def explicit_skill_names(task: str) -> list[str]:
    """Extract explicit names, including unknown ones, without loading anything.

    A request to use/load a named skill, ``$name``, or its ``name/SKILL.md`` path
    is explicit. Words
    such as "debugging" in an ordinary coding task are not activation signals.
    Unknown names remain visible so callers can report a missing requested
    skill rather than quietly proceed without it.
    """
    if not isinstance(task, str):
        raise ValueError("Skill request task must be a string")
    matches = []
    patterns = (
        rf"(?<![\w-])\$(?P<name>{_NAME})(?![\w-])",
        rf"(?<![\w-])(?:skills/)?(?P<name>{_NAME})/SKILL\.md(?![\w/-]|\.[\w/-])",
        rf"(?<![\w-])(?P<name>{_NAME})[`'\"]?\s+(?:trusted\s+)?skills?\b",
        rf"\bskills?\s+(?:(?:named|called)\s+)?[`'\"]?(?P<name>{_NAME})(?![\w-])",
    )
    for index, pattern in enumerate(patterns):
        # Uppercase shell variables such as $HOME are not skill tags.
        for match in re.finditer(pattern, task, 0 if index == 0 else re.IGNORECASE):
            name = match.group("name").lower()
            prefix = task[:match.start()]
            if (name in _GENERIC or len(name) > 64 or _declined(prefix)
                    or index == 0 and not _dollar_activation(prefix)
                    or index >= 2 and not _named_activation(prefix)):
                continue
            matches.append((match.start(), name))
    plural = (rf"\b(?:use|using|apply|load|consult)\s+(?:the\s+)?"
              rf"(?P<names>{_NAME}(?:\s+(?:and|or)\s+{_NAME}|,\s*{_NAME})+)\s+skills?\b")
    for match in re.finditer(plural, task, re.IGNORECASE):
        if _declined(task[:match.start()]):
            continue
        for name_match in re.finditer(_NAME, match.group("names"), re.IGNORECASE):
            name = name_match.group().lower()
            if name not in _GENERIC and len(name) <= 64:
                matches.append((match.start("names") + name_match.start(), name))
    return list(dict.fromkeys(name for _, name in sorted(matches)))


def requested_skills(task: str, index: list[dict]) -> list[str]:
    """Filter explicit requests against the frozen trusted catalog only."""
    known = {item.get("name") for item in index if isinstance(item, dict)
             and isinstance(item.get("name"), str)}
    return [name for name in explicit_skill_names(task) if name in known]


def skill_access_requested(task: str) -> bool:
    """Whether the user requested skill use/discovery, rather than skill code."""
    if explicit_skill_names(task):
        return True
    modifiers = r"(?:(?:a|an|the|any|all|available|trusted|relevant|appropriate|suitable|known|my|our|your)\s+)*"
    patterns = (
        rf"\b(?:use|using|load|apply|consult|read|discover|list|show)\s+{modifiers}skills?\b",
        rf"\b(?:which|what)\s+{modifiers}skills?\b",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, task, re.IGNORECASE):
            if _declined(task[:match.start()]):
                continue
            # A task to change the loader is ordinary coding, even when its
            # wording starts with "read skill loading code".
            if re.match(r"\s+(?:loading|loader|handling|support|implementation|code|module|function|class)\b",
                        task[match.end():], re.IGNORECASE):
                continue
            return True
    return False
