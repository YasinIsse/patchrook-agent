---
name: investigation
description: Trace behavior across repository files without unnecessary edits.
activation: Use when locating a function, explaining a code path, or mapping a dependency.
---
# Repository investigation

1. Start from the source and repository paths supplied by the harness.
2. Follow the requested symbol or behavior through observed callers and imports. Request a specific missing file or range with an enabled reader when needed.
3. Read the smallest useful line ranges, including callers when needed.
4. Record exact paths and symbols that support the explanation.
5. Separate observed facts from hypotheses. Do not modify code unless requested.

When an answer needs several files, follow the call chain rather than reading every file.
