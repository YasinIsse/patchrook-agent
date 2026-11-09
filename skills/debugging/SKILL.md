---
name: debugging
description: Locate a bug, make a focused fix, and verify it with relevant tests.
activation: Use when behavior is incorrect or a test fails.
---
# Debugging

1. Diagnose from the supplied source and exact test errors. Request a specific missing file or range with an enabled reader when necessary.
2. When reproduction was requested, use the harness's pre-edit baseline result before proposing a fix.
3. Make one exact edit that addresses the observed cause. Preserve unrelated behavior.
4. In the coding workflow, the harness runs pytest after each patch batch. Use returned failures and current source to correct the cause. In other workflows, run verification through an enabled test tool.
5. Once the requested work is complete, reply DONE in the coding workflow; the harness reports observed changes and verification. Otherwise explain the specific missing evidence.

Do not replace tests merely to make them pass. If the test command collects no tests, locate the real test before reporting success.

For an example only when needed, request `references/example.md` with `read_skill`.
