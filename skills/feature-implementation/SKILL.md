---
name: feature-implementation
description: Implement a small feature using existing repository conventions.
activation: Use when adding a bounded new behavior or API option.
---
# Small feature implementation

1. Identify the existing entry point, nearby examples, and tests from the supplied source. Request only specific missing files or ranges.
2. Keep the requested behavior and constraints explicit. Extend the current design.
3. Make targeted changes and add meaningful coverage for the new behavior.
4. In the coding workflow, the harness verifies each patch batch with pytest. Investigate returned failures before finishing. In other workflows, use an enabled test tool for verification.
5. Reply DONE when the coding task is complete; the harness reports actual changes and tests. State any missing evidence that prevents completion.

Use a new file only when a focused module is clearer than extending an existing one.
