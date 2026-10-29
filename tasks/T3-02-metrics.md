---
id: T3-02
title: Fluency metrics for one task
status: todo
agent: qwen
depends: []
owns: [curio_robot/collab/metrics.py]
reads: [docs/target3-collaboration.md, curio_robot/collab/session.py, curio_robot/collab/metrics.py, tests/test_t3_metrics.py]
accept: uv run pytest tests/test_t3_metrics.py --strict-impl
---

## Goal
Implement `fluency` and `summarize` in `curio_robot/collab/metrics.py`
exactly as the docstrings say. Only the `Session`/`Step` dataclasses are
needed (not the T3-01 functions). Keep `Fluency` unchanged.

## Hints
- Write two small private helpers: `_union(intervals)` (sort by start, merge
  overlapping or touching ones, return a list of (start, end)) and
  `_overlap(a, b)` (total length of the intersection of two unions).
- `statistics.median` for the two delays.
- `summarize`: group by `session.condition`; for each field in
  `dataclasses.fields(Fluency)` take the mean of the non-None values. Return
  plain floats (`float(...)`), with `"n"` last.
- The worked example in the test explains every expected number.

## Done when
The accept command passes, plus the full suite and ruff (see AGENTS.md).
