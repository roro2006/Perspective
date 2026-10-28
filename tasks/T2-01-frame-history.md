---
id: T2-01
title: Frame history ring buffer
status: todo
agent: qwen
depends: []
owns: [curio_robot/grounding/history.py]
reads: [docs/target2-grounding.md, curio_robot/types.py, tests/scene.py, tests/test_t2_history.py]
accept: uv run pytest tests/test_t2_history.py --strict-impl
---

## Goal
Implement `FrameHistory` in `curio_robot/grounding/history.py` exactly as the
docstrings specify.

## Hints
- Keep two parallel lists (times, frames) sorted by time and use `bisect`
  to insert and to find the neighbours of a query time.
- One `threading.Lock` around every method; the perception thread pushes
  while the orchestrator reads.
- Drop old frames on every push: everything older than `newest_t - keep_s`.
- `at(t)`: compare the neighbour just before and just after `t`; on a tie
  the earlier one wins. Exactly `max_gap_s` away still counts.

## Done when
The accept command passes, plus the full suite and ruff (see AGENTS.md).
