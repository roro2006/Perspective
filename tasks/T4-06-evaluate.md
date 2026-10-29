---
id: T4-06
title: Critic metrics, threshold choice and the phase 1 gate
status: done
agent: qwen
depends: []
owns: [curio_robot/critic/evaluate.py]
reads: [docs/target4-critic.md, curio_robot/critic/evaluate.py, curio_robot/critic/episodes.py, tests/critic_util.py, tests/test_t4_evaluate.py]
accept: uv run pytest tests/test_t4_evaluate.py --strict-impl
---

## Goal
Implement every function in `curio_robot/critic/evaluate.py` as its
docstring says. Keep `GATE`, `THRESHOLDS` and `Scored`.

## Hints
- `alarm_frame`: `np.flatnonzero(p >= threshold)`.
- `lead_s` uses `float(np.median(...))`; return plain floats, not numpy
  scalars, so the dict serialises to JSON.
- `threshold_for_precision` returns `float(t)`.

## Done when
The accept command passes, plus the full suite and ruff (see AGENTS.md).
