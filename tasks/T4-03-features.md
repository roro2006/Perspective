---
id: T4-03
title: Critic features
status: todo
agent: qwen
depends: []
owns: [curio_robot/critic/features.py]
reads: [docs/target4-critic.md, curio_robot/critic/features.py, curio_robot/critic/episodes.py, tests/critic_util.py, tests/test_t4_features.py]
accept: uv run pytest tests/test_t4_features.py --strict-impl
---

## Goal
Implement `features(ep)` in `curio_robot/critic/features.py` exactly as the
docstring says. Keep `FEATURES`, `D` and the scale constants.

## Hints
- Angles: `cos = head_fwd @ unit(v).T` gives (F, N); clip to [-1, 1], then
  `np.degrees(np.arccos(...))`.
- Dwell: one pass over frames keeping a per-block run start (or None).
- Speech: find the first `ts` with `text.startswith("no")` once, then
  compare with each `t[f]`.
- Build the result as `np.zeros((F, N, D))` and fill columns by
  `FEATURES.index(name)`.

## Done when
The accept command passes, plus the full suite and ruff (see AGENTS.md).
