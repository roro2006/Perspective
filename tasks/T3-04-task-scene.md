---
id: T3-04
title: Multi-step fetch scene and its ground-truth map
status: doing
agent: qwen
depends: []
owns: [curio_robot/collab/tasks.py]
reads: [docs/target3-collaboration.md, curio_robot/collab/tasks.py, curio_robot/sim/scenarios.py, curio_robot/geometry.py, curio_robot/types.py, tests/test_t3_tasks.py]
accept: uv run pytest tests/test_t3_tasks.py --strict-impl
---

## Goal
Implement `fetch_scene`, `apply_scene` and `true_map` in
`curio_robot/collab/tasks.py` exactly as the docstrings say. Keep
`TaskScene`, `BLOCK_Z` and `PAIR_HALF` unchanged.

## Hints
- Reuse the private helpers in `sim/scenarios.py` (`_sample_human`,
  `_random_xy`, `_free_xy`, `_in_workspace`, `COLORS`, `MIN_GAP`); copy the
  pair placement from `perspective()` and the box from `occlusion()`.
- From `sim/world.py` you need only `BlockSpec`, `SimWorld.set_blocks`,
  `set_occluder`, `set_human`, `ROBOT_CAM_POS` and `ROBOT_CAM_LOOK`; don't
  read the rest of that file.
- Store block xy as `tuple(float(v) for v in xy)`.
- `true_map`: count colours with a dict as you go; `WorldObject(id, color,
  np.array([x, y, BLOCK_Z]), set())`; poses via `geometry.look_at`.

## Done when
The accept command passes, plus the full suite and ruff (see AGENTS.md).
