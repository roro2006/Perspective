---
id: T3-05
title: Gaze target and dwell watcher
status: doing
agent: qwen
depends: []
owns: [curio_robot/collab/gaze.py]
reads: [docs/target3-collaboration.md, curio_robot/collab/gaze.py, curio_robot/types.py, tests/scene.py, tests/conftest.py, tests/test_t3_gaze.py]
accept: uv run pytest tests/test_t3_gaze.py --strict-impl
---

## Goal
Implement `gaze_target` and `GazeWatch.update` / `reset` in
`curio_robot/collab/gaze.py` exactly as the docstrings say. Keep the
constants and `GazeWatch.__init__` (its three private fields are yours to use).

## Hints
- Angle: `cos = dot(v, pose.forward) / norm(v)` with `v = obj.position -
  pose.position`; `np.degrees(np.arccos(np.clip(cos, -1, 1)))`.
- Ties: iterate objects sorted by `object_id` and keep a strictly smaller angle.
- `update`: three cases in the docstring's order; `reset` sets `_look = None`
  and `_fired = False`.

## Done when
The accept command passes, plus the full suite and ruff (see AGENTS.md).
