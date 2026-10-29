---
id: T4-02
title: Reaction episodes, generator and JSONL files
status: done
agent: qwen
depends: [T4-01]
owns: [curio_robot/critic/episodes.py]
reads: [docs/target4-critic.md, curio_robot/critic/episodes.py, curio_robot/critic/reactions.py, curio_robot/collab/tasks.py, tests/test_t4_episodes.py]
accept: uv run pytest tests/test_t4_episodes.py --strict-impl
---

## Goal
Implement every function in `curio_robot/critic/episodes.py` as its
docstring says. Keep `Episode`, `HOME`, `DURATION_S` and the imports.

## Hints
- `frame_times`: `np.arange(round(d * r) + 1) / r`.
- `arm_path`: `s = np.minimum(speed * np.maximum(t, 0), dist)`; then
  `start + np.outer(s / dist, goal - start)`; handle `dist == 0` first.
- `make_episode`: obj_pos is `np.array([o.position for o in truth.objects])`.
  Follow the rng draw order exactly so runs match across machines.
- JSON: `t_no` may be None; `float("inf")` never appears in an episode.
  Add `import json` yourself.

## Done when
The accept command passes, plus the full suite and ruff (see AGENTS.md).
