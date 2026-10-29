---
id: T4-05
title: The gaze rule as a critic (baseline)
status: todo
agent: qwen
depends: []
owns: [curio_robot/critic/baseline.py]
reads: [docs/target4-critic.md, curio_robot/critic/baseline.py, curio_robot/collab/gaze.py, curio_robot/critic/episodes.py, tests/critic_util.py, tests/test_t4_baseline.py]
accept: uv run pytest tests/test_t4_baseline.py --strict-impl
---

## Goal
Implement `rule_scores` in `curio_robot/critic/baseline.py` exactly as the
docstring says, reusing `GazeWatch` from `collab/gaze.py`.

## Hints
- Angles per frame to each candidate as in `gaze_target`, but over indices,
  not an ObjectMap.
- Once the watch fires, keep that block for the rest of the episode.

## Done when
The accept command passes, plus the full suite and ruff (see AGENTS.md).
