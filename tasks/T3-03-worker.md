---
id: T3-03
title: Simulated person's goals and wording for a task
status: doing
agent: qwen
depends: []
owns: [curio_robot/collab/worker.py]
reads: [docs/target3-collaboration.md, curio_robot/collab/worker.py, curio_robot/sim/users.py, curio_robot/perspective.py, curio_robot/types.py, tests/scene.py, tests/conftest.py, tests/test_t3_worker.py]
accept: uv run pytest tests/test_t3_worker.py --strict-impl
---

## Goal
Implement `describe` and `plan_task` in `curio_robot/collab/worker.py`
exactly as the docstrings say. Keep `Goal` unchanged.

## Hints
- Check `object_map.human_pose is None` first (ValueError), then find the
  object: `next(o for o in objects if o.object_id == object_id)` raising
  `KeyError(object_id)` when absent.
- Order matters for reproducibility: deictic and unique-colour goals make no
  rng draws; otherwise exactly two `rng.random()` calls, frame then marker.
- `perspective.pick(same_colour, "left", viewer).object_id == object_id`.
- `plan_task`: `ids = sorted(o.object_id for o in object_map.objects)`,
  `order = rng.permutation(len(ids))`, then for each of the first n_goals:
  `deictic = rng.random() < p_deictic` before calling `describe`.

## Done when
The accept command passes, plus the full suite and ruff (see AGENTS.md).
