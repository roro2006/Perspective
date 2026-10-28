---
id: T1-03
title: Left/right flip probability under head-pose noise
status: todo
agent: qwen
depends: []
owns: [curio_robot/conventions/noise.py]
reads: [docs/target1-conventions.md, curio_robot/geometry.py, curio_robot/perspective.py, curio_robot/types.py, tests/scene.py, tests/test_t1_noise.py]
accept: uv run pytest tests/test_t1_noise.py --strict-impl
---

## Goal
Implement `flip_probability` in `curio_robot/conventions/noise.py` as its
docstring specifies: a seeded Monte Carlo over `geometry.perturb_pose`.

## Hints
- lateral(a) − lateral(b) simplifies to `dot(a − b, viewer.right)`: the
  viewer's position cancels. Only rotation noise can flip the sign.
- `CameraPose.right` is a property (first column of the rotation).
- Use one `np.random.default_rng(seed)` and call `perturb_pose` `samples`
  times with it, so results are deterministic per seed.
- Compare signs with `np.sign`. Return exactly 0.5 when the noise-free
  difference is exactly 0.
- A plain Python loop over 400 samples is fast enough; don't over-engineer.

## Done when
The accept command passes, plus the full suite and ruff (see AGENTS.md).
