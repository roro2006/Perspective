---
id: T1-09
title: Learning-curve harness (fast mode) and `curio-sim learn`
status: done
agent: any
depends: [T1-01, T1-02, T1-03, T1-04, T1-06]
owns: [curio_robot/sim/learn.py, curio_robot/sim/cli.py]
reads: [docs/target1-conventions.md, curio_robot/sim/learn.py, curio_robot/sim/users.py, curio_robot/sim/scenarios.py, curio_robot/sim/cli.py, curio_robot/conventions/belief.py, curio_robot/conventions/decide.py, curio_robot/conventions/noise.py, curio_robot/conventions/readings.py, tests/test_t1_learn.py]
accept: uv run pytest tests/test_t1_learn.py --strict-impl
---

## Goal
Implement `learning_curve` (fast mode, no rendering) as its docstring
specifies, and add a `learn` subcommand to `curio-sim`.

## Hints
- Scenes: reuse the pair placement from `scenarios.perspective`
  (`_sample_human`, `_random_xy`, `_in_workspace`) but build `WorldObject`s
  and `CameraPose`s directly; don't create a `SimWorld`, it's slow. The
  robot pose is `look_at(ROBOT_CAM_POS, ROBOT_CAM_LOOK)` from `sim/world.py`.
- One `np.random.default_rng(seed)` drives everything, so runs are reproducible.
- Ground truth (`SimUser.intended`) uses the true human pose; the robot sees
  the perturbed one when noise is on.
- For speed with noise on, call `flip_probability(..., samples=100)`.
- About 20% of scenes have both frames agree; that's expected (see the tests).

## CLI
`uv run curio-sim learn [--user own-frame|robot-frame|mixed|all] [--requests 15]
[--users 50] [--seed 0] [--noise] [--baseline] [--out out/learn.json]`
prints one row per user type: error and ask rate at requests 1, 3, 5 and the
last, plus `final_p_own_frame`, and writes the full JSON. `--noise` means
pos 0.01 m, rot 2°. `--baseline` adds the `learn=False` rows.

## Done when
The accept command passes, `uv run curio-sim learn --user all` prints the
table, plus the full suite and ruff.
