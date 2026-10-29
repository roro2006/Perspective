---
id: T4-01
title: Simulated reactions to a reach
status: todo
agent: qwen
depends: []
owns: [curio_robot/critic/reactions.py]
reads: [docs/target4-critic.md, curio_robot/critic/reactions.py, tests/critic_util.py, tests/test_t4_reactions.py]
accept: uv run pytest tests/test_t4_reactions.py --strict-impl
---

## Goal
Implement `sample_style` and `simulate_reaction` in
`curio_robot/critic/reactions.py` exactly as the docstrings say. Keep the
constants, `RANGES` and both dataclasses.

## Hints
- Build an (F, 3) array of look points in the docstring order: gripper,
  then arrived -> arm target, then glances (in start order, so later ones
  win), then the wanted block for a wrong reach. Draw the random numbers in
  exactly the order given: poisson, uniform, integers, random, normal.
- `unit(v) = v / np.linalg.norm(v)`; the head update is a loop over frames.
- `math.inf` for a silent person; `math.isfinite` to test it.

## Done when
The accept command passes, plus the full suite and ruff (see AGENTS.md).
