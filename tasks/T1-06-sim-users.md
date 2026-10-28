---
id: T1-06
title: Simulated users with a frame habit
status: done
agent: qwen
depends: []
owns: [curio_robot/sim/users.py]
reads: [curio_robot/perspective.py, curio_robot/types.py, curio_robot/conventions/readings.py, tests/scene.py, tests/conftest.py, tests/test_t1_users.py]
accept: uv run pytest tests/test_t1_users.py --strict-impl
---

## Goal
Implement `SimUser.utter`, `intended`, `answer` and `react` in
`curio_robot/sim/users.py` as the docstrings specify. Keep `Utterance`,
the `SimUser` fields and `POPULATION` unchanged.

## Hints
- In `utter`, draw `rng.random()` for the frame first, then once more for
  the marker, in that order (a test checks this).
- `intended` uses `perspective.pick(candidates, utt.side, viewer)`, where
  side is "left" or "right" and so is a valid relation.
- `readings.py` is only needed for the `Frame` type; don't call its functions.

## Done when
The accept command passes, plus the full suite and ruff (see AGENTS.md).
