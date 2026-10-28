---
id: T1-02
title: Convention belief and learning from outcomes
status: todo
agent: qwen
depends: []
owns: [curio_robot/conventions/belief.py]
reads: [docs/target1-conventions.md, curio_robot/conventions/readings.py, tests/test_t1_belief.py]
accept: uv run pytest tests/test_t1_belief.py --strict-impl
---

## Goal
Implement every method of `ConventionBelief` and `learn_from_outcome` in
`curio_robot/conventions/belief.py`, exactly as the docstrings specify. Keep
the field names, defaults (1.6, 0.4) and the weight constants.

## Hints
- `readings.py` is only a stub for you. Use the `Reading` dataclass; don't
  call its functions, the tests build readings by hand.
- `from_dict` must ignore unknown keys. `dataclasses.fields(cls)` gives the
  known ones; missing keys fall back to the defaults.
- In `learn_from_outcome`, check "nothing to learn" before computing weights,
  and return the weight actually applied (0.0 if none).
- `update` is the only place `own_frame_a` / `own_frame_b` change.

## Done when
The accept command passes, plus the full suite and ruff (see AGENTS.md).
