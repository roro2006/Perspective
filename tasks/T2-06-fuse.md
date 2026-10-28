---
id: T2-06
title: Fuse VLM probabilities with the belief
status: todo
agent: qwen
depends: []
owns: [curio_robot/grounding/fuse.py]
reads: [docs/target2-grounding.md, tests/test_t2_fuse.py]
accept: uv run pytest tests/test_t2_fuse.py --strict-impl
---

## Goal
Implement `fuse` in `curio_robot/grounding/fuse.py` exactly as its docstring
specifies. Keep `VLM_WEIGHT` and `VLM_FLOOR`.

## Hints
- Check the special cases in the docstring's order before multiplying.
- Only the belief's keys survive; `vlm.get(k, 0.0)` for missing ones.
- Return new dicts; don't mutate the inputs.

## Done when
The accept command passes, plus the full suite and ruff (see AGENTS.md).
