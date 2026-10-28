---
id: T1-04
title: Act, hedge or ask
status: todo
agent: qwen
depends: []
owns: [curio_robot/conventions/decide.py]
reads: [docs/target1-conventions.md, curio_robot/types.py, curio_robot/resolver.py, tests/scene.py, tests/conftest.py, tests/test_t1_decide.py]
accept: uv run pytest tests/test_t1_decide.py --strict-impl
---

## Goal
Implement `decide` in `curio_robot/conventions/decide.py` as its docstring
specifies. Keep `Thresholds` (defaults 0.7 / 0.5) and `Decision` unchanged.

## Hints
- Filter the posterior to ids present in `object_map.objects`, then
  renormalize. Nothing left (or total 0) means kind "none".
- Pick the top object with `min(post, key=lambda k: (-post[k], k))` so ties
  go to the smallest id.
- "act" returns a copy of the object's position; "hedge" returns the
  probability-weighted mean position as a float numpy array.
- The ask question uses `resolver.parse_request(text).relation`.

## Done when
The accept command passes, plus the full suite and ruff (see AGENTS.md).
