---
id: T2-04
title: Grounding request builder
status: todo
agent: qwen
depends: []
owns: [curio_robot/grounding/prompt.py]
reads: [docs/target2-grounding.md, curio_robot/grounding/marks.py, curio_robot/perspective.py, curio_robot/resolver.py, curio_robot/types.py, tests/scene.py, tests/conftest.py, tests/test_t2_prompt.py]
accept: uv run pytest tests/test_t2_prompt.py --strict-impl
---

## Goal
Implement `build_request` in `curio_robot/grounding/prompt.py` exactly as
its docstring specifies. Keep `GroundingRequest` and `PROMPT_VERSION`.

## Hints
- `marks.py` is a stub for you; only use the `Mark` dataclass.
- PNG encode: `Image.fromarray(image).save(buf, format="PNG")` into an
  `io.BytesIO`, then `base64.b64encode(...).decode()`.
- Side words: `perspective.lateral(obj, pose) < 0` is "left", else "right".
  Speaker side first, then robot side; drop the speaker part when
  `object_map.human_pose is None`.
- The habit line is `resolver.frame_habit(p_own_frame)`; don't rephrase it.
- The test compares the prompt line by line, so match the wording exactly.

## Done when
The accept command passes, plus the full suite and ruff (see AGENTS.md).
