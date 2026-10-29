---
id: T5-02
title: Grounding conditions (geometry, paired, robot-only)
status: todo
agent: qwen
depends: [T5-01]
owns: [curio_robot/bench/conditions.py]
reads: [docs/target5-perspective-policy.md, curio_robot/bench/conditions.py, curio_robot/bench/items.py, curio_robot/grounding/prompt.py, curio_robot/grounding/marks.py, curio_robot/resolver.py, tests/test_t5_bench.py]
accept: uv run pytest tests/test_t5_bench.py -k conditions --strict-impl
---

## Goal
Implement build(condition, item) in curio_robot/bench/conditions.py exactly as
the docstring says.

## Hints
- Reuse place_marks and render_marks (grounding/marks.py), build_request and
  _encode_png (grounding/prompt.py), frame_habit (resolver.py).
- Mark is a frozen dataclass: build new Mark(letter, object_id, pixel) objects
  to change letters.

## Done when
The accept command passes, plus the full suite and ruff (see AGENTS.md).
