---
id: T1-01
title: Readings and frame markers
status: todo
agent: qwen
depends: []
owns: [curio_robot/conventions/readings.py]
reads: [curio_robot/resolver.py, curio_robot/perspective.py, curio_robot/types.py, tests/scene.py, tests/conftest.py, tests/test_t1_readings.py]
accept: uv run pytest tests/test_t1_readings.py --strict-impl
---

## Goal
Implement `frame_marker` and `enumerate_readings` in
`curio_robot/conventions/readings.py`, exactly as their docstrings specify.
Keep the `Reading` dataclass and `Frame` type unchanged.

## Hints
- Tokenize like `resolver.parse_request` does: `re.findall(r"[a-z]+", text.lower())`.
- "to me" / "to you" are two consecutive words; a bare "me" or "you" is not a
  marker. The last marker in the sentence wins.
- Use `resolver.parse_request` for colour and relation and `perspective.pick`
  to choose the object from a viewer. Don't reimplement them.
- Order matters in the result: the speaker reading first, then the robot one.
  Frame-less readings follow the order of `object_map.objects`.

## Done when
The accept command passes, plus the full suite and ruff (see AGENTS.md).
