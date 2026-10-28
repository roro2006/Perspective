---
id: T2-07
title: Sim grounding dataset generator
status: done
agent: any
depends: [T2-03, T2-04]
owns: [curio_robot/sim/grounding_data.py]
reads: [docs/target2-grounding.md, curio_robot/sim/scenarios.py, curio_robot/sim/world.py, curio_robot/grounding/marks.py, curio_robot/grounding/prompt.py, curio_robot/perspective.py, tests/test_t2_dataset.py]
accept: uv run pytest tests/test_t2_dataset.py --strict-impl
---

## Goal
Implement `make_example`, `write_dataset` and `load_dataset` in
`curio_robot/sim/grounding_data.py` exactly as the docstrings specify, plus
the `python -m curio_robot.sim.grounding_data <out_dir> --n --seed` entry
point from the module docstring.

## Hints
- Scenes come from `scenarios.perspective` / `scenarios.cluttered`; both
  return a `Scenario` with `human_eye`. For "deictic", re-aim the person
  with `world.set_human(sc.human_eye, target_position)`.
- Render after the scene is final: `world.render_camera(HUMAN_CAM)` and
  `world.render_camera(ROBOT_CAM)` give frames with the poses for the map.
- Map ids: count colours in `world.block_positions()` order ("red_1",
  "red_2", ...). `WorldObject(id, colour, position, set())`.
- One `SimWorld(320, 240)` for the whole dataset; creating one is slow.
- `write_dataset` stores the image as a relative path in the messages
  (`copy.deepcopy` the request messages first) so the JSONL stays small.

## Done when
The accept command passes, `uv run python -m curio_robot.sim.grounding_data
out/grounding --n 30` writes 30 examples, plus the full suite and ruff.
