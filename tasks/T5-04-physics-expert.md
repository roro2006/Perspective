---
id: T5-04
title: Physics world with actuated SO-101, free blocks, and a scripted pick expert
status: todo
agent: claude
depends: []
owns: [curio_robot/sim/physics.py, curio_robot/sim/expert.py, tests/test_t5_physics.py]
reads: [docs/target5-perspective-policy.md, curio_robot/sim/world.py, curio_robot/assets/so101/so101.xml, curio_robot/collab/tasks.py]
accept: uv run pytest tests/test_t5_physics.py
---

## Goal
A MuJoCo world that steps physics so the SO-101 can actually grasp: the arm
driven through its position actuators (ctrl), blocks on free joints with mass
and friction, a table, and the same robot and person cameras as SimWorld. A
scripted expert picks the block at a given index.

## Behaviour (you design the API; keep it small)
- physics.py: PhysWorld(width, height) with reset(scene: collab.tasks.TaskScene),
  step(ctrl, n), arm_qpos(), block_position(i), render_camera(name) and
  lifted(i, height=0.03).
- expert.py: pick(world, i) -> PickResult(success, steps, ctrl_log). Waypoints:
  above the block, descend, close the jaw, lift; IK on a copy of the data, then
  ramp ctrl towards the joint targets. success = block i lifted >= 3 cm and
  still held 0.5 s later.
- tests/test_t5_physics.py (yours): blocks settle on the table; a commanded
  joint reaches its target; the expert picks in >= 90% of 100 seeded
  fetch_scene scenes; a pick of an out-of-reach block reports success False.
- Leave SimWorld unchanged; the live runtime keeps using it.

## Done when
The accept command passes, plus the full suite and ruff (see AGENTS.md). Note
in tasks/notes/T5-04.md the success rate, the physics settings and anything
that needed tuning.
