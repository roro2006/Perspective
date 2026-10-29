---
id: T5-05
title: Recorded pick demonstrations for policy training
status: todo
agent: claude
depends: [T5-04]
owns: [curio_robot/sim/demos.py, tests/test_t5_demos.py]
reads: [docs/target5-perspective-policy.md, curio_robot/sim/physics.py, curio_robot/sim/expert.py, curio_robot/collab/export.py, tasks/notes/T5-04.md]
accept: uv run pytest tests/test_t5_demos.py
---

## Goal
Run the expert on seeded scenes and record picks as training episodes for a
target-conditioned policy, in a LeRobot-style folder (JSON and PNG, no new
dependencies), laid out like collab/export.py. Per step at 10 Hz: robot camera
image, person camera image, arm joint state (6), action (6 ctrl), the target
pixel in the robot image, and the words. Failed picks only with a flag. Test: a
3-episode recording round-trips and every step has every field.
