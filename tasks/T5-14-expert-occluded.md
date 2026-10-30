---
id: T5-14
title: Scripted pick expert that also picks the block hidden behind the box
status: done
agent: claude
depends: [T5-04]
owns: [curio_robot/sim/expert.py, tests/test_t5_physics.py, pyproject.toml]
reads: [tasks/T5-04-physics-expert.md, tasks/notes/T5-04.md, curio_robot/sim/physics.py, curio_robot/collab/tasks.py, curio_robot/sim/world.py]
accept: uv run pytest tests/test_t5_physics.py
---

## Goal
The T5-04 expert picks 99.4% of blocks in open scenes but only 45% of the
hidden single block when `fetch_scene(rng, occlude=True)` puts the 9 cm box
4.5 cm in front of it (between it and the robot camera): the gripper or arm
clips the box. Hidden blocks are exactly the case Curio is for, so the
expert must be able to demonstrate them.

## Behaviour
- Keep `pick(world, i, on_tick=None) -> PickResult` unchanged (demos.py and
  the eval scripts use it), and the open-scene behaviour where the plain
  grasp is clear.
- Plan against the box: collision queries on a copy of the model, with a
  clearance margin, for the move to the pre-grasp and the path down; when no
  plain grasp is clear, try grasps that suit a box (higher pre-grasp, the
  gripper's thin side towards the box, narrower jaw, ways round).
- Leave PhysWorld and fetch_scene unchanged.

## Gate (fixed before the work)
- 100 seeded occluded scenes (`fetch_scene(default_rng(s), occlude=True)`,
  target index 4, the hidden single): the expert picks >= 85%.
- The existing open-scene gate test still passes (>= 90%) and every block of
  the 100 open scenes as target stays >= 99% (as T5-04 measured it).
- Report scenes where the box overlaps other blocks separately, with a count.

## Done when
The accept command passes, plus the full suite and ruff (see AGENTS.md).
tests/test_t5_physics.py gains an occluded-scene gate test (slow, but in the
suite). Note in tasks/notes/T5-14.md the rates before and after, what
changed and what still limits it.
