---
id: T5-23
title: Teach by doing through the head camera (ego demo -> person-frame plan -> robot repeats)
status: todo
agent: claude
budget: 100000
depends: [T5-04, T5-20]
owns: [curio_robot/ego_plan.py, curio_robot/sim/ego_world.py, curio_robot/sim/ego_tasks.py, curio_robot/sim/ego_exec.py, curio_robot/sim/place.py, scripts/ego_eval.py, scripts/poc/make_ego_demo.py, tests/test_t5_ego.py, tasks/T5-23-ego-demo.md, tasks/notes/T5-23.md]
reads: [curio_robot/sim/physics.py, curio_robot/sim/expert.py, curio_robot/sim/demos.py, curio_robot/sim/world.py, curio_robot/sim/scenarios.py, curio_robot/fusion.py, curio_robot/perception.py, curio_robot/geometry.py, scripts/poc/phys_core.py, scripts/poc/encode_cv2.py]
accept: uv run pytest -q tests/test_t5_ego.py
---

## Goal
A person sits across the table from the SO-101 and does a tabletop task once with a scripted
hand. The robot sees the demonstration only through the person's head camera (RGB-D frames at
10 fps plus the head pose, which stands in for the glasses' own tracking), extracts a symbolic
plan whose placements are in the person's frame and relative to objects, converts it to its own
frame for a new, shuffled layout of the same objects and carries it out physically on PhysWorld
with the scripted pick expert and a new place().

Task families: (a) sort two colours into "my left" / "my right" zones; (b) stack 2-3 blocks in
a given colour order; (c) put X to the left (or right) of Y, the person's left/right; (d) two
steps mixing the above. Positions, colours, duplicates and the person's seat are randomised.

## GATE (written before any evaluation; not to be changed afterwards)
Evaluation: `scripts/ego_eval.py`, families a, b, c, d, 50 episodes each (seeds 0-49 per
family, 200 in all). Each episode: one recorded demonstration, then one new layout of the same
blocks (same colours and count, start positions resampled, same seat), on which the robot runs
the plan it extracted (never the ground-truth plan).

Definitions, fixed in code before any run:
- Person frame: left axis L = horizontal part of minus the head camera's right axis, averaged
  over the demo; C = table point (0.24, 0). Zone left: dot(p - C, L) >= 0.05 m; zone right:
  <= -0.05 m.
- left_of(x, y): horizontal offset d = x - y with 0.03 <= |d| <= 0.10 m, angle(d, L) <= 45 deg,
  both on the table (block centre z < 0.02 m); right_of uses -L.
- on(x, y): horizontal |x - y| <= 0.015 m and z_x - z_y in [0.018, 0.032] m.
- Plan exact match: the extracted plan has the same number of steps as the ground-truth plan,
  and every step has the same picked object id, relation and reference (object id or zone).
  Object ids are colour_k, k counting same-coloured blocks from the person's left in the demo's
  start layout.
- Task success: after the robot finishes (arm back home, 0.5 s settle), the true block state
  satisfies the task's goal, checked as: some one-to-one assignment of the task's variables to
  blocks of the right colours has every goal relation true and every moved variable's block
  displaced >= 0.02 m from its start; and no block left the table.
- Baseline: the same perception and extracted plan, with relation words read in the robot's
  own frame (robot's left = +y, zones about C along +y) instead of converted from the person's.

Pass if all three hold:
1. Plan-extraction exact match >= 0.90 over all 200 demos.
2. End-to-end task success on the new layout >= 0.80 over all 200 episodes.
3. On family (c): baseline success <= Curio success - 0.40.

Report per family: exact match, Curio success and baseline success with 95% Wilson CIs, and
failures by cause. Failing numbers are reported, not hidden.

Demo: out/t5_23/ego_demo.mp4 (~20 s, 1080p) and a 720p copy under 30 MB: head-camera view
while the person demonstrates, the extracted plan as text steps, then the robot repeating it on
a shuffled layout.
