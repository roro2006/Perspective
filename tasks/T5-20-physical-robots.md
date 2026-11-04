---
id: T5-20
title: Physically simulated robots carry out what the live grounder decides (both videos)
status: todo
agent: claude
budget: 100000
depends: [T5-19]
owns: [scripts/poc/export_policies.py, scripts/poc/phys_policy.py, scripts/poc/phys_core.py, scripts/poc/phys_run.py, scripts/poc/phys_robots.py, scripts/poc/phys_g1.py, scripts/poc/phys_go2.py, scripts/poc/phys_scenes.py, scripts/poc/phys_reel.py, scripts/poc/phys_eval.py, scripts/poc/make_physical.py, tests/test_t5_physical.py, third_party/README.md, .gitignore, tasks/T5-20-physical-robots.md, tasks/notes/T5-20.md]
reads: [scripts/poc/kit.py, scripts/poc/robots.py, scripts/poc/scene_base.py, scripts/poc/scenes_a.py, scripts/poc/scenes_b.py, scripts/poc/make_video.py, scripts/poc/reel_kit.py, scripts/poc/reel_scenes_a.py, scripts/poc/reel_scenes_b.py, scripts/poc/make_reel.py, tasks/notes/T5-18.md, tasks/notes/T5-19.md, docs/bringup-7900xt.md]
accept: uv run pytest -q tests/test_t5_physical.py
---

## Goal
In the live-grounder scenes of the PoC video (warehouse, dog, drone, humanoid, ROV, space) and
the reel (visibility, your frame), the robot physically goes to whichever object the real
grounder chose. Robots run under MuJoCo physics: no teleporting, nothing set kinematically
except held props (which may attach on contact or proximity) and the scripted people.

- Home humanoid: Unitree G1 12-dof with unitree_rl_gym's motion.pt, reproducing deploy_mujoco
  (47-d obs, its scales, kps/kds, default angles, 50 Hz policy, 0.002 s step). Arms are added
  as PD-held joints at default angles, with a scripted reach at the end. A pure-pursuit layer
  turns a target point and heading into velocity commands and stops.
- Rescue dog: Menagerie Go2 with the cagataydev Isaac Lab rough policy (ONNX, CC-BY-4.0)
  ported to MuJoCo; if it can't walk stably within about 2-3 hours, a hand-written trot
  (phase oscillator, foot trajectories, IK, PD) on the same Go2, and the notes say so.
- Warehouse and visibility: a wheeled diff-drive base with wheel actuators under a velocity
  controller. Drone, ROV, space: a free body under force/torque PD toward the waypoint
  (gravity plus thrust for the drone, drag for the ROV, nothing in space).
- The wrong robot-frame beats stay: the naive pick is executed the same physical way.
- New renders, not overwriting out/poc or out/poc_reel: out/physical/curio_poc_physical.mp4
  and out/physical/curio_reel_physical.mp4 (1080p) plus 720p copies under 30 MB. Scripted reel
  scenes keep their scripted targets; where their robot is a G1 or Go2 it walks physically.
- Downloads only from the approved list, under third_party/ (gitignored except README.md).

## GATE (written before any evaluation; not to be changed afterwards)
Scenes: warehouse, dog, drone, humanoid, rov, space, visibility, yourframe.

1. The grounder is called once per scene (speaker-frame and robot-frame queries, as the
   existing scenes do) and its answers are reused for all 10 physical runs of that scene.
2. Each scene runs 10 times, seeds 0-9. Seed k perturbs the robot's start: x and y uniform in
   +-0.25 m, yaw uniform in +-0.25 rad (flyers: also z in +-0.25 m), and for the G1/Go2 the
   initial joint angles by uniform +-0.05 rad.
3. A run succeeds when, at the end of every grounder-driven leg, the robot's base reference
   point is within 0.35 m of the goal point the scene defines for the grounder-chosen object
   (horizontal distance for ground robots, 3-D for drone, ROV and space flyer). The goal points
   are the scenes' existing stand-off points (GOALS / APPROACH / goal()), fixed in code before
   grounding. The warehouse has two such legs (geo, then geo2); the naive leg is not scored.
   The home humanoid also needs its right hand within 0.10 m of the reach point at the end of
   the reach. Each leg has a 40 s simulated-time cap; running out counts as a failure.
4. Falls: the G1 pelvis must stay above 0.50 m and the Go2 base above 0.18 m for the whole
   run (every physics step). Any fall fails that run.
5. A scene passes with at least 9 of 10 successful runs and zero falls. Report every scene's
   count, falls and pass/fail; failing scenes are reported, not hidden.

Standalone checks (reported, not part of the scene gate): the G1 walker and the Go2 walker
each reach 10 random targets (seed 0, 1-4 m away, random heading) on flat ground, ending
within 0.35 m without falling.
