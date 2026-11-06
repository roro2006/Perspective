---
id: T5-34
title: Live dog demo, where the firefighter's words send the physical Go2 to the room they mean, in the same viewer as the arm demo
status: doing
agent: claude
budget: 120000
depends: [T5-20, T5-33]
owns: [curio_robot/demo/dog.py, curio_robot/demo/viewer.py, curio_robot/demo/voice.py, scripts/poc/phys_core.py, scripts/poc/export_policies.py, scripts/dog_demo_gate.py, tests/test_t5_34_dog_demo.py, docs/demo.md, third_party/README.md, tasks/T5-34-dog-live-demo.md, tasks/notes/T5-34.md]
reads: [curio_robot/demo/viewer.py, curio_robot/demo/voice.py, curio_robot/resolver.py, curio_robot/perspective.py, curio_robot/types.py, scripts/poc/phys_scenes.py, scripts/poc/phys_go2.py, scripts/poc/phys_run.py, scripts/poc/phys_eval.py, scripts/poc/scenes_a.py, tasks/notes/T5-20.md, tasks/notes/T5-32.md, tasks/notes/T5-33.md]
accept: uv run pytest -q tests/test_t5_34_dog_demo.py
---

## Why
The pitch has two live demos: the SO-101 arm (T5-33) and the rescue dog. The dog scene shows
the case the arm can't: the robot can't see the person, the two face each other down a
corridor, so "the room on my left" is the dog's right. The user chose (2026-11-06): re-download
the Go2 assets on this laptop, choose the room with geometry on a floor map (no VLM here), and
put both demos in one viewer behind `--robot arm|dog`.

## What it does
- `uv run python -m curio_robot.demo.viewer --robot dog`: the T5-20 rescue scene (PhysDog) in
  MuJoCo's viewer, physics stepped live, insets for the firefighter's head camera and the dog's
  camera, voice ("Curio, the survivor's in the room on my left") and the terminal.
- Decision: the rooms come from the building's floor map (two `WorldObject`s, seen_by "map").
  `resolver.resolve` picks one with the firefighter's head-camera pose (Curio on) or the dog's
  camera pose (Curio off). No detector runs: rooms aren't objects a camera detects. The overlay
  says "rooms from floor map, geometry, no VLM".
- Motion: T5-20's path follower (ours) through the scene's door waypoints, on the downloaded
  cagataydev walking policy (CC-BY-4.0, credited on screen). Labelled "our path follower +
  pretrained walking policy". T5-32's navigation policy is not used (FAIL on the port).
- Truth: from the simulator, the firefighter's (or for "your", the dog's) true body yaw; only
  for left/right requests, "unknown" otherwise.
- `turn` makes the firefighter face the other way, so the same words mean the other room.
- Each request starts the dog from the scene's start pose.

## GATE (written before building; not to be changed afterwards)
Run once by `scripts/dog_demo_gate.py` on this laptop, results to out/t5_34/gate.json and
gate.md, using the same code path the viewer runs (decision, route, live stepping).
- **Q (walker on this machine, no Curio):** both rooms x 10 jittered starts (seeds
  54000-54009, T5-20's jitter), the dog stops within 0.35 m (T5-20's GOAL_TOL) of the room
  goal on >= 19 of 20, with 0 falls. If Q fails, the dog demo is reported unqualified and is
  not shown live.
- **D (decision):** "Survivor's in the room on my left" / "... on my right", firefighter facing
  the corridor and turned round, 4 cases: Curio on matches the truth on 4 of 4; Curio off is
  reported as it comes out (expected wrong when the two face each other).
- **E (end to end):** Curio on, the 4 cases x 3 seeds: the dog ends within 0.5 m of the true
  room's goal and closer to it than to the other room's, on >= 11 of 12, 0 falls.
- **L (live):** in the viewer loop with insets on, simulated time / wall time >= 1.0 over a
  whole run (median of 3 runs).
A FAIL is acceptable and is reported as such.

## Rules
1. The target is whatever `resolver.resolve` returns on the map and the poses; nothing
   hardcoded.
2. The motion is never called Curio's or learned end to end: path follower (ours) + walking
   policy (cagataydev). Curio only chooses the room.
3. The arm demo is unchanged when `--robot` is left out.

## Done when
- The accept command passes (CPU; the physics tests skip when the Go2 assets are missing).
- The full suite, ruff, and tasks.py check pass.
- docs/demo.md has the dog's start command and a walkthrough; third_party/README.md says how
  this laptop's copy was fetched.
