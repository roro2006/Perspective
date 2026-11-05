---
id: T5-29
title: Fix the grasp with residual RL on top of SmolVLA run 5
status: todo
agent: claude
budget: 120000
depends: [T5-17]
owns: [curio_robot/sim/residual_env.py, scripts/residual_train.py, scripts/residual_eval.py, scripts/scripted_grasp.py, tests/test_t5_29_residual.py, tasks/T5-29-residual-grasp.md, tasks/notes/T5-29.md]
reads: [scripts/smolvla_eval.py, curio_robot/sim/physics.py, curio_robot/sim/expert.py, curio_robot/sim/demos.py, tasks/notes/T5-17.md, docs/bringup-7900xt.md]
accept: uv run pytest -q tests/test_t5_29_residual.py
---

## Why
Run 5 (T5-17) understood the target but missed the grasp: ring-on-target was 0.57.
The 30-scene diagnosis found 6 jaws on the block but too high, 4 off sideways,
1 mid-air close and 3 early closes. The policy reaches the right block, and only
the last few centimetres fail. That suits RL fine-tuning: the goal is given, the
search is small, and the reward can be read from the sim.

## Method
- **Base**: the SmolVLA run-5 checkpoint, frozen.
- **Residual**: a small MLP policy outputs a bounded correction added to each base
  action (at most 1 cm per step for the end effector, and the gripper too). Its
  inputs are the proprio state, the base action, and the ring's position in the
  wrist image. It never gets the true block pose.
- **Training**: PPO in parallel MuJoCo environments built from smolvla_eval's
  trials. Training seeds 600000+ are disjoint from every eval range.
- **Reward**, from sim state only:
  - +1 when the ringed block is lifted 3 cm at the end;
  - -1 when another block is lifted;
  - a small shaping term on the jaw-to-block distance in the final phase only;
  - a penalty on residual size.
- **Budget**: at most 10 h of training, one run (one rerun after a transient
  crash, before any eval score has been seen).
- **Scripted baseline** (`scripts/scripted_grasp.py`): base policy until the
  gripper is within 4 cm of the ring, then a scripted descend-centre-close using the
  ring's wrist-image position. Reported, not gated, to show whether RL beat the
  simple fix.

## GATE (written before any training or evaluation; not to be changed afterwards)
100 fresh scenes, seeds 52000-52099, using smolvla_eval's conditions and outcomes:
1. marker (ring on target): residual target rate >= 0.75.
2. swapped (ring on the partner): residual partner rate >= 0.75. The ring must
   still steer the arm.
3. words only (no ring): residual rate of lifting any block <= 0.10. No blind
   grabbing.
4. Residual marker target rate >= base run-5 marker target rate on the same seeds
   + 0.10.
Report base run 5 and the scripted baseline on the same seeds, and end to end with
the v4 grounder giving the ring. Every rate gets a 95% Wilson CI, and each
difference a paired bootstrap CI. Results go to out/t5_29/gate.json and gate.md.

## GPU
Use the lock (out/gpu.lock), and release it on any exit. T5-28 also needs the GPU,
so the two run one after the other, never together.

## Done when
The accept command passes (CPU tests for the env, the reward and the residual
bound), along with the full suite, ruff, and tasks.py check. The gate has run once,
and the notes are in tasks/notes/T5-29.md.
