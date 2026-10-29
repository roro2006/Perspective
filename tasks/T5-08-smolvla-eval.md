---
id: T5-08
title: Closed-loop evaluation of the fine-tuned SmolVLA in the physics sim
status: done
agent: claude
depends: [T5-07]
owns: [scripts/smolvla_eval.py, docs/target5-perspective-policy.md]
reads: [docs/target5-perspective-policy.md, scripts/smolvla_train.py, curio_robot/sim/demos.py, curio_robot/sim/physics.py, curio_robot/sim/expert.py, tasks/notes/T5-07.md]
accept: uv run pytest -q
---

## Goal
Does the policy pick the block the marker points at, from the robot camera,
in closed loop? Run it in PhysWorld on scenes it never trained on.

## Behaviour
- scripts/smolvla_eval.py (lerobot env): load a checkpoint, roll out on fresh
  seeded fetch_scene scenes (seeds disjoint from training), same camera size,
  marker and control timing as the demonstrations, executing action chunks.
- Three conditions on the same scenes and targets:
  1. marker on the target, words name it (the intended use);
  2. words only, no marker;
  3. marker on the other block of the pair, words name the target (does the
     policy follow the grounding layer or the words?).
- Per rollout: picked the target, picked another block, picked nothing, plus
  time to lift. Report rates per condition, a JSON summary, and a few saved
  frames from one rollout per condition.
- Works on CPU for a short smoke check; the real run is on the GPU after
  training finishes.
