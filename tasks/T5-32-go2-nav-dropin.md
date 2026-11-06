---
id: T5-32
title: Drop-in test on the Go2 dog, where Curio's goal drives a pretrained navigation policy instead of our path follower
status: todo
agent: claude
budget: 100000
depends: [T5-27]
owns: [scripts/poc/phys_go2_nav.py, scripts/poc/export_go2_nav.py, scripts/go2_nav_eval.py, tests/test_t5_32_go2_nav.py, third_party/README.md, tasks/T5-32-go2-nav-dropin.md, tasks/notes/T5-32.md]
reads: [tasks/T5-27-curio-layer.md, docs/curio-layer.md, curio_robot/layer/adapters.py, scripts/poc/phys_go2.py, scripts/poc/phys_policy.py, scripts/poc/phys_run.py, scripts/poc/phys_scenes.py, scripts/poc/phys_eval.py, scripts/poc/export_policies.py, tasks/notes/T5-20.md]
accept: uv run pytest -q tests/test_t5_32_go2_nav.py
---

## Goal
In T5-20 the dog reached Curio's target through our own hand-written path follower
on top of a pretrained walking policy. This card replaces that layer with a
pretrained **navigation** policy: `kouylty/Isaac-Go2-Navigation-Rough-v0` (MIT), which
takes a goal position and outputs velocity commands, plus its own walking policy.
Curio's pose adapter supplies the goal, so the dog goes from the person's words to its
legs with no hand-written control in between.

## Setup (approved by the user)
- Download the checkpoint (under 0.1 GB) to third_party/ the way T5-20 handled its
  policies.
- Convert the TorchScript or PyTorch weights to numpy, so the sim needs no torch
  (export_go2_nav.py).
- Credit and licence go in third_party/README.md.
- The policies were trained in Isaac Lab. Port their observation and action layout to
  the MuJoCo Menagerie Go2 already in use (joint order, default pose, gains, observation
  scales, command format, and control rate). Every choice is written down before any
  gate run.

## GATE (written before any download or evaluation; not to be changed afterwards)
Preconditions, checked on random goals with no Curio involved:
- **P1 (walking):** the ported walking policy, on flat ground with commanded
  velocities from the policy's training range, runs 10 runs of 20 s each with 0 falls.
- **P2 (navigation):** the ported navigation policy plus its walking policy reaches
  random goals 1-4 m away, within 0.3 m in 20 s, on >= 16 of 20, with 0 falls.
If P1 or P2 fails, no Curio lines are run. Report the port diagnosis instead, and the
card is a FAIL on the port.

Curio test, using the T5-20 dog scenes (rescue-dog, and the other Go2 live scenes).
Each scene is run 10 times with jittered start poses (seeds 54000+). Success means the
dog stops within 0.5 m of the true target and closer to it than to any look-alike.
- **(A) raw:** the goal comes from a robot-frame reading of the words, using the
  T5-20 "naive" baseline.
- **(B) Curio:** the goal comes from the layer's pose adapter, with v4 grounding on
  the head camera.
- **(O) oracle:** the goal is the true target position.

Lines:
1. B success >= A success + 0.50.
2. B success >= 0.85 x O success.
3. Zero falls across all Curio runs.
Report a Wilson 95% CI for each rate. Also report T5-20's path-follower result on the
same scenes, for comparison. Results go to out/t5_32/gate.json and gate.md. One run,
and nothing is tuned after a score. A FAIL is acceptable.

## Done when
- The accept command passes (CPU: the observation layout, numpy-vs-torch equality on
  stored samples, and the goal hand-off).
- The full suite, ruff, and tasks.py check pass.
- Notes are in tasks/notes/T5-32.md, crediting kouylty, Isaac Lab and Unitree.
