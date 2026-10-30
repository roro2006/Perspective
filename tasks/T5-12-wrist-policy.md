---
id: T5-12
title: SmolVLA run 2 with the wrist camera and varied wording
status: done
agent: claude
depends: [T5-07, T5-08, T5-09]
owns: [curio_robot/sim/demos.py, scripts/smolvla_dataset.py, scripts/smolvla_train.py, scripts/smolvla_eval.py, scripts/e2e_eval.py, tests/test_t5_demos.py, product/results.md]
reads: [docs/target5-perspective-policy.md, product/results.md, tasks/notes/T5-07.md, tasks/notes/T5-08.md, tasks/notes/T5-09.md]
accept: uv run pytest -q
---

## Goal
Run 1 follows the ring but rarely completes the grasp (oracle ring: 14%
target picks). Add the SO-101 wrist camera (wrist_cam in so101.xml) as the
third view, record demonstrations with varied wording, train run 2 and
evaluate it the same way as run 1.

## Gate (fixed before training)
On the T5-08 evaluation (50 unseen scenes, default settings):
- ring on target: target picks >= 0.40 (run 1: 0.12);
- ring on the partner: partner picks > target picks (the ring still steers);
- words only: target picks stay below ring-on-target picks.
End to end with the lora2 grounder is reported, not gated.
