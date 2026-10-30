---
id: T5-15
title: More accurate grasps, SmolVLA run 3
status: todo
agent: claude
depends: [T5-12, T5-14]
owns: [curio_robot/sim/demos.py, scripts/smolvla_dataset.py, scripts/smolvla_train.py, scripts/smolvla_eval.py, scripts/e2e_eval.py, tests/test_t5_demos.py, product/results.md, tasks/notes/T5-15.md]
reads: [docs/target5-perspective-policy.md, product/results.md, tasks/notes/T5-12.md, tasks/notes/T5-14.md, curio_robot/sim/expert.py, curio_robot/sim/physics.py]
accept: uv run pytest -q
---

## Goal
Run 2 (wrist camera, chunk 30, 10 actions per call, 400 varied demos) picks the ringed
target 0.36 of the time; in 19 of 32 misses the jaws close just short of or beside the
block. Measure where the jaws close relative to the block in run-2 rollouts, choose fixes
from that evidence (data, recoveries, image size, loss weighting, training length), train
run 3 and evaluate it. Keep the ring on camera1 and the varied wording.

## Gate (fixed before training)
On the T5-08 evaluation with n = 100 scenes (seed 50000, default run-3 settings):
- ring on target: target picks >= 0.55 (run 2: 0.36 on 50 scenes);
- ring on the partner: partner picks > target picks (the ring still steers);
- words only: target picks stay below ring-on-target picks.
End to end (setups words, lora2, oracle; out/e2e/run3) is reported, not gated.
