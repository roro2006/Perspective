---
id: T5-26
title: Flywheel turn 2, v4 with deictic replay so front/behind doesn't cost gaze
status: todo
agent: claude
budget: 100000
depends: [T5-24]
owns: [scripts/flywheel_data.py, scripts/flywheel_gate.py, curio_robot/flywheel/gate.py, curio_robot/flywheel/registry.json, tests/test_t5_26_flywheel_v4.py, tasks/T5-26-flywheel-v4.md, tasks/notes/T5-26.md]
reads: [curio_robot/flywheel/log.py, curio_robot/flywheel/request.py, curio_robot/flywheel/scenes.py, curio_robot/flywheel/serve.py, curio_robot/sim/grounding_data.py, scripts/bench_lora_data.py, scripts/train_lora.py, scripts/flywheel_deploy.py, tests/test_t5_flywheel.py, tasks/T5-24-correction-flywheel.md, tasks/notes/T5-24.md, docs/bringup-7900xt.md]
accept: uv run pytest -q tests/test_t5_26_flywheel_v4.py
---

## Why this card exists
T5-24's gate rejected v3, and that decision stands. v3 learned front/behind (0.78 ->
0.985) but forgot gaze: deictic ("hand me that one") went 0.45 -> 0.23. Deictic is
12.5% of v3's training rows, and it was a third of the target-2 adapter's rows, which
scored 0.55. This card trains v4 on the same data plus deictic replay, then runs it
through a promotion gate fixed below.

## Data (one build, one training run)
v4 data is:
- v2's data rebuilt exactly as in T5-24, checked against v2's data_sha256;
- plus the T5-24 flywheel rows, built from the same events in out/t5_24/deploy (no new
  deployment);
- plus deictic replay:
  - every deictic row in out/grounding/data.jsonl that isn't in the 302 target-2
    held-out ids and that v2's data doesn't already include;
  - fresh deictic rows from `curio_robot.sim.grounding_data.write_dataset(...,
    seed=31, kinds=("deictic",))`, as many as needed for deictic to make up at least
    25% of all v4 rows.
Training runs `train_lora.py` with its defaults (the v2 recipe) and is converted to
out/t5_26/curio-ground-r16-v4-f16.gguf. Never touch the v2 GGUF. If ROCm crashes
transiently, one rerun of the identical command is allowed, but only before any test
score has been seen, as in T5-24.

Seed ranges: deictic train seed 31, fresh deictic test seed 41 (the data builder must
refuse 41), front/behind fresh test 625000-625999 (the data builder must refuse it,
along with T5-24's 624000-624999).

## Test sets (built before training)
- (a) out/bench5c, all 350 items, geometry condition, as in T5-24.
- (d1) the 101 deictic rows among the 302 target-2 held-out ids, as in T5-24.
- (d2) a fresh deictic set: 200 rows from grounding_data seed 41, deictic only.
- (b2) a fresh front/behind set: 200 items from seeds 625000+, built the way T5-24 built
  its (100 with training phrasings, 100 with held-out phrasings, speaker to robot frame
  3:1).
- Reported only: T5-24's front/behind set (b), already seen; the warehouse query (c).
Score v2, v3 and v4 on every set. Each adapter runs alone on a fresh llama-server.

## GATE (written before any training or evaluation; not to be changed afterwards)
v4 is PROMOTED only if all three hold, on point estimates:
1. On the fresh front/behind set (b2), v4 >= 0.90 and v4 >= v2 + 0.5 * (1 - v2). In
   words, v4 removes at least half of v2's errors.
2. bench5c overall: v4 >= v2 - 0.01.
3. No bench5c kind, d1 or d2 drops by more than 0.05 (v4 >= v2 - 0.05 on each).
Otherwise REJECT, which is an acceptable reported outcome. On promote, `flywheel_gate.py
decide --register` copies the GGUF into llama/models/lora and adds it to
`curio_robot/flywheel/registry.json`. It doesn't restart or reconfigure the :8081
server; the coordinator does that after review.

Report each accuracy with a 95% Wilson CI, and each v4 - v2 and v4 - v3 difference with
a paired bootstrap 95% CI (10,000 resamples, seed 0). Include per-kind tables and the
train_lora held-out accuracy.

Line 1 differs from T5-24's (+0.25 absolute). Once v2's baseline turned out to be 0.78,
that bar needed v2 + 0.25 = 1.03, which no adapter can reach. "Half of v2's errors" is
set now, from v2's known baseline and before any v4 exists, and it is scored on a fresh
set v2 and v3 haven't been tuned against. Lines 2 and 3 are unchanged, plus the fresh
deictic set.

## Tests (tests/test_t5_26_flywheel_v4.py, CPU only)
- The v4 data builder: v2 sha check, deictic share >= 0.25, no held-out id, refuses
  seed 41 and both front/behind test ranges.
- The new line-1 rule in `flywheel/gate.py`, tested on edge values, alongside T5-24's
  rule (which stays available and its tests unchanged).

## GPU
Use the lock C:/Users/Rohan/dev/Perspective/out/gpu.lock (create atomically with
noclobber, write who and why, and remove when done). Build all data, including
rendered images, before you take the lock for training. T5-25 needs about 5 minutes
of GPU, so let it go first if it's waiting.

## Done when
The accept command passes, along with the full suite, ruff, and tasks.py check. The
gate has run once, and the notes are in tasks/notes/T5-26.md.
