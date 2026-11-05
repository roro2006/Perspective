---
id: T5-30
title: Find why the trained pi0 does nothing at test time, then rerun the T5-28 gate on fresh seeds
status: todo
agent: claude
budget: 120000
depends: [T5-28]
owns: [scripts/pi0_data.py, scripts/pi0_train.py, scripts/pi0_eval.py, scripts/pi0_check.py, tests/test_t5_30_pi0_fix.py, tasks/T5-30-pi0-fix.md, tasks/notes/T5-30.md]
reads: [tasks/T5-28-pi0-plugin.md, tasks/notes/T5-28.md, tests/test_t5_28_pi0.py, scripts/smolvla_eval.py, scripts/smolvla_dataset.py, docs/bringup-7900xt.md]
accept: uv run pytest -q tests/test_t5_30_pi0_fix.py
---

## Why
T5-28's gate failed: pi0 lifted no block in 300 rollouts, across raw, Curio and
oracle alike. That result stands. The evidence points to a fault between
training and inference, not to pi0's ability:
- training loss ended near 0.001, and the logged held-out eval_loss was 0.968 at
  step 22,000;
- loaded through the eval path, the checkpoint gives a training-path loss of 3.76
  on a training frame;
- the loaded model predicts joint targets near 0 rad where the demos command about
  1.5 rad. One normalized output is about -7.

Weights load exactly as saved, and the normalizer stats look right. So the fault
is likely in the pipeline around the weights, for example:
- the pre- or post-processor (normalize/unnormalize mode, or which stats are
  applied);
- the image key mapping or image range;
- the state vs action layout;
- the task-text tokenization;
- the action chunk indexing;
- the train-only expert config not being restored at load.

## Diagnosis (no eval seeds: use the training and held-out demo frames only)
1. `scripts/pi0_check.py`: load out/t5_28/pi0/checkpoints/022000 exactly as
   pi0_eval does. Then recompute the held-out loss on the 80 held-out episodes and
   the training-path loss on training frames. Next, compare predicted vs demo
   actions on the same frames, open loop.
2. Find the mismatch by comparing the training-time path with the eval path step
   by step, tensor by tensor, on the same frame. Fix it in the owned scripts. Then
   add a regression test that fails on the old code path (CPU, with a tiny fake
   model or saved small tensors).
3. Retraining is allowed only if the diagnosis shows the checkpoint itself is
   unusable (for example, it was saved wrong). In that case: one training run with
   T5-28's recipe and budget (at most 8 h), and the reason goes in the notes.

## GATE (written before any diagnosis; not to be changed afterwards)
Precondition, checked before any rollout:
- P. Under the fixed eval loading path, the checkpoint's held-out loss is within
  15% of the value logged in training (0.968 for 022000, or the retrained run's own
  logged value). Open-loop predicted joint targets on held-out frames have a median
  absolute error of 0.15 rad or less against the demos.
If P fails, no rollouts are run. The card FAILs with the diagnosis written up.

Then T5-28's three lines, on **fresh** seeds 53000-53099 (built as T5-28 built
51000-51099), unchanged:
1. Curio (B) target rate >= raw (A) target rate + 0.25.
2. Curio (B) target rate >= 0.85 x oracle (O) target rate.
3. Oracle (O) target rate >= 0.30.
SmolVLA run 5 on the same seeds is reported for context. Report a Wilson 95% CI
for each rate and a paired bootstrap CI for each difference. Results go to
out/t5_30/gate.json and gate.md. One eval run.

## GPU
The GPU is free: T5-29's training was stopped by the user. Take the lock
(out/gpu.lock) atomically and remove it on ANY exit. Don't stop the :8081 grounder
unless memory requires it, and restart it afterwards with the v4 command if you do.

## Done when
The accept command passes, along with the full suite, ruff, and tasks.py check.
The notes go in tasks/notes/T5-30.md, naming the root cause in one sentence.
