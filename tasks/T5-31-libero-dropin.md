---
id: T5-31
title: Drop-in test on LIBERO, where Curio drives SmolVLA and pi0.5 policies trained by others, with no training
status: todo
agent: claude
budget: 120000
depends: [T5-27]
owns: [curio_robot/layer/libero_words.py, scripts/libero_setup.md, scripts/libero_scenes.py, scripts/libero_eval.py, tests/test_t5_31_libero.py, tasks/T5-31-libero-dropin.md, tasks/notes/T5-31.md]
reads: [tasks/T5-27-curio-layer.md, docs/curio-layer.md, tasks/notes/T5-30.md, tasks/notes/T5-28.md, docs/bringup-7900xt.md]
accept: uv run pytest -q tests/test_t5_31_libero.py
---

## Goal
This is the real customer case: a policy someone else has already trained for their
robot, with Curio placed in front of it and nothing retrained. The policies are
`lerobot/smolvla_libero` (Apache-2.0) and `lerobot/pi05_libero_finetuned_v044`
(Gemma terms, already accepted by the user). Both control a Franka in LIBERO
(robosuite/MuJoCo).

LIBERO-Spatial already has two identical black bowls that are told apart only by
spatial words ("pick up the black bowl between the plate and the ramekin and place it
on the plate"). We add a person, who is a **camera only** (a head camera at a seat
beside or across the table, with no body mesh), so the policy's own camera views stay
exactly as in training. The person asks from their side: "the bowl on my left", "the
one closest to me".
- **(A) raw**: the policy gets the person's words.
- **(B) Curio**: the layer grounds the request on the head camera (v4, unchanged), and
  `layer/libero_words.py` maps the chosen bowl to the LIBERO-Spatial task string the
  policy was trained on for that bowl.
- **(O) oracle**: the task string for the true target.

## Setup (approved by the user)
- A NEW venv `C:/Users/Rohan/dev/libero-venv` with LeRobot (the same version as the
  lerobot venv, 0.6.1), torch ROCm (the same wheel source as the lerobot venv), and
  `hf-libero` 0.1.4 with robosuite.
- Don't touch the existing lerobot venv. Record every step in `scripts/libero_setup.md`.
- Downloads: the two checkpoints above and the LIBERO assets. Nothing else without
  asking.

## GATE (written before any setup or evaluation; not to be changed afterwards)
Precondition (lesson from T5-28, where a broken harness gave zeros):
- **P.** On unmodified LIBERO-Spatial (no person, the policy's own task strings, the
  first 5 init states of each of the 10 tasks, 50 episodes), each policy succeeds on
  >= 0.60. If P fails for a policy, that policy's Curio lines aren't run; report the
  harness diagnosis instead.

Curio test, per policy that passes P:
- 100 episodes over the LIBERO-Spatial tasks whose target is one of the two black
  bowls.
- Init states from index 10 on: disjoint from P's, and never seen while building
  `libero_words` or the scenes.
- Person seats are chosen so the request's frame differs from the robot's, so a
  robot-frame reading of the words points at the other bowl.
- Success means LIBERO's own task success for the target bowl's task.

Lines:
1. Curio (B) success >= raw (A) success + 0.25.
2. Curio (B) success >= 0.85 x oracle (O) success.
3. Curio's grounding picks the true bowl on >= 0.90 of the 100 requests. This is
   reported per policy, and it is the same number for both.

Report:
- a Wilson 95% CI for each rate, and paired bootstrap CIs (10,000 resamples, seed 0)
  for the differences;
- the head-camera renders the grounder saw (5 examples).

Results go to out/t5_31/gate.json and gate.md. One eval run per policy, with nothing
tuned after a score. A FAIL is acceptable.

## GPU
Use the lock (out/gpu.lock), removed on ANY exit. The :8081 grounder is shared, so
keep it up. pi0.5 inference needs about 9 GB, which fits beside it.

## Done when
- The accept command passes (CPU: the libero_words mapping, the seat geometry, and
  the harness with a fake policy).
- The full suite, ruff, and tasks.py check pass.
- Notes are in tasks/notes/T5-31.md, crediting the LIBERO authors, Hugging Face's
  LeRobot team and Physical Intelligence.
