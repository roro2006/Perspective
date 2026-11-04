---
id: T5-22
title: Ask back only when it's worth it (one short question in the person's frame)
status: done
agent: claude
budget: 100000
depends: [T5-20]
owns: [curio_robot/clarify.py, scripts/ask_back_eval.py, tests/test_t5_22_clarify.py, tests/test_t5_22_eval.py, tasks/T5-22-ask-back.md, tasks/notes/T5-22.md]
reads: [curio_robot/bench/items.py, curio_robot/bench/score.py, curio_robot/bench/conditions.py, curio_robot/grounding/client.py, curio_robot/perspective.py, scripts/e2e_eval.py, docs/bringup-7900xt.md]
accept: uv run pytest -q tests/test_t5_22_clarify.py tests/test_t5_22_eval.py
---

## Goal
When the grounder is unsure, the robot asks the single most informative short question,
phrased in the speaker's frame, instead of guessing.

- Confidence = the grounder's letter probabilities (VllmGrounder.score, the bench5c/e2e
  scoring), geometry condition, Qwen3-VL-4B Q8_0 + curio-ground-r16-v2 on llama-server :8081.
- curio_robot/clarify.py: candidates (id, colour, speaker-frame lateral offset, distance from
  the speaker, nearby landmarks with their relation as the speaker sees it) -> yes/no
  questions from templates ("The one on your left?", "The one nearer you?", "The one behind
  the boxes from you?", "The red one?"). Side and distance questions are relative among the
  contenders of one colour. Pick the question with the highest expected information gain
  under the posterior and a noisy-answer model (eps = 0.1); Bayesian update on the answer.
- Policy: act if max p >= tau_act; otherwise ask (k <= 2 questions, stop early once max p >=
  tau_act or no question is worth >= 0.1 bit), then act on the argmax.
- Simulated answerers: truthful (from ground truth) and noisy (each answer flipped w.p. 0.1).
- (Demo clip dropped from scope by the coordinator.)

## GATE (written before any evaluation; not to be changed afterwards)
Data: out/bench5c (350 items, 50 per kind), geometry condition, v2 adapter. Every item is
scored once; the letter probabilities are cached in out/t5_22/scores.json and every number
below is computed from that cache.

Split: within each kind, items in file order; even positions -> dev (175), odd -> test (175).
Fixed before evaluation: k = 2, answer-noise model eps = 0.1, minimum gain 0.1 bit,
contenders = objects with p >= 0.02.

Tuning (dev only, truthful answerer): tau_act from {0.50, 0.55, ..., 0.95, 0.97, 0.98, 0.99}.
Choose the tau with the highest dev ask-back accuracy among those with dev ask rate <= 0.25
and dev mean questions <= 0.33; ties -> the smallest tau. If none qualifies, the smallest tau
with the highest dev accuracy. That single tau is then applied once to test.

On the TEST split (175 items):
1. Truthful answerer: ask-back accuracy >= 0.97.
2. Truthful answerer: asks on <= 0.30 of items.
3. Truthful answerer: mean questions per item <= 0.4.
4. Noisy answerer (10% wrong answers): accuracy, averaged over noise seeds 0-19, >= the no-ask
   baseline accuracy on test.

Report 95% CIs: Wilson for proportions, bootstrap (2000 resamples, seed 0) for mean questions
and for noisy minus baseline. Also report: per-kind breakdown (including free-heldout) of
baseline and ask-back accuracy and ask rate, examples of chosen questions. Failures are
reported, not hidden.
