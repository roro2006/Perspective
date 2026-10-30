---
id: T5-13
title: Benchmark kind free-heldout with wording no training set uses
status: done
agent: claude
depends: [T5-06, T5-10]
owns: [curio_robot/bench/items.py, tests/test_t5_bench.py, scripts/bench_lora_data.py, tests/test_t5_lora_data.py, docs/target5-perspective-policy.md]
reads: [docs/target5-perspective-policy.md, curio_robot/bench/items.py, curio_robot/bench/score.py, scripts/bench_lora_data.py, tests/test_t5_bench.py, tests/test_t5_lora_data.py, tasks/notes/T5-06.md, tasks/notes/T5-10.md, docs/bringup-7900xt.md]
accept: uv run pytest tests/test_t5_bench.py tests/test_t5_lora_data.py --strict-impl
---

## Goal
The retrained grounder (r16 v2, T5-06) scores 0.90 on "free", but its training
data used the same FREE_TEMPLATES, FREE_LEADS and FREE_NOUNS as the test items,
so 0.90 shows generalisation to new scenes, not new phrasings. Add a held-out
kind whose wording shares nothing with training:
- "free-heldout", appended at the END of KINDS (earlier items stay
  byte-identical per seed). Its own HELDOUT_TEMPLATES (15-20, natural, no gaze,
  pointing or "that one"), HELDOUT_LEADS and HELDOUT_NOUNS, none shared with the
  free ones; the same exact-answer logic as free (side from a frame,
  nearest/farthest from the speaker or the robot, 0.02 m distance gap, the two
  frames must disagree).
- items.HELDOUT_KINDS lists it; scripts/bench_lora_data.py refuses any held-out
  kind, so future grounder training cannot leak it.

Contract change: KINDS gains a last entry and items gains HELDOUT_* constants;
every signature and rule_predict unchanged.

## Done when
The accept command passes (including tests that the first six kinds' items are
unchanged and that no template, free relation phrase, lead word or noun is
shared), out/bench5c (50 per kind, seed 0) is generated with its first 300
items and PNGs byte-identical to out/bench5b, the rule is scored on it, the GPU
commands to score both adapters on bench5c are in the notes, plus the full
suite, ruff and `scripts/tasks.py check`.
