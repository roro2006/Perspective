---
id: T5-10
title: Benchmark kinds hidden-pair and free
status: done
agent: claude
depends: [T5-03, T5-08]
owns: [curio_robot/bench/items.py, tests/test_t5_bench.py, docs/target5-perspective-policy.md]
reads: [docs/target5-perspective-policy.md, curio_robot/bench/items.py, curio_robot/bench/conditions.py, curio_robot/bench/score.py, curio_robot/collab/tasks.py, curio_robot/perspective.py, tests/test_t5_bench.py]
accept: uv run pytest tests/test_t5_bench.py --strict-impl
---

## Goal
Fix two weaknesses the first scores showed (product/results.md, "Perspective
benchmark"):
- "hidden" targets are a unique colour, so any condition with the person view
  finds them and geometry ties paired. Add "hidden-pair": the target is a
  same-colour pair block hidden from the robot by the box, its partner
  visible, requested as "on my/your left/right" (ambiguous between frames).
- Every request has an exact keyword answer, so the no-VLM rule beats every
  VLM. Add "free": templated natural wording with exact answers from
  perspective.pick (side, nearest, farthest, from the speaker or the robot),
  synonyms for the block, no gaze or "that one".

Contract change: append both kinds at the END of KINDS so items of the old
kinds stay identical per seed; keep every signature; rule_predict unchanged
(it must not crash on the new wording). The depends on T5-08 only orders the
shared ownership of the design doc.

## Done when
The accept command passes, the first 4*n items of generate() equal the old
kinds' items, out/bench5b (50 per kind, seed 0) is generated and the rule is
scored on it, plus the full suite, ruff and `scripts/tasks.py check`.
