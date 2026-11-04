---
id: T5-25
title: Theory of mind, retested on items where the current view really disagrees
status: todo
agent: claude
budget: 80000
depends: [T5-21]
owns: [curio_robot/belief_bench.py, scripts/belief_eval.py, tests/test_t5_25_belief_split.py, tasks/T5-25-belief-split.md, tasks/notes/T5-25.md]
reads: [curio_robot/belief.py, curio_robot/belief_bench.py, scripts/belief_eval.py, tests/test_t5_21_belief.py, tasks/T5-21-theory-of-mind.md, tasks/notes/T5-21.md, curio_robot/grounding/prompt.py, curio_robot/grounding/marks.py, docs/bringup-7900xt.md]
accept: uv run pytest -q tests/test_t5_25_belief_split.py
---

## Why this card exists
T5-21 failed gate line 2: the belief-aware system was right on 0.992 of the
false-belief items, but the current-state baseline was right on 0.733, so the gap
was 0.258, short of the 0.30 the gate asked for. That result stands and is not
rerun or re-scored. The diagnosis in tasks/notes/T5-21.md: in 84 of the 120 items,
both blocks of the pair ended on the same side in the words' frame after the move.
The prompt then gave them the same side label, and v2, reading the current view,
picked the old target by accident in 79 of those 84 items. Those items didn't test
whether the robot tracks belief, because the current view gave no clear competing
answer.

This card asks the question T5-21 meant to ask, on fresh seeds: when the person's
belief and the current view point at different objects, and the current view is
unambiguous, does the belief tracker get the right one? The belief tracker
(`curio_robot/belief.py`) and the grounder (v2) are unchanged.

## Bench change (curio_robot/belief_bench.py)
Add `split_labels: bool = False` to `generate` (default keeps T5-21 byte-identical;
the T5-21 tests must still pass). With `split_labels=True`, a false-belief or
present-tense item is kept only if, after the move:
- the words' relation picks the other block of the pair, as T5-21 already requires, and
- in the words' frame, the target and its partner get different side labels in the
  grounder prompt, built with the same prompt code the eval uses.
Draws that fail are redrawn from the same rng, and the redraw count is recorded.
Controls are generated exactly as in T5-21. Seeds 2111 (false-belief), 2112 (control),
2113 (present). Counts are 120 / 120 / 60, as in T5-21. Also generate, and report but
don't gate, a tied-label set of 60 false-belief items (seed 2114) built as in T5-21
but keeping only the items where the labels tie. This keeps the case where v2 is right
by accident visible.

## GATE (written before any evaluation; not to be changed afterwards)
The four lines are T5-21's, applied to the split-label sets. Each grounder error or
empty answer counts as wrong. Accuracy means the resolved object id equals the target id.
1. Belief-aware accuracy on the 120 split-label false-belief items >= 0.85.
2. Current-state-only accuracy on the same items <= belief-aware accuracy - 0.30.
3. On the 120 controls, belief-aware accuracy >= R - 0.03. R is the higher of
   current-state-only accuracy on the controls and the v2 "marked"-wording reference.
4. The spoken-note flag equals "the target moved while unseen" on >= 0.95 of the 240
   false-belief and control items.
Report every number with a 95% CI: Wilson for rates, and a paired bootstrap for
differences (10,000 resamples, seed 0). The gate is on the point estimates.

Reported, not gated:
- the tied-label set (both systems);
- present-tense accuracy after a move (both systems);
- for current-state-only on false-belief items, how often it picked the block the
  current view's words point to. This shows whether it reads the current view
  correctly and simply lacks the belief.

One run. Don't change the bench, wording or system after any score has been seen.
Write results to out/t5_25/gate.json and out/t5_25/gate.md.

## Tests (tests/test_t5_25_belief_split.py, CPU only, no server)
- With split_labels=True, every false-belief and present-tense item has distinct side
  labels for target and partner after the move, and the words pick the partner.
- Controls equal T5-21's generator output for the same seed.
- split_labels=False reproduces T5-21's items for seeds 2101-2103 exactly.
- The tied-label set has tied labels on every item.

## Done when
The accept command passes, along with the full suite, ruff on curio_robot, tests and
scripts, and tasks.py check. The gate has run once on the real v2 grounder, and the
notes are in tasks/notes/T5-25.md. Calling the result a fix of T5-21 is wrong: it's a
retest with a corrected item design, and the notes must say so.
