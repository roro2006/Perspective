---
id: T4-07
title: curio-sim critic (gen, train, eval, phase1)
status: done
agent: any
depends: [T4-02, T4-03, T4-04, T4-05, T4-06]
owns: [curio_robot/critic/cli.py, curio_robot/sim/cli.py, tests/test_t4_cli.py]
reads: [docs/target4-critic.md, curio_robot/critic/episodes.py, curio_robot/critic/model.py, curio_robot/critic/baseline.py, curio_robot/critic/evaluate.py, curio_robot/sim/cli.py]
accept: uv run pytest tests/test_t4_cli.py --strict-impl
---

## Goal
One command family to make data, train, and score, plus `phase1`, which runs
the whole split from the design doc for the learned critic and the gaze rule.

## Behaviour
- `curio_robot/critic/cli.py`: `add_parser(sub)` registers `critic` with
  subcommands; `curio_robot/sim/cli.py` calls it.
- `critic gen --n N --seed S [--families a,b] [--p-wrong P] --out FILE`.
- `critic train --data FILE --out MODEL [--epochs E] [--lr L] [--seed S]`:
  prints the first and last epoch loss.
- `critic eval --data FILE (--model MODEL | --baseline) [--threshold T]
  [--val FILE] [--json OUT]`: threshold from `--val` via
  `threshold_for_precision`, else `--threshold` (default 0.5); prints
  overall metrics, per-family metrics, AUROC and the gate.
- `critic phase1 [--n-train 4000] [--n-val 1000] [--n-test 2000] [--epochs 30]
  --out DIR`: train set seed 0 (attentive, slow, distracted), val seed 1
  (same), test seed 2 (all four); trains, picks the threshold on val,
  evaluates the critic and the baseline (threshold 0.5) on test overall and
  per family, writes `DIR/model.npz` and `DIR/summary.json`, and prints a
  Markdown table: one row per metric, columns critic and gaze rule, then the
  gate.
- `tests/test_t4_cli.py` (yours to write): phase1 with tiny sizes (for
  example 40 / 20 / 20, 2 epochs) into `tmp_path` writes both files, and the
  summary has `critic`, `baseline` and `gate` keys.

## Done when
The accept command passes, plus the full suite and ruff (see AGENTS.md).
