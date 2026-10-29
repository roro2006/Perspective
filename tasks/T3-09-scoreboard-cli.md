---
id: T3-09
title: curio-sim collab, run tasks and print the scoreboard
status: doing
agent: any
depends: [T3-02, T3-06, T3-08]
owns: [curio_robot/collab/report.py, curio_robot/sim/cli.py, tests/test_t3_report.py]
reads: [docs/target3-collaboration.md, curio_robot/collab/metrics.py, curio_robot/collab/run.py, curio_robot/collab/session.py, curio_robot/collab/export.py, curio_robot/sim/cli.py]
accept: uv run pytest tests/test_t3_report.py --strict-impl
---

## Goal
One command that runs tasks per condition and prints the scoreboard.

## Behaviour
- `collab/report.py`: `format_table(summary) -> str`, a Markdown table with
  one row per metric (Fluency field order, then n), one column per condition,
  numbers to 2 decimals, "-" for None, and the "better" direction from the
  design doc's table in a last column.
- `curio-sim collab --user {own-frame,robot-frame,mixed} --tasks N --goals 4
  --conditions view-off,view-on --seed 0 --deictic 0.0 --occlude
  --grounder-url URL --out out/collab [--export DIR]`: runs `run_task` for
  seeds seed..seed+N-1 per condition, saves `sessions.jsonl` and
  `summary.json` in `--out`, prints `format_table(summarize(...))`, and with
  `--export` writes the dataset.

## New tests (`tests/test_t3_report.py`, yours)
`format_table` on a hand-made summary: headers, row order, "-" for None.

## Done when
The accept command passes, plus the full suite and ruff.
