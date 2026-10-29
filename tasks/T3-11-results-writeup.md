---
id: T3-11
title: Collaboration results write-up
status: doing
agent: claude
depends: [T3-07, T3-09]
owns: [product/results.md]
reads: [docs/target3-collaboration.md, product/results.md, curio_robot/collab/report.py]
accept: uv run pytest -q
---

## Goal
Run `curio-sim collab` for each simulated-user type over the three
conditions (enough tasks per cell for stable means) and add a "Working
together (target 3)" section to `product/results.md`: the scoreboard, what
changed between conditions, what improved across repeated tasks, and the
honest limits (simulated people, head direction not eye tracking).

## Done when
The section is written from a recorded run (commands included so it can be
reproduced); the full suite and ruff still pass.
