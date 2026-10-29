---
id: T4-08
title: Phase 1 run, the gate, and the write-up
status: done
agent: claude
depends: [T4-07]
owns: [product/results.md]
reads: [docs/target4-critic.md, curio_robot/critic/evaluate.py]
accept: uv run pytest -q
---

## Goal
Run `curio-sim critic phase1` at full size, check the gate, and add a
"Reaction critic (target 4, phase 1)" section to `product/results.md`: the
critic against the gaze rule, overall and per family (silent separately),
what passed or failed, and the limits (simulated reactions written by us).
If the gate fails, say which bar and why, and what would change it.
