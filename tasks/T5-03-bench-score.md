---
id: T5-03
title: Benchmark scoring and curio-sim bench
status: todo
agent: any
depends: [T5-02]
owns: [curio_robot/bench/score.py, curio_robot/bench/cli.py, curio_robot/sim/cli.py]
reads: [docs/target5-perspective-policy.md, curio_robot/bench/score.py, curio_robot/bench/conditions.py, curio_robot/grounding/client.py, curio_robot/sim/cli.py, tests/test_t5_bench.py]
accept: uv run pytest tests/test_t5_bench.py -k score --strict-impl
---

## Goal
Implement rule_predict, vlm_predict and score as the docstrings say, and a
command: `curio-sim bench gen --n-per-kind N --seed S --out DIR` (generate and
save) and `curio-sim bench score --data DIR [--url URL] [--conditions
geometry,paired,robot-only] [--json OUT]`: a Markdown table, one row per kind
plus overall, one column for the rule and one per condition (conditions only
when --url is given, through a VllmGrounder on that URL).
Wire it like curio_robot/critic/cli.py: bench/cli.py has add_parser(sub) and
run(args); sim/cli.py calls them.

## Done when
The accept command passes, plus the full suite and ruff (see AGENTS.md).
