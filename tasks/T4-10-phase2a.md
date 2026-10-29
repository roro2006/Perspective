---
id: T4-10
title: Phase 2a, critic labels as a training signal for a target-choice policy
status: todo
agent: claude
depends: [T4-09]
owns: [curio_robot/critic/learn.py, curio_robot/critic/cli.py, tests/test_t4_learn.py, tests/test_t4_cli.py, product/results.md]
reads: [docs/target4-critic.md, curio_robot/critic/learn.py]
accept: uv run pytest tests/test_t4_learn.py tests/test_t4_cli.py --strict-impl
---

The lighter phase 2 agreed with the user: same claim as the brief's SmolVLA
phase 2 (reactions are a better training signal than spoken corrections), on a
learned target-choice policy in numpy, no downloads. Gate in the design doc,
section 7, fixed before the first run.
