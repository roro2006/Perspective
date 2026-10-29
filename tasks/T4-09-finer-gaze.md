---
id: T4-09
title: Finer gaze features and a slow-and-ask operating point
status: done
agent: claude
depends: [T4-08]
owns: [curio_robot/critic/features.py, curio_robot/critic/cli.py, tests/test_t4_features.py, tests/test_t4_cli.py]
reads: [docs/target4-critic.md, curio_robot/critic/features.py, curio_robot/critic/cli.py]
accept: uv run pytest tests/test_t4_features.py tests/test_t4_cli.py --strict-impl
---

Contract change after the phase 1 gate missed on lead time: six features
measured against the gripper direction (FEATURES grows from 10 to 16, old
columns unchanged), and a second reported operating point at 75% precision in
`critic phase1`. Results in product/results.md.
