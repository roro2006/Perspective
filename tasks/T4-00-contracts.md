---
id: T4-00
title: Freeze target-4 (reaction critic, phase 1) contracts, tests and cards
status: done
agent: claude
depends: []
owns: [docs/target4-critic.md, curio_robot/critic/__init__.py, tests/critic_util.py]
reads: []
accept: uv run pytest -q
---

Done in the target-4 scaffold commit: the design doc, typed stubs in
`curio_robot/critic/` (reactions, episodes, features, model, baseline,
evaluate), the shared test helper `tests/critic_util.py`, acceptance tests
`tests/test_t4_*.py` (skipped until implemented), and this board.

Any later change to a stub dataclass, signature or docstring spec is a new
contract task, not part of an implementation card.
