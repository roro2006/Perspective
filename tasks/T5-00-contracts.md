---
id: T5-00
title: Freeze target-5 (perspective grounding and policy) contracts, tests and cards
status: done
agent: claude
depends: []
owns: [docs/target5-perspective-policy.md, curio_robot/bench/__init__.py, tests/test_t5_bench.py]
reads: []
accept: uv run pytest -q
---

Done in the target-5 scaffold commit: the design doc, typed stubs in
curio_robot/bench (items, conditions, score), acceptance tests in
tests/test_t5_bench.py (skipped until implemented), and these cards.
