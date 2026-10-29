---
id: T3-00
title: Freeze target-3 contracts, acceptance tests and task cards
status: done
agent: claude
depends: []
owns: [curio_robot/collab/__init__.py, curio_robot/runtime/trace.py, docs/target3-collaboration.md]
reads: []
accept: uv run pytest -q
---

Done in the target-3 scaffold commit: the design doc, typed stubs in
`curio_robot/collab/` (session, metrics, worker, tasks, gaze, export, run, report),
acceptance tests `tests/test_t3_*.py` (skipped until implemented),
`Tracer.marks()` for absolute stage times, and this board.

Any later change to a stub's dataclass, signature or docstring spec is a new
contract task, not part of an implementation card.
