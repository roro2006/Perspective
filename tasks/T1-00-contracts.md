---
id: T1-00
title: Freeze target-1 contracts, acceptance tests and task cards
status: done
agent: claude
depends: []
owns: [curio_robot/conventions/__init__.py, tests/conftest.py, tests/scene.py, docs/target1-conventions.md]
reads: []
accept: uv run pytest -q
---

Done in the scaffold commit: typed stubs for every target-1 module, acceptance
tests `tests/test_t1_*.py` (verified against a private reference implementation,
82/82 passing, then reverted to stubs), the `--strict-impl` switch, this board.

Any later change to a stub's dataclass, signature or docstring spec is a new
contract task, not part of an implementation card.
