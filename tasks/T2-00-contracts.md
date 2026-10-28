---
id: T2-00
title: Freeze target-2 contracts, acceptance tests and task cards
status: done
agent: claude
depends: []
owns: [curio_robot/grounding/__init__.py, curio_robot/events.py, docs/target2-grounding.md]
reads: []
accept: uv run pytest -q
---

Done in the target-2 scaffold commit: typed stubs in `curio_robot/grounding/`
and `curio_robot/sim/grounding_data.py`, acceptance tests `tests/test_t2_*.py`
(verified against a private reference implementation, 51/51 passing, then
reverted to stubs), `Heard.words` and the `GroundingReply` event, this board.

Any later change to a stub's dataclass, signature or docstring spec is a new
contract task, not part of an implementation card.
