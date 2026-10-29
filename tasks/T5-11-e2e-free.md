---
id: T5-11
title: Free wording in the end-to-end trials
status: done
agent: claude
depends: [T5-09, T5-10]
owns: [scripts/e2e_eval.py, tests/test_t5_e2e.py]
reads: [scripts/e2e_eval.py, curio_robot/bench/items.py]
accept: uv run pytest tests/test_t5_e2e.py
---

Adds a fifth e2e request kind, free, drawn with the benchmark free-wording
templates (side and nearest/farthest relations). For distance wording the
target is the block the relation picks. The trial test checks free items by
relation as well as by side.
