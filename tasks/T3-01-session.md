---
id: T3-01
title: Session records to and from JSON lines
status: todo
agent: qwen
depends: []
owns: [curio_robot/collab/session.py]
reads: [docs/target3-collaboration.md, curio_robot/collab/session.py, tests/test_t3_session.py]
accept: uv run pytest tests/test_t3_session.py --strict-impl
---

## Goal
Implement `session_to_dict`, `session_from_dict`, `save_sessions` and
`load_sessions` in `curio_robot/collab/session.py` exactly as the docstrings
say. Keep `Step`, `Session` and `StepKind` unchanged.

## Hints
- Build the dicts by hand (field by field) rather than `dataclasses.asdict`,
  so tuples become lists and `marks` is a fresh `dict`.
- `session_from_dict`: `tuple(...)` for goals, done and steps;
  `{k: float(v) for k, v in marks.items()}` for marks.
- `save_sessions`: `path.parent.mkdir(parents=True, exist_ok=True)`, open with
  `"w", encoding="utf-8"`, one `json.dumps(...)` per line.

## Done when
The accept command passes, plus the full suite and ruff (see AGENTS.md).
