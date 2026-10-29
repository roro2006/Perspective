---
id: T3-08
title: Export sessions as a dataset
status: done
agent: qwen
depends: []
owns: [curio_robot/collab/export.py]
reads: [docs/target3-collaboration.md, curio_robot/collab/session.py, curio_robot/collab/export.py, tests/test_t3_session.py, tests/test_t3_export.py]
accept: uv run pytest tests/test_t3_export.py --strict-impl
---

## Goal
Implement `export_episodes` in `curio_robot/collab/export.py` exactly as the
docstring says. Only the `Session`/`Step` dataclasses are needed (not the
T3-01 functions). Keep `FORMAT`.

## Hints
- Check `(out_dir / "meta" / "info.json").exists()` first.
- `mkdir(parents=True, exist_ok=True)` for `meta` and `data`.
- Write JSON with `json.dumps(...)`, files with `encoding="utf-8"`.
- `"t": {stage: round(mark - session.t_start, 4) for stage, mark in step.marks.items()}`.

## Done when
The accept command passes, plus the full suite and ruff (see AGENTS.md).
