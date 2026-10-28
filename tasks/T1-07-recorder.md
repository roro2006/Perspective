---
id: T1-07
title: Versioned decision recorder
status: todo
agent: qwen
depends: []
owns: [curio_robot/recorder.py]
reads: [docs/target1-conventions.md, curio_robot/conventions/readings.py, curio_robot/conventions/decide.py, curio_robot/episode_log.py, tests/test_t1_recorder.py]
accept: uv run pytest tests/test_t1_recorder.py --strict-impl
---

## Goal
Implement `DecisionRecorder` and `load_records` in `curio_robot/recorder.py`
as the docstrings specify. Keep `VERSIONS`, `DecisionRecord` and the line
format unchanged.

## Hints
- `episode_log.py` shows the house style for JSONL (append, flush per line,
  `_plain` for numpy). Extend that idea: dataclasses become dicts via
  `dataclasses.fields`, recursively, and numpy arrays become lists rounded to 4.
- The dataclasses use `slots=True`, so use `dataclasses.fields` and
  `getattr` rather than `__dict__`.
- Create the parent directory on open; open in append mode.
- In `load_records`, an utterance's last outcome line wins, and it applies to
  every decision line of that utterance.

## Done when
The accept command passes, plus the full suite and ruff (see AGENTS.md).
