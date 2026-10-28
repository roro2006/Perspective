---
id: T2-08
title: Evaluation and the replacement gate
status: todo
agent: qwen
depends: []
owns: [curio_robot/grounding/evaluate.py]
reads: [docs/target2-grounding.md, tests/test_t2_evaluate.py]
accept: uv run pytest tests/test_t2_evaluate.py --strict-impl
---

## Goal
Implement `evaluate` and `passes_gate` in `curio_robot/grounding/evaluate.py`
exactly as the docstrings specify. Keep `Example` and `Prediction`.

## Hints
- Percentiles: `float(np.percentile(latencies, 50))` and `95` (numpy's
  default is linear interpolation).
- `by_kind` keeps kinds in first-seen order; the test compares dicts, so
  order doesn't matter there.
- Gate checks in the docstring's order: latency, then no current model, then
  each kind of the *current* result in sorted order. Message formats are
  exact: accuracies with `:.3f`, milliseconds with `:.0f`.

## Done when
The accept command passes, plus the full suite and ruff (see AGENTS.md).
