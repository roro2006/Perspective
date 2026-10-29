---
id: T4-04
title: The learned critic (numpy MLP with masked softmax)
status: todo
agent: qwen
depends: []
owns: [curio_robot/critic/model.py]
reads: [docs/target4-critic.md, curio_robot/critic/model.py, curio_robot/critic/features.py, tests/critic_util.py, tests/test_t4_model.py]
accept: uv run pytest tests/test_t4_model.py --strict-impl
---

## Goal
Implement `Critic` in `curio_robot/critic/model.py` exactly as the
docstrings say. Numpy only.

## Hints
- `probs`: `s = np.where(mask, scores, -np.inf)`; subtract the max over
  masked entries; `e = np.where(mask, np.exp(s - m), 0)`; divide by the sum
  where it is > 0. Broadcast the mask with `np.broadcast_to`.
- `fit`: stack `features(ep)` for every episode (M = episodes x frames), the
  mask from `ep.candidates` repeated per frame, and `y = ep.intended`.
- The forward pass, the gradient and Adam all work on arrays shaped
  (B, N, d), (B, N, H) and (B, N).
- Check the gradient once by finite differences while developing (not in
  the file).

## Done when
The accept command passes, plus the full suite and ruff (see AGENTS.md).
