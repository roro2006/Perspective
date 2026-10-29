---
id: T5-06
title: Retrain the grounding LoRA on benchmark-style data and score phase 2
status: todo
agent: claude
depends: [T5-03]
owns: [scripts/train_lora.py, product/results.md]
reads: [docs/target5-perspective-policy.md, curio_robot/bench/score.py, docs/bringup-7900xt.md]
accept: uv run pytest -q
---

GPU work, after the Qwen waves (they share the 7900 XT). Generate training
items from seeds disjoint from the benchmark seed, retrain, convert and serve as
in docs/bringup-7900xt.md, score base and retrained on every condition, and
write the phase 2 gate result.
