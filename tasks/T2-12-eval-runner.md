---
id: T2-12
title: `curio-sim ground-eval`: score a grounder on a dataset and gate it
status: todo
agent: any
depends: [T2-05, T2-07, T2-08]
owns: [curio_robot/sim/ground_eval.py, curio_robot/sim/cli.py, tests/test_t2_ground_eval.py]
reads: [docs/target2-grounding.md, curio_robot/grounding/client.py, curio_robot/grounding/evaluate.py, curio_robot/grounding/prompt.py, curio_robot/sim/grounding_data.py, curio_robot/sim/cli.py]
accept: uv run pytest tests/test_t2_ground_eval.py --strict-impl
---

## Goal
`uv run curio-sim ground-eval --data out/grounding/data.jsonl --url <vllm>
[--ids eval_ids.json] [--current out/eval/current.json] [--max-p95-ms 800]
--out out/eval/<name>.json` rebuilds each request from the JSONL (image path →
data URL), scores it with `VllmGrounder`, runs `evaluate`, prints overall and
per-kind accuracy and p50/p95, and with `--current` prints the gate verdict
(exit code 1 if it fails).

## Hints
- Put the logic in `curio_robot/sim/ground_eval.py` with a function that
  takes any object with `.score(request)`, so tests pass a fake grounder; the
  CLI just parses arguments and calls it.
- Rebuild the request from the stored messages: swap the relative image path
  back to a base64 PNG data URL; drop the assistant message.

## New tests (`tests/test_t2_ground_eval.py`, yours to write)
A tiny dataset from `write_dataset(tmp_path, n=6)`, a fake grounder that
answers correctly except for one kind, check the per-kind numbers, the
written JSON and that a worse candidate fails the gate.

## Done when
The accept command passes, plus the full suite and ruff.
