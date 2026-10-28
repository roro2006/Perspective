---
id: T2-09
title: LoRA training script (runs on the 7900 XT in WSL2)
status: todo
agent: claude
depends: [T2-07]
owns: [scripts/train_lora.py, tests/test_t2_train.py, pyproject.toml]
reads: [docs/target2-grounding.md, curio_robot/sim/grounding_data.py, curio_robot/grounding/prompt.py, tasks/README.md]
accept: uv run pytest tests/test_t2_train.py --strict-impl
---

## Goal
`scripts/train_lora.py --data out/grounding/data.jsonl --model <qwen-vl-4b>
--out out/lora/<name>` fine-tunes a LoRA adapter on the grounding dataset:
the user message (image + prompt) in, the single answer letter out, loss on
the answer token only.

## Requirements
- Plain PyTorch + transformers + peft, so the same script runs on the 7900 XT
  (WSL2 + ROCm), a rented GPU or the H100s. Add them as an optional
  dependency group `train` in `pyproject.toml`; never import them at module
  top level (the dry run and CI don't have them).
- `--dry-run`: load the JSONL with `grounding_data.load_dataset`, check every
  image exists, print counts by kind and the answer-letter distribution, and
  exit 0 without importing torch. This is what the tests cover.
- Defaults sized for 20 GB: bf16, gradient checkpointing, batch 1 with
  gradient accumulation, LoRA r=16 on attention projections, images already
  ≤ 448 px from the generator.
- Hold out 10% by id for eval and save `eval_ids.json` next to the adapter.
- Write `train_config.json` (model, data hash, PROMPT_VERSION, hyperparameters)
  into the output dir so evaluation can attribute results.

## New tests (`tests/test_t2_train.py`, yours to write)
Generate a tiny dataset with `write_dataset(tmp_path, n=6)`, run the script
with `--dry-run` via `subprocess` and check the exit code and the printed
counts; a missing image makes the dry run fail with a clear message.

## Done when
The accept command passes, plus the full suite and ruff. A real training run
on the 7900 XT is part of T2-10.
