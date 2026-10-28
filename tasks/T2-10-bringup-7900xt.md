---
id: T2-10
title: Serving bring-up and latency on the RX 7900 XT
status: todo
agent: claude
depends: [T2-05]
owns: [docs/bringup-7900xt.md, scripts/bench_grounding.py]
reads: [docs/target2-grounding.md, tasks/README.md, curio_robot/grounding/client.py, curio_robot/grounding/anchor.py, curio_robot/voice/stt.py]
accept: manual on the 7900 XT machine, see the card
---

## Goal
Get the target-2 stack running on the 7900 XT and replace every
**[VERIFY]** in `docs/target2-grounding.md` with a measured answer. Needs the
person at the machine for installs; run it there, not on the laptop.

## Steps
1. WSL2 + ROCm + vLLM. Serve a ~4B Qwen-VL (and the 8B for comparison).
   Record exact versions and the serve command in `docs/bringup-7900xt.md`.
2. Confirm the guided-choice parameter name for this vLLM build
   (`VllmGrounder(choice_param=...)`), that capital letters are single tokens,
   and that `top_logprobs` come back.
3. `scripts/bench_grounding.py --url ... --n 50`: renders sim examples with
   `grounding_data.make_example`, calls `VllmGrounder.score`, prints p50/p95
   latency and accuracy. Run it alone, then with whisper-server, Kokoro and
   the detector loaded at the same time (GPU contention).
4. Check whisper-server's word timestamps against `anchor.parse_words`; if the
   format differs, write down the exact JSON and open a contract task.
5. One LoRA training step with `scripts/train_lora.py` (after T2-09) at
   batch 1: peak VRAM and step time. Decide 4B vs 8B for training.

## Done when
`docs/bringup-7900xt.md` has versions, commands, the latency table (alone and
under contention), the choice-param name, the word-timestamp format and the
training-step numbers, and the bench script is committed.
