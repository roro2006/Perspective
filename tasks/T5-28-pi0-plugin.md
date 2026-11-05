---
id: T5-28
title: Curio drives pi0, a foundation policy we didn't build
status: todo
agent: claude
budget: 120000
depends: [T5-27]
owns: [scripts/pi0_data.py, scripts/pi0_train.py, scripts/pi0_eval.py, tests/test_t5_28_pi0.py, tasks/T5-28-pi0-plugin.md, tasks/notes/T5-28.md]
reads: [tasks/T5-27-curio-layer.md, scripts/smolvla_dataset.py, scripts/smolvla_train.py, scripts/smolvla_eval.py, tasks/notes/T5-17.md, docs/bringup-7900xt.md]
accept: uv run pytest -q tests/test_t5_28_pi0.py
---

## Goal
Show that the Curio layer plugs into a general robot policy from another lab:
Physical Intelligence's pi0 (`lerobot/pi0_base`, run through LeRobot 0.6 in the
lerobot venv). pi0 can't move our sim SO-101 out of the box, so it gets a short
fine-tune on our own demos. Its task text is the layer's **robot-words**
instruction, so pi0 only ever learns robot-frame language and never sees the head
camera or a ring.

Then the person speaks from their own side ("the red one on my left"):
- **(A) raw**: pi0 gets the person's words as said.
- **(B) Curio**: the layer grounds the words on the head camera (v4) and gives pi0
  its robot-words instruction.
- **(O) oracle**: pi0 gets the robot-words instruction built from the true target.

## Requirements
- Access to the gated PaliGemma tokenizer (`google/paligemma-3b-pt-224`). The user
  accepts Google's terms and logs in to Hugging Face themselves. Until then the card
  is blocked; no agent asks for or handles a token.
- Data: the run-5 demos (out/demos800b), relabelled with robot-words task text from
  each episode's true target. Same cameras, minus the ring overlay.
- Training: `train_expert_only=True` (the VLM stays frozen; only the action expert
  and projections train), bf16, sized to fit 20 GB alongside nothing else. Stop the
  :8081 server for training, and use the GPU lock. Budget: at most 8 h of training,
  one run. As in earlier cards, one rerun of the identical command is allowed after
  a transient crash, before any test score is seen.

## GATE (written before any training or evaluation; not to be changed afterwards)
Evaluation: 100 fresh scenes, seeds 51000-51099, built the way smolvla_eval builds
its trials, with the person's words in a frame where the robot's own reading
picks the partner. An outcome is "target" when the target block is lifted.
1. Curio (B) target rate >= raw (A) target rate + 0.25.
2. Curio (B) target rate >= 0.85 x oracle (O) target rate. In words, the layer
   costs pi0 at most 15% relative to being told the true answer.
3. Oracle (O) target rate >= 0.30. This shows the fine-tuned pi0 can do the task
   at all. If line 3 fails, lines 1 and 2 are reported but the card is a FAIL on
   pi0's competence, not on Curio's.
Report 95% Wilson CIs for each rate and paired bootstrap CIs for the differences.
Report SmolVLA run 5 on the same seeds for context. Results go to
out/t5_28/gate.json and gate.md.

## Done when
The accept command passes (CPU tests for data relabelling and the eval harness,
using a fake policy), along with the full suite, ruff, and tasks.py check. The gate
has run once, and the notes are in tasks/notes/T5-28.md, credited to pi0's authors
(Physical Intelligence) and licensed under the Gemma terms.
