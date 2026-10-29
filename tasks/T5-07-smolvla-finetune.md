---
id: T5-07
title: Fine-tune SmolVLA on target-marked pick demonstrations
status: todo
agent: claude
depends: [T5-05]
owns: [curio_robot/sim/demos.py, scripts/smolvla_dataset.py, scripts/smolvla_train.py, docs/target5-perspective-policy.md, tests/test_t5_demos.py]
reads: [docs/target5-perspective-policy.md, docs/bringup-7900xt.md, curio_robot/sim/demos.py, curio_robot/sim/physics.py, curio_robot/sim/expert.py]
accept: uv run pytest -q
---

## Goal
SmolVLA picks the block the grounding layer points at. The target reaches the
policy as a visual prompt: a ring drawn on the robot camera image at the target
pixel (camera1). The words stay as the task text.

## Behaviour
- demos.py: option to draw the target marker on the robot image at record time
  (or at conversion; say which), radius and colour as named constants.
- scripts/smolvla_dataset.py (runs in the lerobot venv): convert a recorded
  demos folder into a local LeRobotDataset (images, not video; no hub push).
  Features: observation.state (6), action (6), observation.images.camera1
  (robot view with marker), observation.images.camera2 (person view).
- scripts/smolvla_train.py or a documented lerobot-train command: fine-tune
  from the local smolvla_base snapshot with load_vlm_weights=False,
  HF_HUB_OFFLINE=1, wandb off, on the 7900 XT. Smoke run first (about 100
  steps) to measure step time and memory, then the full run in the
  background with a log and periodic checkpoints under out/smolvla/.
- Approved downloads only: nothing beyond lerobot/smolvla_base and the
  SmolVLM2 config/tokenizer already in the HF cache. If anything tries to
  fetch more, stop and report.

## Done when
Dataset built (report episodes, frames, size), smoke run works, the full run
started with its log path, step time, memory and expected duration recorded in
the design doc and tasks/notes/T5-07.md.
