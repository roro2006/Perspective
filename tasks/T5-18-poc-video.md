---
id: T5-18
title: Simulation-only proof-of-concept video (six scenes, real zero-shot grounding)
status: done
agent: claude
depends: [T5-17]
owns: [scripts/poc/kit.py, scripts/poc/robots.py, scripts/poc/scene_base.py, scripts/poc/scenes_a.py, scripts/poc/scenes_b.py, scripts/poc/compose.py, scripts/poc/make_video.py, scripts/poc/encode_cv2.py, tests/test_t5_poc.py, tasks/T5-18-poc-video.md, tasks/notes/T5-18.md]
reads: [curio_robot/grounding/prompt.py, curio_robot/grounding/marks.py, curio_robot/grounding/client.py, curio_robot/bench/conditions.py, curio_robot/types.py, curio_robot/perspective.py, docs/bringup-7900xt.md]
accept: uv run pytest -q tests/test_t5_poc.py
---

## Goal
A ~50 s 1920x1080 video made only in MuJoCo from hand-written MJCF primitives: an intro,
a face-to-face explainer, six scenes (warehouse, rescue dog, drone, home humanoid, ROV,
space free-flyer) and an end grid. In each scene the object map is built from the person's
head pose (the geometry condition) and sent to Qwen3-VL-4B Q8_0 with the r16 v2 adapter on
llama-server; the robot goes to whatever the grounder chose (min-jerk, scripted). The
warehouse shows the robot-frame reading first (wrong) and a correction. Every clip carries
the label "Simulation · real perspective grounding (zero-shot) · scripted robot motion".
Wrong picks stay in the video and are reported. No downloads, no retraining, no product edits.

Outputs (main checkout): out/poc/curio_poc.mp4, out/poc/scene_<name>.mp4,
out/poc/grounding.json.
