---
id: T5-01
title: Benchmark items with paired views
status: todo
agent: qwen
depends: []
owns: [curio_robot/bench/items.py]
reads: [docs/target5-perspective-policy.md, curio_robot/bench/items.py, curio_robot/collab/tasks.py, curio_robot/perspective.py, curio_robot/types.py, tests/test_t5_bench.py]
accept: uv run pytest tests/test_t5_bench.py -k items --strict-impl
---

## Goal
Implement make_item, generate, save and load in curio_robot/bench/items.py
exactly as the docstrings say. Keep the constants and BenchItem.

## Hints
- Renders: world.render_camera(HUMAN_CAM) and
  world.render_camera(ROBOT_CAM, segmentation=True), names from curio_robot.types.
  Drop the segmentation with dataclasses.replace(frame, segmentation=None).
- Visible pixels: int((robot.segmentation == world._block_geoms[i]).sum()).
- PNG via PIL.Image.fromarray(rgb).save(path); load with
  np.asarray(Image.open(p).convert("RGB")).
- Follow the rng draw order exactly so the determinism test passes.

## Done when
The accept command passes, plus the full suite and ruff (see AGENTS.md).
