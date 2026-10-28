---
id: T2-03
title: Set-of-marks placement and rendering
status: todo
agent: qwen
depends: []
owns: [curio_robot/grounding/marks.py]
reads: [docs/target2-grounding.md, curio_robot/geometry.py, curio_robot/types.py, tests/scene.py, tests/conftest.py, tests/test_t2_marks.py]
accept: uv run pytest tests/test_t2_marks.py --strict-impl
---

## Goal
Implement `place_marks` and `render_marks` in `curio_robot/grounding/marks.py`
exactly as the docstrings specify. Keep `Mark`, `Crop` and the constants.

## Hints
- `geometry.project(position, frame.intrinsics, frame.pose)` gives (u, v) or
  None. Keep only 0 <= u < width and 0 <= v < height.
- Sort objects by `object_id` *before* assigning letters, and stop at 26.
- Draw on a copy: `Image.fromarray(frame.rgb.copy())`, then
  `ImageDraw.Draw(img).ellipse((u-10, v-10, u+10, v+10), fill="white", outline="black")`
  and `draw.text((u, v), letter, fill="black", anchor="mm")` (PIL's default font is fine).
- Crop with `img.crop((x0, y0, x1, y1))`; resize only when scale < 1, with
  `Image.BILINEAR`. Return `np.asarray(img, dtype=np.uint8)`.
- Use `math.floor` / `math.ceil` for the crop box, exactly as specified.

## Done when
The accept command passes, plus the full suite and ruff (see AGENTS.md).
