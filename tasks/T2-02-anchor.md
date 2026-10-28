---
id: T2-02
title: Word timings and the anchor moment
status: done
agent: qwen
depends: []
owns: [curio_robot/grounding/anchor.py]
reads: [docs/target2-grounding.md, curio_robot/resolver.py, tests/test_t2_anchor.py]
accept: uv run pytest tests/test_t2_anchor.py --strict-impl
---

## Goal
Implement `parse_words` and `anchor_time` in `curio_robot/grounding/anchor.py`
exactly as the docstrings specify. Keep `Word` and `DEICTIC` unchanged.

## Hints
- Colour words are `resolver.COLORS`; import it, don't copy it.
- Normalize with `re.sub(r"[^a-z]", "", text.lower())` so " That," matches "that".
- Keep `Word.text` as transcribed (leading spaces and punctuation included);
  only skip words that are empty after `.strip()`.
- "Last" means the last in the list, which is the latest in time.

## Done when
The accept command passes, plus the full suite and ruff (see AGENTS.md).
