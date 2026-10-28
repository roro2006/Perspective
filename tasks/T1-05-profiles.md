---
id: T1-05
title: Per-person belief profiles on disk
status: done
agent: qwen
depends: [T1-02]
owns: [curio_robot/conventions/profiles.py]
reads: [curio_robot/conventions/belief.py, tests/test_t1_profiles.py]
accept: uv run pytest tests/test_t1_profiles.py --strict-impl
---

## Goal
Implement `ProfileStore` in `curio_robot/conventions/profiles.py` as its
docstrings specify. Depends on T1-02 because the tests round-trip a real
`ConventionBelief`.

## Hints
- Validate names with `re.fullmatch(r"[A-Za-z0-9_-]{1,64}", person)` in every
  method that takes a name, before touching the filesystem.
- Atomic write: `tempfile.mkstemp(dir=root, suffix=".tmp")`, write the JSON,
  then `os.replace(tmp, final)`. No temp files may be left behind.
- `load` never raises for missing or corrupt files: log a warning with
  `logging.getLogger(__name__)` and return `ConventionBelief()`.

## Done when
The accept command passes, plus the full suite and ruff (see AGENTS.md).
