# Instructions for coding agents

Read this whole file before touching anything. It applies to every agent:
Claude sessions and local Qwen agents alike.

## The project in five lines
Curio fuses a person's first-person camera with a robot's cameras so the robot
understands "on my left" from the person's side of the table. Python 3.12,
managed with `uv`. Product code is in `curio_robot/`, tests in `tests/`, a
MuJoCo sim stands in for hardware. Current work is **target 1**: the robot
learns which "left" each person means from their corrections
(`docs/target1-conventions.md`).

## How work is split
Work is broken into task cards in `tasks/` (e.g. `tasks/T1-01-readings.md`).
You work on exactly one card. The card's frontmatter is the contract:

- `owns`: the only files you may create or edit.
- `reads`: what you need to understand. Don't go exploring beyond it.
- `accept`: the command that must pass when you're done.
- `depends`: cards that must be finished first (you'll be told if they aren't).

Get everything for your card in one file (recommended for local models):

    uv run python scripts/tasks.py pack T1-01      # writes out/context/T1-01.md

## Rules
1. **Edit only files listed in `owns`.** If you need a change elsewhere, stop
   and say so in your notes file instead.
2. **Contracts are frozen.** Dataclasses, function names, signatures and
   docstring specs in stub modules were agreed so tasks can run in parallel.
   Implement them exactly. Don't rename, reorder fields or add required params.
3. **Tests are read-only.** Acceptance tests (`tests/test_t1_*.py`) define done.
   If a test looks wrong, don't edit it: explain why in your notes and stop.
   (A card may own a *new* test file; then you write it.)
4. **No new dependencies** unless your card says so.
5. **Match the surrounding code**: short module docstrings that explain why,
   type hints, `from __future__ import annotations`, numpy for maths, no
   clever metaprogramming. Line length 100.
6. **Before you finish**, all three must pass:

       <the card's accept command>
       uv run pytest -q
       uv run ruff check curio_robot tests

7. **Hand off** by writing `tasks/notes/<ID>.md` (always owned by you): what you
   did, any decision the spec left open, anything that worried you. Five to
   fifteen lines. Don't edit the card's `status`; the coordinator does that.
8. **Git**: work on branch `task/<ID>` (lowercase, e.g. `task/t1-01`), one
   or a few commits, message `T1-01: <what>`. Don't push or merge unless told.

## Test switch
Stubs raise `NotImplementedError("T1-xx")` and their tests are *skipped* by
default so the suite stays green. Add `--strict-impl` to see real failures:

    uv run pytest tests/test_t1_readings.py --strict-impl

## Context discipline (local models have 32-64k tokens)
- Don't read `product/`, `curio_robot/assets/`, `uv.lock`, or `out/` unless
  your card lists them under `reads`.
- Don't print whole large files when a function is enough; use grep/sed ranges.
- Don't paste long test output back into context; rerun with `-x -q` instead.

## Shared test scene
`tests/scene.py` and `tests/conftest.py`: a robot at x=-0.2 facing +x (its left
is +y) and a person at x=0.7 facing -x (their left is -y). Fixture `pair_map`
has `red_1` at y=+0.08 (robot's left) and `red_2` at y=-0.08 (person's left).
