---
id: T1-08
title: Wire conventions, profiles and the recorder into the live runtime
status: todo
agent: claude
depends: [T1-01, T1-02, T1-03, T1-04, T1-05, T1-07]
owns: [curio_robot/runtime/orchestrator.py, curio_robot/runtime/__init__.py, curio_robot/sim/live.py, tests/test_t1_runtime.py]
reads: [docs/target1-conventions.md, curio_robot/conventions/readings.py, curio_robot/conventions/belief.py, curio_robot/conventions/noise.py, curio_robot/conventions/decide.py, curio_robot/conventions/profiles.py, curio_robot/recorder.py, curio_robot/events.py, curio_robot/runtime/orchestrator.py, curio_robot/runtime/__init__.py, curio_robot/sim/live.py, tests/test_runtime.py]
accept: uv run pytest tests/test_t1_runtime.py tests/test_runtime.py tests/test_sim.py --strict-impl
---

## Goal
The orchestrator stops calling `resolve()` directly for targets and uses
readings → `belief.posterior` → `decide`. It learns from outcomes, keeps a
profile per person, and records every decision.

## Behaviour
- **Config**: `RuntimeConfig` gains `person: str = "anonymous"`,
  `profiles_dir: Path | None = None` and `decisions_path: Path | None = None`.
  With no paths, nothing touches disk (tests and the current CLI stay as they are).
- **Partial**: decide on the live map. `act` → speculative go; `hedge` →
  approach the weighted centre; `ask`/`none` → don't move. Record a "partial"
  decision only when its kind or object changes.
- **Final**: `act` → go (tag "belief"); `hedge`/`ask` → stop, say
  `decision.question` (or "Your left or mine?" for a hedged left/right), and
  remember the pending readings. If the next final utterance contains
  mine/my/yours/your, learn with outcome "answered", say a confirmation and go
  to the answered object.
- **Outcomes**: "no, the other one" → learn "corrected" with `p_flip` from
  `noise.flip_probability` on the two framed readings' objects, using the
  speaker viewer. Arriving with no interrupt before the next utterance starts
  (or 2 s, whichever is first) → learn "accepted". Save the profile and write
  `recorder.outcome` after every update.
- **VLM**: unchanged for now (confirm or redirect). Record its reply as stage "vlm".
- Keep `confirmation()` phrasing and every existing test in `tests/test_runtime.py`.

## New tests (`tests/test_t1_runtime.py`, yours to write)
Reuse the FakeMotion / `_brain` pattern from `tests/test_runtime.py`:
1. Fresh belief + "the red one on the left" acts on the speaker's left.
2. "no, the other one" lowers `p_own_frame` and saves the profile (tmp_path).
3. With a belief at 0.5, the final asks "Your left or mine?"; "yours" goes
   to the robot-frame object and moves the belief.
4. With `decisions_path` set, `load_records` shows the decisions with their outcomes joined.

## Done when
The accept command passes, `uv run curio-sim live perspective --seed 3` still
succeeds, plus the full suite and ruff.
