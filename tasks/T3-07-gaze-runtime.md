---
id: T3-07
title: Implicit feedback from where the person looks
status: doing
agent: claude
depends: [T3-05, T3-06]
owns: [curio_robot/runtime/orchestrator.py, curio_robot/runtime/__init__.py, curio_robot/runtime/perception_stage.py, curio_robot/collab/run.py, tests/test_t3_gaze_runtime.py]
reads: [docs/target3-collaboration.md, curio_robot/collab/gaze.py, curio_robot/collab/run.py, curio_robot/runtime/orchestrator.py, curio_robot/runtime/__init__.py, curio_robot/runtime/perception_stage.py, curio_robot/recorder.py, tests/test_t1_runtime.py]
accept: uv run pytest tests/test_t3_gaze_runtime.py tests/test_t1_runtime.py tests/test_t2_runtime.py --strict-impl
---

## Goal
With `RuntimeConfig.implicit_feedback = True`, a steady look at another
candidate while the arm moves re-targets the arm before the person says "no".

## Behaviour
- Each tick with a human pose: `gaze_target(human_pose, live map)` into a
  `GazeWatch(arm_goal=brain.goal_id)`; when it fires, post an event the
  orchestrator handles on its thread.
- The orchestrator re-targets only if the gazed object is a reading of the
  current request (never to an unrelated object), records stage
  "correction" with source "gaze", marks `gaze_redirected`, and learns from
  it as from a correction.
- `collab/run.py`: enable `"view-on+gaze"`; when the arm heads to the wrong
  object, the simulated person turns their head to the goal for a moment
  before saying "no" (so with gaze on, the "no" often isn't needed).
- Off by default: every existing test and condition behaves as before.

## New tests (`tests/test_t3_gaze_runtime.py`, yours)
A gaze at the other reading re-targets; a gaze at a non-candidate does not;
a glance shorter than the dwell does not; with the flag off nothing changes.

## Done when
The accept command passes, plus the full suite and ruff.
