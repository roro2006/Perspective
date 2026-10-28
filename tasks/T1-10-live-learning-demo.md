---
id: T1-10
title: Live learning demo in the full runtime
status: todo
agent: claude
depends: [T1-08, T1-09]
owns: [curio_robot/sim/learn.py, curio_robot/sim/cli.py, curio_robot/sim/live.py, tests/test_t1_live_learning.py]
reads: [docs/target1-conventions.md, curio_robot/sim/learn.py, curio_robot/sim/live.py, curio_robot/sim/users.py, curio_robot/voice/scripted.py, curio_robot/runtime/__init__.py, curio_robot/runtime/orchestrator.py]
accept: uv run pytest tests/test_t1_live_learning.py --strict-impl
---

## Goal
`curio-sim learn --live --user robot-frame --requests 6` runs the real
runtime (rendering, motion, scripted voice) for one simulated person. The
person speaks marker-free requests, reacts with "no, the other one" when the
arm heads for the wrong block, and answers "yours"/"mine" when asked. Print
the per-request transcript, the belief after each request and the waterfall.
This is the stage demo of "it learns from you".

## Notes
- `ScriptedVoice` plays a fixed list of lines; the person here has to react to
  what the robot does. Add a reactive voice source (in `sim/live.py`) that
  watches `ArmEvent`s / spoken lines and decides what to say next.
- The scene changes between requests (new block pair), so reset
  blocks via the scenario functions between utterances, with the same world.
- Keep it under ~60 s for 6 requests on the dev laptop.

## Done when
A smoke test with 3 requests passes (the robot-frame person is corrected at
least once and `p_own_frame` ends below its prior), plus the full suite and ruff.
