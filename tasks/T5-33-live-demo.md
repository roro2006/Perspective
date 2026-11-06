---
id: T5-33
title: Live demo console, where a typed request goes through the real Curio pipeline in the SO-101 sim, with honest status for the drop-ins
status: doing
agent: claude
budget: 120000
depends: [T5-27]
owns: [curio_robot/sim/world.py, curio_robot/demo/__init__.py, curio_robot/demo/__main__.py, curio_robot/demo/viewer.py, curio_robot/demo/handover.py, curio_robot/demo/voice.py, curio_robot/demo/listen.ps1, curio_robot/demo/session.py, curio_robot/demo/server.py, curio_robot/demo/status.py, curio_robot/demo/static/index.html, tests/test_t5_33_demo.py, docs/demo.md, tasks/T5-33-live-demo.md, tasks/notes/T5-33.md]
reads: [curio_robot/pipeline.py, curio_robot/resolver.py, curio_robot/sim/world.py, curio_robot/sim/scenarios.py, curio_robot/sim/shots.py, curio_robot/sim/ab.py, product/pitch-strategy.md, docs/booth-script.md, tasks/notes/T5-32.md, tasks/T5-31-libero-dropin.md]
accept: uv run pytest -q tests/test_t5_33_demo.py
---

## Why
The pitch needs a live demo that an observer understands in under a minute: a person asks
for "the red one on my left", Curio works out whose left, and the robot goes there. It has
to run the real pipeline, not an animation. The demo machine is a laptop with no grounder
adapter on it, so this version runs Curio's geometry pipeline (`pipeline.step`: detector,
fusion with `seen_by`, perspective, rule resolver) on live MuJoCo renders. The trained-VLM
grounder and the LIBERO drop-in (T5-31) plug in later; until then, the console says so.

## What it does
- One `SimWorld` owned by one worker thread. Commands queue to it, and a second command
  while one runs is refused, so two commands never drive the same arm.
- Scenes: the three `sim.scenarios` kinds by seed. The scene's own request is pre-filled,
  and any typed request is accepted.
- Run: `pipeline.step` on fresh renders, Curio on (robot + head camera) or off (robot only),
  then the arm moves to the chosen block in small IK steps while frames stream.
- Oracle: from the simulator's ground truth (true block positions, the person's true eye
  and gaze, the scenarios' own left/right rule). It is computed only for requests whose
  meaning the rule covers, and reported as unknown otherwise.
- Compare: the same scene and words with Curio off, then on, next to the oracle.
- A/B: `sim.ab.run` measured live on this machine, labelled with its trial count.
- Replay: every live run is recorded (frames and result) under `out/demo/recordings/`,
  and replay plays a recording back under a REPLAY banner. Replay never stands in for a
  live run.
- Status panel: preflight (MuJoCo render, grounder URL if given, GPU), plus the drop-ins as
  recorded in the repo: T5-31 not run, T5-32 FAIL on the port, SmolVLA run 5 gate missed.

## Rules (fixed before building)
1. No hardcoded target selection. The target is whatever `pipeline.step` returns.
2. No confidence, reasoning or numbers the pipeline doesn't produce. The rule resolver
   gives a target and a reason string, and that is what is shown.
3. The arm motion is IK toward the chosen block, and the console labels it
   "scripted motion". It is never called a learned policy.
4. Live and replay look different, and replay is labelled on every frame.
5. The server binds to 127.0.0.1 by default and runs no shell commands from requests.

## Done when
- The accept command passes (CPU: the session runs scenes, on/off, oracle, cancel,
  busy-refusal and replay with a small world; the server answers its routes).
- The full suite, ruff, and tasks.py check pass.
- docs/demo.md has the start command, the architecture and a one-minute walkthrough.
