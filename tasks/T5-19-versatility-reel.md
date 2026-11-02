---
id: T5-19
title: Versatility reel, a second simulation video (nine scenes, two with live grounding)
status: todo
agent: claude
depends: [T5-18]
owns: [scripts/poc/reel_kit.py, scripts/poc/reel_scenes_a.py, scripts/poc/reel_scenes_b.py, scripts/poc/make_reel.py, tests/test_t5_reel.py, tasks/T5-19-versatility-reel.md, tasks/notes/T5-19.md]
reads: [scripts/poc/kit.py, scripts/poc/robots.py, scripts/poc/scene_base.py, scripts/poc/compose.py, scripts/poc/make_video.py, curio_robot/bench/items.py, curio_robot/bench/conditions.py, curio_robot/resolver.py, docs/bringup-7900xt.md]
accept: uv run pytest -q tests/test_t5_reel.py
---

## Goal
"Whatever you mean": a ~75 s 1920x1080 30 fps reel built on the T5-18 kit. A 5 s cold open,
nine 6-7 s scenes (visibility, reach, gaze, pointing, memory, placement, your frame, moving
frame, habits) and a 10 s 3x3 payoff grid. Each scene has the head-camera inset, the spoken
line, a red X on the robot-only beat, a green check on Curio's beat and a caption. The
visibility and ROV scenes ask the real r16 v2 grounder on llama-server :8081 (geometry
condition, logged to out/poc_reel/grounding.json); the habits scene was to ask it for beat 2 if
the habit mechanism can express the request; every other beat is scripted. No on-screen
disclaimers. curio_poc.mp4 must render exactly as before: no T5-18 file changes.

Outputs (main checkout): out/poc_reel/curio_reel.mp4, out/poc_reel/curio_reel_720p.mp4
(< 30 MB), out/poc_reel/scene_<name>.mp4, out/poc_reel/grounding.json.
