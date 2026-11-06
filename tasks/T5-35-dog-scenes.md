---
id: T5-35
title: Seeded building layouts for the live dog demo (corridor, doorway, closest), like the arm's scenario kinds
status: doing
agent: claude
budget: 120000
depends: [T5-34]
owns: [curio_robot/demo/dog_scenes.py, curio_robot/demo/dog.py, curio_robot/demo/viewer.py, curio_robot/demo/voice.py, scripts/dog_scenes_gate.py, tests/test_t5_35_dog_scenes.py, tests/test_t5_34_dog_demo.py, docs/demo.md, tasks/T5-35-dog-scenes.md, tasks/notes/T5-35.md]
reads: [tasks/T5-34-dog-live-demo.md, tasks/notes/T5-34.md, curio_robot/demo/dog.py, curio_robot/resolver.py, curio_robot/sim/scenarios.py, scripts/poc/phys_scenes.py, scripts/poc/scenes_a.py, scripts/poc/kit.py, scripts/poc/phys_core.py, scripts/dog_demo_gate.py]
accept: uv run pytest -q tests/test_t5_35_dog_scenes.py
---

## Why
The user wants the dog demo to have several different scenes (2026-11-06), the way the arm
has scenario kinds and seeds. T5-34 had one fixed corridor.

## What it does
`dog_scenes.py` generates a building from (kind, seed): a straight corridor (11-14 m) with
rooms on both sides in 3 m slots, each room's door marked with a coloured frame and floor
stripe (red, blue, green, yellow). The dog enters at the east end facing west; smoke sits
between it and the firefighter. Routes go along a corridor lane clear of the firefighter, then
through the door as in T5-20 (door at x, points x+0.8 / x, 0.6 / x, 1.7 / goal x+0.25, 2.3).
- **corridor**: the firefighter at the west end facing the dog. Two same-colour rooms face each
  other across the corridor, plus 1-3 other-colour rooms. "The <c> room on my <left|right>".
- **doorway**: the firefighter stands against one corridor wall facing across it. Two
  same-colour rooms on the far wall, one each side of them along the corridor, plus others.
  "The <c> room on my <left|right>": their left is along the corridor; the dog's left is a wall.
- **closest**: 2-3 same-colour rooms spread along the corridor, the firefighter at the west
  end. "The <c> room closest to me" / "furthest from me".
- **rescue**: T5-34's scene, unchanged (its gate stands).
Truth is the simulator's: the speaker's (or "your": the dog's) true position and body yaw,
with the resolver's own relations, over rooms of the named colour. The survivor is placed in
the true room. Viewer: `scene KIND SEED` and `n` (next showcase scene), as the arm.

## GATE (written before building; not to be changed afterwards)
Run once by `scripts/dog_scenes_gate.py`, results to out/t5_35/gate.json and gate.md.
For each new kind (corridor, doorway, closest) separately:
- **Q (walking):** seeds 0-9, the dog walks to every room in the layout from the start pose:
  within 0.35 m of the room goal on >= 95% of walks, 0 falls. A kind that fails Q is not
  offered in the viewer.
- **D (decision):** seeds 0-49: Curio on matches the truth on >= 49 of 50. Curio off is
  reported as it comes.
- **E (end to end):** seeds 0-9, Curio on: the dog ends within 0.5 m of the true room's goal
  and nearer to it than to any other room's goal on >= 9 of 10, 0 falls.
Showcase seeds are then picked by a rule fixed here: per kind, the two lowest seeds in 0-49
where Curio off is wrong, Curio on is correct, and (if in 0-9) E succeeded; for seeds above 9
one end-to-end walk is run to confirm before listing. A FAIL is acceptable.

## Rules
As T5-34: the target is whatever `resolver.resolve` returns on the map and poses; the motion
is our path follower on the cagataydev walking policy; the rooms come from the floor map.

## Done when
- The accept command passes (CPU; physics tests skip without the Go2 assets).
- The full suite, ruff, and tasks.py check pass. docs/demo.md lists the dog scenes.
