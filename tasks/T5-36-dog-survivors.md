---
id: T5-36
title: Survivor variety in the dog demo's buildings (how many, what they wear, how they lie or sit)
status: doing
agent: claude
budget: 80000
depends: [T5-35]
owns: [curio_robot/demo/dog_scenes.py, curio_robot/demo/dog.py, scripts/dog_survivors_gate.py, tests/test_t5_36_dog_survivors.py, docs/demo.md, tasks/T5-36-dog-survivors.md, tasks/notes/T5-36.md]
reads: [tasks/T5-35-dog-scenes.md, tasks/notes/T5-35.md, curio_robot/demo/dog_scenes.py, curio_robot/demo/dog.py, scripts/poc/kit.py, scripts/dog_scenes_gate.py]
accept: uv run pytest -q tests/test_t5_36_dog_survivors.py
---

## Why
The user asked for variation in how many survivors there are and what they are wearing
(2026-11-06). They chose survivor variety over other missions (sampling, inspection).

## What it does
For the generated buildings (corridor, doorway, closest), each seed also draws its survivors
from a separate random stream, so rooms and requests stay exactly as in T5-35:
- 1-3 survivors in the meant room (p = 0.40 / 0.35 / 0.25), and with p = 0.35 one more in
  another room.
- Outfits: red jacket, navy coat, green hoodie, denim, orange overalls, white lab coat,
  hi-vis vest (stripes), hard hat with hi-vis (helmet). Skin tones drawn from five.
- Poses at fixed spots chosen to stay off the dog's route: at the back of the room lying on
  their back or lying waving one arm; in the front corners sitting against the wall with legs
  out, or sitting waving.
- When the meaning changes (`turn`), the survivors move to the newly meant room.
- On arrival the dog reports how many survivors are in the room it reached, counted from the
  simulator ("Found two survivors" / "Nobody here"), and the overlay lists who is where.
The rescue scene (T5-34) is unchanged. Survivors are scenery: Curio still chooses the room
from the floor map and never sees them.

## GATE (written before building; not to be changed afterwards)
Run once by `scripts/dog_survivors_gate.py`, results to out/t5_36/gate.json and gate.md. For
each kind (corridor, doorway, closest):
- **U (unchanged):** seeds 0-49, each layout's request, rooms and both decisions (Curio on and
  off) equal T5-35's recorded rows in out/t5_35/gate.json: 50 of 50.
- **V (variety):** over seeds 0-49, all three counts 1, 2 and 3 occur, at least 5 outfits
  occur, and all four poses occur.
- **Q (walking with survivors):** seeds 0-9, walk to every room: >= 95% within 0.35 m, 0 falls,
  0 contacts between the dog and any survivor.
- **E (end to end):** seeds 0-9, Curio on, as faced and turned round (20 runs): >= 19 of 20
  end in the meant room, the meant room holds 1-3 survivors, 0 falls, 0 survivor contacts.
A FAIL is acceptable and is reported as such.

## Done when
- The accept command passes; the full suite, ruff and tasks.py check pass.
- docs/demo.md mentions the survivor variety.
