---
id: T5-27
title: Curio as a plug-in layer, one goal interface for any robot policy
status: done
agent: claude
budget: 100000
depends: [T5-26]
owns: [curio_robot/layer/__init__.py, curio_robot/layer/goal.py, curio_robot/layer/resolve.py, curio_robot/layer/adapters.py, curio_robot/layer/robot_words.py, scripts/layer_eval.py, tests/test_t5_27_layer.py, docs/curio-layer.md, tasks/T5-27-curio-layer.md, tasks/notes/T5-27.md]
reads: [curio_robot/grounding/client.py, curio_robot/grounding/marks.py, curio_robot/grounding/prompt.py, curio_robot/clarify.py, curio_robot/belief.py, curio_robot/perspective.py, curio_robot/types.py, curio_robot/pipeline.py, curio_robot/flywheel/registry.json, curio_robot/bench/items.py, curio_robot/bench/conditions.py, curio_robot/bench/score.py, scripts/smolvla_eval.py, scripts/poc/phys_run.py, scripts/poc/phys_robots.py, docs/bringup-7900xt.md]
accept: uv run pytest -q tests/test_t5_27_layer.py
---

## Goal
Big robotics labs build general robot policies. Curio's job is narrower: it tells
any policy *what the person means*. This card turns the pieces built so far into one
interface that other policies can consume:

    resolve(utterance, head_frame, robot_frames, object_map, speaker) -> Goal

`Goal` (frozen dataclass, `layer/goal.py`) holds:
- `target_id`;
- `target_xyz_robot`, the target's position in the robot base frame;
- `frame`, meaning "speaker" or "robot";
- `confidence`, the grounder's probability for the chosen mark;
- `alternatives`, the ids and probabilities of the other marks;
- `ask`, the T5-22 question to ask when the layer should ask instead of act, or None;
- `note`, the T5-21 moved-it note, or None;
- `adapter`, the grounder adapter used, read from `flywheel/registry.json`.

`resolve` wraps the existing pieces and changes none of them: grounding (v4, the
geometry condition), clarify (ask-back), and belief (memory wording).

## Adapters (`layer/adapters.py`): a Goal in each policy's input format
- **ring**: the target marker image the SmolVLA policies were trained on. It must
  reproduce `smolvla_eval`'s marker images pixel for pixel, for the same scene and
  target.
- **pose**: `target_xyz_robot` as a goal for motion-level controllers (T5-20's G1
  and Go2 walkers, AMR, drone, ROV, flyer). It must give the same goals the
  T5-20 scene code computes today.
- **words** (`layer/robot_words.py`): a robot-frame instruction that is unambiguous
  without the head camera, for any language-conditioned policy (pi0 in T5-28). The
  rules are deterministic, built from the object map in the robot frame. Use colour
  plus the relation that singles the target out from its look-alike in the robot's
  own frame, for example "pick up the red block on your left". Use "closest to you" /
  "furthest from you" when left/right doesn't separate them. Never mention the
  speaker.

## GATE (written before any evaluation; not to be changed afterwards)
Grounder: the production adapter (r16-v4) on :8081. The layer must run against the
live server without starting its own.
1. Equivalence, CPU only. On every bench5c item and every T5-20 live scene, the
   layer's target, ring image and pose goal equal what the existing code produces:
   10/10 scenes and all items. Any mismatch fails.
2. Words are unambiguous, CPU only. On every bench5c item, the robot-words rule
   checker resolves the instruction back to the target alone: >= 0.99.
3. Words are understood by a generic reader. The v4 grounder, run in the robot-view
   condition with the robot-words instruction (no head camera), picks the target on
   >= 0.95 of bench5c items. As a comparison, the same reader with the human's
   original words scores whatever it scores; report it, it isn't gated.
4. Latency: p95 of `resolve` on bench5c <= 250 ms, measured on the live server.
Report a 95% Wilson CI for each rate. Results go to out/t5_27/gate.json and gate.md.

## Docs
`docs/curio-layer.md`: one page covering the interface, the three adapters, an
example call, and what each field means. It should be written for an engineer at
another robotics company.

## Done when
The accept command passes, along with the full suite, ruff on curio_robot, tests
and scripts, and tasks.py check. The gate has run once, and the notes are in
tasks/notes/T5-27.md.
