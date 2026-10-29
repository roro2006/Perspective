---
id: T3-06
title: Run a multi-step task live and record it as a Session
status: doing
agent: claude
depends: [T3-01, T3-03, T3-04]
owns: [curio_robot/collab/run.py, tests/test_t3_run.py]
reads: [docs/target3-collaboration.md, curio_robot/collab/session.py, curio_robot/collab/worker.py, curio_robot/collab/tasks.py, curio_robot/sim/learn.py, curio_robot/sim/live.py, curio_robot/runtime/__init__.py, curio_robot/runtime/trace.py, curio_robot/runtime/orchestrator.py]
accept: uv run pytest tests/test_t3_run.py tests/test_t3_session.py --strict-impl
---

## Goal
`curio_robot/collab/run.py`: one simulated person does a whole fetch task
with the live runtime, and the run comes back as a `Session`.

## Behaviour
- `CONDITIONS = ("view-off", "view-on", "view-on+gaze")`.
- `run_task(user, condition, seed, *, n_goals=4, p_deictic=0.0,
  occlude=False, world=None, profiles_dir=None, grounder_url=None,
  quiet=True) -> Session`: `fetch_scene` + `apply_scene`, `plan_task` on
  `true_map`, then one `Runtime` for the whole task (so the belief and the
  person's profile carry over), `use_human_view = condition != "view-off"`.
- The person says each goal in turn and reacts like `sim.learn._Reactor`:
  answers "your left or mine?", says "no, the other one" when the arm commits
  to the wrong object, otherwise waits for arrival and moves on. For a
  deictic goal the person looks at the goal while speaking
  (`world.set_human(eye, goal position)` under `world.lock`).
- Map live ids to goals by nearest position (the tracker names objects its
  own way). A goal is done when the arm arrives at it and the person accepts.
- One `Step` per thing said, `marks = rt.tracer.marks(utterance)`,
  `asked` from the robot's lines. `session_id = f"{user.name}-{condition}-{seed}"`.
- `"view-on+gaze"` raises `NotImplementedError("T3-07")` until T3-07.
- A per-task timeout ends a stuck task rather than hanging (record what got done).

## New tests (`tests/test_t3_run.py`, yours)
Short tasks (2 goals): a session comes back with the right goals and one
step per utterance; view-off vs view-on both run; the profile carries over
between goals (a robot-frame user is corrected at most once per side).

## Done when
The accept command passes, plus the full suite and ruff.
