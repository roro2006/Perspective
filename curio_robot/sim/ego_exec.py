"""One teach-by-doing episode end to end (task T5-23).

1. The person demonstrates on the demo layout; only head-camera RGB-D frames (with a noisy
   head pose) reach the extractor, which streams them, so no video is kept in memory.
2. The blocks are reset to a new layout. The robot looks with its own camera (same colour and
   depth detector), names what it sees the way the plan names objects (colour_k from the
   person's left), and for each step works out a goal point from the step's words, read in
   the person's frame (Curio) or in its own (the baseline), then picks with the scripted
   expert and puts the block down with place(). Perceived blocks are matched to the nearest
   simulated block for the expert, which plans its grasp from the true pose (as in T5-04).
3. Arm home, 0.5 s settle, and the goal is checked on the true block positions."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from curio_robot.ego_plan import (
    ROBOT_FRAME,
    START_MAX_Z,
    Blob,
    Extractor,
    PersonFrame,
    Plan,
    detect,
    goal_for,
    name_objects,
)
from curio_robot.sim.ego_tasks import Episode, satisfied
from curio_robot.sim.ego_world import Demonstrator, EgoWorld, true_poses
from curio_robot.sim.expert import pick
from curio_robot.sim.place import go_home, place
from curio_robot.types import ROBOT_CAM, CameraPose

MERGE = 0.03  # m
ROBOT_FULL = 200  # points: the robot camera sees the whole block
HEAD_FRAMES = 8
MATCH = 0.03  # m from a perceived block to the simulated one it is
PICK_TRIES = 2
SETTLE = 0.5  # s


@dataclass(slots=True)
class Demo:
    plan: Plan
    true_frame: PersonFrame  # from the exact head poses: the frame the goal is checked in
    frames: int
    seconds: float


@dataclass(slots=True)
class Run:
    success: bool
    reason: str = ""
    steps_done: int = 0
    final: list[np.ndarray] = field(default_factory=list)


def demonstrate(world: EgoWorld, ep: Episode, on_frame=None) -> Demo:
    world.reset(ep.scene(new=False))
    rng = np.random.default_rng([7, ep.seed, len(ep.family)])
    ex = Extractor()
    poses: list[CameraPose] = []

    def frame(w: EgoWorld) -> None:
        f = w.ego_frame(rng)
        poses.append(true_poses(w))
        ex.add(f)
        if on_frame is not None:
            on_frame(w, f)

    Demonstrator(world, ep.seat, rng, frame).run(list(ep.moves))
    world.held = None
    return Demo(ex.plan(), PersonFrame.from_poses(poses), len(poses), world.time)


def _on_table(f) -> list[Blob]:
    return [b for b in detect(f)[0] if b.position[2] <= START_MAX_Z]


def _same(a: Blob, b: Blob) -> bool:
    return a.color == b.color and float(np.linalg.norm(a.position - b.position)) < MERGE


def perceive(world: EgoWorld, frame: PersonFrame, rng: np.random.Generator
             ) -> dict[str, np.ndarray]:
    """The new layout from the robot camera and the head camera together (the parked arm
    hides the near edge of the table from the robot camera), named as the plan names
    objects. The robot camera is calibrated, so a block it sees whole (ROBOT_FULL points)
    is placed from it; any other is placed by the mean of HEAD_FRAMES head frames, whose
    poses each carry the glasses' tracking error."""
    robot = _on_table(world.rgbd(ROBOT_CAM))
    head: list[list[Blob]] = []
    for _ in range(HEAD_FRAMES):
        for b in _on_table(world.ego_frame(rng)):
            group = next((g for g in head if _same(g[0], b)), None)
            if group is None:
                head.append([b])
            else:
                group.append(b)
    blobs = [b for b in robot if b.points >= ROBOT_FULL]
    for g in head:
        mean = Blob(g[0].color, np.mean([b.position for b in g], axis=0), g[0].points)
        if len(g) * 2 > HEAD_FRAMES and not any(_same(b, mean) for b in blobs):
            blobs.append(mean)
    for b in robot:  # seen only, and only partly, by the robot camera
        if not any(_same(o, b) for o in blobs):
            blobs.append(b)
    names = name_objects([(b.color, b.position) for b in blobs], frame)
    return {n: b.position for n, b in zip(names, blobs)}


def execute(world: EgoWorld, ep: Episode, plan: Plan, relation_frame: PersonFrame,
            on_tick=None) -> Run:
    """Reset to the new layout and carry the plan out, reading relations in relation_frame."""
    world.reset(ep.scene(new=True))
    n = len(ep.colors)
    seen = perceive(world, plan.frame, np.random.default_rng([11, ep.seed]))
    index = {}
    for name, p in seen.items():
        d = [np.linalg.norm(p[:2] - world.block_position(i)[:2]) for i in range(n)]
        if min(d) <= MATCH:
            index[name] = int(np.argmin(d))
    run = Run(False)
    for k, step in enumerate(plan.steps):
        if step.obj not in index:
            run.reason = f"step {k + 1}: {step.obj} not seen"
            break
        goal = goal_for(step, seen, relation_frame)
        if goal is None:
            run.reason = f"step {k + 1}: no goal"
            break
        i = index[step.obj]
        for _ in range(PICK_TRIES):
            picked = pick(world, i, on_tick)
            if picked.success:
                break
        if not picked.success:
            run.reason = f"step {k + 1}: pick {picked.reason}"
            break
        placed = place(world, i, goal, on_tick)
        if not placed.success:
            run.reason = f"step {k + 1}: place {placed.reason}"
        seen[step.obj] = world.block_position(i) if not placed.success else goal
        run.steps_done = k + 1
    go_home(world, on_tick)
    world.step(world.ctrl(), round(SETTLE / 0.002))
    run.final = [world.block_position(i) for i in range(n)]
    return run


def check(ep: Episode, run: Run, frame: PersonFrame, start: list[np.ndarray]) -> bool:
    return satisfied(ep.goals, ep.colors, start, run.final, frame)


def new_start(ep: Episode) -> list[np.ndarray]:
    from curio_robot.sim.ego_tasks import _xyz

    return [_xyz(p) for p in ep.new_xy]


def episode(world: EgoWorld, ep: Episode) -> dict:
    """Demo, extraction, then Curio and the baseline each on a fresh copy of the new layout."""
    demo = demonstrate(world, ep)
    exact = demo.plan.steps == ep.plan
    out = {"family": ep.family, "seed": ep.seed, "exact": bool(exact),
           "plan": demo.plan.text(), "truth": [s.text() for s in ep.plan],
           "demo_s": round(demo.seconds, 2), "frames": demo.frames}
    for label, frame in (("curio", demo.plan.frame), ("baseline", ROBOT_FRAME)):
        run = execute(world, ep, demo.plan, frame)
        ok = check(ep, run, demo.true_frame, new_start(ep))
        out[label] = {"success": bool(ok), "reason": run.reason or ("" if ok else "goal not met"),
                      "steps": run.steps_done}
    return out
