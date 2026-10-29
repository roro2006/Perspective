"""The scene for a multi-step task: two same-coloured pairs (so some requests
need "left"/"right") and single blocks, a seated person, and optionally a box
hiding the first single block from the robot camera. Task T3-04."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from curio_robot.geometry import look_at
from curio_robot.sim.scenarios import (
    COLORS,
    MIN_GAP,
    _free_xy,
    _in_workspace,
    _random_xy,
    _sample_human,
)
from curio_robot.sim.world import ROBOT_CAM_LOOK, ROBOT_CAM_POS, BlockSpec, SimWorld
from curio_robot.types import CameraPose, ObjectMap, WorldObject

BLOCK_Z = 0.0125  # block centre height (sim.world.BLOCK_HALF)
PAIR_HALF = (0.045, 0.07)  # half the distance between a pair's blocks, m


@dataclass(frozen=True, slots=True)
class TaskScene:
    kind: str  # "fetch"
    blocks: tuple[BlockSpec, ...]  # pairs first (both blocks of a pair together), then singles
    human_eye: np.ndarray  # (3,)
    human_look: np.ndarray  # (3,)
    occluder: tuple[float, float] | None  # box centre xy, None: no box
    occluder_yaw: float


def fetch_scene(
    rng: np.random.Generator, n_pairs: int = 2, n_single: int = 1, occlude: bool = False
) -> TaskScene:
    """A random fetch scene; pure (no world needed).

    - Person: sim.scenarios._sample_human(rng).
    - Colours: pair colours = rng.choice(scenarios.COLORS, n_pairs,
      replace=False); single colours = rng.choice(the remaining colours,
      n_single, replace=False). ValueError if n_pairs + n_single > 4.
    - Each pair as scenarios.perspective places one: mid = _random_xy(rng),
      angle = rng.uniform(0, pi), half = rng.uniform(*PAIR_HALF) along the
      angle, blocks at mid ± half; retry (up to 200 tries, then RuntimeError)
      until both are in the workspace (_in_workspace) and more than
      scenarios.MIN_GAP from every block already placed.
    - Singles: _free_xy(rng, taken), appending each to taken.
    - occlude: a box 0.045 m from the first single block toward
      sim.world.ROBOT_CAM_POS (xy), yaw = atan2 of that direction, exactly as
      scenarios.occlusion. ValueError if occlude and n_single == 0.
      No box: occluder None, yaw 0.0."""
    if n_pairs + n_single > 4:
        raise ValueError("n_pairs + n_single must be <= 4")
    if occlude and n_single == 0:
        raise ValueError("occlude requires at least one single block")

    eye, look = _sample_human(rng)

    pair_colours = rng.choice(COLORS, n_pairs, replace=False)
    remaining = [c for c in COLORS if c not in pair_colours]
    single_colours = rng.choice(remaining, n_single, replace=False)

    pair_colours = [str(c) for c in pair_colours]
    single_colours = [str(c) for c in single_colours]

    blocks: list[BlockSpec] = []
    taken: list[np.ndarray] = []

    for colour in pair_colours:
        for _ in range(200):
            mid = _random_xy(rng)
            angle = rng.uniform(0, np.pi)
            half = rng.uniform(*PAIR_HALF) * np.array([np.cos(angle), np.sin(angle)])
            a = mid - half
            b = mid + half
            if not (_in_workspace(a) and _in_workspace(b)):
                continue
            ok = True
            for pt in taken:
                if np.linalg.norm(a - pt) <= MIN_GAP or np.linalg.norm(b - pt) <= MIN_GAP:
                    ok = False
                    break
            if ok:
                taken.append(a)
                taken.append(b)
                blocks.append(BlockSpec(colour, (float(a[0]), float(a[1]))))
                blocks.append(BlockSpec(colour, (float(b[0]), float(b[1]))))
                break
        else:
            raise RuntimeError("could not place pair block")

    for colour in single_colours:
        xy = _free_xy(rng, taken)
        taken.append(xy)
        blocks.append(BlockSpec(colour, (float(xy[0]), float(xy[1]))))

    if occlude:
        target = np.asarray(blocks[n_pairs * 2].xy)
        toward = ROBOT_CAM_POS[:2] - target
        toward = toward / np.linalg.norm(toward)
        occ = target + 0.045 * toward
        yaw = float(np.arctan2(toward[1], toward[0]))
        occluder = (float(occ[0]), float(occ[1]))
    else:
        occluder = None
        yaw = 0.0

    return TaskScene("fetch", tuple(blocks), eye, look, occluder, yaw)


def apply_scene(world: SimWorld, scene: TaskScene) -> None:
    """Put the scene in the world: set_blocks(list(blocks)), set_occluder
    (occluder, occluder_yaw) or set_occluder(None), set_human(eye, look)."""
    world.set_blocks(list(scene.blocks))
    if scene.occluder is None:
        world.set_occluder(None)
    else:
        world.set_occluder(scene.occluder, scene.occluder_yaw)
    world.set_human(scene.human_eye, scene.human_look)


def true_map(scene: TaskScene) -> ObjectMap:
    """The ground-truth map the simulated person reasons with.

    One WorldObject per block, in block order, id f"{color}_{k}" with k
    counting that colour from 1 in block order, position [x, y, BLOCK_Z],
    seen_by empty. robot_pose = CameraPose(ROBOT_CAM_POS, look_at(
    ROBOT_CAM_POS, ROBOT_CAM_LOOK)); human_pose = CameraPose(human_eye,
    look_at(human_eye, human_look)) (geometry.look_at)."""
    counts: dict[str, int] = {}
    objects: list[WorldObject] = []
    for block in scene.blocks:
        counts[block.color] = counts.get(block.color, 0) + 1
        k = counts[block.color]
        x, y = block.xy
        objects.append(
            WorldObject(f"{block.color}_{k}", block.color, np.array([x, y, BLOCK_Z]), set())
        )
    robot_pose = CameraPose(ROBOT_CAM_POS, look_at(ROBOT_CAM_POS, ROBOT_CAM_LOOK))
    human_pose = CameraPose(scene.human_eye, look_at(scene.human_eye, scene.human_look))
    return ObjectMap(objects, robot_pose, human_pose)
