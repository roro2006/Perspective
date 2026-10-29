"""The scene for a multi-step task: two same-coloured pairs (so some requests
need "left"/"right") and single blocks, a seated person, and optionally a box
hiding the first single block from the robot camera. Task T3-04."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from curio_robot.sim.world import BlockSpec, SimWorld
from curio_robot.types import ObjectMap

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
    raise NotImplementedError("T3-04")


def apply_scene(world: SimWorld, scene: TaskScene) -> None:
    """Put the scene in the world: set_blocks(list(blocks)), set_occluder
    (occluder, occluder_yaw) or set_occluder(None), set_human(eye, look)."""
    raise NotImplementedError("T3-04")


def true_map(scene: TaskScene) -> ObjectMap:
    """The ground-truth map the simulated person reasons with.

    One WorldObject per block, in block order, id f"{color}_{k}" with k
    counting that colour from 1 in block order, position [x, y, BLOCK_Z],
    seen_by empty. robot_pose = CameraPose(ROBOT_CAM_POS, look_at(
    ROBOT_CAM_POS, ROBOT_CAM_LOOK)); human_pose = CameraPose(human_eye,
    look_at(human_eye, human_look)) (geometry.look_at)."""
    raise NotImplementedError("T3-04")
