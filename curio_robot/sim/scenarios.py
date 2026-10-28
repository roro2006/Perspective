"""Randomized demo scenarios. Each one places blocks, an optional occluder and
the human, and knows which block the request should resolve to."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from curio_robot.geometry import look_at
from curio_robot.sim.world import ROBOT_CAM_POS, TABLE_CENTER, BlockSpec, SimWorld

WORKSPACE_X = (0.15, 0.33)
WORKSPACE_Y = (-0.15, 0.15)
MIN_GAP = 0.06
COLORS = ("red", "blue", "green", "yellow")


@dataclass(slots=True)
class Scenario:
    kind: str
    request: str
    expected_xy: np.ndarray  # where the correct block is
    human_eye: np.ndarray
    human_look: np.ndarray


def _in_workspace(xy: np.ndarray) -> bool:
    return WORKSPACE_X[0] <= xy[0] <= WORKSPACE_X[1] and WORKSPACE_Y[0] <= xy[1] <= WORKSPACE_Y[1]


def _random_xy(rng: np.random.Generator) -> np.ndarray:
    return np.array([rng.uniform(*WORKSPACE_X), rng.uniform(*WORKSPACE_Y)])


def _free_xy(rng: np.random.Generator, taken: list[np.ndarray]) -> np.ndarray:
    for _ in range(200):
        xy = _random_xy(rng)
        if all(np.linalg.norm(xy - t) > MIN_GAP for t in taken):
            return xy
    raise RuntimeError("could not place block")


def _sample_human(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """A seated person on the far side of the table, facing the robot."""
    theta = np.radians(rng.uniform(-70, 70))
    eye_xy = TABLE_CENTER + 0.48 * np.array([np.cos(theta), np.sin(theta)])
    eye = np.array([*eye_xy, 0.38 + rng.uniform(-0.05, 0.05)])
    look = np.array([*(TABLE_CENTER + rng.normal(0, 0.03, 2)), 0.0])
    return eye, look


def _distractors(rng, taken: list[np.ndarray], exclude: str, n: int) -> list[BlockSpec]:
    out = []
    for _ in range(n):
        xy = _free_xy(rng, taken)
        taken.append(xy)
        out.append(BlockSpec(rng.choice([c for c in COLORS if c != exclude]), tuple(xy)))
    return out


def perspective(world: SimWorld, rng: np.random.Generator) -> Scenario:
    """Two same-coloured blocks; "the <colour> one on my <side>"."""
    eye, look = _sample_human(rng)
    color = str(rng.choice(COLORS))
    while True:
        mid = _random_xy(rng)
        angle = rng.uniform(0, np.pi)
        half = rng.uniform(0.045, 0.07) * np.array([np.cos(angle), np.sin(angle)])
        a, b = mid - half, mid + half
        if _in_workspace(a) and _in_workspace(b):
            break
    taken = [a, b]
    blocks = [BlockSpec(color, tuple(a)), BlockSpec(color, tuple(b))]
    blocks += _distractors(rng, taken, color, int(rng.integers(1, 3)))
    world.set_blocks(blocks)
    world.set_occluder(None)
    world.set_human(eye, look)

    side = str(rng.choice(["left", "right"]))
    right = look_at(eye, look)[:, 0]
    lat = [float(np.dot(np.array([*p, 0]) - eye, right)) for p in (a, b)]
    expected = (a, b)[int(np.argmin(lat))] if side == "left" else (a, b)[int(np.argmax(lat))]
    return Scenario("perspective", f"Hand me the {color} one on my {side}", expected, eye, look)


def occlusion(world: SimWorld, rng: np.random.Generator) -> Scenario:
    """A unique blue block with a box between it and the robot's camera."""
    eye, look = _sample_human(rng)
    target = _random_xy(rng)
    toward_robot = ROBOT_CAM_POS[:2] - target
    toward_robot /= np.linalg.norm(toward_robot)
    occ = target + 0.045 * toward_robot
    yaw = float(np.arctan2(toward_robot[1], toward_robot[0]))
    taken = [target, occ]
    blocks = [BlockSpec("blue", tuple(target))]
    blocks += _distractors(rng, taken, "blue", int(rng.integers(1, 3)))
    world.set_blocks(blocks)
    world.set_occluder(tuple(occ), yaw)
    world.set_human(eye, look)
    return Scenario("occlusion", "Hand me the blue one", target, eye, look)


def cluttered(world: SimWorld, rng: np.random.Generator) -> Scenario:
    """Nothing staged: a random box somewhere, 4-5 blocks, and either a
    unique-colour request or a pair with "my left/right". The honest overall number."""
    eye, look = _sample_human(rng)
    right = look_at(eye, look)[:, 0]
    taken: list[np.ndarray] = []
    if rng.random() < 0.5:
        color = str(rng.choice(COLORS))
        a = _free_xy(rng, taken)
        taken.append(a)
        blocks = [BlockSpec(color, tuple(a))]
        blocks += _distractors(rng, taken, color, int(rng.integers(3, 5)))
        expected, request = a, f"Hand me the {color} one"
    else:
        color = str(rng.choice(COLORS))
        a = _free_xy(rng, taken)
        taken.append(a)
        b = _free_xy(rng, taken)
        taken.append(b)
        blocks = [BlockSpec(color, tuple(a)), BlockSpec(color, tuple(b))]
        blocks += _distractors(rng, taken, color, int(rng.integers(2, 4)))
        side = str(rng.choice(["left", "right"]))
        lat = [float(np.dot(np.array([*p, 0]) - eye, right)) for p in (a, b)]
        pick = int(np.argmin(lat)) if side == "left" else int(np.argmax(lat))
        expected, request = (a, b)[pick], f"Hand me the {color} one on my {side}"
    world.set_blocks(blocks)
    occ = _free_xy(rng, taken)
    world.set_occluder(tuple(occ), float(rng.uniform(0, np.pi)))
    world.set_human(eye, look)
    return Scenario("cluttered", request, expected, eye, look)


SCENARIOS = {"perspective": perspective, "occlusion": occlusion, "cluttered": cluttered}
