"""Hand-built critic episodes for the T4 tests, independent of the episode
generator. Head at (0.7, 0, 0.4); from there, looking at block 1 puts block 0
15.34 deg off and block 2 (blue) 8.37 deg off."""

from __future__ import annotations

import numpy as np

from curio_robot.critic.episodes import Episode

HEAD = np.array([0.7, 0.0, 0.4])
OBJ = np.array([[0.25, 0.08, 0.0125], [0.25, -0.08, 0.0125], [0.30, 0.0, 0.0125]])
COLORS = ("red", "red", "blue")
START = np.array([0.05, 0.25, 0.15])
SPEED = 0.25
AWAY = np.array([0.25, 0.45, 0.0])  # nothing within 10 deg of this


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def times(n: int = 31) -> np.ndarray:
    return np.arange(n) / 10.0


def path(t: np.ndarray, goal: np.ndarray) -> np.ndarray:
    delta = goal - START
    dist = np.linalg.norm(delta)
    s = np.minimum(SPEED * np.maximum(t, 0.0), dist)
    return START + np.outer(s / dist, delta)


def make_ep(
    looks: list[int | None] | None = None,
    *,
    intended: int = 1,
    arm_target: int = 0,
    speech: tuple[tuple[float, str], ...] = (),
    t_no: float | None = None,
    n: int = 31,
    family: str = "attentive",
) -> Episode:
    """looks[f]: block index the head points at in frame f, None = AWAY.
    Default: always the arm's target."""
    t = times(n)
    looks = [arm_target] * n if looks is None else looks
    points = np.array([AWAY if j is None else OBJ[j] for j in looks])
    return Episode(
        t=t, obj_pos=OBJ.copy(), colors=COLORS, request_color="red", intended=intended,
        arm_target=arm_target, gripper=path(t, OBJ[arm_target] + [0, 0, 0.03]),
        head_pos=HEAD.copy(), head_fwd=unit(points - HEAD), speech=speech, t_no=t_no,
        family=family, seed=0,
    )
