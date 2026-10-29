"""Reaction episodes: one right-or-wrong reach with the person's head and
speech, sampled without MuJoCo so thousands are cheap. The arm is a straight
line at the runtime's speed, not the real kinematics. Task T4-02."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from curio_robot.collab.tasks import fetch_scene, true_map
from curio_robot.critic import reactions
from curio_robot.critic.reactions import FAMILIES, RATE_HZ
from curio_robot.runtime.motion import SPEED
from curio_robot.sim.world import HOVER

HOME = np.array([0.05, 0.25, 0.15])  # gripper start: off to the robot's left, above the table
DURATION_S = 3.0  # 31 frames at RATE_HZ


@dataclass(frozen=True, slots=True)
class Episode:
    t: np.ndarray  # (F,) seconds from the reach start
    obj_pos: np.ndarray  # (N, 3)
    colors: tuple[str, ...]  # (N,)
    request_color: str
    intended: int  # index of the block the person wants
    arm_target: int  # index of the block the arm goes to
    gripper: np.ndarray  # (F, 3)
    head_pos: np.ndarray  # (3,)
    head_fwd: np.ndarray  # (F, 3) unit vectors
    speech: tuple[tuple[float, str], ...]
    t_no: float | None
    family: str
    seed: int

    @property
    def wrong(self) -> bool:
        return self.arm_target != self.intended

    @property
    def candidates(self) -> np.ndarray:
        """(N,) bool: blocks of the requested colour."""
        return np.array([c == self.request_color for c in self.colors])


def frame_times(duration_s: float = DURATION_S, rate_hz: float = RATE_HZ) -> np.ndarray:
    """0, 1/rate, ..., duration (inclusive): round(duration * rate) + 1 frames."""
    return np.arange(round(duration_s * rate_hz) + 1) / rate_hz


def arm_path(t: np.ndarray, start: np.ndarray, goal: np.ndarray, speed: float = SPEED) -> np.ndarray:
    """(F, 3) gripper positions: start + direction * min(speed * t[f], distance),
    so a straight line at constant speed that holds at goal. t < 0 -> start.
    start == goal -> every row is start."""
    dist = np.linalg.norm(goal - start)
    if dist == 0:
        return np.tile(start, (len(t), 1))
    s = np.minimum(speed * np.maximum(t, 0), dist)
    return start + np.outer(s / dist, goal - start)


def make_episode(seed: int, family: str, wrong: bool) -> Episode:
    """One episode, fully determined by its arguments.

    rng = np.random.default_rng(seed), then in this order:
    1. scene = collab.tasks.fetch_scene(rng) (2 pairs, 1 single, no box);
       truth = collab.tasks.true_map(scene); obj_pos (N, 3) and colors in
       truth.objects order (pairs first: blocks 0-1 and 2-3 are the pairs).
    2. intended = int(rng.integers(0, 4)); request_color = colors[intended];
       arm_target = intended ^ 1 if wrong else intended (the pair partner).
    3. t = frame_times(); gripper = arm_path(t, HOME, obj_pos[arm_target] +
       [0, 0, HOVER]).
    4. style = reactions.sample_style(rng, family); reaction =
       reactions.simulate_reaction(t, scene.human_eye, obj_pos, intended,
       arm_target, gripper, style, rng).
    head_pos = scene.human_eye; family and seed as given."""
    rng = np.random.default_rng(seed)
    scene = fetch_scene(rng)
    truth = true_map(scene)
    obj_pos = np.array([o.position for o in truth.objects])
    colors = tuple(o.color for o in truth.objects)

    intended = int(rng.integers(0, 4))
    request_color = colors[intended]
    arm_target = intended ^ 1 if wrong else intended

    t = frame_times()
    gripper = arm_path(t, HOME, obj_pos[arm_target] + [0, 0, HOVER])

    style = reactions.sample_style(rng, family)
    reaction = reactions.simulate_reaction(
        t, scene.human_eye, obj_pos, intended, arm_target, gripper, style, rng
    )

    return Episode(
        t=t,
        obj_pos=obj_pos,
        colors=colors,
        request_color=request_color,
        intended=intended,
        arm_target=arm_target,
        gripper=gripper,
        head_pos=scene.human_eye,
        head_fwd=reaction.head_fwd,
        speech=reaction.speech,
        t_no=reaction.t_no,
        family=family,
        seed=seed,
    )


def generate(
    n: int, seed: int, families: tuple[str, ...] = FAMILIES, p_wrong: float = 0.5
) -> list[Episode]:
    """n episodes. rng = np.random.default_rng(seed); for i in range(n):
    family = families[i % len(families)]; wrong = rng.random() < p_wrong;
    ep_seed = int(rng.integers(0, 2**31 - 1)); make_episode(ep_seed, family,
    wrong). Same arguments -> same episodes."""
    rng = np.random.default_rng(seed)
    episodes: list[Episode] = []
    for i in range(n):
        family = families[i % len(families)]
        wrong = rng.random() < p_wrong
        ep_seed = int(rng.integers(0, 2**31 - 1))
        episodes.append(make_episode(ep_seed, family, wrong))
    return episodes


def to_dict(ep: Episode) -> dict:
    """JSON-safe dict: arrays as nested lists, speech as [[t, text], ...],
    every field by its name."""
    return {
        "t": ep.t.tolist(),
        "obj_pos": ep.obj_pos.tolist(),
        "colors": list(ep.colors),
        "request_color": ep.request_color,
        "intended": int(ep.intended),
        "arm_target": int(ep.arm_target),
        "gripper": ep.gripper.tolist(),
        "head_pos": ep.head_pos.tolist(),
        "head_fwd": ep.head_fwd.tolist(),
        "speech": [[float(t), text] for t, text in ep.speech],
        "t_no": None if ep.t_no is None else float(ep.t_no),
        "family": ep.family,
        "seed": int(ep.seed),
    }


def from_dict(d: dict) -> Episode:
    """Inverse of to_dict: arrays back to float numpy arrays, colors and
    speech back to tuples (speech items as (float, str) tuples)."""
    return Episode(
        t=np.array(d["t"], dtype=float),
        obj_pos=np.array(d["obj_pos"], dtype=float),
        colors=tuple(d["colors"]),
        request_color=d["request_color"],
        intended=int(d["intended"]),
        arm_target=int(d["arm_target"]),
        gripper=np.array(d["gripper"], dtype=float),
        head_pos=np.array(d["head_pos"], dtype=float),
        head_fwd=np.array(d["head_fwd"], dtype=float),
        speech=tuple((float(t), str(text)) for t, text in d["speech"]),
        t_no=None if d["t_no"] is None else float(d["t_no"]),
        family=d["family"],
        seed=int(d["seed"]),
    )


def save(episodes: list[Episode], path: Path) -> None:
    """One json.dumps(to_dict(ep)) per line, UTF-8; creates parent folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(to_dict(ep)) + "\n" for ep in episodes)


def load(path: Path) -> list[Episode]:
    """Inverse of save; blank lines skipped."""
    episodes: list[Episode] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(from_dict(json.loads(line)))
    return episodes
