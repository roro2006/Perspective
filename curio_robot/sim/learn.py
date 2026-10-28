"""Learning curves for target 1: simulated users, many requests each, error
rate against request number. Task T1-09 (fast mode); live mode after T1-08.

    uv run curio-sim learn --requests 15 --users 50 [--noise] [--live]"""

from __future__ import annotations

import numpy as np

from curio_robot.conventions.belief import ConventionBelief, learn_from_outcome
from curio_robot.conventions.decide import decide
from curio_robot.conventions.noise import flip_probability
from curio_robot.conventions.readings import enumerate_readings
from curio_robot.geometry import OBJECT_PLANE_Z, look_at, perturb_pose
from curio_robot.sim.scenarios import COLORS, _in_workspace, _random_xy, _sample_human
from curio_robot.sim.users import SimUser
from curio_robot.sim.world import ROBOT_CAM_LOOK, ROBOT_CAM_POS
from curio_robot.types import CameraPose, ObjectMap, WorldObject

HALF_LEN_MIN = 0.045
HALF_LEN_MAX = 0.07
FLIP_SAMPLES = 100


def _sample_pair(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Two block centres on the table, at a random angle, both inside the workspace."""
    while True:
        mid = _random_xy(rng)
        angle = rng.uniform(0, np.pi)
        half_len = rng.uniform(HALF_LEN_MIN, HALF_LEN_MAX)
        half = half_len * np.array([np.cos(angle), np.sin(angle)])
        a, b = mid - half, mid + half
        if _in_workspace(a) and _in_workspace(b):
            return a, b


def learning_curve(
    user: SimUser,
    *,
    requests: int = 15,
    users: int = 50,
    seed: int = 0,
    pos_noise: float = 0.0,
    rot_noise_deg: float = 0.0,
    learn: bool = True,
) -> dict:
    """Fast mode, no runtime or arm: for each of `users` fresh copies of
    `user` (fresh belief each), run `requests` perspective scenes. Per scene:
    utter -> readings -> posterior -> decide; the user reacts to the chosen
    object; learn_from_outcome (unless learn=False, the baseline).
    The decision is the final one, so only "act" moves the arm: "hedge" and
    "ask" both mean the robot asks "your left or mine?", the user answers
    (SimUser.answer), and the robot learns with outcome "answered" and then
    takes the intended object (an ask is never an error).
    Scenes: sim.scenarios.perspective geometry without rendering is fine;
    two same-colour blocks, a random side, a random human pose.
    p_flip: noise.flip_probability for the pair when pos/rot noise > 0, and
    the human pose given to the robot is perturbed with geometry.perturb_pose.

    Returns {"user", "requests", "users", "learn", "error_rate": [per request
    index, fraction of users wrong], "ask_rate": [...], "final_p_own_frame":
    mean over users}. "ask" counts as not wrong and not right; track it apart."""
    rng = np.random.default_rng(seed)
    robot_pose = CameraPose(ROBOT_CAM_POS, look_at(ROBOT_CAM_POS, ROBOT_CAM_LOOK))
    noisy = pos_noise > 0 or rot_noise_deg > 0

    errors = [0] * requests
    asks = [0] * requests
    finals = []

    for _ in range(users):
        belief = ConventionBelief()
        for i in range(requests):
            eye, look = _sample_human(rng)
            true_human = CameraPose(eye, look_at(eye, look))
            color = str(rng.choice(COLORS))

            a, b = _sample_pair(rng)
            pa = np.array([*a, OBJECT_PLANE_Z])
            pb = np.array([*b, OBJECT_PLANE_Z])
            objects = [
                WorldObject(f"{color}_1", color, pa),
                WorldObject(f"{color}_2", color, pb),
            ]

            side = str(rng.choice(["left", "right"]))
            utt = user.utter(color, side, rng)
            # Ground truth uses the true pose; the robot only sees the perturbed one.
            intended = user.intended(utt, ObjectMap(objects, robot_pose, true_human))
            seen = perturb_pose(true_human, rng, pos_noise, rot_noise_deg) if noisy else true_human

            omap = ObjectMap(objects, robot_pose, seen)
            readings = enumerate_readings(utt.text, omap)
            d = decide(belief.posterior(readings), omap, utt.text)

            p_flip = 0.0
            if noisy:
                p_flip = flip_probability(
                    pa, pb, seen,
                    pos_sigma=pos_noise,
                    rot_sigma_deg=rot_noise_deg,
                    samples=FLIP_SAMPLES,
                    seed=int(rng.integers(2**31)),
                )

            if d.kind == "act":
                if d.object_id == intended:
                    outcome, chosen = "accepted", d.object_id
                else:
                    errors[i] += 1
                    outcome, chosen = "corrected", d.object_id
                right = None
            else:
                # hedge / ask / none: robot asks, user answers, robot takes intended
                asks[i] += 1
                user.answer(utt)
                outcome, chosen, right = "answered", intended, intended

            if learn:
                learn_from_outcome(belief, readings, chosen, outcome, right_id=right, p_flip=p_flip)

        finals.append(belief.p_own_frame())

    return {
        "user": user.name,
        "requests": requests,
        "users": users,
        "learn": learn,
        "error_rate": [e / users for e in errors],
        "ask_rate": [n / users for n in asks],
        "final_p_own_frame": float(np.mean(finals)),
    }
