"""Learning curves for target 1: simulated users, many requests each, error
rate against request number. Task T1-09 (fast mode); live mode after T1-08.

    uv run curio-sim learn --requests 15 --users 50 [--noise] [--live]"""

from __future__ import annotations

from curio_robot.sim.users import SimUser


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
    raise NotImplementedError("T1-09")
