"""Per-frame, per-block features for the critic. Causal: frame f uses only
what a running robot has at t[f]. Task T4-03."""

from __future__ import annotations

import numpy as np

from curio_robot.collab.gaze import GAZE_MAX_DEG  # noqa: F401 (used by the implementation)
from curio_robot.critic.episodes import HOME, Episode  # noqa: F401 (used by the implementation)
from curio_robot.runtime.motion import SPEED  # noqa: F401 (used by the implementation)

FEATURES = (
    "angle",  # head direction to block, deg / GAZE_MAX_DEG, clipped to ANGLE_CLIP
    "dwell",  # seconds the head has been within GAZE_MAX_DEG of the block, capped at DWELL_CAP
    "is_arm_target",  # 1.0 for the arm's target
    "grip_dist",  # gripper-to-block distance / DIST_SCALE, clipped to DIST_CLIP
    "approach",  # how fast the gripper closes on the block, / SPEED
    "candidate",  # 1.0 if the block has the requested colour
    "progress",  # reach progress, same for every block
    "said_no",  # 1.0 once a transcript starting with "no" has arrived
    "since_no",  # seconds since that transcript, capped at SINCE_CAP
    "head_speed",  # head turn since the last frame, deg/s / HEAD_SPEED_SCALE
)
D = len(FEATURES)

ANGLE_CLIP = 3.0
DWELL_CAP = 1.0
DIST_SCALE = 0.3
DIST_CLIP = 2.0
SINCE_CAP = 2.0
HEAD_SPEED_SCALE = 100.0


def features(ep: Episode) -> np.ndarray:
    """(F, N, D) float array, columns in FEATURES order.

    With dt = t[1] - t[0], v_j = obj_pos[j] - head_pos, and angle(f, j) =
    degrees between head_fwd[f] and v_j:
    - angle: min(angle / GAZE_MAX_DEG, ANGLE_CLIP).
    - dwell: 0 when angle > GAZE_MAX_DEG; otherwise t[f] - t[s], s the first
      frame of the current unbroken run of frames within the cone (so 0 on
      the first such frame); capped at DWELL_CAP.
    - is_arm_target: 1.0 where j == ep.arm_target.
    - grip_dist: min(norm(gripper[f] - obj_pos[j]) / DIST_SCALE, DIST_CLIP).
    - approach: 0 at f = 0; else (d[f-1] - d[f]) / dt / SPEED with d the
      unscaled gripper-to-block distance.
    - candidate: 1.0 where ep.candidates.
    - progress: 1 - norm(gripper[f] - gripper[-1]) / norm(HOME -
      gripper[-1]), clipped to [0, 1]; 1.0 if that denominator is 0.
    - said_no: 1.0 if some speech (ts, text) has ts <= t[f] and
      text.startswith("no").
    - since_no: min(t[f] - first such ts, SINCE_CAP) once said_no, else 0.
    - head_speed: 0 at f = 0; else degrees between head_fwd[f-1] and
      head_fwd[f], / dt / HEAD_SPEED_SCALE. Same for every block."""
    raise NotImplementedError("T4-03")

