"""Per-frame, per-block features for the critic. Causal: frame f uses only
what a running robot has at t[f]. Task T4-03."""

from __future__ import annotations

import numpy as np

from curio_robot.collab.gaze import GAZE_MAX_DEG
from curio_robot.critic.episodes import HOME, Episode
from curio_robot.runtime.motion import SPEED

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


def _angles_deg(head_fwd: np.ndarray, v: np.ndarray) -> np.ndarray:
    """(F, N) degrees between each head_fwd[f] (F, 3) and each v[j] (N, 3)."""
    unit_v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    cos = np.clip(head_fwd @ unit_v.T, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def _dwell(t: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """(F, N) seconds each block has been inside the gaze cone, unbroken."""
    n_frames, n_blocks = angles.shape
    dwell = np.zeros((n_frames, n_blocks))
    for j in range(n_blocks):
        run_start: int | None = None
        for f in range(n_frames):
            if angles[f, j] > GAZE_MAX_DEG:
                run_start = None
                continue
            if run_start is None:
                run_start = f
            dwell[f, j] = min(t[f] - t[run_start], DWELL_CAP)
    return dwell


def _approach(dist: np.ndarray, dt: float) -> np.ndarray:
    """(F, N) closing speed in units of SPEED from unscaled distances (F, N)."""
    approach = np.zeros_like(dist)
    approach[1:] = (dist[:-1] - dist[1:]) / dt / SPEED
    return approach


def _progress(gripper: np.ndarray) -> np.ndarray:
    """(F,) reach progress in [0, 1], 1.0 everywhere if HOME is at the goal."""
    denom = np.linalg.norm(HOME - gripper[-1])
    if denom == 0:
        return np.ones(len(gripper))
    moved = np.linalg.norm(gripper - gripper[-1], axis=-1)
    return np.clip(1.0 - moved / denom, 0.0, 1.0)


def _speech_flags(
    t: np.ndarray, speech: tuple[tuple[float, str], ...]
) -> tuple[np.ndarray, np.ndarray]:
    """(F,) said_no and since_no, from the first transcript starting with "no"."""
    no_times = [ts for ts, text in speech if text.startswith("no")]
    said = np.zeros(len(t))
    since = np.zeros(len(t))
    if not no_times:
        return said, since
    first = min(no_times)
    heard = t >= first
    said[heard] = 1.0
    since[heard] = np.minimum(t[heard] - first, SINCE_CAP)
    return said, since


def _head_speed(t: np.ndarray, head_fwd: np.ndarray) -> np.ndarray:
    """(F,) head turn since the previous frame, deg/s / HEAD_SPEED_SCALE."""
    speed = np.zeros(len(t))
    if len(t) > 1:
        cos = np.clip(np.sum(head_fwd[1:] * head_fwd[:-1], axis=-1), -1.0, 1.0)
        speed[1:] = np.degrees(np.arccos(cos)) / (t[1] - t[0]) / HEAD_SPEED_SCALE
    return speed


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
    n_frames, n_blocks = len(ep.t), len(ep.obj_pos)
    angles = _angles_deg(ep.head_fwd, ep.obj_pos - ep.head_pos)
    dist = np.linalg.norm(ep.gripper[:, None, :] - ep.obj_pos[None, :, :], axis=-1)
    dt = ep.t[1] - ep.t[0] if n_frames > 1 else 1.0
    said, since = _speech_flags(ep.t, ep.speech)

    out = np.zeros((n_frames, n_blocks, D))
    out[:, :, FEATURES.index("angle")] = np.minimum(angles / GAZE_MAX_DEG, ANGLE_CLIP)
    out[:, :, FEATURES.index("dwell")] = _dwell(ep.t, angles)
    out[:, :, FEATURES.index("is_arm_target")] = np.arange(n_blocks) == ep.arm_target
    out[:, :, FEATURES.index("grip_dist")] = np.minimum(dist / DIST_SCALE, DIST_CLIP)
    out[:, :, FEATURES.index("approach")] = _approach(dist, dt)
    out[:, :, FEATURES.index("candidate")] = ep.candidates
    out[:, :, FEATURES.index("progress")] = _progress(ep.gripper)[:, None]
    out[:, :, FEATURES.index("said_no")] = said[:, None]
    out[:, :, FEATURES.index("since_no")] = since[:, None]
    out[:, :, FEATURES.index("head_speed")] = _head_speed(ep.t, ep.head_fwd)[:, None]
    return out
