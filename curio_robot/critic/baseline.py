"""The target-3 gaze rule as a critic, for comparison: a steady look at
another block of the requested colour, or a heard "no", means wrong.
Task T4-05."""

from __future__ import annotations

import numpy as np

from curio_robot.collab.gaze import DWELL_S, GAZE_MAX_DEG, GazeWatch
from curio_robot.critic.episodes import Episode


def _angle_deg(v: np.ndarray, fwd: np.ndarray) -> float:
    """Angle in degrees between vector v and unit vector fwd. A zero vector is
    never looked at (as in gaze_target), so its angle is infinite."""
    norm_v = np.linalg.norm(v)
    if norm_v == 0:
        return float("inf")
    cos_theta = np.clip(np.dot(v, fwd) / norm_v, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def _nearest_candidate(
    ep: Episode, f: int, cand_idx: np.ndarray, skip: int | None
) -> tuple[int | None, float]:
    """Candidate (ignoring index skip) with the smallest angle to head_fwd[f]; ties go to the
    smaller index. Returns (None, inf) if there is no such candidate."""
    best_idx: int | None = None
    best_angle = float("inf")
    for j in cand_idx:
        if j == skip:
            continue
        angle = _angle_deg(ep.obj_pos[j] - ep.head_pos, ep.head_fwd[f])
        if angle < best_angle:
            best_angle = angle
            best_idx = int(j)
    return best_idx, best_angle


def _looked_at(ep: Episode, f: int, cand_idx: np.ndarray, max_deg: float) -> int | None:
    """Candidate the head points at in frame f, if within max_deg; else None."""
    idx, angle = _nearest_candidate(ep, f, cand_idx, skip=None)
    if idx is not None and angle <= max_deg:
        return idx
    return None


def rule_scores(
    ep: Episode, dwell_s: float = DWELL_S, max_deg: float = GAZE_MAX_DEG
) -> tuple[np.ndarray, np.ndarray]:
    """(p_wrong (F,), intended (F, N)) in the same shape the learned critic gives.

    Candidates are the blocks with ep.candidates True. Per frame f, in order:
    1. looked = the candidate index with the smallest angle between
       head_fwd[f] and obj_pos[j] - head_pos, if that angle <= max_deg, else
       None (ties: smaller index).
    2. fired = watch.update(t[f], f"o{looked}" or None, f"o{arm_target}") with
       one GazeWatch(dwell_s) for the episode; remember the first block it
       fires on (gaze_pick) and keep it.
    3. heard_no = some speech (ts, text) with ts <= t[f] and text starting
       with "no".
    Then:
    - gaze_pick set: p_wrong 1.0, intended one-hot gaze_pick;
    - else heard_no: p_wrong 1.0, intended one-hot on the candidate other than
      arm_target with the smallest angle at frame f (ties: smaller index);
      if there is no other candidate, intended one-hot arm_target;
    - else p_wrong 0.0, intended one-hot arm_target."""
    n_frames = len(ep.t)
    n_blocks = len(ep.colors)
    p_wrong = np.zeros(n_frames, dtype=float)
    intended = np.zeros((n_frames, n_blocks), dtype=float)

    cand_idx = np.flatnonzero(ep.candidates)
    watch = GazeWatch(dwell_s)
    arm_goal = f"o{ep.arm_target}"
    gaze_pick: int | None = None

    for f in range(n_frames):
        looked = _looked_at(ep, f, cand_idx, max_deg)
        looking_at = f"o{looked}" if looked is not None else None
        fired = watch.update(float(ep.t[f]), looking_at, arm_goal)
        if fired is not None and gaze_pick is None:
            gaze_pick = int(fired[1:])

        heard_no = any(ts <= ep.t[f] and text.startswith("no") for ts, text in ep.speech)

        if gaze_pick is not None:
            p_wrong[f] = 1.0
            intended[f, gaze_pick] = 1.0
        elif heard_no:
            p_wrong[f] = 1.0
            other, _ = _nearest_candidate(ep, f, cand_idx, skip=ep.arm_target)
            intended[f, other if other is not None else ep.arm_target] = 1.0
        else:
            intended[f, ep.arm_target] = 1.0

    return p_wrong, intended
