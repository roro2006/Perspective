"""The target-3 gaze rule as a critic, for comparison: a steady look at
another block of the requested colour, or a heard "no", means wrong.
Task T4-05."""

from __future__ import annotations

import numpy as np

from curio_robot.collab.gaze import DWELL_S, GAZE_MAX_DEG, GazeWatch  # noqa: F401 (used by the implementation)
from curio_robot.critic.episodes import Episode


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
    raise NotImplementedError("T4-05")

