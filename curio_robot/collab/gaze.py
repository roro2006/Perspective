"""Implicit feedback: which object the person's head points at, and when a
look at something other than the arm's goal has lasted long enough to mean
"not that one". Head direction, not eye tracking. Task T3-05."""

from __future__ import annotations

import numpy as np

from curio_robot.types import CameraPose, ObjectMap

GAZE_MAX_DEG = 10.0  # an object further than this off the line of sight isn't looked at
DWELL_S = 0.4  # a look this long at something else is a signal, not a glance
_CLOCK_SLACK_S = 1e-9  # float slack so t - start == dwell_s counts as elapsed


def gaze_target(
    pose: CameraPose, object_map: ObjectMap, max_deg: float = GAZE_MAX_DEG
) -> str | None:
    """The object whose direction from pose.position makes the smallest angle
    with pose.forward, if that angle is <= max_deg degrees; else None. Ties go
    to the smaller object_id. No objects -> None."""
    if not object_map.objects:
        return None

    best_id: str | None = None
    best_angle = float("inf")

    for obj in sorted(object_map.objects, key=lambda o: o.object_id):
        v = obj.position - pose.position
        norm_v = np.linalg.norm(v)
        if norm_v == 0:
            continue
        cos = np.dot(v, pose.forward) / norm_v
        cos = np.clip(cos, -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos))
        if angle_deg < best_angle:
            best_angle = angle_deg
            best_id = obj.object_id

    if best_id is not None and best_angle <= max_deg:
        return best_id
    return None


class GazeWatch:
    """Fires once per steady look at an object that isn't the arm's goal."""

    def __init__(self, dwell_s: float = DWELL_S) -> None:
        self.dwell_s = dwell_s
        self._look: str | None = None  # the object being timed
        self._since = 0.0  # when the look started
        self._fired = False

    def update(self, t: float, looking_at: str | None, arm_goal: str | None) -> str | None:
        """Feed one observation at time t (seconds, increasing).

        - looking_at None, arm_goal None, or looking_at == arm_goal: forget the
          current look; return None.
        - looking_at differs from the look being timed: start timing it at t;
          return None.
        - Same look, not fired yet, and t - start >= dwell_s: mark it fired and
          return looking_at.
        - Otherwise None (a fired look stays quiet until it is forgotten or
          replaced)."""
        if looking_at is None or arm_goal is None or looking_at == arm_goal:
            self._look = None
            self._fired = False
            return None

        if looking_at != self._look:
            self._look = looking_at
            self._since = t
            self._fired = False
            return None

        if not self._fired and t - self._since + _CLOCK_SLACK_S >= self.dwell_s:
            self._fired = True
            return looking_at

        return None

    def reset(self) -> None:
        """Forget the current look."""
        self._look = None
        self._fired = False
