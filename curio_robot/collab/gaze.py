"""Implicit feedback: which object the person's head points at, and when a
look at something other than the arm's goal has lasted long enough to mean
"not that one". Head direction, not eye tracking. Task T3-05."""

from __future__ import annotations

from curio_robot.types import CameraPose, ObjectMap

GAZE_MAX_DEG = 10.0  # an object further than this off the line of sight isn't looked at
DWELL_S = 0.4  # a look this long at something else is a signal, not a glance


def gaze_target(
    pose: CameraPose, object_map: ObjectMap, max_deg: float = GAZE_MAX_DEG
) -> str | None:
    """The object whose direction from pose.position makes the smallest angle
    with pose.forward, if that angle is <= max_deg degrees; else None. Ties go
    to the smaller object_id. No objects -> None."""
    raise NotImplementedError("T3-05")


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
        raise NotImplementedError("T3-05")

    def reset(self) -> None:
        """Forget the current look."""
        raise NotImplementedError("T3-05")
