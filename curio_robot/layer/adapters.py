"""A Goal in each policy's input format. Task T5-27.

- ring: the robot image with the magenta target ring the SmolVLA policies were trained on
  (sim.demos.draw_marker at the target's pixel, rounded as demos.target_pixel rounds it).
- pose: the target position as a goal for motion-level controllers (walkers, AMR, drone,
  ROV, flyer), in the robot base frame or, given the base pose, in the world frame.
- words: robot-frame text for language-conditioned policies (robot_words.py)."""

from __future__ import annotations

import numpy as np

from curio_robot.geometry import project
from curio_robot.layer.goal import Base, Goal, base_from_camera
from curio_robot.layer.robot_words import robot_words as words
from curio_robot.sim.demos import draw_marker
from curio_robot.types import CameraFrame

__all__ = ["pose", "ring", "ring_pixel", "words"]


def ring_pixel(goal: Goal, robot: CameraFrame, base: Base | None = None
               ) -> tuple[float, float] | None:
    """The target's pixel in the robot image, rounded to 0.01 px; None outside the image.
    base defaults to base_from_camera(robot.pose), the same default resolve() uses."""
    xyz = pose(goal, base or base_from_camera(robot.pose))
    uv = project(xyz, robot.intrinsics, robot.pose)
    if uv is None:
        return None
    u, v = uv
    if not (0 <= u < robot.intrinsics.width and 0 <= v < robot.intrinsics.height):
        return None
    return (round(float(u), 2), round(float(v), 2))


def ring(goal: Goal, robot: CameraFrame, base: Base | None = None) -> np.ndarray:
    """A copy of the robot image with the target ring; unmarked if the target is off-image."""
    return draw_marker(robot.rgb, ring_pixel(goal, robot, base))


def pose(goal: Goal, base: Base | None = None) -> np.ndarray:
    """The target position (3,): robot base frame, or the world frame when base is given."""
    xyz = np.asarray(goal.target_xyz_robot, float)
    return xyz if base is None else base.to_world(xyz)
