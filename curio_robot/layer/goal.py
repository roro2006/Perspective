"""The one thing Curio hands to a robot policy: which object the person means, where it is in
the robot's own base frame, and how sure the layer is. Task T5-27.

A Goal is plain data (strings, floats, tuples) so it can cross a process or language boundary
as JSON. Positions are in the robot base frame (x forward, y left, z up, metres), the frame
any controller already plans in; Base converts to and from the world frame Curio fuses in."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from curio_robot.types import CameraPose

Frame = str  # "speaker" (grounded on the person's view) or "robot" (robot cameras only)


@dataclass(frozen=True, slots=True)
class Goal:
    target_id: str
    target_xyz_robot: tuple[float, float, float]  # robot base frame, m
    frame: Frame
    confidence: float  # the grounder's probability for the chosen mark
    alternatives: tuple[tuple[str, float], ...]  # the other marks, most likely first
    ask: str | None  # a yes/no question to ask instead of acting (T5-22), else None
    note: str | None  # "I moved it; ..." when the object moved unseen (T5-21), else None
    adapter: str  # grounder adapter name from flywheel/registry.json

    @property
    def should_ask(self) -> bool:
        return self.ask is not None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Base:
    """The robot base pose in the world: origin and heading (yaw about world z, rad).
    Base axes: x along the heading, y to its left, z up."""

    position: tuple[float, float, float]
    yaw: float

    def _rotation(self) -> np.ndarray:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def to_base(self, xyz_world) -> np.ndarray:
        return self._rotation().T @ (np.asarray(xyz_world, float) - np.asarray(self.position))

    def to_world(self, xyz_base) -> np.ndarray:
        return self._rotation() @ np.asarray(xyz_base, float) + np.asarray(self.position)


def base_from_camera(pose: CameraPose) -> Base:
    """A base for robots that only report a camera pose: directly below the camera at world
    z = 0, heading along the camera's horizontal forward direction. Pass the real base pose
    when the robot has one."""
    forward = pose.forward
    yaw = math.atan2(float(forward[1]), float(forward[0]))
    x, y = float(pose.position[0]), float(pose.position[1])
    return Base((x, y, 0.0), yaw)
