"""The swap point between simulation and the real bench. Everything above this
interface (perception, fusion, resolver, monitor, logging) is product code that
runs unchanged on either backend."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from curio_robot.types import CameraFrame


class World(Protocol):
    def capture(self) -> dict[str, CameraFrame]:
        """Latest synchronized frames, keyed ROBOT_CAM / HUMAN_CAM."""
        ...

    def move_gripper_to(self, position: np.ndarray) -> bool:
        """Move the gripper above a table point. False if unreachable."""
        ...
