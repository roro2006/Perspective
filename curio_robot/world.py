"""The swap point between simulation and the real bench. Everything above this
interface (perception, fusion, resolver, monitor, logging) is product code that
runs unchanged on either backend."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from curio_robot.types import CameraFrame


class World(Protocol):
    """The live runtime calls capture() from the main thread and the arm methods
    from the motion thread, at the same time; implementations make that safe."""

    def capture(self) -> dict[str, CameraFrame]:
        """Latest synchronized frames, keyed ROBOT_CAM / HUMAN_CAM."""
        ...

    def move_gripper_to(self, position: np.ndarray) -> bool:
        """Move the gripper above a table point. False if unreachable."""
        ...

    def gripper_position(self) -> np.ndarray:
        """Table point the gripper is currently above (inverse of move_gripper_to)."""
        ...
