"""Real bench: a robot-side webcam, a phone head cam and (later) the SO-101.

TODO(Track 2, owner B):
- capture: two OpenCV streams (webcam + DroidCam/IP Webcam), timestamped.
- calibration: ArUco board on the table -> solvePnP per camera every frame,
  giving CameraPose in table coordinates (see calibration.py).
- move_gripper_to: until the arm arrives, mirror the target into the sim twin
  (curio_robot.sim.world.SimWorld.move_gripper_to); then LeRobot SO-101 driver.
"""

from __future__ import annotations

import numpy as np

from curio_robot.types import CameraFrame


class RealBench:
    def capture(self) -> dict[str, CameraFrame]:
        raise NotImplementedError

    def move_gripper_to(self, position: np.ndarray) -> bool:
        raise NotImplementedError

    def gripper_position(self) -> np.ndarray:
        raise NotImplementedError
