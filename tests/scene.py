"""A robot and a person facing each other across a table, shared by tests."""

from __future__ import annotations

import numpy as np

from curio_robot.geometry import look_at
from curio_robot.types import ROBOT_CAM, CameraPose, WorldObject


def make_pose(eye, look) -> CameraPose:
    eye = np.asarray(eye, float)
    return CameraPose(eye, look_at(eye, np.asarray(look, float)))


ROBOT = make_pose([-0.2, 0, 0.5], [0.25, 0, 0])  # robot faces +x: its left is +y
HUMAN = make_pose([0.7, 0, 0.4], [0.25, 0, 0])  # person faces -x: their left is -y


def obj(oid: str, color: str, xy, seen=(ROBOT_CAM,)) -> WorldObject:
    return WorldObject(oid, color, np.array([*xy, 0.0125]), set(seen))
