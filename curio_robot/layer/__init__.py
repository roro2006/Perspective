"""Curio as a plug-in layer: resolve() turns what a person said into a Goal, and the adapters
put that Goal in the input format of any robot policy. Task T5-27; see docs/curio-layer.md."""

from __future__ import annotations

from curio_robot.layer.adapters import pose, ring, words
from curio_robot.layer.goal import Base, Goal, base_from_camera
from curio_robot.layer.resolve import GroundingUnavailable, resolve
from curio_robot.layer.robot_words import check, robot_words

__all__ = ["Base", "Goal", "GroundingUnavailable", "base_from_camera", "check", "pose",
           "resolve", "ring", "robot_words", "words"]
