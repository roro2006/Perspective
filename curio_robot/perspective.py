"""Spatial relations from a chosen viewpoint. "My left" is computed in the
speaker's camera frame; a robot-only system has to fall back to its own."""

from __future__ import annotations

import numpy as np

from curio_robot.types import CameraPose, WorldObject


def lateral(obj: WorldObject, viewer: CameraPose) -> float:
    """Signed distance to the viewer's right (negative = viewer's left)."""
    return float(np.dot(obj.position - viewer.position, viewer.right))


def horizontal_distance(obj: WorldObject, viewer: CameraPose) -> float:
    return float(np.linalg.norm((obj.position - viewer.position)[:2]))


def pick(candidates: list[WorldObject], relation: str, viewer: CameraPose) -> WorldObject:
    if relation == "left":
        return min(candidates, key=lambda o: lateral(o, viewer))
    if relation == "right":
        return max(candidates, key=lambda o: lateral(o, viewer))
    if relation == "nearest":
        return min(candidates, key=lambda o: horizontal_distance(o, viewer))
    if relation == "farthest":
        return max(candidates, key=lambda o: horizontal_distance(o, viewer))
    raise ValueError(f"unknown relation: {relation}")
