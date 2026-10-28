"""Shared data types. Every world backend (sim, real cameras) produces these,
and every pipeline stage consumes them, so the two backends stay swappable."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ROBOT_CAM = "robot"
HUMAN_CAM = "human"


@dataclass(frozen=True, slots=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class CameraPose:
    """Camera-to-world transform. Convention (MuJoCo/OpenGL): the camera looks
    along its -z axis, +x is right in the image, +y is up."""

    position: np.ndarray  # (3,)
    rotation: np.ndarray  # (3, 3), columns are the camera axes in world coordinates

    @property
    def right(self) -> np.ndarray:
        return self.rotation[:, 0]

    @property
    def forward(self) -> np.ndarray:
        return -self.rotation[:, 2]


@dataclass(slots=True)
class CameraFrame:
    camera: str  # ROBOT_CAM or HUMAN_CAM
    rgb: np.ndarray  # (H, W, 3) uint8
    intrinsics: Intrinsics
    pose: CameraPose
    timestamp: float
    # Sim only: per-pixel object ids, used by the ground-truth detector.
    segmentation: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class Detection:
    camera: str
    color: str
    pixel: tuple[float, float]  # (u, v) centroid of the visible pixels
    area: int  # visible pixels


@dataclass(slots=True)
class WorldObject:
    """One object in the fused map of the table, in table (world) coordinates."""

    object_id: str
    color: str
    position: np.ndarray  # (3,)
    seen_by: set[str] = field(default_factory=set)


@dataclass(slots=True)
class ObjectMap:
    objects: list[WorldObject]
    robot_pose: CameraPose
    human_pose: CameraPose | None  # None when the human view is off

    def by_color(self, color: str) -> list[WorldObject]:
        return [o for o in self.objects if o.color == color]


@dataclass(frozen=True, slots=True)
class Target:
    object_id: str | None  # None: nothing matched the request
    position: np.ndarray | None
    reason: str
