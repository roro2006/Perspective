"""Fuse detections from the robot's and the human's cameras into one map of
the table. Turning the human view off is the "Curio off" switch in the demo."""

from __future__ import annotations

import numpy as np

from curio_robot.geometry import lift_to_plane
from curio_robot.types import HUMAN_CAM, ROBOT_CAM, CameraFrame, Detection, ObjectMap, WorldObject

MERGE_RADIUS = 0.04  # metres: same colour within this distance = same object


def build_object_map(
    frames: dict[str, CameraFrame],
    detections: dict[str, list[Detection]],
    *,
    use_human_view: bool,
) -> ObjectMap:
    cameras = [ROBOT_CAM] + ([HUMAN_CAM] if use_human_view and HUMAN_CAM in frames else [])
    objects: list[WorldObject] = []
    counts: dict[str, int] = {}
    # Robot camera first, so merged positions prefer the robot's own estimate.
    for cam in cameras:
        frame = frames[cam]
        for det in detections.get(cam, []):
            p = lift_to_plane(det.pixel, frame.intrinsics, frame.pose)
            if p is None:
                continue
            match = next(
                (
                    o
                    for o in objects
                    if o.color == det.color
                    and np.linalg.norm(o.position[:2] - p[:2]) < MERGE_RADIUS
                ),
                None,
            )
            if match:
                match.seen_by.add(cam)
                continue
            counts[det.color] = counts.get(det.color, 0) + 1
            objects.append(
                WorldObject(
                    object_id=f"{det.color}_{counts[det.color]}",
                    color=det.color,
                    position=p,
                    seen_by={cam},
                )
            )
    return ObjectMap(
        objects=objects,
        robot_pose=frames[ROBOT_CAM].pose,
        human_pose=frames[HUMAN_CAM].pose if HUMAN_CAM in cameras else None,
    )
