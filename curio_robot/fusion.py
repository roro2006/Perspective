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


TRACK_RADIUS = 0.05  # metres: a detection this close to a same-colour track continues it
MAX_MISSED = 15  # frames a track survives unseen (e.g. while the arm covers it)


class Tracker:
    """Keeps object ids stable across frames. The live runtime rebuilds the map
    every frame, and a [TARGET:red_2] the VLM sends back a second later must
    still name the same block."""

    def __init__(self, radius: float = TRACK_RADIUS, max_missed: int = MAX_MISSED) -> None:
        self._radius = radius
        self._max_missed = max_missed
        self._tracks: dict[str, WorldObject] = {}
        self._missed: dict[str, int] = {}
        self._counts: dict[str, int] = {}

    def update(self, fresh: ObjectMap) -> ObjectMap:
        matched: set[str] = set()
        for obj in fresh.objects:
            best = min(
                (
                    t
                    for t in self._tracks.values()
                    if t.color == obj.color and t.object_id not in matched
                ),
                key=lambda t: np.linalg.norm(t.position[:2] - obj.position[:2]),
                default=None,
            )
            if best is not None and (
                np.linalg.norm(best.position[:2] - obj.position[:2]) < self._radius
            ):
                best.position = 0.5 * best.position + 0.5 * obj.position
                best.seen_by = set(obj.seen_by)
            else:
                self._counts[obj.color] = self._counts.get(obj.color, 0) + 1
                best = WorldObject(
                    f"{obj.color}_{self._counts[obj.color]}", obj.color, obj.position.copy(),
                    set(obj.seen_by),
                )
                self._tracks[best.object_id] = best
            matched.add(best.object_id)
            self._missed[best.object_id] = 0
        for oid in list(self._tracks):
            if oid in matched:
                continue
            self._missed[oid] += 1
            self._tracks[oid].seen_by = set()
            if self._missed[oid] > self._max_missed:
                del self._tracks[oid], self._missed[oid]
        objects = [
            WorldObject(t.object_id, t.color, t.position.copy(), set(t.seen_by))
            for t in self._tracks.values()
        ]
        return ObjectMap(objects, fresh.robot_pose, fresh.human_pose)
