"""One step of Curio: frames from both cameras + a request -> fused map -> target.
Identical for the sim and for real cameras; only the World changes."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from curio_robot.fusion import build_object_map
from curio_robot.perception import Detector
from curio_robot.resolver import resolve
from curio_robot.types import CameraFrame, ObjectMap, Target


@dataclass(slots=True)
class StepResult:
    object_map: ObjectMap
    target: Target
    timings_ms: dict[str, float] = field(default_factory=dict)


def step(
    frames: dict[str, CameraFrame], request: str, detector: Detector, *, use_human_view: bool
) -> StepResult:
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    detections = {cam: detector.detect(frame) for cam, frame in frames.items()}
    t1 = time.perf_counter()
    object_map = build_object_map(frames, detections, use_human_view=use_human_view)
    t2 = time.perf_counter()
    target = resolve(request, object_map)
    t3 = time.perf_counter()
    timings["detect"] = (t1 - t0) * 1000
    timings["fuse"] = (t2 - t1) * 1000
    timings["resolve"] = (t3 - t2) * 1000
    return StepResult(object_map, target, timings)
