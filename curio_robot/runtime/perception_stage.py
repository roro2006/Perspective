"""The fast half of the cascade: detector on every new frame from both
cameras, fused and tracked into the live object map. Runs on its own thread
behind a latest-value slot, so capture never waits for the GPU."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import numpy as np

from curio_robot.events import Interrupt
from curio_robot.fusion import Tracker, build_object_map
from curio_robot.perception import Detector
from curio_robot.runtime.slot import Latest
from curio_robot.runtime.trace import Tracer
from curio_robot.types import HUMAN_CAM, CameraFrame, Detection, ObjectMap

GATE_PIXEL_DIFF = 2.0  # mean abs grey-level change (0-255) on a 1/8 thumbnail
GATE_POSE_M = 0.005
GATE_POSE_DEG = 0.5


def _thumb(rgb: np.ndarray) -> np.ndarray:
    return rgb[::8, ::8].mean(axis=2)


class FrameGate:
    """Skip detection on human-camera frames that barely changed: the head cam
    is the expensive, mostly-still stream. Detections are reused instead."""

    def __init__(self) -> None:
        self._thumb: np.ndarray | None = None
        self._pose = None

    def changed(self, frame: CameraFrame) -> bool:
        thumb = _thumb(frame.rgb)
        if self._thumb is None or self._pose is None or thumb.shape != self._thumb.shape:
            changed = True
        else:
            moved = np.linalg.norm(frame.pose.position - self._pose.position) > GATE_POSE_M
            r = self._pose.rotation.T @ frame.pose.rotation
            angle = np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1)))
            diff = float(np.abs(thumb - self._thumb).mean())
            changed = moved or angle > GATE_POSE_DEG or diff > GATE_PIXEL_DIFF
        if changed:
            self._thumb, self._pose = thumb, frame.pose
        return changed


class PerceptionStage:
    def __init__(
        self,
        frames: Latest[dict[str, CameraFrame]],
        maps: Latest[ObjectMap],
        detector: Detector,
        tracer: Tracer,
        *,
        use_human_view: Callable[[], bool],
        hand_in_workspace: Callable[[CameraFrame], bool] | None = None,
        emit: Callable[[Interrupt], None] | None = None,
    ) -> None:
        self._frames, self._maps, self._detector, self._tracer = frames, maps, detector, tracer
        self._use_human_view = use_human_view
        self._hand = hand_in_workspace
        self._emit = emit
        self._gate = FrameGate()
        self._tracker = Tracker()
        self._last: dict[str, list[Detection]] = {}
        self._hand_was_in = False
        self._running = False
        self._thread = threading.Thread(target=self._loop, name="perception", daemon=True)

    def start(self) -> None:
        self._running = True
        self._thread.start()

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=2)

    def _loop(self) -> None:
        seen = 0
        while self._running:
            version, frames = self._frames.wait_newer(seen, timeout=0.1)
            if frames is None or version == seen:
                continue
            if version > seen + 1:
                self._tracer.frame("dropped", version - seen - 1)
            seen = version
            self._process(frames)

    def _process(self, frames: dict[str, CameraFrame]) -> None:
        t0 = time.perf_counter()
        todo = [
            f for cam, f in frames.items()
            if cam != HUMAN_CAM or cam not in self._last or self._gate.changed(f)
        ]
        self._tracer.frame("gated", float(len(frames) - len(todo)))
        batch = getattr(self._detector, "detect_batch", None)
        results = batch(todo) if batch else [self._detector.detect(f) for f in todo]
        for f, dets in zip(todo, results):
            self._last[f.camera] = dets
        t1 = time.perf_counter()
        fresh = build_object_map(frames, self._last, use_human_view=self._use_human_view())
        object_map = self._tracker.update(fresh)
        t2 = time.perf_counter()
        self._maps.put(object_map)
        self._tracer.frame("detect", (t1 - t0) * 1000)
        self._tracer.frame("fuse", (t2 - t1) * 1000)
        if self._hand and self._emit and HUMAN_CAM in frames:
            hand_in = self._hand(frames[HUMAN_CAM])
            if hand_in and not self._hand_was_in:
                self._emit(Interrupt("hand", "hand entered workspace", time.perf_counter()))
            self._hand_was_in = hand_in
