"""Arm motion on its own thread, in small steps toward the current goal. A new
goal (the VLM redirecting a speculative start) or a stop (the human reacting)
takes effect on the next step, mid-motion, instead of after a blocking move."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable

import numpy as np

from curio_robot.events import ArmEvent
from curio_robot.world import World

SPEED = 0.25  # m/s at the gripper
RATE_HZ = 50
ARRIVE_RADIUS = 0.006  # metres
STALL_TICKS = 25  # no progress for this many steps = unreachable


class Motion:
    def __init__(
        self,
        world: World,
        emit: Callable[[ArmEvent], None],
        *,
        speed: float = SPEED,
        rate_hz: float = RATE_HZ,
    ) -> None:
        self._world, self._emit = world, emit
        self._step = speed / rate_hz
        self._period = 1.0 / rate_hz
        self._commands: queue.Queue[tuple[np.ndarray | None, str]] = queue.Queue()
        self._running = False
        self._thread = threading.Thread(target=self._loop, name="motion", daemon=True)

    # Called from any thread.
    def go(self, goal: np.ndarray, tag: str) -> None:
        self._commands.put((np.asarray(goal, float).copy(), tag))

    def stop(self) -> None:
        self._commands.put((None, "stop"))

    def start(self) -> None:
        self._running = True
        self._thread.start()

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=2)

    def _loop(self) -> None:
        goal: np.ndarray | None = None
        tag = ""
        best, stalled = np.inf, 0
        while self._running:
            try:
                if goal is None:
                    new_goal, new_tag = self._commands.get(timeout=0.1)
                else:
                    new_goal, new_tag = self._commands.get_nowait()
                while not self._commands.empty():  # only the newest command matters
                    new_goal, new_tag = self._commands.get_nowait()
                if new_goal is None:
                    if goal is not None:
                        self._emit(ArmEvent("stopped", goal, time.perf_counter(), tag))
                    goal = None
                    continue
                if goal is None or not np.allclose(new_goal, goal):
                    self._emit(ArmEvent("started", new_goal, time.perf_counter(), new_tag))
                    best, stalled = np.inf, 0
                goal, tag = new_goal, new_tag
            except queue.Empty:
                pass
            if goal is None:
                continue
            t0 = time.perf_counter()
            here = np.asarray(self._world.gripper_position(), float)
            delta = goal - here
            dist = float(np.linalg.norm(delta))
            if dist >= ARRIVE_RADIUS:
                waypoint = goal if dist <= self._step else here + delta / dist * self._step
                self._world.move_gripper_to(waypoint)
            else:
                self._emit(ArmEvent("arrived", goal, time.perf_counter(), tag))
                goal = None
                continue
            if dist < best - 1e-4:
                best, stalled = dist, 0
            else:
                stalled += 1
                if stalled > STALL_TICKS:
                    self._emit(ArmEvent("unreachable", goal, time.perf_counter(), tag))
                    goal = None
                    continue
            time.sleep(max(0.0, self._period - (time.perf_counter() - t0)))
