"""Latest-value slot: the hand-off between pipelined stages. A slow consumer
always gets the newest frame or map and silently skips the ones it missed,
so a slow detector adds staleness, never a growing backlog."""

from __future__ import annotations

import threading


class Latest[T]:
    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._value: T | None = None
        self._version = 0

    def put(self, value: T) -> None:
        with self._cond:
            self._value = value
            self._version += 1
            self._cond.notify_all()

    def get(self) -> tuple[int, T | None]:
        with self._cond:
            return self._version, self._value

    def wait_newer(self, version: int, timeout: float) -> tuple[int, T | None]:
        """Block until something newer than `version` arrives (or timeout)."""
        with self._cond:
            self._cond.wait_for(lambda: self._version > version, timeout)
            return self._version, self._value
