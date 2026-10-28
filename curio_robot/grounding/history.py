"""A few seconds of recent frames, so the grounder can look at what the person
saw when they said "that", not what the camera sees now. Task T2-01."""

from __future__ import annotations

import bisect
import threading

from curio_robot.types import CameraFrame

KEEP_S = 3.0
MAX_GAP_S = 0.25  # a frame further than this from the asked time doesn't count


class FrameHistory:
    """Thread-safe ring of (t, frame), t on the runtime clock (perf_counter).
    CameraFrame.timestamp is the capture source's own clock (sim time on the
    sim), so the caller passes t explicitly."""

    def __init__(self, keep_s: float = KEEP_S, max_gap_s: float = MAX_GAP_S) -> None:
        self._keep_s = keep_s
        self._max_gap_s = max_gap_s
        self._lock = threading.Lock()
        self._times: list[float] = []
        self._frames: list[CameraFrame] = []

    def push(self, frame: CameraFrame, t: float) -> None:
        """Add a frame. Drop frames older than newest t - keep_s. Pushes with t older than the newest are allowed and kept in time order."""
        with self._lock:
            idx = bisect.bisect_left(self._times, t)
            self._times.insert(idx, t)
            self._frames.insert(idx, frame)

            # Drop frames older than the window
            newest_t = self._times[-1]
            cutoff = newest_t - self._keep_s

            # Find the first index where time >= cutoff
            # We want to keep frames with time >= cutoff
            # bisect_left finds the insertion point for cutoff
            # If we have [1.0, 2.0] and cutoff is 1.0, bisect_left returns 0.
            # If we have [1.0, 2.0] and cutoff is 1.5, bisect_left returns 1.
            # We want to remove everything before this index.
            start_idx = bisect.bisect_left(self._times, cutoff)

            if start_idx > 0:
                del self._times[:start_idx]
                del self._frames[:start_idx]

    def at(self, t: float) -> CameraFrame | None:
        """The frame whose t is nearest to `t` (ties: the earlier one), or None if empty or the nearest is more than max_gap_s away."""
        with self._lock:
            if not self._times:
                return None

            # Find the insertion point for t
            idx = bisect.bisect_left(self._times, t)

            candidates = []

            # Check the element just before idx (if it exists)
            if idx > 0:
                candidates.append(idx - 1)

            # Check the element at idx (if it exists)
            if idx < len(self._times):
                candidates.append(idx)

            if not candidates:
                return None

            best_idx = None
            best_dist = float('inf')

            for i in candidates:
                dist = abs(self._times[i] - t)
                # Tie-breaking: if dist is equal, we prefer the earlier one (smaller index)
                # Since we check idx-1 before idx, and we use strict < for updating best_dist,
                # the earlier one (idx-1) will be kept if distances are equal.
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i

            if best_idx is None:
                return None

            # Check if the best candidate is within the max gap
            if best_dist > self._max_gap_s:
                return None

            return self._frames[best_idx]

    def latest(self) -> CameraFrame | None:
        """Returns the newest frame or None."""
        with self._lock:
            if not self._frames:
                return None
            return self._frames[-1]

    def __len__(self) -> int:
        """Number of frames."""
        with self._lock:
            return len(self._frames)
