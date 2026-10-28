"""A few seconds of recent frames, so the grounder can look at what the person
saw when they said "that", not what the camera sees now. Task T2-01."""

from __future__ import annotations

from curio_robot.types import CameraFrame

KEEP_S = 3.0
MAX_GAP_S = 0.25  # a frame further than this from the asked time doesn't count


class FrameHistory:
    """Thread-safe ring of (t, frame), t on the runtime clock (perf_counter).
    CameraFrame.timestamp is the capture source's own clock (sim time on the
    sim), so the caller passes t explicitly."""

    def __init__(self, keep_s: float = KEEP_S, max_gap_s: float = MAX_GAP_S) -> None:
        raise NotImplementedError("T2-01")

    def push(self, frame: CameraFrame, t: float) -> None:
        """Add a frame. Drop frames older than newest t - keep_s. Pushes with
        t older than the newest are allowed and kept in time order."""
        raise NotImplementedError("T2-01")

    def at(self, t: float) -> CameraFrame | None:
        """The frame whose t is nearest to `t` (ties: the earlier one), or None
        if empty or the nearest is more than max_gap_s away."""
        raise NotImplementedError("T2-01")

    def latest(self) -> CameraFrame | None:
        raise NotImplementedError("T2-01")

    def __len__(self) -> int:
        raise NotImplementedError("T2-01")
