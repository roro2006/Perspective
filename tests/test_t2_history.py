"""Acceptance tests for T2-01 (grounding/history.py)."""

import threading

import numpy as np
from scene import HUMAN

from curio_robot.geometry import intrinsics_from_fovy
from curio_robot.grounding.history import FrameHistory
from curio_robot.types import HUMAN_CAM, CameraFrame

K = intrinsics_from_fovy(60, 8, 6)


def _frame(tag: float) -> CameraFrame:
    return CameraFrame(HUMAN_CAM, np.zeros((6, 8, 3), np.uint8), K, HUMAN, tag)


def test_empty():
    h = FrameHistory()
    assert h.at(1.0) is None and h.latest() is None and len(h) == 0


def test_nearest_frame():
    h = FrameHistory()
    for t in (1.0, 1.1, 1.2):
        h.push(_frame(t), t)
    assert h.at(1.12).timestamp == 1.1
    assert h.at(1.19).timestamp == 1.2
    assert h.latest().timestamp == 1.2


def test_tie_goes_to_the_earlier_frame():
    h = FrameHistory()
    h.push(_frame(1.0), 1.0)
    h.push(_frame(1.5), 1.5)
    assert h.at(1.25).timestamp == 1.0


def test_too_far_is_none():
    h = FrameHistory(max_gap_s=0.25)
    h.push(_frame(1.0), 1.0)
    assert h.at(1.25).timestamp == 1.0  # exactly at the gap still counts
    assert h.at(1.5) is None
    assert h.at(0.5) is None


def test_old_frames_are_dropped():
    h = FrameHistory(keep_s=3.0)
    for i in range(9):
        h.push(_frame(i * 0.5), i * 0.5)  # 0.0 .. 4.0
    assert len(h) == 7  # 1.0 .. 4.0
    assert h.at(0.9).timestamp == 1.0


def test_out_of_order_push():
    h = FrameHistory()
    h.push(_frame(2.0), 2.0)
    h.push(_frame(1.9), 1.9)
    assert h.at(1.9).timestamp == 1.9
    assert h.latest().timestamp == 2.0


def test_concurrent_pushes():
    h = FrameHistory(keep_s=1e9)

    def worker(k):
        for i in range(200):
            t = k * 1000 + i
            h.push(_frame(t), float(t))

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert len(h) == 800
