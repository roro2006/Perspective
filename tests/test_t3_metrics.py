"""T3-02: fluency numbers from tracer marks, on a hand-built session whose
answers are worked out in the comments."""

from __future__ import annotations

from dataclasses import fields

import pytest

from curio_robot.collab.metrics import Fluency, fluency, summarize
from curio_robot.collab.session import Session, Step


def worked() -> Session:
    steps = (
        # speaks 0-2, arm moves 1.5-4 (started before they finished), reply at 2.3
        Step(1, "request", "Hand me the blue one", "a", "a", False,
             {"speech_start": 0.0, "speech_end": 2.0, "motion_start": 1.5, "arrived": 4.0,
              "tts_start": 2.3}),
        # speaks 5-6, the robot asks, arm moves 6.5-8 to the wrong one, reply at 6.4
        Step(2, "request", "Hand me the red one on the left", "b", "c", True,
             {"speech_start": 5.0, "speech_end": 6.0, "motion_start": 6.5, "arrived": 8.0,
              "tts_start": 6.4}),
        # "no" said 7-7.5 while the arm is still moving
        Step(3, "correction", "no, the other one", "b", None, False,
             {"speech_start": 7.0, "speech_end": 7.5}),
    )
    return Session("w", "p", "view-on", "fetch", 0, 0.0, 10.0, ("a", "b"), ("a",), steps)


def quiet(condition: str = "view-on") -> Session:
    """20 s, one request said 0-1, nothing moved."""
    step = Step(1, "request", "Hand me the blue one", "a", None, False,
                {"speech_start": 0.0, "speech_end": 1.0})
    return Session("q", "p", condition, "fetch", 0, 0.0, 20.0, ("a",), ("a",), (step,))


def test_worked_example():
    f = fluency(worked())
    assert f.completion_s == pytest.approx(10.0)
    assert f.success == pytest.approx(0.5)
    assert (f.turns, f.corrections, f.questions) == (3, 1, 1)
    assert f.human_idle_s == pytest.approx(4.0)  # (4 - 2) + (8 - 6)
    assert f.robot_idle_frac == pytest.approx(0.6)  # moving 2.5 + 1.5 of 10
    assert f.concurrent_frac == pytest.approx(0.1)  # [1.5, 2] and [7, 7.5]
    assert f.functional_delay_s == pytest.approx(0.0)  # median(-0.5, 0.5)
    assert f.reply_s == pytest.approx(0.35)  # median(0.3, 0.4)


def test_overlapping_moves_are_merged():
    steps = (
        Step(1, "request", "x", "a", "a", False, {"motion_start": 1.0, "arrived": 3.0}),
        Step(2, "request", "y", "b", "b", False, {"motion_start": 2.0, "arrived": 4.0}),
    )
    s = Session("m", "p", "view-on", "fetch", 0, 0.0, 10.0, ("a", "b"), ("a", "b"), steps)
    assert fluency(s).robot_idle_frac == pytest.approx(0.7)  # union [1, 4]


def test_arriving_before_they_finish_is_not_waiting():
    step = Step(1, "request", "x", "a", "a", False,
                {"speech_start": 0.0, "speech_end": 3.0, "motion_start": 0.5, "arrived": 2.0})
    s = Session("e", "p", "view-on", "fetch", 0, 0.0, 5.0, ("a",), ("a",), (step,))
    f = fluency(s)
    assert f.human_idle_s == 0.0
    assert f.functional_delay_s == pytest.approx(-2.5)
    assert f.concurrent_frac == pytest.approx(0.3)  # [0.5, 2] of 5


def test_empty_session():
    f = fluency(Session("z", "p", "view-off", "fetch", 0, 5.0, 5.0, (), (), ()))
    assert f.completion_s == 0.0 and f.success == 0.0
    assert (f.turns, f.corrections, f.questions) == (0, 0, 0)
    assert f.human_idle_s == 0.0 and f.robot_idle_frac == 0.0 and f.concurrent_frac == 0.0
    assert f.functional_delay_s is None and f.reply_s is None


def test_quiet_session_has_no_delays():
    f = fluency(quiet())
    assert f.robot_idle_frac == pytest.approx(1.0) and f.concurrent_frac == 0.0
    assert f.functional_delay_s is None and f.reply_s is None


def test_summarize_by_condition():
    out = summarize([worked(), quiet(), quiet("view-off")])
    assert list(out) == ["view-off", "view-on"]
    on = out["view-on"]
    assert list(on) == [f.name for f in fields(Fluency)] + ["n"]
    assert on["n"] == 2.0
    assert on["completion_s"] == pytest.approx(15.0)
    assert on["success"] == pytest.approx(0.75)
    assert on["turns"] == pytest.approx(2.0)
    assert on["human_idle_s"] == pytest.approx(2.0)
    assert on["robot_idle_frac"] == pytest.approx(0.8)
    assert on["functional_delay_s"] == pytest.approx(0.0)  # quiet() has None: ignored
    assert on["reply_s"] == pytest.approx(0.35)
    off = out["view-off"]
    assert off["n"] == 1.0 and off["reply_s"] is None and off["success"] == pytest.approx(1.0)


def test_summarize_nothing():
    assert summarize([]) == {}
