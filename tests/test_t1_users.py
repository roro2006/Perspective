"""Acceptance tests for T1-06 (sim/users.py)."""

import numpy as np
import pytest

from curio_robot.sim.users import POPULATION, SimUser, Utterance


def test_own_frame_unmarked():
    u = SimUser("x", p_own_frame=1.0, p_marker=0.0).utter("red", "left", np.random.default_rng(0))
    assert u == Utterance("Hand me the red one on the left", "speaker", "red", "left")


def test_robot_frame_marked():
    u = SimUser("x", p_own_frame=0.0, p_marker=1.0).utter("blue", "right", np.random.default_rng(0))
    assert u == Utterance("Hand me the blue one on your right", "robot", "blue", "right")


def test_own_frame_marked():
    u = SimUser("x", p_own_frame=1.0, p_marker=1.0).utter("red", "right", np.random.default_rng(0))
    assert u.text == "Hand me the red one on my right"


def test_frame_frequency():
    rng = np.random.default_rng(1)
    user = SimUser("x", p_own_frame=0.3)
    frames = [user.utter("red", "left", rng).frame for _ in range(2000)]
    assert abs(frames.count("speaker") / 2000 - 0.3) < 0.04


def test_rng_order_is_frame_then_marker():
    draws = np.random.default_rng(5).random(2)
    u = SimUser("x", p_own_frame=0.5, p_marker=0.5).utter("red", "left", np.random.default_rng(5))
    assert u.frame == ("speaker" if draws[0] < 0.5 else "robot")
    words = u.text.split()
    assert ("my" in words or "your" in words) == (draws[1] < 0.5)


def test_intended(pair_map):
    user = SimUser("x", p_own_frame=0.5)
    assert user.intended(Utterance("", "speaker", "red", "left"), pair_map) == "red_2"
    assert user.intended(Utterance("", "robot", "red", "left"), pair_map) == "red_1"
    assert user.intended(Utterance("", "speaker", "red", "right"), pair_map) == "red_1"


def test_intended_errors(pair_map, robot_only_map):
    user = SimUser("x", p_own_frame=0.5)
    with pytest.raises(ValueError):
        user.intended(Utterance("", "speaker", "green", "left"), pair_map)
    with pytest.raises(ValueError):
        user.intended(Utterance("", "speaker", "red", "left"), robot_only_map)


def test_answer_and_react():
    user = SimUser("x", p_own_frame=0.5)
    assert user.answer(Utterance("", "speaker", "red", "left")) == "mine"
    assert user.answer(Utterance("", "robot", "red", "left")) == "yours"
    assert user.react("red_1", "red_1") is None
    assert user.react("red_1", "red_2") == "no, the other one"
    assert user.react("red_1", None) == "no, the other one"


def test_population():
    assert [u.name for u in POPULATION] == ["own-frame", "robot-frame", "mixed"]
