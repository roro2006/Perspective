"""Acceptance tests for T1-02 (conventions/belief.py)."""

import json

import pytest

from curio_robot.conventions.belief import (
    ACCEPT_WEIGHT,
    CORRECT_WEIGHT,
    ConventionBelief,
    learn_from_outcome,
)
from curio_robot.conventions.readings import Reading

S = Reading("red_2", "speaker", "left", False)  # speaker's left
R = Reading("red_1", "robot", "left", False)  # robot's left


def test_prior_mean():
    assert ConventionBelief().p_own_frame() == pytest.approx(0.8)


def test_posterior_weighs_frames_by_belief():
    post = ConventionBelief().posterior([S, R])
    assert post == pytest.approx({"red_2": 0.8, "red_1": 0.2})


def test_posterior_explicit_and_frameless():
    b = ConventionBelief()
    assert b.posterior([Reading("red_2", "speaker", "left", True)]) == {"red_2": 1.0}
    frameless = [Reading("red_1", None, None, False), Reading("red_2", None, None, False)]
    assert b.posterior(frameless) == pytest.approx({"red_1": 0.5, "red_2": 0.5})
    assert b.posterior([]) == {}


def test_posterior_merges_readings_of_the_same_object():
    same = [Reading("blue_1", "speaker", "left", False), Reading("blue_1", "robot", "left", False)]
    assert ConventionBelief().posterior(same) == pytest.approx({"blue_1": 1.0})


def test_posterior_all_zero_weights_is_uniform():
    b = ConventionBelief(own_frame_a=1.0, own_frame_b=0.0)  # p_own = 1: robot readings weigh 0
    r2 = Reading("red_2", "robot", "left", False)
    assert b.posterior([R, r2]) == pytest.approx({"red_1": 0.5, "red_2": 0.5})


@pytest.mark.parametrize(
    ("frame", "correct", "da", "db"),
    [("speaker", True, 2, 0), ("robot", False, 2, 0), ("speaker", False, 0, 2), ("robot", True, 0, 2)],
)
def test_update(frame, correct, da, db):
    b = ConventionBelief(own_frame_a=1.0, own_frame_b=1.0)
    b.update(frame, correct, 2.0)
    assert (b.own_frame_a, b.own_frame_b) == pytest.approx((1.0 + da, 1.0 + db))


def test_update_ignores_non_positive_weight():
    b = ConventionBelief()
    b.update("speaker", True, 0.0)
    b.update("speaker", True, -1.0)
    assert (b.own_frame_a, b.own_frame_b) == pytest.approx((1.6, 0.4))


def test_dict_round_trip_is_json_safe():
    b = ConventionBelief(own_frame_a=3.25, own_frame_b=7.5)
    data = json.loads(json.dumps(b.to_dict()))
    assert ConventionBelief.from_dict(data) == b
    assert ConventionBelief.from_dict({**data, "future_field": 1}) == b


def test_corrected_own_frame_moves_toward_robot_frame():
    b = ConventionBelief()
    w = learn_from_outcome(b, [S, R], chosen_id="red_2", outcome="corrected")
    assert w == pytest.approx(CORRECT_WEIGHT)
    assert (b.own_frame_a, b.own_frame_b) == pytest.approx((1.6, 1.4))


def test_corrected_robot_frame_moves_toward_own_frame():
    b = ConventionBelief()
    learn_from_outcome(b, [S, R], chosen_id="red_1", outcome="corrected")
    assert (b.own_frame_a, b.own_frame_b) == pytest.approx((2.6, 0.4))


def test_accepted_is_weak_evidence():
    b = ConventionBelief()
    w = learn_from_outcome(b, [S, R], chosen_id="red_2", outcome="accepted")
    assert w == pytest.approx(ACCEPT_WEIGHT)
    assert b.own_frame_a == pytest.approx(1.6 + ACCEPT_WEIGHT)


def test_answered_uses_the_right_object():
    b = ConventionBelief()
    w = learn_from_outcome(b, [S, R], chosen_id="red_2", outcome="answered", right_id="red_1")
    assert w == pytest.approx(CORRECT_WEIGHT)
    assert b.own_frame_b == pytest.approx(1.4)
    assert learn_from_outcome(b, [S, R], chosen_id="red_2", outcome="answered") == 0.0


def test_flip_probability_discounts_the_update():
    b = ConventionBelief()
    w = learn_from_outcome(b, [S, R], chosen_id="red_2", outcome="corrected", p_flip=0.25)
    assert w == pytest.approx(0.5)
    assert b.own_frame_b == pytest.approx(0.9)
    b2 = ConventionBelief()
    assert learn_from_outcome(b2, [S, R], "red_2", "corrected", p_flip=0.6) == 0.0
    assert b2 == ConventionBelief()


@pytest.mark.parametrize(
    "readings",
    [
        [Reading("red_2", "speaker", "left", True)],  # explicit
        [Reading("red_1", None, None, False), Reading("red_2", None, None, False)],  # no frame
        [Reading("blue_1", "speaker", "left", False), Reading("blue_1", "robot", "left", False)],
    ],
)
def test_nothing_to_learn(readings):
    b = ConventionBelief()
    assert learn_from_outcome(b, readings, readings[0].object_id, "corrected") == 0.0
    assert b == ConventionBelief()


def test_chosen_object_not_in_readings_teaches_nothing():
    b = ConventionBelief()
    assert learn_from_outcome(b, [S, R], chosen_id="blue_1", outcome="corrected") == 0.0
    assert b == ConventionBelief()


def test_robot_frame_person_is_learned_in_a_few_corrections():
    b = ConventionBelief()
    for _ in range(2):
        learn_from_outcome(b, [S, R], chosen_id="red_2", outcome="corrected")
    assert b.p_own_frame() < 0.5
