"""Acceptance tests for T1-09 (sim/learn.py, fast mode)."""

from curio_robot.sim.learn import learning_curve
from curio_robot.sim.users import SimUser

ROBOT_FRAME = SimUser("robot-frame", p_own_frame=0.0)
OWN_FRAME = SimUser("own-frame", p_own_frame=1.0)
MIXED = SimUser("mixed", p_own_frame=0.5)


def _mean(xs):
    return sum(xs) / len(xs)


def test_result_shape():
    r = learning_curve(ROBOT_FRAME, requests=4, users=5, seed=0)
    assert len(r["error_rate"]) == len(r["ask_rate"]) == 4
    assert all(0.0 <= e + a <= 1.0 for e, a in zip(r["error_rate"], r["ask_rate"]))
    assert {"user", "requests", "users", "learn", "final_p_own_frame"} <= r.keys()


# In ~20% of random scenes both frames' "left" is the same block (the person
# sits at an angle), so even an unlearned robot gets those right by luck.


def test_robot_frame_person_is_learned():
    r = learning_curve(ROBOT_FRAME, requests=10, users=40, seed=0)
    assert r["error_rate"][0] > 0.6  # the prior says "own frame": usually wrong at first
    assert _mean(r["error_rate"][-3:]) < 0.1
    assert _mean(r["ask_rate"][-3:]) < 0.2
    assert r["final_p_own_frame"] < 0.3


def test_without_learning_it_stays_wrong():
    r = learning_curve(ROBOT_FRAME, requests=10, users=40, seed=0, learn=False)
    assert min(r["error_rate"]) > 0.6


def test_own_frame_person_is_never_hurt():
    r = learning_curve(OWN_FRAME, requests=10, users=40, seed=0)
    assert max(r["error_rate"]) < 0.05
    assert r["final_p_own_frame"] > 0.8


def test_mixed_person_gets_asked_rather_than_guessed():
    r = learning_curve(MIXED, requests=15, users=40, seed=0)
    assert _mean(r["ask_rate"][-5:]) > _mean(r["error_rate"][-5:])


def test_noise_does_not_break_learning():
    r = learning_curve(ROBOT_FRAME, requests=10, users=40, seed=0, pos_noise=0.01, rot_noise_deg=2)
    assert _mean(r["error_rate"][-3:]) < 0.25
