"""T4-05: the gaze rule as a critic."""

from __future__ import annotations

import numpy as np
from critic_util import make_ep

from curio_robot.critic.baseline import rule_scores


def test_a_steady_look_at_the_other_red_fires_after_the_dwell():
    p, intended = rule_scores(make_ep([0] * 5 + [1] * 26, intended=1, arm_target=0))
    assert np.all(p[:9] == 0.0) and np.all(p[9:] == 1.0)  # look starts 0.5 s, fires 0.9 s
    assert np.array_equal(intended[9], [0.0, 1.0, 0.0])
    assert np.array_equal(intended[0], [1.0, 0.0, 0.0])


def test_looking_at_the_target_or_another_colour_never_fires():
    p, _ = rule_scores(make_ep([0] * 31, arm_target=0))
    assert np.all(p == 0.0)
    p, _ = rule_scores(make_ep([2] * 31, arm_target=0))  # blue is not a candidate
    assert np.all(p == 0.0)


def test_a_heard_no_means_the_other_candidate():
    ep = make_ep([0] * 31, intended=1, arm_target=0, speech=((1.2, "no"),), t_no=1.0)
    p, intended = rule_scores(ep)
    assert np.all(p[:12] == 0.0) and np.all(p[12:] == 1.0)
    assert np.array_equal(intended[12], [0.0, 1.0, 0.0])


def test_shapes():
    p, intended = rule_scores(make_ep())
    assert p.shape == (31,) and intended.shape == (31, 3)
    assert np.allclose(intended.sum(axis=1), 1.0)
