"""T4-04: the learned critic."""

from __future__ import annotations

import numpy as np
import pytest
from critic_util import make_ep

from curio_robot.critic.features import D
from curio_robot.critic.model import Critic


def _dataset(n: int = 40, seed: int = 0):
    """Wrong reaches: the head moves to the wanted block at frame 10. Right
    reaches: it stays on the arm's target."""
    rng = np.random.default_rng(seed)
    eps = []
    for i in range(n):
        intended = int(rng.integers(0, 2))
        wrong = i % 2 == 0
        arm = 1 - intended if wrong else intended
        looks = [arm] * 10 + [intended if wrong else arm] * 21
        eps.append(make_ep(looks, intended=intended, arm_target=arm))
    return eps


def test_probs_are_a_softmax_over_the_mask():
    c = Critic(seed=0)
    x = np.random.default_rng(1).normal(size=(2, 3, D))
    p = c.probs(x, np.array([True, True, False]))
    assert p.shape == (2, 3)
    assert np.allclose(p.sum(axis=1), 1.0) and np.all(p[:, 2] == 0.0)
    assert np.all(c.probs(x, np.zeros(3, bool)) == 0.0)


def test_predict_shapes_and_p_wrong():
    ep = make_ep()
    p_wrong, intended = Critic(seed=0).predict(ep)
    assert p_wrong.shape == (31,) and intended.shape == (31, 3)
    assert np.allclose(p_wrong, 1.0 - intended[:, ep.arm_target])


def test_fit_learns_to_read_a_look_at_the_other_block():
    eps = _dataset()
    c = Critic(seed=0)
    losses = c.fit(eps, epochs=60, lr=0.03, batch=64, seed=0)
    assert len(losses) == 60 and losses[-1] < losses[0]
    end_wrong = [c.predict(e)[0][-1] for e in eps if e.wrong]
    end_right = [c.predict(e)[0][-1] for e in eps if not e.wrong]
    assert np.mean(end_wrong) > 0.8 and np.mean(end_right) < 0.2
    _, intended = c.predict(eps[0])
    assert int(np.argmax(intended[-1])) == eps[0].intended


def test_fit_is_deterministic():
    eps = _dataset(12)
    a, b = Critic(seed=3), Critic(seed=3)
    assert a.fit(eps, epochs=3, seed=1) == pytest.approx(b.fit(eps, epochs=3, seed=1))


def test_save_and_load(tmp_path):
    eps = _dataset(8)
    c = Critic(seed=2)
    c.fit(eps, epochs=2)
    c.save(tmp_path / "m" / "critic.npz")
    back = Critic.load(tmp_path / "m" / "critic.npz")
    assert np.allclose(back.predict(eps[1])[0], c.predict(eps[1])[0])
