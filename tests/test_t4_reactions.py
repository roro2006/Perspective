"""T4-01: simulated reactions to a reach."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from critic_util import HEAD, OBJ, path, times, unit

from curio_robot.critic.reactions import FAMILIES, RANGES, Style, sample_style, simulate_reaction


def _style(**kw) -> Style:
    base = Style(family="attentive", notice_s=0.3, p_look_intended=1.0, look_delay_s=0.2,
                 speak_delay_s=0.6, head_noise_deg=0.0, glance_rate_hz=0.0, glance_s=0.3)
    return replace(base, **kw)


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip(np.dot(unit(a), unit(b)), -1, 1))))


def _run(style: Style, intended: int = 1, arm_target: int = 0, seed: int = 0):
    t = times()
    grip = path(t, OBJ[arm_target] + [0, 0, 0.03])
    return simulate_reaction(t, HEAD, OBJ, intended, arm_target, grip, style,
                             np.random.default_rng(seed))


@pytest.mark.parametrize("family", FAMILIES)
def test_styles_stay_in_their_family_ranges(family):
    ranges = RANGES["attentive" if family == "silent" else family]
    for seed in range(20):
        s = sample_style(np.random.default_rng(seed), family)
        assert s.family == family
        values = (s.notice_s, s.p_look_intended, s.look_delay_s, s.speak_delay_s,
                  s.head_noise_deg, s.glance_rate_hz, s.glance_s)
        for i, (v, (lo, hi)) in enumerate(zip(values, ranges, strict=True)):
            if family == "silent" and i == 3:
                assert v == math.inf
            else:
                assert lo <= v <= hi


def test_style_draws_are_deterministic_and_unknown_family_fails():
    a = sample_style(np.random.default_rng(3), "slow")
    assert a == sample_style(np.random.default_rng(3), "slow")
    with pytest.raises(ValueError):
        sample_style(np.random.default_rng(0), "sleepy")


def test_shapes_and_unit_head_vectors():
    r = _run(_style(head_noise_deg=4.0, glance_rate_hz=1.0))
    assert r.head_fwd.shape == (31, 3)
    assert np.allclose(np.linalg.norm(r.head_fwd, axis=1), 1.0)


def test_a_wrong_reach_gets_a_spoken_no_at_the_styled_time():
    r = _run(_style(notice_s=0.3, speak_delay_s=0.6))
    assert r.t_no == pytest.approx(0.9)
    assert [txt for _, txt in r.speech] == ["no", "no, the other one"]
    assert [ts for ts, _ in r.speech] == pytest.approx([1.1, 1.6])


def test_a_right_reach_and_a_silent_person_say_nothing():
    right = _run(_style(), intended=0, arm_target=0)
    assert right.speech == () and right.t_no is None
    silent = _run(_style(speak_delay_s=math.inf))
    assert silent.speech == () and silent.t_no is None


def test_after_noticing_the_head_turns_to_the_wanted_block():
    r = _run(_style(p_look_intended=1.0))
    assert _angle(r.head_fwd[-1], OBJ[1] - HEAD) < 1.0
    assert _angle(r.head_fwd[2], OBJ[1] - HEAD) > 5.0  # before notice + delay: watching the arm


def test_a_person_who_does_not_look_keeps_watching_the_arm():
    r = _run(_style(p_look_intended=0.0))
    assert _angle(r.head_fwd[-1], OBJ[0] - HEAD) < 1.0  # the arm arrived; they look at it


def test_head_starts_on_the_gripper_and_lags_a_jump():
    t = times()
    grip = path(t, OBJ[0] + [0, 0, 0.03])
    r = simulate_reaction(t, HEAD, OBJ, 1, 0, grip, _style(), np.random.default_rng(0))
    assert _angle(r.head_fwd[0], grip[0] - HEAD) < 1e-6
    # eyes jump to block 1 at notice + delay = 0.5 s; one frame later the head is part way
    eye = OBJ[1] - HEAD
    assert 0.5 < _angle(r.head_fwd[5], eye) < _angle(r.head_fwd[4], eye)


def test_same_rng_same_reaction():
    s = _style(head_noise_deg=3.0, glance_rate_hz=0.8)
    a, b = _run(s, seed=7), _run(s, seed=7)
    assert np.array_equal(a.head_fwd, b.head_fwd) and a.speech == b.speech
