"""T4-02: reaction episodes."""

from __future__ import annotations

import numpy as np
import pytest

from curio_robot.critic.episodes import (
    HOME,
    HOVER,
    arm_path,
    frame_times,
    from_dict,
    generate,
    load,
    make_episode,
    save,
    to_dict,
)


def test_frame_times():
    t = frame_times(3.0, 10.0)
    assert len(t) == 31 and t[0] == 0.0 and t[-1] == pytest.approx(3.0)
    assert np.allclose(np.diff(t), 0.1)


def test_arm_path_is_a_constant_speed_line_that_holds():
    t = np.array([-0.5, 0.0, 0.2, 0.4, 2.0])
    start, goal = np.zeros(3), np.array([0.1, 0.0, 0.0])
    p = arm_path(t, start, goal, speed=0.25)
    assert np.allclose(p[:, 0], [0.0, 0.0, 0.05, 0.1, 0.1])
    assert np.allclose(p[:, 1:], 0.0)
    assert np.allclose(arm_path(t, goal, goal), goal)


def test_make_episode_is_determined_by_its_arguments():
    a, b = make_episode(11, "slow", True), make_episode(11, "slow", True)
    assert np.array_equal(a.head_fwd, b.head_fwd) and a.speech == b.speech
    assert a.intended == b.intended and a.colors == b.colors


@pytest.mark.parametrize("wrong", [False, True])
def test_episode_layout(wrong):
    ep = make_episode(5, "attentive", wrong)
    assert ep.t.shape == (31,) and ep.gripper.shape == (31, 3) and ep.head_fwd.shape == (31, 3)
    assert ep.obj_pos.shape == (5, 3) and len(ep.colors) == 5
    assert 0 <= ep.intended < 4
    assert ep.request_color == ep.colors[ep.intended]
    assert ep.candidates.sum() == 2
    assert ep.wrong == wrong
    assert ep.arm_target == (ep.intended ^ 1 if wrong else ep.intended)
    assert np.allclose(ep.gripper[0], HOME)
    assert np.allclose(ep.gripper[-1], ep.obj_pos[ep.arm_target] + [0, 0, HOVER])
    assert ep.family == "attentive" and ep.seed == 5


def test_generate_cycles_families_and_respects_p_wrong():
    eps = generate(8, seed=3, families=("slow", "silent"))
    assert [e.family for e in eps] == ["slow", "silent"] * 4
    assert not any(e.wrong for e in generate(6, seed=1, p_wrong=0.0))
    assert all(e.wrong for e in generate(6, seed=1, p_wrong=1.0))
    again = generate(8, seed=3, families=("slow", "silent"))
    assert [e.seed for e in eps] == [e.seed for e in again]


def test_dict_and_file_round_trip(tmp_path):
    eps = generate(4, seed=2, p_wrong=1.0)
    back = from_dict(to_dict(eps[0]))
    assert np.array_equal(back.head_fwd, eps[0].head_fwd)
    assert back.speech == eps[0].speech and isinstance(back.colors, tuple)
    path = tmp_path / "sub" / "eps.jsonl"
    save(eps, path)
    loaded = load(path)
    assert len(loaded) == 4
    for a, b in zip(eps, loaded, strict=True):
        assert np.allclose(a.gripper, b.gripper) and a.t_no == b.t_no
        assert a.intended == b.intended and a.family == b.family
