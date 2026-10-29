"""T4-03: critic features."""

from __future__ import annotations

import numpy as np
import pytest
from critic_util import make_ep

from curio_robot.critic.features import FEATURES, D, features

COL = {name: i for i, name in enumerate(FEATURES)}


def test_shape_and_flags():
    x = features(make_ep())
    assert x.shape == (31, 3, D)
    assert np.array_equal(x[:, :, COL["is_arm_target"]][0], [1.0, 0.0, 0.0])
    assert np.array_equal(x[:, :, COL["candidate"]][0], [1.0, 1.0, 0.0])


def test_angle_is_scaled_by_the_gaze_cone():
    x = features(make_ep([1] * 31))
    assert x[0, 1, COL["angle"]] == pytest.approx(0.0, abs=1e-6)
    assert x[0, 0, COL["angle"]] == pytest.approx(1.534, abs=0.01)


def test_dwell_grows_caps_and_resets():
    x = features(make_ep([1] * 31))[:, :, COL["dwell"]]
    assert x[0, 1] == 0.0 and x[3, 1] == pytest.approx(0.3) and x[20, 1] == 1.0
    assert np.all(x[:, 0] == 0.0)  # 15 deg off: never in the cone
    broken = features(make_ep([1] * 6 + [None] + [1] * 24))[:, :, COL["dwell"]]
    assert broken[6, 1] == 0.0 and broken[7, 1] == 0.0 and broken[9, 1] == pytest.approx(0.2)


def test_speech_flags_arrive_at_their_time():
    x = features(make_ep(speech=((1.2, "no"), (1.7, "no, the other one")), t_no=1.0))
    said, since = x[:, 0, COL["said_no"]], x[:, 0, COL["since_no"]]
    assert said[11] == 0.0 and said[12] == 1.0 and said[30] == 1.0
    assert since[11] == 0.0 and since[15] == pytest.approx(0.3) and since[30] == pytest.approx(1.8)


def test_motion_features():
    x = features(make_ep())
    assert x[0, 0, COL["progress"]] == pytest.approx(0.0)
    assert x[-1, 0, COL["progress"]] == pytest.approx(1.0)
    assert x[0, 0, COL["approach"]] == 0.0
    assert x[3, 0, COL["approach"]] > 0.8  # heading straight at the target
    assert x[-1, 0, COL["approach"]] == pytest.approx(0.0, abs=1e-9)  # holding
    assert x[3, 1, COL["approach"]] < x[3, 0, COL["approach"]]
    assert 0.0 < x[0, 0, COL["grip_dist"]] <= 2.0


def test_head_speed():
    steady = features(make_ep([1] * 31))[:, 0, COL["head_speed"]]
    assert np.allclose(steady, 0.0)
    jump = features(make_ep([0] * 10 + [1] * 21))[:, 0, COL["head_speed"]]
    assert jump[10] > 0.5 and jump[9] == pytest.approx(0.0, abs=1e-9)


def test_features_are_causal():
    a = make_ep([0] * 31, speech=((2.5, "no"),), t_no=2.3)
    b = make_ep([0] * 20 + [1] * 11)
    assert np.allclose(features(a)[:20], features(b)[:20])


# ---- T4-09: finer gaze, measured against the gripper ----


def _watching_gripper(ep):
    from dataclasses import replace

    from critic_util import HEAD, unit

    return replace(ep, head_fwd=unit(ep.gripper - HEAD))


def test_toward_is_0_on_the_gripper_and_1_on_the_block():
    on_block = features(make_ep([1] * 31))
    assert on_block[5, 1, COL["toward"]] == pytest.approx(1.0, abs=1e-6)
    on_grip = features(_watching_gripper(make_ep()))
    assert np.allclose(on_grip[:, :, COL["toward"]], 0.0, atol=1e-6)
    assert np.allclose(on_grip[:, 0, COL["off_gripper"]], 0.0, atol=1e-4)
    assert on_block[5, 0, COL["off_gripper"]] > 0.5


def test_toward_ema_lags_a_jump_and_toward_vel_spikes():
    x = features(make_ep([0] * 10 + [1] * 21))
    jump, ema, vel = x[:, 1, COL["toward"]], x[:, 1, COL["toward_ema"]], x[:, 1, COL["toward_vel"]]
    assert jump[10] == pytest.approx(1.0, abs=1e-6)
    assert jump[9] < ema[10] < jump[10]
    assert vel[10] > 0.0 and vel[0] == 0.0


def test_fine_dwell_separates_neighbours_the_wide_cone_does_not():
    x = features(make_ep([1] * 31))
    assert x[10, 2, COL["dwell"]] > 0.0  # blue is 8.37 deg off: inside 10 deg
    assert x[10, 2, COL["dwell_fine"]] == 0.0  # but not inside 4 deg
    assert x[10, 1, COL["dwell_fine"]] == pytest.approx(1.0)


def test_angle_vel_is_negative_when_turning_to_a_block():
    x = features(make_ep([0] * 10 + [1] * 21))
    assert x[10, 1, COL["angle_vel"]] < 0.0 < x[10, 0, COL["angle_vel"]]
    assert x[0, 1, COL["angle_vel"]] == 0.0
