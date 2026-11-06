"""T5-32: the Go2 navigation-policy port. CPU only: the 203-d navigation observation layout,
the pose command, numpy vs torch on the samples export_go2_nav.py stored, and the goal
hand-off from the layer's pose adapter to the navigation policy's last waypoint. Checks that
need the downloaded policies or robot skip when third_party/ is missing."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "poc"))
sys.path.insert(0, str(ROOT / "scripts"))

from phys_core import GO2_XML
from phys_go2 import N_SCAN, OBS_DIM
from phys_go2_nav import (
    GOAL_Z,
    LOCO_NPZ,
    NAV_NPZ,
    NAV_OBS_DIM,
    SAMPLES_NPZ,
    NavPolicy,
    nav_observation,
    pose_command,
    waypoints,
)
from phys_policy import MlpPolicy

from curio_robot.layer import adapters
from curio_robot.layer.goal import Base, Goal

needs_weights = pytest.mark.skipif(not (NAV_NPZ.exists() and LOCO_NPZ.exists()),
                                   reason="export_go2_nav.py not run")
needs_samples = pytest.mark.skipif(not SAMPLES_NPZ.exists(), reason="no stored samples")
needs_robot = pytest.mark.skipif(not (GO2_XML.exists() and NAV_NPZ.exists()),
                                 reason="third_party Go2 files not fetched")


# ---- observation layout ----------------------------------------------------------------------


def test_nav_observation_is_203_in_the_documented_order():
    obs = nav_observation(np.full(3, 1.0), np.full(3, 2.0), np.full(3, 3.0), np.full(4, 4.0),
                          np.full(N_SCAN, 5.0), np.full(3, 6.0))
    assert obs.shape == (NAV_OBS_DIM,) == (203,)
    assert obs.dtype == np.float32
    bounds = [(0, 3, 1), (3, 6, 2), (6, 9, 3), (9, 13, 4), (13, 200, 5), (200, 203, 6)]
    for lo, hi, v in bounds:
        assert np.all(obs[lo:hi] == v), (lo, hi)


def test_pose_command_is_in_the_yaw_frame_with_flat_ground_z_and_wrapped_heading():
    # robot at (1, 1) facing +y; goal 2 m further along +y, heading -pi + 0.1
    cmd = pose_command(1.0, 1.0, 0.32, np.pi / 2, (1.0, 3.0, -np.pi + 0.1))
    assert np.allclose(cmd[:2], [2.0, 0.0], atol=1e-6)  # straight ahead
    assert np.isclose(cmd[2], GOAL_Z - 0.32)
    assert np.isclose(cmd[3], np.pi / 2 + 0.1, atol=1e-6)  # wrap(-pi + 0.1 - pi / 2)
    left = pose_command(0.0, 0.0, 0.3, 0.0, (0.0, 1.5, 0.0))
    assert np.allclose(left[:2], [0.0, 1.5])  # +y is the robot's left at yaw 0


def test_heading_error_wraps_into_minus_pi_pi():
    cmd = pose_command(0.0, 0.0, 0.3, 3.0, (1.0, 0.0, -3.0))
    assert -np.pi <= cmd[3] <= np.pi
    assert np.isclose(cmd[3], 2 * np.pi - 6.0)


def test_waypoints_take_the_heading_of_travel():
    wps = waypoints((0.0, 0.0), [(1.0, 0.0), (1.0, 2.0)])
    assert np.isclose(wps[0][2], 0.0) and np.isclose(wps[1][2], np.pi / 2)
    assert np.isclose(waypoints((0.0, 0.0), [(1.0, 0.0)], heading=1.2)[0][2], 1.2)


@needs_weights
def test_exported_shapes_match_the_layouts():
    assert NavPolicy(NAV_NPZ).obs_dim == NAV_OBS_DIM
    assert MlpPolicy(LOCO_NPZ).obs_dim == OBS_DIM == 235
    assert NavPolicy(NAV_NPZ).mean.shape == (NAV_OBS_DIM,)


# ---- numpy vs torch ------------------------------------------------------------------------


@needs_weights
@needs_samples
def test_numpy_matches_torch_on_stored_samples():
    s = np.load(SAMPLES_NPZ)
    loco, nav = MlpPolicy(LOCO_NPZ), NavPolicy(NAV_NPZ)
    got_l = np.stack([loco(x) for x in s["loco_x"]])
    got_n = np.stack([nav(x) for x in s["nav_x"]])
    assert got_l.shape == s["loco_y"].shape and got_n.shape == s["nav_y"].shape
    assert np.abs(got_l - s["loco_y"]).max() < 1e-4
    assert np.abs(got_n - s["nav_y"]).max() < 1e-4


# ---- the goal hand-off ---------------------------------------------------------------------


def _layer_goal(target: str, world_xyz, base: Base) -> Goal:
    return Goal(target_id=target, target_xyz_robot=tuple(base.to_base(world_xyz)),
                frame="speaker", confidence=0.99, alternatives=(), ask=None, note=None,
                adapter="r16-v4")


def test_pose_adapter_goal_becomes_the_scene_goal():
    from go2_nav_eval import TRUE, NavDog, goal_point, handoff
    scene = NavDog()
    base = Base(np.array([8.7, 0.1, 0.0]), np.pi - 0.2)  # a jittered start
    cand = scene.cand(TRUE).pos
    goal = _layer_goal(TRUE, cand, base)
    world = adapters.pose(goal, base)
    assert np.allclose(world, cand, atol=1e-9)
    leg = handoff(scene, TRUE, world).legs({"geo": TRUE})[0]
    assert np.allclose(leg.goal, goal_point(scene, TRUE), atol=1e-9)
    assert leg.goal[1] > 0  # room_n is the +y room
    assert np.allclose(leg.route[-1], leg.goal)


@needs_robot
def test_navigation_policy_gets_the_goal_as_its_last_waypoint():
    import mujoco
    from go2_nav_eval import TRUE, NavDog, handoff
    scene = NavDog()
    model = scene.spec().compile()
    data = mujoco.MjData(model)
    scene.setup(model, data)
    robot = scene.make_robot(model, data)
    robot.place(8.9, 0.0, np.pi)
    mujoco.mj_forward(model, data)
    leg = handoff(scene, TRUE, scene.cand(TRUE).pos).legs({"geo": TRUE})[0]
    robot.go(leg.route, leg.heading)
    assert robot.goal[:2] == pytest.approx(leg.route[0])
    assert robot.wps[-1][:2] == pytest.approx(leg.goal)
    assert not robot.arrived
    obs = robot.nav_obs()
    assert obs.shape == (NAV_OBS_DIM,)
    x, y, yaw = robot.pose()
    assert np.allclose(obs[9:13], pose_command(x, y, robot.height(), yaw, robot.goal))
