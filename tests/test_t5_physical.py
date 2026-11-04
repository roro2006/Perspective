"""T5-20: the maths under the physical robots (pure pursuit, diff drive, force PD), the G1
observation against g1.yaml and the Go2 joint mapping. Fast and CPU-only; the checks that
need the downloaded robots skip when third_party/ is missing."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "poc"))

from phys_core import (
    G1_YAML,
    GO2_ENV,
    Pursuit,
    RouteFollower,
    closest_on_polyline,
    diff_drive,
    lookahead_point,
    pursuit_command,
    read_flat_yaml,
    wrap,
)
from phys_go2 import (
    DEFAULT_ISAAC,
    ISAAC_ORDER,
    ISAAC_TO_MJ,
    MJ_ORDER,
    N_SCAN,
    OBS_DIM,
    to_isaac,
    to_mj,
)
from phys_robots import Flyer

needs_g1 = pytest.mark.skipif(not G1_YAML.exists(), reason="third_party G1 files not fetched")
needs_go2 = pytest.mark.skipif(not GO2_ENV.exists(), reason="third_party Go2 policy not fetched")


# ---- pure pursuit --------------------------------------------------------------------------


def test_pursuit_steers_toward_a_target_on_the_left():
    v, w, done = pursuit_command(0, 0, 0, (1.0, 0.5), (3.0, 0.0), None, Pursuit())
    assert v > 0 and w > 0 and not done


def test_pursuit_turns_on_the_spot_when_the_target_is_behind():
    v, w, _ = pursuit_command(0, 0, 0, (-1.0, -0.1), (-2.0, 0.0), None, Pursuit())
    assert v == 0.0 and w < 0  # behind and slightly right: shorter to turn clockwise


def test_pursuit_curvature_is_the_textbook_one():
    prm = Pursuit(v_max=0.5, w_max=10.0, slow=0.01)
    v, w, _ = pursuit_command(0, 0, 0, (1.0, 1.0), (5.0, 5.0), None, prm)
    alpha, ld = np.pi / 4, np.sqrt(2)
    assert v == pytest.approx(0.5 * np.cos(alpha))
    assert w == pytest.approx(2 * v * np.sin(alpha) / ld)


def test_pursuit_finishes_with_the_heading():
    prm = Pursuit(stop=0.1, heading_tol=0.05)
    v, w, done = pursuit_command(0, 0, 0, (0.02, 0), (0.02, 0), np.pi / 2, prm)
    assert v == 0.0 and w > 0 and not done
    assert pursuit_command(0, 0, np.pi / 2, (0, 0), (0, 0), np.pi / 2, prm)[2]


def test_polyline_projection_and_lookahead():
    pts = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]])
    seg, frac = closest_on_polyline(pts, np.array([1.5, 0.3]))
    assert seg == 0 and frac == pytest.approx(0.75)
    assert lookahead_point(pts, seg, frac, 1.0) == pytest.approx([2.0, 0.5])
    assert lookahead_point(pts, seg, frac, 9.0) == pytest.approx([2.0, 2.0])


def test_route_follower_drives_a_unicycle_to_the_goal():
    rf = RouteFollower([(0, 0), (2, 0), (2, 2)], heading=np.pi, prm=Pursuit(stop=0.05))
    x = y = yaw = 0.0
    for _ in range(4000):
        v, w = rf.command(x, y, yaw)
        x, y, yaw = x + 0.01 * v * np.cos(yaw), y + 0.01 * v * np.sin(yaw), yaw + 0.01 * w
        if rf.done:
            break
    assert rf.done
    assert np.hypot(x - 2, y - 2) < 0.06 and abs(wrap(yaw - np.pi)) < 0.13


def test_diff_drive_wheel_speeds():
    assert diff_drive(1.0, 0.0, 0.5, 0.1) == pytest.approx((10.0, 10.0))
    left, right = diff_drive(0.0, 2.0, 0.5, 0.1)
    assert left == pytest.approx(-5.0) and right == pytest.approx(5.0)


def test_wrap():
    assert wrap(3 * np.pi / 2) == pytest.approx(-np.pi / 2)
    assert wrap(-3 * np.pi / 2) == pytest.approx(np.pi / 2)


# ---- force PD free body ------------------------------------------------------------------

BOX = """<mujoco><option timestep="0.002" gravity="0 0 {g}"/><worldbody>
<body name="fb_root" pos="0 0 1"><freejoint name="fb_free"/>
<geom type="box" size="0.15 0.1 0.1" mass="5"/></body></worldbody></mujoco>"""


@pytest.mark.parametrize("mode,g,water", [("holonomic", 0.0, False),
                                          ("holonomic", -9.81, True),
                                          ("thrust", -9.81, False)])
def test_force_pd_free_body_converges(mode, g, water):
    m = mujoco.MjModel.from_xml_string(BOX.format(g=g))
    d = mujoco.MjData(m)
    f = Flyer(m, d, "fb_", mode, yaw=0.0, speed=1.0, buoyant=water, drag_lin=0.5,
              drag_quad=0.3)
    f.place(0, 0, 0.0, z=1.0)
    mujoco.mj_forward(m, d)
    goal = np.array([1.0, -0.5, 1.5])
    f.go([goal], heading=0.6)
    for _ in range(int(6 / m.opt.timestep)):
        f.control()
        mujoco.mj_step(m, d)
        f.after_step()
    assert np.linalg.norm(f.position() - goal) < 0.05
    assert abs(wrap(f.pose()[2] - 0.6)) < 0.05
    assert f.arrived


# ---- G1 observation against g1.yaml -------------------------------------------------------


@needs_g1
def test_g1_observation_matches_the_deploy_config():
    from phys_g1 import config, gravity_orientation, observation
    cfg = config()
    assert (cfg["num_obs"], cfg["num_actions"]) == (47, 12)
    assert cfg["simulation_dt"] == 0.002 and cfg["control_decimation"] == 10
    raw = read_flat_yaml(G1_YAML)
    assert list(cfg["kps"]) == raw["kps"] and list(cfg["default_angles"]) == raw["default_angles"]
    rng = np.random.default_rng(0)
    qj, dqj, omega = rng.normal(size=12), rng.normal(size=12), rng.normal(size=3)
    cmd, action = np.array([0.5, 0.1, -0.3]), rng.normal(size=12).astype(np.float32)
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    obs = observation(cfg, qj, dqj, quat, omega, cmd, action, t=0.2)
    assert obs.shape == (cfg["num_obs"],)
    assert obs[:3] == pytest.approx(omega * raw["ang_vel_scale"], rel=1e-6)
    assert obs[3:6] == pytest.approx([0, 0, -1])
    assert obs[6:9] == pytest.approx(cmd * np.array(raw["cmd_scale"]), rel=1e-6)
    assert obs[9:21] == pytest.approx((qj - np.array(raw["default_angles"]))
                                      * raw["dof_pos_scale"], rel=1e-5)
    assert obs[21:33] == pytest.approx(dqj * raw["dof_vel_scale"], rel=1e-5)
    assert obs[33:45] == pytest.approx(action)
    assert obs[45:] == pytest.approx([np.sin(2 * np.pi * 0.25), np.cos(2 * np.pi * 0.25)],
                                     abs=1e-6)
    assert gravity_orientation([0.0, 1.0, 0.0, 0.0]) == pytest.approx([0, 0, 1])


# ---- Go2 joint mapping ---------------------------------------------------------------------


def test_go2_joint_map_is_a_permutation():
    assert sorted(ISAAC_TO_MJ) == list(range(12))
    assert all(MJ_ORDER[ISAAC_TO_MJ[i]] == ISAAC_ORDER[i] for i in range(12))
    assert ISAAC_ORDER[:4] == ("FL_hip", "FR_hip", "RL_hip", "RR_hip")
    v = np.arange(12.0)
    assert to_mj(to_isaac(v)) == pytest.approx(v)
    assert OBS_DIM == 235 and N_SCAN == 17 * 11


@needs_go2
def test_go2_defaults_and_gains_match_env_yaml():
    env = GO2_ENV.read_text()
    init = env[env.index("init_state:"):env.index("joint_vel:", env.index("init_state:"))]
    assert re.search(r"\.\*L_hip_joint: 0\.1\b", init)
    assert re.search(r"R\[L,R\]_thigh_joint: 1\.0", init)
    for name, q in zip(ISAAC_ORDER, DEFAULT_ISAAC, strict=True):
        leg, part = name.split("_")
        if part == "hip":
            want = 0.1 if leg[1] == "L" else -0.1
        elif part == "thigh":
            want = 0.8 if leg[0] == "F" else 1.0
        else:
            want = -1.5
        assert q == pytest.approx(want)
    assert "stiffness: 25.0" in env and "damping: 0.5" in env and "scale: 0.25" in env
