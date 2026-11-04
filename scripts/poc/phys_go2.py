"""T5-20: the Menagerie Unitree Go2 walking under the cagataydev Isaac Lab rough-terrain policy
(rsl_rl MLP 235 -> 512 -> 256 -> 128 -> 12, ELU, exported to ONNX, run here in numpy).

The port follows params/env.yaml: Isaac Lab's breadth-first joint order (all hips, then all
thighs, then all calves) mapped onto Menagerie's per-leg order; default pose from init_state;
JointPositionAction with scale 0.25 around the default; DCMotor PD (stiffness 25, damping 0.5,
saturation 23.7/23.7/45.43 Nm, velocity limits 30.1/30.1/15.7 rad/s); 50 Hz policy; no
observation history. Observation terms in order: base linear velocity and angular velocity
(base frame), projected gravity, the (vx, vy, wz) command, joint positions relative to the
default, joint velocities, the last action and a 17 x 11 height scan (1.6 m x 1.0 m at 0.1 m)
of base height minus ground height minus 0.5, clipped to [-1, 1]."""

from __future__ import annotations

import mujoco
import numpy as np
from phys_core import GO2_NPZ, GO2_XML, Pursuit, RouteFollower, load_child, quat_yaw
from phys_policy import MlpPolicy

LEGS = ("FL", "FR", "RL", "RR")
PARTS = ("hip", "thigh", "calf")
MJ_ORDER = tuple(f"{leg}_{part}" for leg in LEGS for part in PARTS)  # Menagerie actuators
ISAAC_ORDER = tuple(f"{leg}_{part}" for part in PARTS for leg in LEGS)  # policy vectors
# ISAAC_TO_MJ[i] = Menagerie index of the policy's i-th joint
ISAAC_TO_MJ = tuple(MJ_ORDER.index(n) for n in ISAAC_ORDER)
DEFAULT = {"FL_hip": 0.1, "FR_hip": -0.1, "RL_hip": 0.1, "RR_hip": -0.1,
           "FL_thigh": 0.8, "FR_thigh": 0.8, "RL_thigh": 1.0, "RR_thigh": 1.0,
           "FL_calf": -1.5, "FR_calf": -1.5, "RL_calf": -1.5, "RR_calf": -1.5}
DEFAULT_ISAAC = np.array([DEFAULT[n] for n in ISAAC_ORDER], np.float32)
KP, KD, ACTION_SCALE = 25.0, 0.5, 0.25
EFFORT = np.array([{"hip": 23.7, "thigh": 23.7, "calf": 45.43}[n.split("_")[1]]
                   for n in ISAAC_ORDER])
VEL_LIMIT = np.array([{"hip": 30.1, "thigh": 30.1, "calf": 15.7}[n.split("_")[1]]
                      for n in ISAAC_ORDER])
SCAN_X, SCAN_Y = np.arange(-0.8, 0.8 + 1e-6, 0.1), np.arange(-0.5, 0.5 + 1e-6, 0.1)
N_SCAN = len(SCAN_X) * len(SCAN_Y)  # 187
OBS_DIM = 3 + 3 + 3 + 3 + 12 + 12 + 12 + N_SCAN  # 235
POLICY_HZ = 50
FALL_Z = 0.18
STAND_Z = 0.30


def to_isaac(mj_vec) -> np.ndarray:
    return np.asarray(mj_vec)[list(ISAAC_TO_MJ)]


def to_mj(isaac_vec) -> np.ndarray:
    out = np.zeros(12, np.asarray(isaac_vec).dtype)
    out[list(ISAAC_TO_MJ)] = isaac_vec
    return out


def dc_motor(target, q, qd) -> np.ndarray:
    """Isaac Lab's DCMotor: PD torque clipped to the torque-speed envelope (Isaac order)."""
    tau = KP * (target - q) - KD * qd
    hi = np.clip(EFFORT * (1.0 - qd / VEL_LIMIT), 0.0, EFFORT)
    lo = np.clip(EFFORT * (-1.0 - qd / VEL_LIMIT), -EFFORT, 0.0)
    return np.clip(tau, lo, hi)


def observation(lin_b, ang_b, grav_b, cmd, q_rel, qd, action, scan) -> np.ndarray:
    return np.concatenate([lin_b, ang_b, grav_b, cmd, q_rel, qd, action, scan]).astype(
        np.float32)


def go2_spec(zero_joint_friction: bool = True, color=None) -> mujoco.MjSpec:
    """Menagerie Go2 with a named free joint and a forward camera ('cam'). Isaac Lab trained
    with no joint friction or damping beyond the PD and with plain Coulomb friction 1.0 at the
    feet, so those are matched (Menagerie's feet also resist rolling and twisting)."""
    spec = load_child(GO2_XML)
    if color is not None:  # Menagerie's grey shell parts, e.g. rescue yellow
        spec.material("gray").rgba = list(color)
    spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL  # the scene's solver settings apply
    spec.option.impratio = 1.0
    base = spec.body("base")
    base.first_joint().name = "free"
    base.add_camera(name="cam", pos=[0.32, 0, 0.05], xyaxes=[0, -1, 0, 0, 0, 1], fovy=75)
    for leg in LEGS:  # Isaac Lab: friction 1.0, no torsional or rolling friction at the feet
        foot = spec.geom(leg)
        foot.condim = 3
        foot.friction = [1.0, 0.0, 0.0]
    if zero_joint_friction:
        for j in spec.joints:
            if j.type == mujoco.mjtJoint.mjJNT_HINGE:
                j.damping = np.zeros(3)
                j.frictionloss = 0.0
    return spec


class Go2Walker:
    """One Go2 in a compiled model (names prefixed `p`). control() before every mj_step,
    after_step() after it, as G1Walker."""

    def __init__(self, model, data, p: str = "go2_", prm: Pursuit | None = None,
                 scan_mode: str = "flat") -> None:
        self.m, self.d, self.p = model, data, p
        self.policy = MlpPolicy(GO2_NPZ)
        assert self.policy.obs_dim == OBS_DIM
        self.free = model.joint(p + "free")
        self.base = model.body(p + "base").id
        self.qadr = np.array([model.joint(f"{p}{n}_joint").qposadr[0] for n in MJ_ORDER])
        self.vadr = np.array([model.joint(f"{p}{n}_joint").dofadr[0] for n in MJ_ORDER])
        self.act = np.array([model.actuator(f"{p}{n}").id for n in MJ_ORDER])
        self.decim = max(1, round(1.0 / (POLICY_HZ * model.opt.timestep)))
        self.scan_mode = scan_mode
        # The policy turns on the spot to the right but barely to the left (yaw rate 0.1 for
        # a 0.6 command), and both ways once it walks at 0.2 m/s; so it turns while walking.
        self.prm = prm or Pursuit(lookahead=0.6, v_max=0.8, w_max=0.9, stop=0.12, slow=0.6,
                                  turn_first=0.9, k_heading=1.5, turn_speed=0.25)
        self.reset()

    def reset(self) -> None:
        self.action = np.zeros(12, np.float32)
        self.target = DEFAULT_ISAAC.copy()
        self.counter = 0
        self.route: RouteFollower | None = None
        self.cmd = np.zeros(3, np.float32)

    def place(self, x: float, y: float, yaw: float, rng=None, jitter: float = 0.0) -> None:
        a = self.free.qposadr[0]
        self.d.qpos[a:a + 3] = (x, y, 0.33)
        self.d.qpos[a + 3:a + 7] = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
        q = DEFAULT_ISAAC.astype(float)
        if rng is not None and jitter > 0:
            q = q + rng.uniform(-jitter, jitter, q.shape)
        self.d.qpos[self.qadr] = to_mj(q)
        self.d.qvel[self.free.dofadr[0]:self.free.dofadr[0] + 6] = 0
        self.d.qvel[self.vadr] = 0

    def pose(self) -> tuple[float, float, float]:
        a = self.free.qposadr[0]
        q = self.d.qpos[a:a + 7]
        return float(q[0]), float(q[1]), quat_yaw(q[3:])

    def height(self) -> float:
        return float(self.d.qpos[self.free.qposadr[0] + 2])

    def go(self, points, heading: float | None = None) -> None:
        x, y, _ = self.pose()
        self.route = RouteFollower([(x, y), *points], heading, self.prm)

    @property
    def arrived(self) -> bool:
        return self.route is None or self.route.done

    def _scan(self, z: float) -> np.ndarray:
        if self.scan_mode == "flat":  # ground at z = 0 under every ray
            return np.full(N_SCAN, np.clip(z - 0.5, -1.0, 1.0), np.float32)
        return np.ones(N_SCAN, np.float32)

    def obs(self) -> np.ndarray:
        d = self.d
        a, v = self.free.qposadr[0], self.free.dofadr[0]
        rot = d.xmat[self.base].reshape(3, 3)
        lin_b = rot.T @ d.qvel[v:v + 3]
        ang_b = d.qvel[v + 3:v + 6]
        grav_b = rot.T @ np.array([0.0, 0.0, -1.0])
        q = to_isaac(d.qpos[self.qadr])
        qd = to_isaac(d.qvel[self.vadr])
        return observation(lin_b, ang_b, grav_b, self.cmd, q - DEFAULT_ISAAC, qd, self.action,
                           self._scan(float(d.qpos[a + 2])))

    def control(self) -> None:
        d = self.d
        q, qd = to_isaac(d.qpos[self.qadr]), to_isaac(d.qvel[self.vadr])
        d.ctrl[self.act] = to_mj(dc_motor(self.target, q, qd))

    def after_step(self) -> None:
        self.counter += 1
        if self.counter % self.decim:
            return
        if self.route is not None:
            x, y, yaw = self.pose()
            vx, wz = self.route.command(x, y, yaw)
            self.cmd = np.array([vx, 0.0, wz], np.float32)
        self.action = self.policy(self.obs()).astype(np.float32)
        self.target = DEFAULT_ISAAC + ACTION_SCALE * self.action
