"""T5-20: the Unitree G1 (12-dof legs) walking under unitree_rl_gym's motion.pt, reproducing
deploy/deploy_mujoco: the 47-d observation, its scales, kps/kds, default angles, 50 Hz policy
on a 0.002 s physics step, PD torques on the leg motors. The downloaded 12-dof model has its
arms fused to the pelvis; here they become hinge chains held by PD at the default (zero)
angles, so a scripted reach can move the right arm at the end. A pure-pursuit layer
(phys_core.RouteFollower) turns a route into the (vx, 0, wz) command."""

from __future__ import annotations

import mujoco
import numpy as np
from phys_core import G1_NPZ, G1_XML, G1_YAML, Pursuit, RouteFollower, load_child, quat_yaw
from phys_core import read_flat_yaml as _yaml
from phys_policy import G1Policy

LEGS = ("left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch",
        "left_ankle_roll", "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee",
        "right_ankle_pitch", "right_ankle_roll")
# Link origins of the fused arm, read off the geoms on the pelvis in g1_12dof.xml (pelvis frame).
ARM_LINKS = {
    "r": {"sh_pitch": (0.0, -0.10021, 0.29178), "sh_roll": (0.0, -0.14055, 0.288961),
          "sh_yaw": (0.0, -0.146797, 0.185762), "elbow": (0.0157745, -0.146798, 0.105243),
          "hand": (0.115774, -0.148668, 0.0952374)},
    "l": {"sh_pitch": (0.0, 0.10022, 0.29178), "sh_roll": (0.0, 0.14056, 0.288961),
          "sh_yaw": (0.0, 0.146807, 0.185762), "elbow": (0.0157745, 0.146808, 0.105243),
          "hand": (0.115774, 0.148678, 0.0952374)},
}
ARM_MESH = {"sh_roll": "shoulder_roll_link", "sh_yaw": "shoulder_yaw_link",
            "elbow": "elbow_link", "hand": "wrist_roll_rubber_hand"}
ARM_AXES = {"sh_pitch": (0, 1, 0), "sh_roll": (1, 0, 0), "sh_yaw": (0, 0, 1), "elbow": (0, 1, 0)}
ARM_KP, ARM_KD = 60.0, 3.0
PALM = (0.08, 0.0, -0.01)  # palm site, from the wrist origin (the hand points along +x)
FALL_Z = 0.50


def config() -> dict:
    """g1.yaml as numpy arrays where deploy_mujoco makes them arrays."""
    c = _yaml(G1_YAML)
    for k in ("kps", "kds", "default_angles", "cmd_scale", "cmd_init"):
        c[k] = np.array(c[k], np.float32)
    return c


def gravity_orientation(q) -> np.ndarray:
    """deploy_mujoco.get_gravity_orientation, verbatim."""
    qw, qx, qy, qz = q
    return np.array([2 * (-qz * qx + qw * qy), -2 * (qz * qy + qw * qx),
                     1 - 2 * (qw * qw + qz * qz)])


def observation(cfg: dict, qj, dqj, quat, omega, cmd, action, t: float) -> np.ndarray:
    """The 47-d policy input: omega, gravity, scaled command, joint pos/vel, last action,
    sin/cos of the 0.8 s gait phase."""
    n = cfg["num_actions"]
    obs = np.zeros(cfg["num_obs"], np.float32)
    phase = t % 0.8 / 0.8
    obs[:3] = np.asarray(omega) * cfg["ang_vel_scale"]
    obs[3:6] = gravity_orientation(quat)
    obs[6:9] = np.asarray(cmd, np.float32) * cfg["cmd_scale"]
    obs[9:9 + n] = (np.asarray(qj) - cfg["default_angles"]) * cfg["dof_pos_scale"]
    obs[9 + n:9 + 2 * n] = np.asarray(dqj) * cfg["dof_vel_scale"]
    obs[9 + 2 * n:9 + 3 * n] = action
    obs[9 + 3 * n:9 + 3 * n + 2] = (np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase))
    return obs


def _move_arm(spec: mujoco.MjSpec, side: str) -> None:
    """Re-parent one fused arm into a pitch-roll-yaw-elbow chain on the pelvis."""
    pelvis = spec.body("pelvis")
    links = ARM_LINKS[side]
    full = "right" if side == "r" else "left"
    geoms = {}
    for g in list(pelvis.geoms):
        if g.type == mujoco.mjtGeom.mjGEOM_MESH:
            for k, mesh in ARM_MESH.items():
                if g.meshname == f"{full}_{mesh}":
                    geoms.setdefault(k, []).append(g)
        if g.type == mujoco.mjtGeom.mjGEOM_MESH and g.meshname == f"{full}_shoulder_pitch_link":
            geoms.setdefault("sh_pitch", []).append(g)
    parent, origin = pelvis, np.zeros(3)
    for k in ("sh_pitch", "sh_roll", "sh_yaw", "elbow"):
        o = np.array(links[k])
        b = parent.add_body(name=f"{side}_{k}", pos=list(o - origin), mass=0.15,
                            inertia=[3e-4, 3e-4, 3e-4], explicitinertial=True)
        b.add_joint(name=f"{side}_{k}_joint", axis=list(ARM_AXES[k]), damping=0.5,
                    armature=0.01, range=[-3.0, 3.0])
        for g in geoms.get(k, []) + (geoms.get("hand", []) if k == "elbow" else []):
            b.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=g.meshname,
                            pos=list(np.array(g.pos) - o), quat=list(g.quat), rgba=list(g.rgba),
                            contype=0, conaffinity=0, group=g.group, density=0)
        if k == "elbow":
            hand = np.array(links["hand"]) - o
            b.add_site(name=f"{side}_palm", pos=list(hand + PALM), size=[0.015, 0, 0])
        parent, origin = b, o
    for gl in geoms.values():
        for g in gl:
            spec.delete(g)
    for k in ("sh_pitch", "sh_roll", "sh_yaw", "elbow"):
        spec.add_actuator(name=f"{side}_{k}", target=f"{side}_{k}_joint",
                          trntype=mujoco.mjtTrn.mjTRN_JOINT, ctrlrange=[-40, 40],
                          ctrllimited=True)


def g1_spec(rgba: tuple | None = None) -> mujoco.MjSpec:
    """The 12-dof G1 with movable PD-held arms, a named free joint, a forward head camera
    ('cam') and palm sites."""
    spec = load_child(G1_XML)
    pelvis = spec.body("pelvis")
    pelvis.first_joint().name = "free"
    _move_arm(spec, "r")
    _move_arm(spec, "l")
    pelvis.add_camera(name="cam", pos=[0.1, 0, 0.43], xyaxes=[0, -1, 0, 0, 0, 1], fovy=75)
    if rgba is not None:
        for b in spec.bodies:
            for g in b.geoms:
                if g.group == 1 and list(g.rgba)[:3] == [0.7, 0.7, 0.7]:
                    g.rgba = list(rgba)
    return spec


class G1Walker:
    """One G1 in a compiled model (joints prefixed `p`), driven at 50 Hz by the motion policy.
    Call control() before every mj_step; give it a route with go() and a reach with reach()."""

    def __init__(self, model, data, p: str = "g1_", prm: Pursuit | None = None) -> None:
        self.m, self.d, self.p = model, data, p
        self.cfg = config()
        self.policy = G1Policy(G1_NPZ)
        self.free = model.joint(p + "free")
        self.qadr = np.array([model.joint(f"{p}{j}_joint").qposadr[0] for j in LEGS])
        self.vadr = np.array([model.joint(f"{p}{j}_joint").dofadr[0] for j in LEGS])
        self.act = np.array([model.actuator(f"{p}{j}_joint").id for j in LEGS])
        self.arm = {s: [model.joint(f"{p}{s}_{k}_joint") for k in ARM_AXES] for s in "rl"}
        self.arm_act = {s: [model.actuator(f"{p}{s}_{k}").id for k in ARM_AXES] for s in "rl"}
        self.palm = model.site(p + "r_palm").id
        self.decim = int(self.cfg["control_decimation"])
        self.prm = prm or Pursuit(lookahead=0.5, v_max=0.8, w_max=0.8, stop=0.1, slow=0.6,
                                  turn_first=0.9, k_heading=1.2, heading_tol=0.12, creep=0.15)
        self.reset()

    def reset(self) -> None:
        self.policy.reset()
        self.action = np.zeros(self.cfg["num_actions"], np.float32)
        self.target = self.cfg["default_angles"].copy()
        self.counter = 0
        self.route: RouteFollower | None = None
        self.cmd = np.zeros(3, np.float32)
        self.arm_goal = {s: np.zeros(4) for s in "rl"}
        self.reach_point: np.ndarray | None = None

    # -- state -----------------------------------------------------------------------------

    def place(self, x: float, y: float, yaw: float, rng=None, jitter: float = 0.0) -> None:
        a = self.free.qposadr[0]
        self.d.qpos[a:a + 3] = (x, y, 0.793)
        self.d.qpos[a + 3:a + 7] = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
        q = self.cfg["default_angles"].astype(float)
        if rng is not None and jitter > 0:
            q = q + rng.uniform(-jitter, jitter, q.shape)
        self.d.qpos[self.qadr] = q
        self.d.qvel[self.free.dofadr[0]:self.free.dofadr[0] + 6] = 0
        self.d.qvel[self.vadr] = 0

    def pose(self) -> tuple[float, float, float]:
        a = self.free.qposadr[0]
        q = self.d.qpos[a:a + 7]
        return float(q[0]), float(q[1]), quat_yaw(q[3:])

    def height(self) -> float:
        return float(self.d.qpos[self.free.qposadr[0] + 2])

    def palm_pos(self) -> np.ndarray:
        return self.d.site_xpos[self.palm].copy()

    # -- commands --------------------------------------------------------------------------

    def go(self, points, heading: float | None = None) -> None:
        x, y, _ = self.pose()
        self.route = RouteFollower([(x, y), *points], heading, self.prm)

    @property
    def arrived(self) -> bool:
        return self.route is None or self.route.done

    def reach(self, point) -> None:
        self.reach_point = np.asarray(point, float)

    def lower_arm(self) -> None:
        self.reach_point = None
        self.arm_goal["r"] = np.zeros(4)

    # -- control ---------------------------------------------------------------------------

    def _policy_step(self) -> None:
        cfg, d = self.cfg, self.d
        a, v = self.free.qposadr[0], self.free.dofadr[0]
        if self.route is not None:
            x, y, yaw = self.pose()
            vx, wz = self.route.command(x, y, yaw)
            self.cmd = np.array([vx, 0.0, wz], np.float32)
        obs = observation(cfg, d.qpos[self.qadr], d.qvel[self.vadr], d.qpos[a + 3:a + 7],
                          d.qvel[v + 3:v + 6], self.cmd, self.action,
                          self.counter * self.m.opt.timestep)
        self.action = self.policy(obs).astype(np.float32)
        self.target = self.action * cfg["action_scale"] + cfg["default_angles"]
        if self.reach_point is not None:
            self.arm_goal["r"] = self._ik_step(self.reach_point)

    def _ik_step(self, point: np.ndarray) -> np.ndarray:
        """One damped-least-squares step of the right arm toward point, from the current
        arm angles (closed loop at 50 Hz, so the hand tracks while the legs keep stepping)."""
        m, d = self.m, self.d
        jac = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jac, None, self.palm)
        cols = [j.dofadr[0] for j in self.arm["r"]]
        jm = jac[:, cols]
        err = point - d.site_xpos[self.palm]
        step = jm.T @ np.linalg.solve(jm @ jm.T + 0.02 * np.eye(3), err)
        q = np.array([d.qpos[j.qposadr[0]] for j in self.arm["r"]])
        return np.clip(q + np.clip(step, -0.15, 0.15), -2.6, 2.6)

    def control(self) -> None:
        """deploy_mujoco's inner loop, before mj_step: PD torques on the legs (and the arms)."""
        d, cfg = self.d, self.cfg
        tau = (self.target - d.qpos[self.qadr]) * cfg["kps"] - d.qvel[self.vadr] * cfg["kds"]
        d.ctrl[self.act] = tau
        for s in "rl":
            q = np.array([d.qpos[j.qposadr[0]] for j in self.arm[s]])
            dq = np.array([d.qvel[j.dofadr[0]] for j in self.arm[s]])
            d.ctrl[self.arm_act[s]] = ARM_KP * (self.arm_goal[s] - q) - ARM_KD * dq

    def after_step(self) -> None:
        """deploy_mujoco's inner loop, after mj_step: count the step and run the policy every
        control_decimation steps on the new state."""
        self.counter += 1
        if self.counter % self.decim == 0:
            self._policy_step()
