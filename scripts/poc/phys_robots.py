"""T5-20: the non-legged physical robots. A diff-drive AMR (two driven wheels under velocity
actuators, frictionless casters) steered by pure pursuit, and free bodies (drone, ROV, space
flyer) pushed by a force/torque PD that chases a carrot sliding along the route. All share the
walkers' interface: place, pose, go, arrived, control (before mj_step), after_step."""

from __future__ import annotations

import mujoco
import numpy as np
from phys_core import (
    Carrot,
    Pursuit,
    RouteFollower,
    attitude_torque,
    body_state,
    diff_drive,
    drag,
    force_pd,
    quat_yaw,
    wrap,
)

ORANGE = "0.95 0.45 0.12 1"
DARK = "0.14 0.15 0.17 1"
WHEEL_R, TRACK = 0.09, 0.56


def amr_phys(p: str = "amr_") -> str:
    """The warehouse AMR of robots.amr with real wheels: two driven wheels on hinges at mid
    length, four frictionless caster balls under the corners (1 cm clear of the floor, so
    the wheels always carry the weight and the casters only stop it tipping), ~60 kg."""
    casters = "".join(
        f'<geom type="sphere" pos="{x} {y} 0.05" size="0.04" rgba="{DARK}" condim="1" '
        f'priority="1" friction="0 0 0" mass="0.5"/>' for x in (0.33, -0.33) for y in (0.18, -0.18))
    wheels = "".join(f"""
    <body name="{p}wheel_{s}" pos="0 {y} {WHEEL_R}">
      <joint name="{p}wheel_{s}" axis="0 1 0" damping="0.05" armature="0.1"/>
      <geom type="cylinder" euler="1.5708 0 0" size="{WHEEL_R} 0.04" rgba="{DARK}" mass="2"
            friction="1.2 0.01 0.001" priority="1"/>
      <geom type="box" pos="0 0.0 0" size="0.06 0.042 0.012" rgba="0.5 0.5 0.52 1"
            contype="0" conaffinity="0" mass="0.01"/>
    </body>""" for s, y in (("l", TRACK / 2), ("r", -TRACK / 2)))
    return f"""
  <body name="{p}root" pos="0 0 0">
    <freejoint name="{p}free"/>
    <geom type="box" pos="0 0 0.2" size="0.42 0.3 0.11" rgba="{ORANGE}" material="gloss"
          mass="55"/>
    <geom type="box" pos="0 0 0.33" size="0.36 0.26 0.02" rgba="{DARK}" mass="2"/>
    <geom type="box" pos="0.425 0 0.2" size="0.01 0.22 0.05" rgba="0.15 0.15 0.15 1" mass="0.1"/>
    <geom type="box" pos="0.432 0.14 0.24" size="0.004 0.04 0.012" rgba="1 1 0.8 1" mass="0"
          contype="0" conaffinity="0"/>
    <geom type="box" pos="0.432 -0.14 0.24" size="0.004 0.04 0.012" rgba="1 1 0.8 1" mass="0"
          contype="0" conaffinity="0"/>
    <geom type="cylinder" pos="0.3 0 0.39" size="0.06 0.04" rgba="{DARK}" mass="0.5"/>
    <geom type="cylinder" pos="0.3 0 0.44" size="0.045 0.015" rgba="0.2 0.8 0.9 1" mass="0.1"/>
    {casters}{wheels}
    <site name="{p}deck" pos="0 0 0.35"/>
    <camera name="{p}cam" pos="0.44 0 0.3" xyaxes="0 -1 0 0 0 1" fovy="70"/>
  </body>"""


def add_amr_actuators(spec: mujoco.MjSpec, p: str = "amr_") -> None:
    for s in "lr":
        spec.add_actuator(name=f"{p}wheel_{s}", target=f"{p}wheel_{s}",
                          trntype=mujoco.mjtTrn.mjTRN_JOINT, gaintype=mujoco.mjtGain.mjGAIN_FIXED,
                          biastype=mujoco.mjtBias.mjBIAS_AFFINE, gainprm=[60.0] + [0] * 9,
                          biasprm=[0, 0, -60.0] + [0] * 7, ctrlrange=[-20, 20],
                          ctrllimited=True, forcerange=[-40, 40], forcelimited=True)


class _Base:
    """Shared free-joint bookkeeping."""

    def __init__(self, model, data, p: str) -> None:
        self.m, self.d, self.p = model, data, p
        self.free = model.joint(p + "free")
        self.root = model.body(p + "root").id

    def place(self, x: float, y: float, yaw: float, z: float = 0.0, rng=None,
              jitter: float = 0.0) -> None:
        a = self.free.qposadr[0]
        self.d.qpos[a:a + 3] = (x, y, z)
        self.d.qpos[a + 3:a + 7] = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
        v = self.free.dofadr[0]
        self.d.qvel[v:v + 6] = 0

    def pose(self) -> tuple[float, float, float]:
        a = self.free.qposadr[0]
        q = self.d.qpos[a:a + 7]
        return float(q[0]), float(q[1]), quat_yaw(q[3:])

    def position(self) -> np.ndarray:
        a = self.free.qposadr[0]
        return self.d.qpos[a:a + 3].copy()

    def height(self) -> float:
        return float(self.d.qpos[self.free.qposadr[0] + 2])

    def after_step(self) -> None:
        pass


class AmrDrive(_Base):
    """Pure pursuit to (v, w), slew-limited, then wheel speeds for the velocity servos."""

    ACC, ALPHA = 1.5, 3.0  # m/s^2, rad/s^2

    def __init__(self, model, data, p: str = "amr_", prm: Pursuit | None = None) -> None:
        super().__init__(model, data, p)
        self.act = [model.actuator(f"{p}wheel_{s}").id for s in "lr"]
        self.prm = prm or Pursuit(lookahead=0.45, v_max=1.3, w_max=1.5, stop=0.1, slow=0.7,
                                  turn_first=0.7, k_heading=2.5, heading_tol=0.06, creep=0.12)
        self.route: RouteFollower | None = None
        self.v = self.w = 0.0

    def go(self, points, heading: float | None = None) -> None:
        x, y, _ = self.pose()
        self.route = RouteFollower([(x, y), *[q[:2] for q in points]], heading, self.prm)

    @property
    def arrived(self) -> bool:
        return (self.route is None or self.route.done) and abs(self.v) < 0.02

    def control(self) -> None:
        dt = self.m.opt.timestep
        v, w = 0.0, 0.0
        if self.route is not None:
            x, y, yaw = self.pose()
            v, w = self.route.command(x, y, yaw)
            # brake in time: never faster than it can stop within what's left
            d_end = float(np.hypot(*(self.route.goal - (x, y))))
            v = min(v, float(np.sqrt(1.6 * self.ACC * max(0.0, d_end - 0.03))))
            if self.route.at_end and self.route.heading is not None:
                err = abs(wrap(self.route.heading - yaw))
                w = float(np.sign(w) * min(abs(w), np.sqrt(1.6 * self.ALPHA * err)))
        self.v += float(np.clip(v - self.v, -self.ACC * dt, self.ACC * dt))
        self.w += float(np.clip(w - self.w, -self.ALPHA * dt, self.ALPHA * dt))
        wl, wr = diff_drive(self.v, self.w, TRACK, WHEEL_R)
        self.d.ctrl[self.act[0]], self.d.ctrl[self.act[1]] = wl, wr


class Flyer(_Base):
    """A free body chasing a carrot with force PD. 'thrust' (the drone) can only push along its
    own z axis and tilts to aim it; 'holonomic' (ROV, space flyer) pushes in any direction.
    buoyancy cancels gravity; drag is the water's (linear + quadratic, also on rotation)."""

    def __init__(self, model, data, p: str, mode: str, yaw: float, speed: float = 0.7,
                 wn: float = 2.2, zeta: float = 1.0, amax: float = 4.0, buoyant: bool = False,
                 drag_lin: float = 0.0, drag_quad: float = 0.0) -> None:
        super().__init__(model, data, p)
        self.mode, self.yaw, self.speed = mode, yaw, speed
        self.mass = float(model.body_subtreemass[self.root])
        self.inertia = float(np.max(model.body_inertia[self.root])) + 1e-3
        self.wn, self.zeta, self.amax = wn, zeta, amax
        self.buoyant, self.drag_lin, self.drag_quad = buoyant, drag_lin, drag_quad
        self.g = -float(model.opt.gravity[2])
        self.carrot: Carrot | None = None
        self.t = 0.0
        self.rotors = [j for j in range(model.njnt)
                       if model.joint(j).name.startswith(p + "rotor")]

    def place(self, x: float, y: float, yaw: float, z: float = 0.0, rng=None,
              jitter: float = 0.0) -> None:
        super().place(x, y, yaw, z)
        self.yaw = yaw

    def go(self, points, heading: float | None = None) -> None:
        self.carrot = Carrot([self.position(), *points], self.speed, self.t)
        if heading is not None:
            self.yaw = heading

    @property
    def goal(self) -> np.ndarray | None:
        return None if self.carrot is None else self.carrot.goal

    @property
    def arrived(self) -> bool:
        if self.carrot is None:
            return True
        _, vel, _, _ = body_state(self.m, self.d, self.m.body(self.root).name)
        done = self.t >= self.carrot.t0 + self.carrot.duration
        return done and float(np.linalg.norm(self.position() - self.carrot.goal)) < 0.08 \
            and float(np.linalg.norm(vel)) < 0.08

    def control(self) -> None:
        m, d = self.m, self.d
        _, vel, rot, omega = body_state(m, d, m.body(self.root).name)
        pos = self.position()
        target = pos if self.carrot is None else self.carrot.at(self.t)
        kp, kd = self.mass * self.wn ** 2, 2 * self.zeta * self.wn * self.mass
        f = force_pd(pos, vel, target, kp, kd, self.mass * self.amax)
        weight = np.array([0.0, 0.0, self.mass * self.g])
        env = np.zeros(3)
        if self.buoyant:  # buoyancy carries the weight, the water drags
            env = weight + drag(vel, self.drag_lin * self.mass, self.drag_quad * self.mass)
        else:  # the thrusters hold the weight too (zero in space)
            f = f + weight
        z_des = np.array([0.0, 0.0, 1.0])
        force = f
        if self.mode == "thrust":  # rotors push along body z only: tilt to aim them
            z_des = f / max(1e-6, np.linalg.norm(f))
            force = max(0.0, float(f @ rot[:, 2])) * rot[:, 2]
        force = force + env
        ki = self.inertia
        tq = attitude_torque(rot, omega, z_des, self.yaw, ki * 60.0, ki * 14.0)
        if self.buoyant:
            tq = tq - 2.0 * ki * omega
        d.xfrc_applied[self.root, :3] = force
        d.xfrc_applied[self.root, 3:] = tq
        for j in self.rotors:  # spin the props with a small motor on each (visual only)
            dof = m.joint(j).dofadr[0]
            d.qfrc_applied[dof] = 0.002 * ((60.0 if j % 2 else -60.0) - d.qvel[dof])

    def after_step(self) -> None:
        self.t += self.m.opt.timestep
