"""T5-20: one physical rollout of a scene. The robot is the only thing driven by its own
controller; the scripted people and scenery are put back on their scripted pose after every
physics step (so they act as static obstacles), held props follow the robot while held, and
everything else is plain MuJoCo dynamics. The rollout records qpos at the video rate so the
renderer can play it back, plus each leg's start and end time and the gate's measurements."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import mujoco
import numpy as np

FPS = 30
CAP = 40.0  # s of simulated time per leg (the gate's cap)
START_XY, START_YAW, START_Z, JOINT_JITTER = 0.25, 0.25, 0.25, 0.05


@dataclass
class Leg:
    """One step of a scene's plan. A route leg drives the robot through `route` (points, or a
    function of the robot's current position) and is scored against `goal` if given; a timed
    leg (route None) just runs `on_start` and lasts `wait` seconds."""

    key: str
    route: list | Callable | None = None
    heading: float | None = None
    goal: tuple | None = None
    dims: int = 2
    pause: float = 0.5
    start: float = 0.0
    wait: float = 0.0
    on_start: Callable | None = None
    on_end: Callable | None = None


@dataclass
class Traj:
    qpos: list = field(default_factory=list)
    mocap: list = field(default_factory=list)
    base: list = field(default_factory=list)  # robot base position per frame
    legs: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)
    min_z: float = 9.0
    fell: bool = False
    timed_out: list = field(default_factory=list)
    end: float = 0.0

    def frame(self, t: float) -> int:
        return int(np.clip(round(t * FPS), 0, len(self.qpos) - 1))

    def smooth_base(self, t: float, half: float = 0.6) -> np.ndarray:
        """The robot base position averaged over +-half seconds (for a follow camera)."""
        i, k = self.frame(t), round(half * FPS)
        lo, hi = max(0, i - k), min(len(self.base), i + k + 1)
        return np.mean(self.base[lo:hi], axis=0)


class Hold:
    """A prop that follows a robot body with a fixed offset (attached on proximity), moving
    there smoothly over `blend` seconds from wherever it was."""

    def __init__(self, model, data, joint: str, body: str, offset, blend: float = 0.0,
                 t0: float = 0.0) -> None:
        self.m, self.d = model, data
        self.adr = model.joint(joint).qposadr[0]
        self.dof = model.joint(joint).dofadr[0]
        self.body = model.body(body).id
        self.offset = np.asarray(offset, float)
        self.blend, self.t0 = blend, t0
        self.p0 = data.qpos[self.adr:self.adr + 3].copy()
        self.q0 = data.qpos[self.adr + 3:self.adr + 7].copy()

    def apply(self, t: float) -> None:
        d = self.d
        rot = d.xmat[self.body].reshape(3, 3)
        dst = d.xpos[self.body] + rot @ self.offset
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rot.ravel())
        f = 1.0 if self.blend <= 0 else float(np.clip((t - self.t0) / self.blend, 0, 1))
        f = f * f * (3 - 2 * f)
        d.qpos[self.adr:self.adr + 3] = self.p0 + f * (dst - self.p0)
        d.qpos[self.adr + 3:self.adr + 7] = quat if f >= 1 else self.q0
        d.qvel[self.dof:self.dof + 6] = 0


class Run:
    """State of one rollout; scenes reach in through `robot`, `holds` and `t`."""

    def __init__(self, scene, answers: dict, seed: int | None) -> None:
        self.scene, self.answers = scene, answers
        self.model = scene.spec().compile()
        self.data = mujoco.MjData(self.model)
        m, d = self.model, self.data
        scene.setup(m, d)
        self.robot = scene.make_robot(m, d)
        x, y, z, yaw = scene.start()
        rng = None
        if seed is not None:
            rng = np.random.default_rng(seed)
            x += rng.uniform(-START_XY, START_XY)
            y += rng.uniform(-START_XY, START_XY)
            yaw += rng.uniform(-START_YAW, START_YAW)
            if scene.kind == "flyer":
                z += rng.uniform(-START_Z, START_Z)
        if scene.kind == "flyer":
            self.robot.place(x, y, yaw, z=z)
        else:
            self.robot.place(x, y, yaw, rng=rng, jitter=JOINT_JITTER if rng is not None else 0)
        mujoco.mj_forward(m, d)
        self.holds: dict[str, Hold] = {}
        self.t = 0.0
        self.free = set(scene.props)
        self._kin()
        scene.begin(self)

    def _kin(self) -> None:
        """qpos/qvel indices of every joint that is neither the robot's nor a physical prop."""
        m = self.model
        qa, va = [], []
        size = {mujoco.mjtJoint.mjJNT_FREE: (7, 6), mujoco.mjtJoint.mjJNT_BALL: (4, 3)}
        for j in range(m.njnt):
            name = m.joint(j).name
            if name.startswith(self.scene.rprefix) or name in self.free:
                continue
            nq, nv = size.get(m.jnt_type[j], (1, 1))
            qa += range(m.jnt_qposadr[j], m.jnt_qposadr[j] + nq)
            va += range(m.jnt_dofadr[j], m.jnt_dofadr[j] + nv)
        self.kq, self.kv = np.array(qa, int), np.array(va, int)

    def hold(self, joint: str, body: str, offset, blend: float = 0.0) -> None:
        self.holds[joint] = Hold(self.model, self.data, joint, body, offset, blend, self.t)

    def release(self, joint: str, vel=None) -> None:
        h = self.holds.pop(joint, None)
        if h is not None and vel is not None:
            self.data.qvel[h.dof:h.dof + 3] = vel


def rollout(scene, answers: dict, seed: int | None = None, record: bool = True,
            tail: float = 1.5) -> Traj:
    run = Run(scene, answers, seed)
    m, d, robot = run.model, run.data, run.robot
    legs = scene.legs(answers)
    traj = Traj()
    dt = m.opt.timestep
    li, active, next_start, done_at = 0, None, legs[0].start if legs else 0.0, None
    saved = d.qpos[run.kq].copy()
    next_frame, step = 0.0, 0
    while True:
        t = step * dt
        run.t = t
        if t >= next_frame - 1e-9:
            scene.scenery(m, d, t)
            saved = d.qpos[run.kq].copy()
            if record:
                traj.qpos.append(d.qpos.copy())
                traj.mocap.append(d.mocap_pos.copy())
                a = robot.free.qposadr[0]
                traj.base.append(d.qpos[a:a + 3].copy())
            next_frame += 1.0 / FPS
        if li < len(legs):
            leg = legs[li]
            if active is None and t >= next_start:
                active = leg
                traj.legs[leg.key] = [round(t, 3), None]
                if leg.on_start:
                    leg.on_start(run)
                if leg.route is not None:
                    pts = leg.route(robot) if callable(leg.route) else leg.route
                    robot.go(pts, leg.heading)
            elif active is not None:
                t0 = traj.legs[leg.key][0]
                finished = (t - t0 >= leg.wait) if leg.route is None else robot.arrived
                if not finished and t - t0 > CAP:
                    traj.timed_out.append(leg.key)
                    finished = True
                if finished:
                    traj.legs[leg.key][1] = round(t, 3)
                    if leg.goal is not None:
                        traj.errors[leg.key] = scene.leg_error(run, leg)
                    if leg.on_end:
                        leg.on_end(run)
                    active, li = None, li + 1
                    if li < len(legs):
                        next_start = max(legs[li].start, t + legs[li].pause)
                    else:
                        done_at = t
        elif done_at is not None and t >= done_at + tail:
            break
        robot.control()
        mujoco.mj_step(m, d)
        d.qpos[run.kq] = saved
        d.qvel[run.kv] = 0
        robot.after_step()
        for h in run.holds.values():
            h.apply(t)
        if scene.kind in ("g1", "go2"):
            z = robot.height()
            traj.min_z = min(traj.min_z, z)
            if z < scene.fall_z:
                traj.fell = True
        step += 1
        if not legs and t > tail:
            break
    traj.end = round(step * dt, 3)
    return traj
