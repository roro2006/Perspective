"""A scripted pick for the physics world (task T5-04): the demonstrator whose
episodes a learned policy imitates (T5-05).

Top-down grasp: the jaws point down and close across two opposite faces of the
block. Blocks at the far edge of the workspace are out of reach with the jaws
vertical, so the gripper leans outwards there, by the least tilt that reaches.
IK runs on a copy of the data; the arm is only ever moved through its
actuators, by ramping ctrl towards the IK joint targets once per control tick.
Waypoints: above the block, straight down, close the jaw, straight up, hold."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import numpy as np

from curio_robot.sim.physics import TIMESTEP, PhysWorld

CONTROL_DT = 0.02  # s per control tick (ctrl_log row)
SUBSTEPS = round(CONTROL_DT / TIMESTEP)
# Where the block centre should sit in the gripper body frame: a little off the
# fixed jaw's inner face (x = -0.0098) so it clears on the way down, and low
# enough that the fingertips stop about 4 mm above the table.
GRASP_POINT = np.array([0.008, 0.0, -0.0925])
JAW_OPEN = 0.45  # rad, about 2 cm clearance beside a 2.5 cm block
JAW_CLOSED = -0.17  # commanded past contact so the servo squeezes at its force limit
APPROACH = 0.05  # m above the grasp height for the pre-grasp and lift
HOLD_TIME = 0.5  # s the block must stay up after the lift
LIFT_HEIGHT = 0.03
N_ARM = 5  # joints the IK moves (not the gripper)
ROT_WEIGHT = 0.05  # m per rad: position wins over orientation when both can't be met
IK_POS_TOL = 0.003  # m
MAX_AXIS_ERR = 0.25  # rad between the jaw axis and the block's face normal
PATH_STEP = 0.01  # m between planned poses on the way down and up
TILTS = np.radians([0, 6, 12, 18, 24, 30, 36])  # outward lean tried, least first


@dataclass(frozen=True, slots=True)
class PickResult:
    success: bool
    steps: int  # control ticks run
    ctrl_log: np.ndarray  # (steps, 6) ctrl sent at each tick
    reason: str = ""  # why it failed, "" on success


Tick = Callable[[PhysWorld, np.ndarray], None]


@dataclass(frozen=True, slots=True)
class Pose:
    tilt: float  # outward lean from vertical, rad
    q: np.ndarray  # (5,) arm joints


@dataclass(frozen=True, slots=True)
class Grasp:
    yaw: float  # jaw closing direction, rad
    path: tuple[Pose, ...]  # jaws around the block first, then up to APPROACH above it

    @property
    def at(self) -> Pose:
        return self.path[0]

    @property
    def above(self) -> Pose:
        return self.path[-1]


class _Ik:
    """Damped least squares on a private MjData, so the world is untouched."""

    def __init__(self, world: PhysWorld) -> None:
        self.model = world.model
        self.data = mujoco.MjData(world.model)
        self.data.qpos[:] = world.data.qpos
        self.body = self.model.body("gripper").id
        self.lo, self.hi = self.model.jnt_range[:N_ARM].T
        self.arm_bodies = set(range(self.model.body("base").id, self.model.nbody)) - {
            self.model.body(n).id for n in world.body_names()
        }

    def collisions(self, q: np.ndarray, jaw: float, ignore_geom: int) -> int:
        """How many contacts the arm at q (jaw open to jaw) would make with
        anything but itself and the geom ignore_geom (the block to grasp)."""
        m, d = self.model, self.data
        d.qpos[:N_ARM] = q
        d.qpos[N_ARM] = jaw
        mujoco.mj_kinematics(m, d)
        mujoco.mj_collision(m, d)
        hits = 0
        for c in d.contact[: d.ncon]:
            arm = [m.geom_bodyid[g] in self.arm_bodies for g in (c.geom1, c.geom2)]
            if arm[0] != arm[1] and ignore_geom not in (c.geom1, c.geom2):
                hits += 1
        return hits

    def solve(
        self, pos: np.ndarray, target: np.ndarray, seed: np.ndarray, iters: int = 150
    ) -> tuple[np.ndarray, float, float]:
        """Joint angles (5) putting GRASP_POINT at pos with the gripper body as
        close to rotation target as five joints allow. Returns (q, position
        error m, jaw-axis error rad)."""
        m, d = self.model, self.data
        d.qpos[:N_ARM] = seed[:N_ARM]
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        for _ in range(iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            r = d.xmat[self.body].reshape(3, 3)
            point = d.xpos[self.body] + r @ GRASP_POINT
            ep = pos - point
            er = 0.5 * sum(np.cross(r[:, k], target[:, k]) for k in range(3))
            if np.linalg.norm(ep) < 2e-4 and np.linalg.norm(er) < 2e-3:
                break
            mujoco.mj_jac(m, d, jacp, jacr, point, self.body)
            j = np.vstack([jacp[:, :N_ARM], ROT_WEIGHT * jacr[:, :N_ARM]])
            e = np.concatenate([ep, ROT_WEIGHT * er])
            dq = j.T @ np.linalg.solve(j @ j.T + 1e-5 * np.eye(6), e)
            d.qpos[:N_ARM] = np.clip(d.qpos[:N_ARM] + np.clip(dq, -0.3, 0.3), self.lo, self.hi)
        mujoco.mj_kinematics(m, d)
        r = d.xmat[self.body].reshape(3, 3)
        ep = pos - (d.xpos[self.body] + r @ GRASP_POINT)
        axis = float(np.arccos(np.clip(abs(r[:, 0] @ target[:, 0]), -1, 1)))
        return d.qpos[:N_ARM].copy(), float(np.linalg.norm(ep)), axis


def jaw_frame(yaw: float, tilt: float = 0.0, outward: np.ndarray | None = None) -> np.ndarray:
    """Gripper body rotation: x (jaw closing) horizontal along yaw, z up (the
    fingertips are on the body's -z side, so they point down), then leaned by
    tilt so the fingertips swing towards outward (a horizontal unit vector)."""
    x = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    z = np.array([0.0, 0.0, 1.0])
    frame = np.column_stack([x, np.cross(z, x), z])
    if tilt == 0.0 or outward is None:
        return frame
    axis = np.cross(outward, z)
    lean = np.zeros(9)
    quat = np.zeros(4)
    mujoco.mju_axisAngle2Quat(quat, axis, tilt)
    mujoco.mju_quat2Mat(lean, quat)
    return lean.reshape(3, 3) @ frame


def _seed(xy: np.ndarray) -> np.ndarray:
    # Arm pointing at the block with the wrist bent down: converges from here.
    return np.array([np.arctan2(xy[1], xy[0]), 0.0, 0.5, 1.2, 0.0])


def _reach(
    ik: _Ik, pos: np.ndarray, yaw: float, outward: np.ndarray, seed: np.ndarray,
    min_tilt: float = 0.0,
) -> tuple[Pose, float] | None:
    """The least-tilted pose (from min_tilt) with GRASP_POINT at pos and the
    jaws along yaw, and its jaw-axis error; None if no tilt in TILTS reaches."""
    for tilt in TILTS[TILTS >= min_tilt - 1e-9]:
        q, ep, axis = ik.solve(pos, jaw_frame(yaw, tilt, outward), seed)
        if ep <= IK_POS_TOL and axis <= MAX_AXIS_ERR:
            return Pose(float(tilt), q), axis
    return None


def plan_grasp(world: PhysWorld, i: int) -> Grasp | None:
    """The grasp of block i, or None if it is out of reach.

    A pose every PATH_STEP up the vertical line from the block to APPROACH
    above it, each with the least tilt that reaches (higher up the wrist runs
    out of bend sooner, so the tilt grows on the way up). Of the four yaws
    that close across the block's faces, one whose path (jaws open) touches
    nothing but the block wins, then the least tilt at the block, then the
    least jaw-axis error, then (when tilted) the fixed jaw on the outer side,
    then the smallest wrist roll."""
    centre = world.block_position(i)
    outward = np.array([centre[0], centre[1], 0.0])
    outward /= np.linalg.norm(outward)
    heights = np.linspace(0.0, APPROACH, round(APPROACH / PATH_STEP) + 1)
    ik = _Ik(world)
    best: tuple[tuple[bool, float, float, bool, float], Grasp] | None = None
    for k in range(4):
        yaw = world.block_yaw(i) + k * np.pi / 2
        at = _reach(ik, centre, yaw, outward, _seed(centre[:2]))
        if at is None:
            continue
        path = [at[0]]
        for h in heights[1:]:
            pose = _reach(ik, centre + [0, 0, h], yaw, outward, path[-1].q, path[-1].tilt)
            if pose is None:
                break
            path.append(pose[0])
        if len(path) < len(heights):
            continue
        # Leaning outwards swings the fingertips outwards; with the fixed jaw on
        # the outer side it swings away from the block instead of onto it.
        inward = -float(np.cos(yaw) * outward[0] + np.sin(yaw) * outward[1])
        fixed_inside = at[0].tilt > 0 and inward < 0
        hits = sum(ik.collisions(pose.q, JAW_OPEN, world.block_ids()[i]) for pose in path)
        score = (hits > 0, at[0].tilt, round(at[1], 1), fixed_inside, abs(at[0].q[4]))
        if best is None or score < best[0]:
            best = (score, Grasp(yaw, tuple(path)))
    return None if best is None else best[1]


def pick(world: PhysWorld, i: int, on_tick: Tick | None = None) -> PickResult:
    """Pick block i: above it, straight down, close, straight up, hold.

    success: block i lifted LIFT_HEIGHT and still lifted HOLD_TIME later.
    on_tick(world, ctrl) runs after every control tick (for recording)."""
    log: list[np.ndarray] = []
    grasp = plan_grasp(world, i)
    if grasp is None:
        return PickResult(False, 0, np.zeros((0, 6)), "out of reach")
    path = np.array([pose.q for pose in grasp.path])

    def tick(ctrl: np.ndarray) -> None:
        world.step(ctrl, SUBSTEPS)
        log.append(world.ctrl())
        if on_tick is not None:
            on_tick(world, log[-1])

    def ramp(target: np.ndarray, seconds: float, jaw: float) -> None:
        start = world.ctrl()
        goal = np.append(target, jaw)
        n = max(1, round(seconds / CONTROL_DT))
        for t in range(1, n + 1):
            tick(start + (goal - start) * _smooth(t / n))

    def vertical(down: bool, seconds: float, jaw: float) -> None:
        # Joint-space blend between poses PATH_STEP apart: close to a straight line.
        n = max(1, round(seconds / CONTROL_DT))
        for t in range(1, n + 1):
            s = _smooth(t / n)
            tick(np.append(_along(path, 1 - s if down else s), jaw))

    def hold(seconds: float) -> None:
        for _ in range(round(seconds / CONTROL_DT)):
            tick(world.ctrl())

    ramp(grasp.above.q, 1.2, JAW_OPEN)
    hold(0.2)
    vertical(True, 0.6, JAW_OPEN)
    hold(0.2)
    ramp(grasp.at.q, 0.3, JAW_CLOSED)
    hold(0.2)
    vertical(False, 0.8, JAW_CLOSED)
    hold(0.1)
    if not world.lifted(i, LIFT_HEIGHT):
        return PickResult(False, len(log), np.array(log), "not lifted")
    hold(HOLD_TIME)
    if not world.lifted(i, LIFT_HEIGHT):
        return PickResult(False, len(log), np.array(log), "dropped")
    return PickResult(True, len(log), np.array(log))


def _along(path: np.ndarray, s: float) -> np.ndarray:
    """Piecewise-linear point at fraction s (0: path[0], 1: path[-1])."""
    x = np.clip(s, 0.0, 1.0) * (len(path) - 1)
    k = min(int(x), len(path) - 2)
    return path[k] + (path[k + 1] - path[k]) * (x - k)


def _smooth(s: float) -> float:
    """Smoothstep: zero velocity at both ends of a ramp."""
    return s * s * (3 - 2 * s)
