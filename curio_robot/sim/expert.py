"""A scripted pick for the physics world (task T5-04): the demonstrator whose
episodes a learned policy imitates (T5-05).

Top-down grasp: the jaws point down and close across two opposite faces of the
block. Blocks at the far edge of the workspace are out of reach with the jaws
vertical, so the gripper leans outwards there, by the least tilt that reaches.
IK runs on a copy of the data; the arm is only ever moved through its
actuators, by ramping ctrl towards the IK joint targets once per control tick.
Waypoints: above the block, straight down, close the jaw, straight up, hold.

Hidden blocks (task T5-14): an occluded scene puts a 9 cm box 2 cm from the
block, between it and the robot, and the plain grasp clips it. The planner
checks every move against the box on a copy of the model (with a clearance
margin) and, when no plain grasp is clear, tries grasps that suit a box: a
higher pre-grasp, the gripper's thin side turned towards the box, a narrower
jaw, a grasp a little higher up the block, and a way round to the pre-grasp
(also out of the box when fetch_scene puts it against the parked arm)."""

from __future__ import annotations

import copy
import itertools
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
# For a block beside something tall (_stages); only tried when no plain grasp is clear.
HIGH_APPROACH = 0.11  # m: fingertips about 2 cm above the box top
YAW_STEP = np.radians(6)  # between closing yaws from a block face to along the box
BOX_NEAR = 0.12  # m from the block centre: the box is in the way
JAW_NARROW = 0.3  # rad, about 1 cm clearance beside the block
GRASP_RAISE = 0.005  # m higher up the block
FINE_TILTS = np.radians(np.arange(0, 39, 3))
CLEARANCE = 0.003  # m the planned path keeps from the box
RAMP_SAMPLES = 16  # poses checked on the joint-space move to the pre-grasp
VIA_FOLDS = (0.5, 0.3, 0.7)  # how far back towards folded the via pose is, tried in turn
RAISED_ELBOW_WRIST = (0.7, 1.0)  # rad: from the parked pose, lifts the gripper to ~20 cm
VIA_SECONDS = 0.6  # per leg of a move through via poses
STUCK_PENALTY = 1000  # contacts charged to a straight move out of a box
ESCAPE_PAN = 0.4  # rad turned to get the parked arm out of a box placed against it
ESCAPE_GAIN = 2e-4  # m further from the box each checked step on the way out of it
DEFLECTED = 0.02  # rad off its ctrl: the parked arm has been pushed by a box
DESCENT_SPEED = APPROACH / 0.6  # m/s down to the block
HIGH_DESCENT_SPEED = DESCENT_SPEED / 3  # beside the box: the servos lag less
LIFT_SPEED = APPROACH / 0.8  # m/s back up


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
    path: tuple[Pose, ...]  # jaws around the block first, then up to the pre-grasp above it
    jaw: float = JAW_OPEN  # opening on the way down
    height: float = APPROACH  # pre-grasp height above the grasp, m
    vias: tuple[np.ndarray, ...] = ()  # (5,) arm poses passed on the way to the pre-grasp

    @property
    def at(self) -> Pose:
        return self.path[0]

    @property
    def above(self) -> Pose:
        return self.path[-1]


class Ik:
    """Damped least squares on a private MjData, so the world is untouched.

    The model is a copy whose box has a CLEARANCE collision margin, so the
    collision checks keep the arm that far from it."""

    def __init__(self, world: PhysWorld) -> None:
        self.model = copy.deepcopy(world.model)
        self.model.geom_margin[self.model.geom("occluder").id] = CLEARANCE
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:] = world.data.qpos
        self.body = self.model.body("gripper").id
        self.lo, self.hi = self.model.jnt_range[:N_ARM].T
        self.deepest = np.inf
        self.arm_bodies = set(range(self.model.body("base").id, self.model.nbody)) - {
            self.model.body(n).id for n in world.body_names()
        }

    def collisions(self, q: np.ndarray, jaw: float, ignore_geom: int) -> int:
        """How many contacts the arm at q (jaw open to jaw) would make with
        anything but itself and the geom ignore_geom (the block to grasp;
        -1 for none). Within CLEARANCE of the box counts as a contact.
        Sets self.deepest: the least contact distance (m; inf if none)."""
        m, d = self.model, self.data
        d.qpos[:N_ARM] = q
        d.qpos[N_ARM] = jaw
        mujoco.mj_kinematics(m, d)
        mujoco.mj_collision(m, d)
        hits = 0
        self.deepest = np.inf
        for c in d.contact[: d.ncon]:
            arm = [m.geom_bodyid[g] in self.arm_bodies for g in (c.geom1, c.geom2)]
            if arm[0] != arm[1] and ignore_geom not in (c.geom1, c.geom2):
                hits += 1
                self.deepest = min(self.deepest, float(c.dist))
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

    def grasp_pose(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Where GRASP_POINT is (world, m) and the gripper body rotation with the arm at q."""
        self.data.qpos[:N_ARM] = q[:N_ARM]
        mujoco.mj_kinematics(self.model, self.data)
        r = self.data.xmat[self.body].reshape(3, 3).copy()
        return self.data.xpos[self.body] + r @ GRASP_POINT, r


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
    ik: Ik, pos: np.ndarray, yaw: float, outward: np.ndarray, seed: np.ndarray,
    min_tilt: float = 0.0, tilts: np.ndarray = TILTS,
) -> tuple[Pose, float] | None:
    """The least-tilted pose (from min_tilt) with GRASP_POINT at pos and the
    jaws along yaw, and its jaw-axis error; None if no tilt in tilts reaches."""
    for tilt in tilts[tilts >= min_tilt - 1e-9]:
        q, ep, axis = ik.solve(pos, jaw_frame(yaw, tilt, outward), seed)
        if ep <= IK_POS_TOL and axis <= MAX_AXIS_ERR:
            return Pose(float(tilt), q), axis
    return None


def _path(
    ik: Ik, centre: np.ndarray, yaw: float, outward: np.ndarray, height: float,
    tilts: np.ndarray = TILTS,
) -> tuple[list[Pose], float] | None:
    """A pose every PATH_STEP up the vertical line from centre (the grasp) to
    height above it, each with the least tilt that reaches (higher up the
    wrist runs out of bend sooner, so the tilt grows on the way up), and the
    jaw-axis error at the block. None if some height is out of reach."""
    at = _reach(ik, centre, yaw, outward, _seed(centre[:2]), tilts=tilts)
    if at is None:
        return None
    path = [at[0]]
    for h in np.linspace(0.0, height, round(height / PATH_STEP) + 1)[1:]:
        pose = _reach(ik, centre + [0, 0, h], yaw, outward, path[-1].q, path[-1].tilt, tilts)
        if pose is None:
            return None
        path.append(pose[0])
    return path, at[1]


def _ramp_hits(ik: Ik, start: np.ndarray, q: np.ndarray, jaw: float) -> int:
    """Contacts on the joint-space move from ctrl start to pre-grasp q.

    A box can be placed against the parked arm; contacts the move starts in
    don't count until it has got clear of them, as long as every step moves
    away from the box by ESCAPE_GAIN (else STUCK_PENALTY: pressed against
    the box, the arm jams rather than slide along it)."""
    goal = np.append(q, jaw)
    hits = 0
    clear = ik.collisions(start[:N_ARM], float(start[N_ARM]), -1) == 0
    depth = ik.deepest
    for s in np.linspace(0.0, 1.0, RAMP_SAMPLES)[1:]:
        c = start + (goal - start) * s
        n = ik.collisions(c[:N_ARM], float(c[N_ARM]), -1)
        if not clear and n and ik.deepest < depth + ESCAPE_GAIN:
            hits += STUCK_PENALTY
        depth = ik.deepest
        clear = clear or n == 0
        hits += n if clear else 0
    return hits


def _via_chains(
    start: np.ndarray, above: np.ndarray, stuck: bool
) -> list[tuple[np.ndarray, ...]]:
    """Ways round to the pre-grasp above when the straight move hits something.

    Part-folded: turned and rolled as at the pre-grasp, with shoulder, elbow
    and wrist part-way back to start's, so the arm unfolds onto the
    pre-grasp. Raised: first the elbow and wrist up at start's turn (the
    parked gripper is low, beside the table), then (or not) turned, then
    down. When the arm starts stuck against a box: first a turn either way
    out of the box and then raised, else the raised way. (Pressed against
    the box, the arm only comes away along the contact normal, and the
    collision check can't tell sliding from separating well, so the turn,
    which separates in most such scenes, goes first.)"""
    def raise_(q: np.ndarray) -> np.ndarray:
        up = q.copy()
        up[2:4] = RAISED_ELBOW_WRIST
        return up

    turned = raise_(start[:N_ARM])
    turned[[0, 4]] = above[[0, 4]]
    chains: list[tuple[np.ndarray, ...]] = [(raise_(start[:N_ARM]), turned)]
    if stuck:
        for sign in (-1, 1):
            out = start[:N_ARM].copy()
            out[0] += sign * ESCAPE_PAN
            chains.insert(0, (out, raise_(out), turned))
        return chains
    for fold in VIA_FOLDS:
        q = above.copy()
        q[1:4] += fold * (start[1:4] - above[1:4])
        chains.insert(-1, (q,))
    return chains


def _approach(
    ik: Ik, start: np.ndarray, above: np.ndarray, jaw: float
) -> tuple[tuple[np.ndarray, ...], int]:
    """The move from ctrl start to the pre-grasp above: straight in joint
    space if that is clear, else the _via_chains way that touches least.
    A box can be placed against the parked arm (fetch_scene doesn't know
    where it is); then the arm first has to move straight away from it (see
    _ramp_hits), so the straight move costs STUCK_PENALTY.
    Returns (via poses, contacts)."""
    stuck = ik.collisions(start[:N_ARM], float(start[N_ARM]), -1) > 0
    direct = _ramp_hits(ik, start, above, jaw)
    best: tuple[tuple[np.ndarray, ...], int] = ((), direct + STUCK_PENALTY * stuck)
    for chain in _via_chains(start, above, stuck):
        if best[1] == 0:
            break
        hits, at = 0, start
        for q in (*chain, above):
            hits += _ramp_hits(ik, at, q, jaw)
            at = np.append(q, jaw)
        if hits < best[1]:
            best = (chain, hits)
    return best


def _start(world: PhysWorld) -> np.ndarray:
    """The ctrl the first move starts from: the current ctrl, unless the arm
    has been pushed more than DEFLECTED off it (a box placed against the
    parked arm), then where the arm actually is."""
    ctrl = world.ctrl()
    q = world.arm_qpos()
    if np.abs(q[:N_ARM] - ctrl[:N_ARM]).max() <= DEFLECTED:
        return ctrl
    return np.append(q[:N_ARM], ctrl[N_ARM])


@dataclass(frozen=True, slots=True)
class _Stage:
    """One round of grasp candidates (see plan_grasp)."""

    height: float  # pre-grasp height
    yaws: tuple[tuple[float, float], ...]  # (closing yaw, angle off the block's faces)
    jaws: tuple[float, ...]
    lifts: tuple[float, ...]  # grasp heights above the block centre
    tilts: np.ndarray
    precheck: bool  # test the jaws at the block before planning a whole path


def _wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def _box_along(world: PhysWorld, i: int) -> float | None:
    """The yaw of the box's long side if the box is on the table beside
    block i (within BOX_NEAR), else None."""
    occ = world.model.body("occluder").id
    pos = world.model.body_pos[occ]
    if pos[2] < 0 or np.linalg.norm(pos[:2] - world.block_position(i)[:2]) > BOX_NEAR:
        return None
    w, _, _, z = world.model.body_quat[occ]
    return 2 * float(np.arctan2(z, w)) + np.pi / 2  # the box is long along its y axis


def _box_yaws(
    world: PhysWorld, i: int, along: float | None
) -> tuple[tuple[float, float], ...]:
    """Closing yaws for a block beside a box whose long side runs along yaw
    along, least off-face first: from the face nearest each of the two
    along-side directions, in YAW_STEP steps, to that direction (where the
    gripper's long side is parallel to the box, and furthest from it). Just
    the faces if there's no box (along None)."""
    faces = [world.block_yaw(i) + k * np.pi / 2 for k in range(4)]
    if along is None:
        return tuple((f, 0.0) for f in faces)
    out: list[tuple[float, float]] = []
    for d in (along, along + np.pi):
        face = min(faces, key=lambda f, d=d: abs(_wrap(d - f)))
        delta = _wrap(d - face)
        n = int(np.ceil(abs(delta) / YAW_STEP))
        out += [(face + delta * k / max(n, 1), abs(delta) * k / max(n, 1)) for k in range(n + 1)]
    return tuple(sorted(out, key=lambda c: c[1]))


def _stages(world: PhysWorld, i: int) -> tuple[_Stage, ...]:
    """Plain grasps first; then the grasps for a block beside something tall
    (the 9 cm box of an occluded scene): a pre-grasp high enough that the
    fingertips pass over the box, yaws turning the gripper's long side along
    it, a narrower jaw, finer tilts, and a grasp a little higher up the block
    (less reach, so less lean towards the box). With no box near, the second
    stage keeps the plain pre-grasp height (it helps far blocks whose tilted
    fingertips touch the table)."""
    faces = tuple((world.block_yaw(i) + k * np.pi / 2, 0.0) for k in range(4))
    along = _box_along(world, i)
    return (
        _Stage(APPROACH, faces, (JAW_OPEN,), (0.0,), TILTS, precheck=False),
        _Stage(APPROACH if along is None else HIGH_APPROACH, _box_yaws(world, i, along),
               (JAW_OPEN, JAW_NARROW), (0.0, GRASP_RAISE), FINE_TILTS, precheck=True),
    )


def plan_grasp(world: PhysWorld, i: int) -> Grasp | None:
    """The grasp of block i, or None if it is out of reach.

    Stage by stage (_stages), until a stage has a clear grasp: the move there
    and the path down (jaws open) touch nothing but the block and keep
    CLEARANCE from the box. A clear grasp wins, then the least angle off the
    block's faces, the lower grasp, the least tilt at the block, the least
    jaw-axis error, (when tilted) the fixed jaw on the outer side, the
    smallest wrist roll, and the wider jaw."""
    centre = world.block_position(i)
    outward = np.array([centre[0], centre[1], 0.0])
    outward /= np.linalg.norm(outward)
    ik = Ik(world)
    start = _start(world)
    block = world.block_ids()[i]
    best: tuple[tuple, Grasp] | None = None
    for stage in _stages(world, i):
        for (yaw, off), lift in itertools.product(stage.yaws, stage.lifts):
            if best is not None and not best[0][0] and off > best[0][1] + 1e-9:
                break  # a clear grasp closer to the faces is already in hand
            pos = centre + [0, 0, lift]
            if _blocked_at(ik, pos, yaw, outward, stage, block):
                continue
            planned = _path(ik, pos, yaw, outward, stage.height, stage.tilts)
            if planned is None:
                continue
            path, axis = planned
            # Leaning outwards swings the fingertips outwards; with the fixed jaw on
            # the outer side it swings away from the block instead of onto it.
            inward = -float(np.cos(yaw) * outward[0] + np.sin(yaw) * outward[1])
            fixed_inside = path[0].tilt > 0 and inward < 0
            for jaw in stage.jaws:
                vias, hits = _approach(ik, start, path[-1].q, jaw)
                hits += sum(ik.collisions(p.q, jaw, block) for p in path)
                score = (hits > 0, round(off, 3), lift, path[0].tilt, round(axis, 1),
                         fixed_inside, abs(path[0].q[4]), jaw != JAW_OPEN)
                if best is None or score < best[0]:
                    best = (score, Grasp(yaw, tuple(path), jaw, stage.height, vias))
        if best is not None and not best[0][0]:
            break
    return None if best is None else best[1]


def _blocked_at(
    ik: Ik, pos: np.ndarray, yaw: float, outward: np.ndarray, stage: _Stage, block: int
) -> bool:
    """True if the stage prechecks and the jaws can't be around the block
    without a contact at any of the openings: a quick test before planning
    the whole path (the plain stage's candidates are few, and usually clear)."""
    if not stage.precheck:
        return False
    at = _reach(ik, pos, yaw, outward, _seed(pos[:2]), tilts=stage.tilts)
    return at is None or all(ik.collisions(at[0].q, j, block) for j in stage.jaws)


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
            tick(start + (goal - start) * smooth(t / n))

    def vertical(down: bool, seconds: float, jaw: float) -> None:
        # Joint-space blend between poses PATH_STEP apart: close to a straight line.
        n = max(1, round(seconds / CONTROL_DT))
        for t in range(1, n + 1):
            s = smooth(t / n)
            tick(np.append(along(path, 1 - s if down else s), jaw))

    def hold(seconds: float) -> None:
        for _ in range(round(seconds / CONTROL_DT)):
            tick(world.ctrl())

    if not np.array_equal(_start(world), world.ctrl()):
        tick(_start(world))  # pushed off its target: move on from where it is
    for via in grasp.vias:
        ramp(via, VIA_SECONDS, grasp.jaw)
    ramp(grasp.above.q, VIA_SECONDS if grasp.vias else 1.2, grasp.jaw)
    hold(0.2)
    speed = DESCENT_SPEED if grasp.height == APPROACH else HIGH_DESCENT_SPEED
    vertical(True, grasp.height / speed, grasp.jaw)
    hold(0.2)
    ramp(grasp.at.q, 0.3, JAW_CLOSED)
    hold(0.2)
    vertical(False, grasp.height / LIFT_SPEED, JAW_CLOSED)
    hold(0.1)
    if not world.lifted(i, LIFT_HEIGHT):
        return PickResult(False, len(log), np.array(log), "not lifted")
    hold(HOLD_TIME)
    if not world.lifted(i, LIFT_HEIGHT):
        return PickResult(False, len(log), np.array(log), "dropped")
    return PickResult(True, len(log), np.array(log))


def along(path: np.ndarray, s: float) -> np.ndarray:
    """Piecewise-linear point at fraction s (0: path[0], 1: path[-1])."""
    x = np.clip(s, 0.0, 1.0) * (len(path) - 1)
    k = min(int(x), len(path) - 2)
    return path[k] + (path[k + 1] - path[k]) * (x - k)


def smooth(s: float) -> float:
    """Smoothstep: zero velocity at both ends of a ramp."""
    return s * s * (3 - 2 * s)


# ---- noisy-descent demonstrations with a state-checked close (task T5-16) ----------------
# Run 3's policy closed in mid-air: its demos always closed at the same moment of a clean
# descent, so "when" was learnable from time alone. Here the executed arm command carries
# correlated noise on the way down (DART: the recorded label stays the expert's own command,
# which pulls back to the path), the pause above and the descent vary in length, and the
# jaws close only once the block is between them.
NOISE_TAU = 0.25  # s, correlation time of the grasp-point noise
NOISE_CLIP = 0.025  # m, per axis
NOISE_FADE = (0.015, 0.04)  # m above the grasp: no noise below the first, full above the second
TOP_HOLD = (0.1, 0.4)  # s above the block before going down, uniform
DESCENT_TIME = (0.6, 0.8)  # s for the descent, uniform
# Block centre off GRASP_POINT in the gripper frame (x closing, y across the jaws, z up)
# within which the jaws straddle it. Clean picks close within about 2 mm of it.
STRADDLE_TOL = np.array([0.006, 0.010, 0.008])
SETTLE_MIN = 0.1  # s at the bottom before the first straddle check can close
SETTLE_MAX = 0.5  # s waiting to straddle before going up and coming down again
RETRY_RISE = 1.0  # share of the approach height risen for a retry (clears a jaw on the block)
MAX_RETRIES = 1

Label = Callable[[PhysWorld, np.ndarray, np.ndarray], None]


@dataclass(frozen=True, slots=True)
class NoisyPickResult:
    success: bool
    ticks: int
    reason: str = ""
    retries: int = 0  # times it went up and came down again
    settle_ticks: int = 0  # ticks waited at the bottom before closing (all tries)
    noisy: bool = True  # False when the grasp needed vias or the high stage (run clean)


def grasp_error(world: PhysWorld, i: int) -> np.ndarray:
    """Block i's centre relative to GRASP_POINT, in the gripper body frame (m; x is the jaw
    closing axis): zero when the block sits where the grasp wants it."""
    body = world.model.body("gripper").id
    r = world.data.xmat[body].reshape(3, 3)
    return r.T @ (world.block_position(i) - world.data.xpos[body]) - GRASP_POINT


def straddles(world: PhysWorld, i: int, tol: np.ndarray = STRADDLE_TOL) -> bool:
    """True when the open jaws are around block i: grasp_error within tol on every axis."""
    return bool(np.all(np.abs(grasp_error(world, i)) <= tol))


def _fade(height: float) -> float:
    lo, hi = NOISE_FADE
    return float(np.clip((height - lo) / (hi - lo), 0.0, 1.0))


def noisy_pick(
    world: PhysWorld, i: int, rng: np.random.Generator, sigma: float, on_tick: Label
) -> NoisyPickResult:
    """Pick block i as pick() does, with DART-style noise on the way down and a close that
    waits for the jaws to straddle the block.

    Over the pause above the block and the descent, the executed command puts the grasp
    point off the planned one by an Ornstein-Uhlenbeck offset (sigma m per axis, faded out
    over the last NOISE_FADE of the descent); on_tick(world, executed, label) gets both, and
    the label is the noise-free command. At the bottom it holds the grasp pose (no noise)
    and closes on the first tick the jaws straddle the block (after SETTLE_MIN); if they
    don't within SETTLE_MAX it rises, re-plans for where the block is now and comes down
    again without noise (up to MAX_RETRIES). Grasps through vias or from the high stage run as pick()."""
    grasp = plan_grasp(world, i)
    if grasp is None:
        return NoisyPickResult(False, 0, "out of reach", noisy=False)
    if grasp.vias or grasp.height != APPROACH:
        result = pick(world, i, lambda w, c: on_tick(w, c, c))
        return NoisyPickResult(result.success, result.steps, result.reason, noisy=False)
    ik = Ik(world)
    ticks = settle = retries = 0
    noise = np.zeros(3)
    amplitude = sigma  # 0 on a retry: the second descent is clean
    decay = CONTROL_DT / NOISE_TAU

    def tick(label: np.ndarray, executed: np.ndarray | None = None) -> None:
        nonlocal ticks
        ticks += 1
        executed = label if executed is None else executed
        world.step(executed, SUBSTEPS)
        on_tick(world, executed.copy(), label.copy())

    def noisy(q: np.ndarray, jaw: float, height: float) -> None:
        nonlocal noise
        if amplitude == 0.0:
            tick(np.append(q, jaw))
            return
        noise = noise * (1 - decay) + amplitude * np.sqrt(2 * decay) * rng.standard_normal(3)
        noise = np.clip(noise, -NOISE_CLIP, NOISE_CLIP)
        point, r = ik.grasp_pose(q)
        moved, err, _ = ik.solve(point + noise * _fade(height), r, q, iters=30)
        tick(np.append(q, jaw), np.append(moved if err <= IK_POS_TOL else q, jaw))

    def ramp(target: np.ndarray, seconds: float, jaw: float) -> None:
        start = world.ctrl()
        goal = np.append(target, jaw)
        n = max(1, round(seconds / CONTROL_DT))
        for t in range(1, n + 1):
            tick(start + (goal - start) * smooth(t / n))

    def hold(seconds: float) -> None:
        for _ in range(round(seconds / CONTROL_DT)):
            tick(world.ctrl())

    def descend(path: np.ndarray, top: float, seconds: float, jaw: float) -> None:
        """Noisy, from fraction top of the path (1: the pre-grasp) down to the grasp."""
        n = max(1, round(seconds / CONTROL_DT))
        for t in range(1, n + 1):
            s = top * (1 - smooth(t / n))
            noisy(along(path, s), jaw, s * APPROACH)

    def settle_ok(q: np.ndarray, jaw: float) -> bool:
        nonlocal settle
        for k in range(round(SETTLE_MAX / CONTROL_DT)):
            if k >= round(SETTLE_MIN / CONTROL_DT) and straddles(world, i):
                return True
            tick(np.append(q, jaw))
            settle += 1
        return straddles(world, i)

    path = np.array([pose.q for pose in grasp.path])
    ramp(grasp.above.q, 1.2, grasp.jaw)
    for _ in range(round(rng.uniform(*TOP_HOLD) / CONTROL_DT)):
        noisy(grasp.above.q, grasp.jaw, APPROACH)
    descend(path, 1.0, rng.uniform(*DESCENT_TIME), grasp.jaw)
    while not settle_ok(path[0], grasp.jaw):
        if retries == MAX_RETRIES:
            return NoisyPickResult(False, ticks, "no straddle", retries, settle)
        retries += 1
        amplitude = 0.0
        n = round(0.3 / CONTROL_DT)
        for t in range(1, n + 1):  # straight up, clean
            tick(np.append(along(path, RETRY_RISE * smooth(t / n)), grasp.jaw))
        again = plan_grasp(world, i)
        if again is None or again.vias or again.height != APPROACH:
            return NoisyPickResult(False, ticks, "no re-plan", retries, settle)
        grasp, path = again, np.array([pose.q for pose in again.path])
        ramp(along(path, RETRY_RISE), 0.3, grasp.jaw)
        descend(path, RETRY_RISE, 0.5, grasp.jaw)
    ramp(path[0], 0.3, JAW_CLOSED)
    hold(0.2)
    n = round(APPROACH / LIFT_SPEED / CONTROL_DT)
    for t in range(1, n + 1):
        tick(np.append(along(path, smooth(t / n)), JAW_CLOSED))
    hold(0.1)
    if not world.lifted(i, LIFT_HEIGHT):
        return NoisyPickResult(False, ticks, "not lifted", retries, settle)
    hold(HOLD_TIME)
    if not world.lifted(i, LIFT_HEIGHT):
        return NoisyPickResult(False, ticks, "dropped", retries, settle)
    return NoisyPickResult(True, ticks, "", retries, settle)
