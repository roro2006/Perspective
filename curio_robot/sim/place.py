"""Put a held block down at a point (task T5-23): the other half of expert.pick.

The expert picks and holds a block 5 cm up. Here the arm carries it at a height that clears
every block on the table (a stack included), along a straight horizontal line with an IK pose
every CARRY_STEP, the jaws keeping their world yaw so the block keeps its orientation (a block
stacked on another lands face-aligned). Over the goal it goes straight down the same way the
expert goes down to a block, opens the jaw and rises. The block's offset in the jaws is
measured after the pick, so the block, not the grasp point, lands on the goal.

Like the expert, the arm only ever moves through its actuators; IK runs on a copy of the data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from curio_robot.sim.expert import (
    CONTROL_DT,
    JAW_CLOSED,
    N_ARM,
    RAISED_ELBOW_WRIST,
    SUBSTEPS,
    Ik,
    Tick,
    _path,
    _reach,
    along,
    smooth,
)
from curio_robot.sim.physics import PhysWorld
from curio_robot.sim.world import BLOCK_HALF, HOME_QPOS

CARRY_CLEARANCE = 0.06  # m from the top of the highest block to the grasp point while carrying
LOW_CLEARANCES = (0.045, 0.035)  # tried in turn when the wrist can't reach that high
CARRY_MIN = 0.08  # m, lowest carry height of the grasp point
CARRY_STEP = 0.02  # m between IK poses on the carry line
CARRY_SPEED = 0.15  # m/s along the carry line
DROP = 0.003  # m: the block is let go this far above where it will rest
DESCENT_SPEED = 0.05 / 0.7  # m/s, a little slower than the expert's descent to a block
RISE_SPEED = 0.05 / 1.5
RELEASE_TIME = 0.3  # s to open the jaw
# Wider than the expert's 0.45 approach: with the gripper leaning at the far edge, a jaw opened
# only that far drags a block it has just put on another back off it on the way up.
RELEASE_JAW = 0.7
PLACE_TOL_XY = 0.015  # m: the block rests this close to the goal
PLACE_TOL_Z = 0.006
OFFSET_ITERS = 3
RISE_SHIFTS = (0.0, 0.03, 0.06)  # m outwards the rise may lean, tried in turn
HOME_SECONDS = 0.8  # per leg of the move home


@dataclass(frozen=True, slots=True)
class PlaceResult:
    success: bool
    steps: int  # control ticks run
    reason: str = ""


def resting_z(support_top: float = 0.0) -> float:
    """Height of a block's centre resting on a surface at support_top."""
    return support_top + BLOCK_HALF


def carry_height(world: PhysWorld, held: int, goal_z: float,
                 clearance: float = CARRY_CLEARANCE) -> float:
    """Grasp-point height for the carry: clear of every block but the held one."""
    tops = [world.block_position(j)[2] + BLOCK_HALF for j in range(len(world.block_colors))
            if j != held]
    return max(CARRY_MIN, max(tops, default=0.0) + clearance, goal_z + 0.05)


def _yaw(ik: Ik, q: np.ndarray) -> float:
    _, r = ik.grasp_pose(q)
    return float(np.arctan2(r[1, 0], r[0, 0]))


def _outward(p: np.ndarray) -> np.ndarray:
    out = np.array([p[0], p[1], 0.0])
    return out / np.linalg.norm(out)


def _line(ik: Ik, start: np.ndarray, end: np.ndarray, yaw: float, seed: np.ndarray
          ) -> list[np.ndarray] | None:
    """IK poses every CARRY_STEP from start to end (grasp-point positions), least tilt each."""
    n = max(1, int(np.ceil(np.linalg.norm(end - start) / CARRY_STEP)))
    poses, q = [], seed
    for s in np.linspace(0.0, 1.0, n + 1):
        p = start + (end - start) * s
        pose = _reach(ik, p, yaw, _outward(p), q)
        if pose is None:
            return None
        q = pose[0].q
        poses.append(q)
    return poses


def place(world: PhysWorld, held: int, goal: np.ndarray, on_tick: Tick | None = None
          ) -> PlaceResult:
    """Put the held block `held` down with its centre at goal (world, m; goal z is where the
    centre will rest). success: afterwards the block rests within PLACE_TOL of goal."""
    ticks = 0

    def tick(ctrl: np.ndarray) -> None:
        nonlocal ticks
        world.step(ctrl, SUBSTEPS)
        ticks += 1
        if on_tick is not None:
            on_tick(world, world.ctrl())

    def follow(path: list[np.ndarray] | np.ndarray, seconds: float, jaw: float) -> None:
        path = np.asarray(path)
        n = max(1, round(seconds / CONTROL_DT))
        for t in range(1, n + 1):
            q = along(path, smooth(t / n)) if len(path) > 1 else path[0]
            tick(np.append(q, jaw))

    def hold(seconds: float, jaw: float) -> None:
        q = world.ctrl()[:N_ARM]
        for _ in range(round(seconds / CONTROL_DT)):
            tick(np.append(q, jaw))

    ik = Ik(world)
    q0 = world.ctrl()[:N_ARM]
    point, r = ik.grasp_pose(world.arm_qpos())
    local = r.T @ (world.block_position(held) - point)  # block centre in the gripper frame
    yaw = _yaw(ik, world.arm_qpos())
    target, down, z_carry = None, None, 0.0
    offset = r @ local
    for _ in range(OFFSET_ITERS):  # the lean at the goal turns the offset with the gripper
        target = np.asarray(goal, float) + [0, 0, DROP] - offset
        for clearance in (CARRY_CLEARANCE, *LOW_CLEARANCES):
            z_carry = carry_height(world, held, target[2], clearance)
            down = _path(ik, target, yaw, _outward(target), z_carry - target[2])
            if down is not None:
                break
        if down is None:
            return PlaceResult(False, 0, "out of reach")
        offset = ik.grasp_pose(down[0][0].q)[1] @ local
    assert target is not None and down is not None
    over = np.array([target[0], target[1], z_carry])
    up, rise = point, None
    for shift in RISE_SHIFTS:  # close to the base the wrist can't rise straight up
        up = np.array([*(point[:2] + shift * _outward(point)[:2]), z_carry])
        rise = _line(ik, point, up, yaw, world.arm_qpos()[:N_ARM])
        if rise is not None:
            break
    carry = None if rise is None else _line(ik, up, over, yaw, rise[-1])
    if rise is None or carry is None:
        return PlaceResult(False, 0, "out of reach")
    down_q = np.array([p.q for p in down[0]])
    follow([q0, *rise[1:]], max(0.3, (z_carry - point[2]) / RISE_SPEED), JAW_CLOSED)
    dist = float(np.linalg.norm(over - up))
    follow([rise[-1], *carry[1:], down_q[-1]], max(0.6, dist / CARRY_SPEED), JAW_CLOSED)
    hold(0.15, JAW_CLOSED)
    follow(down_q[::-1], (z_carry - target[2]) / DESCENT_SPEED, JAW_CLOSED)
    hold(0.15, JAW_CLOSED)
    follow([down_q[0]], RELEASE_TIME, RELEASE_JAW)
    hold(0.15, RELEASE_JAW)
    follow(down_q, (z_carry - target[2]) / RISE_SPEED, RELEASE_JAW)
    hold(0.2, RELEASE_JAW)
    p = world.block_position(held)
    if np.linalg.norm(p[:2] - goal[:2]) > PLACE_TOL_XY or abs(p[2] - goal[2]) > PLACE_TOL_Z:
        return PlaceResult(False, ticks, "missed the goal")
    return PlaceResult(True, ticks)


def go_home(world: PhysWorld, on_tick: Tick | None = None) -> int:
    """Back to the parked pose the way pick() leaves it: turn at height, then fold down.
    Returns the control ticks run."""
    ticks = 0
    start = world.ctrl()
    turned = start.copy()
    turned[0] = HOME_QPOS[0]
    turned[4] = HOME_QPOS[4]
    raised = HOME_QPOS.copy()
    raised[2:4] = RAISED_ELBOW_WRIST
    raised[N_ARM] = start[N_ARM]
    for a, b in ((start, turned), (turned, raised), (raised, HOME_QPOS)):
        n = max(1, round(HOME_SECONDS / CONTROL_DT))
        for t in range(1, n + 1):
            world.step(a + (b - a) * smooth(t / n), SUBSTEPS)
            ticks += 1
            if on_tick is not None:
                on_tick(world, world.ctrl())
    return ticks
