"""Pick the chosen block up and hand it to the person (T5-33). Scripted, kinematic motion.

The demo world has no contact physics: blocks are fixed bodies and the arm is moved by
position-only IK (`SimWorld.move_gripper_to`). So the "grasp" is staged: the arm opens over the
block, descends, closes its jaws, and from then on the block is carried with the gripper until
it is set down in front of the person and released. This shows what happens after Curio has
decided; it is not a learned policy and not a physics grasp, and the console says so. (The
physics sim with the scripted pick expert is `sim/physics.py` + `sim/expert.py`.)

The hand-over point is on the table between the person's hands, pulled in to the arm's
comfortable reach (REACH) when they sit further away, and kept within the arm's working arc
(MAX_BEARING): someone standing beside or behind the robot gets it on the robot's side
nearest them."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from curio_robot.sim.world import BLOCK_HALF, HOME_QPOS, SimWorld

OPEN, CLOSED, HOME_GRIP = 1.0, 0.2, 0.6  # gripper joint (rad): 0 is shut, 1.2 wide open
GRIP_RATE = 3.0  # rad/s
JOINT_RATE = 1.5  # rad/s per joint on the way home
SPEED = 0.25  # m/s at the gripper, as runtime.motion
ARRIVE = 0.006  # metres
LIFT = 0.10  # carry height above the table
GRASP_Z = BLOCK_HALF + 0.004  # gripper site height when the jaws close on a block
REACH = (0.22, 0.42)  # hand-over distance from the robot base, metres
MAX_BEARING = np.radians(95)  # the arm works in front of its base; beyond this, its side
HANDS_AHEAD = 0.2  # the person's hands are this far in front of their eyes, on the table
STALL_S = 6.0  # give up on a waypoint after this long


@dataclass(slots=True)
class Waypoint:
    pos: np.ndarray | None  # gripper site target, world frame; None: joints back to home
    grip: float  # gripper joint target
    label: str  # shown in the console
    action: str | None = None  # "attach" or "release", done on arrival
    dwell: float = 0.0  # seconds to wait after arriving (and after the gripper settles)


def block_body(world: SimWorld, xy: np.ndarray) -> str:
    """The body of the block nearest `xy` (the chosen target's position)."""
    best = min(world.block_colors, key=lambda g: np.linalg.norm(world.data.geom_xpos[g][:2] - xy))
    return world.model.geom(best).name  # block geoms and bodies share the name block_i


def offer_point(eye: np.ndarray, look: np.ndarray) -> np.ndarray:
    """Where to set the block down: between the person's hands, within the arm's reach."""
    facing = np.asarray(look, float)[:2] - np.asarray(eye, float)[:2]
    facing /= np.linalg.norm(facing)
    hands = np.asarray(eye, float)[:2] + HANDS_AHEAD * facing
    r = float(np.linalg.norm(hands))
    bearing = float(np.arctan2(hands[1], hands[0]))
    if abs(bearing) > MAX_BEARING:  # person beside or behind the robot: its side nearest them
        side = np.sign(hands[1]) or np.sign(eye[1]) or 1.0
        bearing = side * MAX_BEARING
    return np.array([np.cos(bearing), np.sin(bearing)]) * float(np.clip(r, *REACH))


def plan(block_xyz: np.ndarray, offer_xy: np.ndarray) -> list[Waypoint]:
    b = np.asarray(block_xyz, float)
    o = np.array([offer_xy[0], offer_xy[1], 0.0])
    up = np.array([0.0, 0.0, LIFT])
    at = np.array([0.0, 0.0, GRASP_Z])
    return [
        Waypoint(b * [1, 1, 0] + up, OPEN, "reaching"),
        Waypoint(b * [1, 1, 0] + at, OPEN, "grasping"),
        Waypoint(b * [1, 1, 0] + at, CLOSED, "grasping", "attach", 0.2),
        Waypoint(b * [1, 1, 0] + up, CLOSED, "lifting"),
        Waypoint(o + up, CLOSED, "handing over"),
        Waypoint(o + at, CLOSED, "handing over"),
        Waypoint(o + at, OPEN, "releasing", "release", 0.2),
        Waypoint(o + up, OPEN, "returning"),
        Waypoint(None, HOME_GRIP, "returning"),  # in joint space: IK can stall from the side
    ]


class HandOver:
    """Drives the world through the plan, one `step(dt)` per frame. The caller holds whatever
    lock guards the world (the viewer's) while it steps."""

    def __init__(self, world: SimWorld, chosen_xyz: np.ndarray, eye: np.ndarray,
                 look: np.ndarray) -> None:
        self.world = world
        self.body = block_body(world, np.asarray(chosen_xyz, float)[:2])
        self._site = world.model.site("gripperframe").id
        block = world.model.body(self.body).pos.copy()
        self.offer = offer_point(eye, look)
        self.path = plan(block, self.offer)
        self.carrying = False
        self.released = False
        self.result: str | None = None  # "handed over" or "unreachable", when done
        self._i = 0
        self._t = 0.0  # time on the current waypoint
        self._waited = 0.0

    @property
    def label(self) -> str:
        return self.path[min(self._i, len(self.path) - 1)].label

    @property
    def done(self) -> bool:
        return self.result is not None

    def step(self, dt: float) -> str | None:
        """Advance; returns "attach" or "release" on the frame that happens, else None."""
        if self.done:
            return None
        wp = self.path[self._i]
        w = self.world
        self._t += dt
        if wp.pos is None:
            q = w.data.qpos[:5]
            q += np.clip(HOME_QPOS[:5] - q, -JOINT_RATE * dt, JOINT_RATE * dt)
            dist = float(np.abs(HOME_QPOS[:5] - q).max()) * ARRIVE / 0.01  # 0.01 rad counts
        else:
            here = w.data.site_xpos[self._site].copy()
            delta = wp.pos - here
            dist = float(np.linalg.norm(delta))
            stride = SPEED * dt
            if dist >= ARRIVE:
                w.move_gripper_to(wp.pos if dist <= stride else here + delta / dist * stride,
                                  hover=0.0)
        g = w.data.qpos[5]
        w.data.qpos[5] = g + float(np.clip(wp.grip - g, -GRIP_RATE * dt, GRIP_RATE * dt))
        if self.carrying:
            w.model.body(self.body).pos[:] = w.data.site_xpos[self._site] - [0, 0, 0.004]
        mujoco.mj_forward(w.model, w.data)
        arrived = dist < ARRIVE and abs(w.data.qpos[5] - wp.grip) < 1e-3
        if not arrived and self._t < STALL_S:
            return None
        if not arrived:  # stalled: say so rather than pretend
            self.result = "unreachable"
            return None
        self._waited += dt
        if self._waited < wp.dwell:
            return None
        event = wp.action
        if event == "attach":
            self.carrying = True
        elif event == "release":
            self.carrying = False
            self.released = True
            pos = w.model.body(self.body).pos
            pos[2] = BLOCK_HALF
            mujoco.mj_forward(w.model, w.data)
        self._i += 1
        self._t = self._waited = 0.0
        if self._i == len(self.path):
            w.home()
            self.result = "handed over"
        return event
