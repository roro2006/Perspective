"""The demonstrator (task T5-23): PhysWorld plus a person's scripted hand and arm, and the
head camera recorded as RGB-D at 10 fps with the head pose.

The hand and forearm are mocap bodies with no collisions. A block in the hand follows it
(set on its free joint every physics step, as T5-20's held props are); let go just above
where it will rest, it falls the last few millimetres and physics settles it. The head sways
a little and turns towards what the hand is doing, so every frame has its own pose."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import numpy as np

from curio_robot.ego_plan import EgoFrame, PersonFrame
from curio_robot.geometry import intrinsics_from_fovy, look_at, perturb_pose
from curio_robot.sim.expert import CONTROL_DT
from curio_robot.sim.physics import N_BLOCKS, TIMESTEP, PhysWorld, build_spec
from curio_robot.sim.world import (
    BLOCK_HALF,
    HUMAN_GROUP,
    OFF_TABLE,
    OVERVIEW_LOOK,
    OVERVIEW_POS,
    ROBOT_CAM_LOOK,
    ROBOT_CAM_POS,
)
from curio_robot.types import HUMAN_CAM, CameraPose

FPS = 10
TICKS_PER_FRAME = round(1.0 / (FPS * CONTROL_DT))
SUBSTEPS = round(CONTROL_DT / TIMESTEP)
SKIN = (0.86, 0.6, 0.45, 1)
SLEEVE = (0.3, 0.33, 0.4, 1)
PALM_ABOVE = 0.021  # m from a held block's centre up to the palm centre
CARRY_Z = 0.11  # m, palm height while carrying
HAND_SPEED = 0.3  # m/s between points
DROP = 0.003  # m above its rest where the hand lets a block go
POSE_NOISE = (0.003, 0.5)  # m, deg: the glasses' tracking error per frame
FOREARM_HALF = 0.12


@dataclass(frozen=True, slots=True)
class Seat:
    eye: np.ndarray  # (3,)
    look: np.ndarray  # (3,) where the person looks when at rest

    @property
    def frame(self) -> PersonFrame:
        return PersonFrame.from_left(-look_at(self.eye, self.look)[:, 0])


def build_ego_spec() -> mujoco.MjSpec:
    spec = build_spec()
    w = spec.worldbody
    hand = w.add_body(name="ego_hand", mocap=True, pos=OFF_TABLE.tolist())
    kw = {"contype": 0, "conaffinity": 0, "rgba": list(SKIN)}
    hand.add_geom(type=mujoco.mjtGeom.mjGEOM_ELLIPSOID, size=[0.022, 0.03, 0.01], **kw)
    for side in (-1, 1):
        hand.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.0065, 0],
                      fromto=[side * 0.021, 0.0, 0.0, side * 0.021, 0.0, -0.028], **kw)
    arm = w.add_body(name="ego_forearm", mocap=True, pos=OFF_TABLE.tolist())
    arm.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.03, FOREARM_HALF],
                 contype=0, conaffinity=0, rgba=list(SLEEVE))
    torso = w.add_body(name="ego_torso", mocap=True, pos=OFF_TABLE.tolist())
    torso.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.1, 0.18, 0.25], pos=[0, 0, -0.27],
                   contype=0, conaffinity=0, rgba=list(SLEEVE), group=HUMAN_GROUP)
    return spec


class EgoWorld(PhysWorld):
    """PhysWorld with the person's hand. Same API, plus hand control and RGB-D frames."""

    def __init__(self, width: int = 640, height: int = 480) -> None:
        self.model = build_ego_spec().compile()
        self.data = mujoco.MjData(self.model)
        self.width, self.height = width, height
        self._renderer: mujoco.Renderer | None = None
        self.block_colors: list[str] = []
        self._qadr = [self.model.jnt_qposadr[self.model.joint(f"block_{i}").id]
                      for i in range(N_BLOCKS)]
        self._vadr = [self.model.jnt_dofadr[self.model.joint(f"block_{i}").id]
                      for i in range(N_BLOCKS)]
        self._block_body = [self.model.body(f"block_{i}").id for i in range(N_BLOCKS)]
        self._block_geom = [self.model.geom(f"block_{i}").id for i in range(N_BLOCKS)]
        self._set_camera("robot", ROBOT_CAM_POS, ROBOT_CAM_LOOK)
        self._set_camera("overview", OVERVIEW_POS, OVERVIEW_LOOK)
        self._mocap = {n: self.model.body_mocapid[self.model.body(n).id]
                       for n in ("ego_hand", "ego_forearm", "ego_torso")}
        self.held: int | None = None
        self.hand = OFF_TABLE.copy()
        self._home()

    def body_names(self) -> tuple[str, ...]:
        return (*super().body_names(), "ego_hand", "ego_forearm", "ego_torso")

    # ---- the person ------------------------------------------------------------------------

    def set_person(self, seat: Seat, eye: np.ndarray, look: np.ndarray) -> None:
        self._set_human(eye, look)
        back = seat.eye - seat.look
        back[2] = 0
        back /= np.linalg.norm(back)
        yaw = float(np.arctan2(back[1], back[0]))
        self.data.mocap_pos[self._mocap["ego_torso"]] = seat.eye + 0.1 * back
        self.data.mocap_quat[self._mocap["ego_torso"]] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]

    def set_hand(self, palm: np.ndarray, seat: Seat) -> None:
        """Palm centre at palm; the forearm runs from the wrist back to the person's right and a
        little up (low, so it doesn't hide the hand from their own eyes)."""
        self.hand = np.asarray(palm, float).copy()
        fr = seat.frame
        axis = -0.55 * fr.forward - 0.75 * fr.left + np.array([0, 0, 0.3])
        axis /= np.linalg.norm(axis)
        wrist = self.hand + 0.035 * axis
        quat = np.zeros(4)
        mujoco.mju_quatZ2Vec(quat, axis)
        self.data.mocap_pos[self._mocap["ego_forearm"]] = wrist + FOREARM_HALF * axis
        self.data.mocap_quat[self._mocap["ego_forearm"]] = quat
        # Fingers either side of the block across the face nearest the person's sideways axis.
        yaw = np.round(np.arctan2(fr.left[1], fr.left[0]) / (np.pi / 2)) * (np.pi / 2)
        self.data.mocap_pos[self._mocap["ego_hand"]] = self.hand
        self.data.mocap_quat[self._mocap["ego_hand"]] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]

    def step(self, ctrl: np.ndarray, n: int = 1) -> None:
        lo, hi = self.model.actuator_ctrlrange.T
        self.data.ctrl[:] = np.clip(np.asarray(ctrl, float), lo, hi)
        for _ in range(n):
            if self.held is not None:
                a, v = self._qadr[self.held], self._vadr[self.held]
                self.data.qpos[a:a + 3] = self.hand - [0, 0, PALM_ABOVE]
                self.data.qvel[v:v + 6] = 0
            mujoco.mj_step(self.model, self.data)

    # ---- frames ----------------------------------------------------------------------------

    def ego_frame(self, rng: np.random.Generator | None = None) -> EgoFrame:
        """The head camera as RGB-D; with rng, the pose carries the glasses' tracking noise."""
        rgb = self.render_camera(HUMAN_CAM)
        depth = self.render_depth(HUMAN_CAM)
        pose = rgb.pose if rng is None else perturb_pose(rgb.pose, rng, *POSE_NOISE)
        return EgoFrame(rgb.rgb, depth, rgb.intrinsics, pose, self.time)

    def render_depth(self, name: str) -> np.ndarray:
        r = self._renderer
        assert r is not None, "render_camera first"
        option = mujoco.MjvOption()
        option.geomgroup[HUMAN_GROUP] = name != HUMAN_CAM
        r.enable_depth_rendering()
        r.update_scene(self.data, name, scene_option=option)
        depth = r.render().copy()
        r.disable_depth_rendering()
        return depth

    def rgbd(self, name: str) -> EgoFrame:
        """Any camera as RGB-D, exact pose (the robot's own camera is calibrated)."""
        rgb = self.render_camera(name)
        return EgoFrame(rgb.rgb, self.render_depth(name), rgb.intrinsics, rgb.pose, self.time)

    def intrinsics(self, name: str):
        return intrinsics_from_fovy(self.model.cam_fovy[self.model.camera(name).id],
                                    self.width, self.height)


# ---- the scripted demonstration --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HandMove:
    block: int
    goal: np.ndarray  # where the block's centre will rest (world)


Frame = Callable[[EgoWorld, np.ndarray, np.ndarray], None]  # (world, eye, look) per tick


def _smooth(s: float) -> float:
    return s * s * s * (10 - 15 * s + 6 * s * s)  # minimum jerk


class Demonstrator:
    """Runs a scripted demonstration, calling on_frame at FPS with the head pose."""

    def __init__(self, world: EgoWorld, seat: Seat, rng: np.random.Generator,
                 on_frame: Callable[[EgoWorld], None]) -> None:
        self.w, self.seat, self.rng, self.on_frame = world, seat, rng, on_frame
        self._ticks = 0
        self._phase = rng.uniform(0, 2 * np.pi, 3)
        self._period = rng.uniform(3.0, 5.0, 3)
        self._look = seat.look.copy()
        fr = seat.frame
        self.rest = fr.point(-0.14, -0.2, 0.05)  # right hand resting by the table edge

    def _head(self, focus: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        t = self._ticks * CONTROL_DT
        sway = 0.012 * np.sin(2 * np.pi * t / self._period + self._phase)
        aim = self.seat.look if focus is None else 0.6 * self.seat.look + 0.4 * np.array(
            [focus[0], focus[1], 0.0])
        self._look += 0.15 * (aim - self._look)  # the eyes lead, the head follows
        return self.seat.eye + sway, self._look.copy()

    def tick(self, focus: np.ndarray | None = None) -> None:
        eye, look = self._head(focus)
        self.w.set_person(self.seat, eye, look)
        self.w.set_hand(self.w.hand, self.seat)
        self.w.step(self.w.ctrl(), SUBSTEPS)
        if self._ticks % TICKS_PER_FRAME == 0:
            self.on_frame(self.w)
        self._ticks += 1

    def go(self, to: np.ndarray, focus: np.ndarray | None = None, speed: float = HAND_SPEED,
           min_s: float = 0.4) -> None:
        start = self.w.hand.copy()
        n = max(1, round(max(min_s, np.linalg.norm(to - start) / speed) / CONTROL_DT))
        for k in range(1, n + 1):
            self.w.hand = start + (to - start) * _smooth(k / n)
            self.tick(focus)

    def wait(self, seconds: float, focus: np.ndarray | None = None) -> None:
        for _ in range(round(seconds / CONTROL_DT)):
            self.tick(focus)

    def run(self, moves: list[HandMove]) -> None:
        w = self.w
        w.hand = self.rest.copy()
        w.set_hand(w.hand, self.seat)
        self.wait(0.6)
        for m in moves:
            b = w.block_position(m.block)
            over = np.array([b[0], b[1], CARRY_Z])
            self.go(over, b)
            self.go(b + [0, 0, PALM_ABOVE], b, speed=0.12)
            self.wait(self.rng.uniform(0.2, 0.35), b)
            w.held = m.block
            self.wait(0.15, b)
            self.go(over, m.goal, speed=0.15)
            goal_over = np.array([m.goal[0], m.goal[1], CARRY_Z])
            self.go(goal_over, m.goal)
            self.go(m.goal + [0, 0, PALM_ABOVE + DROP], m.goal, speed=0.12)
            self.wait(self.rng.uniform(0.15, 0.3), m.goal)
            w.held = None
            self.wait(0.15, m.goal)
            self.go(goal_over, m.goal, speed=0.15)
        self.go(self.rest)
        self.wait(1.0)


def true_poses(world: EgoWorld) -> CameraPose:
    """The head camera's exact pose right now."""
    cid = world.model.camera(HUMAN_CAM).id
    return CameraPose(world.data.cam_xpos[cid].copy(),
                      world.data.cam_xmat[cid].reshape(3, 3).copy())


__all__ = ["BLOCK_HALF", "Demonstrator", "EgoWorld", "HandMove", "Seat", "true_poses"]
