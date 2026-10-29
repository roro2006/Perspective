"""MuJoCo stand-in for the bench: a table, coloured blocks, an occluding box,
the SO-101 arm, a fixed robot camera and a moving "human head" camera.

The arm moves kinematically (we set joint angles directly): grasping is the
solved part, and this sim exists to test perception and perspective."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import mujoco
import numpy as np

from curio_robot.geometry import intrinsics_from_fovy, look_at
from curio_robot.types import HUMAN_CAM, ROBOT_CAM, CameraFrame, CameraPose

SO101_XML = Path(__file__).resolve().parent.parent / "assets" / "so101" / "so101.xml"

BLOCK_HALF = 0.0125
N_BLOCKS = 6
COLOR_RGBA = {
    "red": (0.85, 0.15, 0.15, 1),
    "blue": (0.15, 0.35, 0.9, 1),
    "green": (0.15, 0.7, 0.3, 1),
    "yellow": (0.95, 0.8, 0.15, 1),
}
TABLE_CENTER = np.array([0.24, 0.0])
OFF_TABLE = np.array([0.0, 0.0, -5.0])
HUMAN_GROUP = 5  # geom group for the head model, hidden from the human's own camera
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")
HOME_QPOS = np.array([1.5, -1.4, 1.4, 1.0, 0.0, 0.6])  # tucked to one side, out of the robot camera's view
ROBOT_CAM_POS = np.array([-0.2, 0.0, 0.55])
ROBOT_CAM_LOOK = np.array([0.3, 0.0, 0.0])
HOVER = 0.03  # gripper height above the commanded table point
OVERVIEW_POS = np.array([0.28, -0.75, 0.6])
OVERVIEW_LOOK = np.array([0.26, 0.0, 0.02])


def _quat_from_rotation(r: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, r.flatten())
    return q


def build_model() -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(SO101_XML))
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.headlight.diffuse = [0.4, 0.4, 0.4]
    spec.visual.quality.shadowsize = 8192
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.93, 0.92, 0.9], rgb2=[0.55, 0.57, 0.6], width=512, height=3072)
    w = spec.worldbody

    w.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1],
               pos=[0, 0, -0.75], rgba=[0.32, 0.34, 0.38, 1])
    w.add_geom(name="wall", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 3, 1.5],
               pos=[1.6, 0, 0], rgba=[0.86, 0.85, 0.82, 1])
    w.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.5, 0.45, 0.01],
               pos=[0.25, 0, -0.01], rgba=[0.80, 0.68, 0.52, 1])
    for name, pos in (("key", [0.3, -0.3, 1.2]), ("fill", [0.6, 0.4, 1.0])):
        light = w.add_light(name=name, pos=pos, dir=[-0.1, 0.1, -1])
        light.diffuse = [0.55, 0.55, 0.55]
        light.castshadow = name == "key"

    for i in range(N_BLOCKS):
        body = w.add_body(name=f"block_{i}", pos=OFF_TABLE.tolist())
        body.add_geom(name=f"block_{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[BLOCK_HALF] * 3, rgba=[0.5, 0.5, 0.5, 1])
    occ = w.add_body(name="occluder", pos=OFF_TABLE.tolist())
    occ.add_geom(name="occluder", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.012, 0.055, 0.045],
                 rgba=[0.62, 0.48, 0.32, 1])

    head = w.add_body(name="human_head", pos=OFF_TABLE.tolist())
    head.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.09, 0, 0],
                  pos=[0, 0, 0.09], rgba=[0.87, 0.72, 0.6, 1], group=HUMAN_GROUP)
    head.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.035, 0.008, 0.02],
                  pos=[0, 0, 0], rgba=[0.1, 0.1, 0.12, 1], group=HUMAN_GROUP)

    w.add_camera(name=ROBOT_CAM, fovy=62)
    w.add_camera(name=HUMAN_CAM, fovy=70)
    w.add_camera(name="overview", fovy=40)
    return spec.compile()


@dataclass(slots=True)
class BlockSpec:
    color: str
    xy: tuple[float, float]


class SimWorld:
    def __init__(self, width: int = 320, height: int = 240) -> None:
        self.model = build_model()
        self.data = mujoco.MjData(self.model)
        self.width, self.height = width, height
        self.renderer = mujoco.Renderer(self.model, height, width)
        # capture() renders a snapshot of the state, so the live runtime's motion
        # thread only waits for a copy, not for four renders.
        self.lock = threading.Lock()
        self._render_data = mujoco.MjData(self.model)
        self._block_geoms = [self.model.geom(f"block_{i}").id for i in range(N_BLOCKS)]
        self.block_colors: dict[int, str] = {}
        self._joint_ids = [self.model.joint(n).id for n in ARM_JOINTS]
        self.data.qpos[: len(HOME_QPOS)] = HOME_QPOS
        self._set_camera(ROBOT_CAM, ROBOT_CAM_POS, ROBOT_CAM_LOOK)
        self._set_camera("overview", OVERVIEW_POS, OVERVIEW_LOOK)
        mujoco.mj_forward(self.model, self.data)

    # ---- scene setup -------------------------------------------------------

    def set_blocks(self, blocks: list[BlockSpec]) -> None:
        self.block_colors = {}
        for i in range(N_BLOCKS):
            body = self.model.body(f"block_{i}")
            geom = self.model.geom(f"block_{i}")
            if i < len(blocks):
                b = blocks[i]
                body.pos[:] = [b.xy[0], b.xy[1], BLOCK_HALF]
                geom.rgba[:] = COLOR_RGBA[b.color]
                self.block_colors[self._block_geoms[i]] = b.color
            else:
                body.pos[:] = OFF_TABLE
        mujoco.mj_forward(self.model, self.data)

    def block_positions(self) -> list[tuple[str, np.ndarray]]:
        return [
            (color, self.data.geom_xpos[gid].copy()) for gid, color in self.block_colors.items()
        ]

    def set_occluder(self, xy: tuple[float, float] | None, yaw: float = 0.0) -> None:
        body = self.model.body("occluder")
        if xy is None:
            body.pos[:] = OFF_TABLE
        else:
            body.pos[:] = [xy[0], xy[1], 0.045]
            body.quat[:] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        mujoco.mj_forward(self.model, self.data)

    def set_human(self, eye: np.ndarray, look: np.ndarray) -> None:
        self._set_camera(HUMAN_CAM, eye, look)
        head = self.model.body("human_head")
        back = eye - look
        back[2] = 0
        back /= np.linalg.norm(back)
        head.pos[:] = eye + 0.04 * back
        mujoco.mj_forward(self.model, self.data)

    def _set_camera(self, name: str, eye: np.ndarray, look: np.ndarray) -> None:
        cam = self.model.camera(name)
        cam.pos[:] = eye
        cam.quat[:] = _quat_from_rotation(look_at(eye, look))

    # ---- World interface ---------------------------------------------------

    def capture(self) -> dict[str, CameraFrame]:
        with self.lock:
            mujoco.mj_copyData(self._render_data, self.model, self.data)
        return {
            name: self.render_camera(name, segmentation=True, data=self._render_data)
            for name in (ROBOT_CAM, HUMAN_CAM)
        }

    def render_camera(
        self,
        name: str,
        *,
        segmentation: bool = False,
        renderer: mujoco.Renderer | None = None,
        data: mujoco.MjData | None = None,
    ) -> CameraFrame:
        r = renderer or self.renderer
        d = data if data is not None else self.data
        option = mujoco.MjvOption()
        option.geomgroup[HUMAN_GROUP] = name != HUMAN_CAM
        r.update_scene(d, name, scene_option=option)
        rgb = r.render().copy()
        seg = None
        if segmentation:
            r.enable_segmentation_rendering()
            r.update_scene(d, name, scene_option=option)
            raw = r.render()
            r.disable_segmentation_rendering()
            ids = np.where(raw[..., 1] == mujoco.mjtObj.mjOBJ_GEOM, raw[..., 0], -1)
            seg = ids
        cam_id = self.model.camera(name).id
        pose = CameraPose(
            position=d.cam_xpos[cam_id].copy(),
            rotation=d.cam_xmat[cam_id].reshape(3, 3).copy(),
        )
        k = intrinsics_from_fovy(self.model.cam_fovy[cam_id], r.width, r.height)
        return CameraFrame(name, rgb, k, pose, float(d.time), seg)

    def gripper_position(self, hover: float = HOVER) -> np.ndarray:
        with self.lock:
            return self.data.site_xpos[self.model.site("gripperframe").id] - [0, 0, hover]

    def move_gripper_to(self, position: np.ndarray, hover: float = HOVER) -> bool:
        """Damped least-squares IK on the gripper site (position only)."""
        with self.lock:
            return self._ik(position, hover)

    def _ik(self, position: np.ndarray, hover: float) -> bool:
        goal = np.asarray(position, float) + [0, 0, hover]
        site = self.model.site("gripperframe").id
        dofs = [self.model.jnt_dofadr[j] for j in self._joint_ids]
        qadr = [self.model.jnt_qposadr[j] for j in self._joint_ids]
        jac = np.zeros((3, self.model.nv))
        for _ in range(200):
            mujoco.mj_forward(self.model, self.data)
            err = goal - self.data.site_xpos[site]
            if np.linalg.norm(err) < 0.004:
                return True
            mujoco.mj_jacSite(self.model, self.data, jac, None, site)
            j = jac[:, dofs]
            dq = j.T @ np.linalg.solve(j @ j.T + 1e-3 * np.eye(3), err)
            for a, jid, d in zip(qadr, self._joint_ids, dq):
                lo, hi = self.model.jnt_range[jid]
                self.data.qpos[a] = np.clip(self.data.qpos[a] + 0.5 * d, lo, hi)
        mujoco.mj_forward(self.model, self.data)
        return bool(np.linalg.norm(goal - self.data.site_xpos[site]) < 0.015)

    def close(self) -> None:
        """Free the renderer's GL context now. Left to the garbage collector it can
        be freed in the middle of another thread's render."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def home(self) -> None:
        self.data.qpos[: len(HOME_QPOS)] = HOME_QPOS
        mujoco.mj_forward(self.model, self.data)


@contextmanager
def ensure_world(world: SimWorld | None, width: int = 320, height: int = 240) -> Iterator[SimWorld]:
    """The caller's world, or a new one that is closed when the block ends."""
    if world is not None:
        yield world
        return
    with SimWorld(width, height) as owned:
        yield owned
