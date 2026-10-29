"""A MuJoCo world that steps physics, so the SO-101 can really grasp (task T5-04).

SimWorld sets joint angles and can only pretend to pick. Here the arm is driven
through its position actuators (ctrl), the blocks sit on free joints with mass
and friction, and mj_step decides whether a grasp holds. The table, the robot
and person cameras and the head model match SimWorld so images look the same;
SimWorld itself stays as it is for the live runtime.

Contact settings were tuned for 2.5 cm blocks in the SO-101 jaws: a 2 ms step,
elliptic cones with a high impratio (from so101.xml) and a few noslip
iterations against slip, condim 4 on the blocks (torsional friction stops them
spinning out of the jaws), and stiffer contacts on jaws and blocks than the
so101.xml default so a squeezed block doesn't sink into the fingers."""

from __future__ import annotations

from typing import Self

import mujoco
import numpy as np

from curio_robot.collab.tasks import TaskScene
from curio_robot.geometry import intrinsics_from_fovy, look_at
from curio_robot.sim.world import (
    BLOCK_HALF,
    COLOR_RGBA,
    HOME_QPOS,
    HUMAN_GROUP,
    OFF_TABLE,
    OVERVIEW_LOOK,
    OVERVIEW_POS,
    ROBOT_CAM_LOOK,
    ROBOT_CAM_POS,
    SO101_XML,
)
from curio_robot.types import HUMAN_CAM, ROBOT_CAM, CameraFrame, CameraPose

N_BLOCKS = 6
TIMESTEP = 0.002  # s; 5 ms (so101.xml) lets the small blocks jitter in the jaws
BLOCK_MASS = 0.015  # kg, a 2.5 cm plastic cube
BLOCK_FRICTION = (1.2, 0.01, 0.0002)  # sliding, torsional, rolling
TABLE_FRICTION = (0.6, 0.005, 0.0001)
JAW_FRICTION = (1.5, 0.01, 0.0002)
CONTACT_SOLREF = (0.004, 1.0)  # stiffer than the 0.01 in so101.xml: blocks don't sink in
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
PARK_Z = -0.75 + BLOCK_HALF  # unused blocks rest on the floor, out of every camera


def _park_xy(i: int) -> tuple[float, float]:
    return (-1.0 - 0.1 * i, 2.0)


def _quat_from_rotation(r: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, r.flatten())
    return q


def build_spec() -> mujoco.MjSpec:
    """SimWorld's scene (sim.world.build_model) with free-jointed blocks and
    physics settings for grasping."""
    spec = mujoco.MjSpec.from_file(str(SO101_XML))
    spec.option.timestep = TIMESTEP
    spec.option.noslip_iterations = 3
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.headlight.diffuse = [0.4, 0.4, 0.4]
    spec.visual.quality.shadowsize = 4096
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.93, 0.92, 0.9], rgb2=[0.55, 0.57, 0.6], width=512, height=3072)
    for g in spec.geoms:
        if g.name.startswith(("fixed_jaw", "moving_jaw")) or g.classname.name.startswith(
            "collision_gripper"
        ):
            g.friction = list(JAW_FRICTION)
            g.solref = list(CONTACT_SOLREF)
    w = spec.worldbody
    w.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1],
               pos=[0, 0, -0.75], rgba=[0.32, 0.34, 0.38, 1])
    w.add_geom(name="wall", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 3, 1.5],
               pos=[1.6, 0, 0], rgba=[0.86, 0.85, 0.82, 1])
    w.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.5, 0.45, 0.01],
               pos=[0.25, 0, -0.01], rgba=[0.80, 0.68, 0.52, 1],
               friction=list(TABLE_FRICTION))
    for name, pos in (("key", [0.3, -0.3, 1.2]), ("fill", [0.6, 0.4, 1.0])):
        light = w.add_light(name=name, pos=pos, dir=[-0.1, 0.1, -1])
        light.diffuse = [0.55, 0.55, 0.55]
        light.castshadow = name == "key"

    for i in range(N_BLOCKS):
        body = w.add_body(name=f"block_{i}", pos=[*_park_xy(i), PARK_Z])
        body.add_freejoint(name=f"block_{i}")
        body.add_geom(name=f"block_{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[BLOCK_HALF] * 3, rgba=[0.5, 0.5, 0.5, 1], mass=BLOCK_MASS,
                      condim=4, friction=list(BLOCK_FRICTION), solref=list(CONTACT_SOLREF))
    occ = w.add_body(name="occluder", pos=OFF_TABLE.tolist())
    occ.add_geom(name="occluder", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.012, 0.055, 0.045],
                 rgba=[0.62, 0.48, 0.32, 1])

    head = w.add_body(name="human_head", pos=OFF_TABLE.tolist())
    head.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.09, 0, 0], pos=[0, 0, 0.09],
                  rgba=[0.87, 0.72, 0.6, 1], group=HUMAN_GROUP, contype=0, conaffinity=0)
    head.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.035, 0.008, 0.02], pos=[0, 0, 0],
                  rgba=[0.1, 0.1, 0.12, 1], group=HUMAN_GROUP, contype=0, conaffinity=0)

    w.add_camera(name=ROBOT_CAM, fovy=62)
    w.add_camera(name=HUMAN_CAM, fovy=70)
    w.add_camera(name="overview", fovy=40)
    return spec


class PhysWorld:
    """The physics bench. ctrl is the six position-actuator targets, in
    ARM_JOINTS order (radians; gripper: -0.17 closed .. 1.75 open)."""

    def __init__(self, width: int = 320, height: int = 240) -> None:
        self.model = build_spec().compile()
        self.data = mujoco.MjData(self.model)
        self.width, self.height = width, height
        self._renderer: mujoco.Renderer | None = None
        self.block_colors: list[str] = []
        self._qadr = [self.model.jnt_qposadr[self.model.joint(f"block_{i}").id]
                      for i in range(N_BLOCKS)]
        self._block_body = [self.model.body(f"block_{i}").id for i in range(N_BLOCKS)]
        self._block_geom = [self.model.geom(f"block_{i}").id for i in range(N_BLOCKS)]
        self._set_camera(ROBOT_CAM, ROBOT_CAM_POS, ROBOT_CAM_LOOK)
        self._set_camera("overview", OVERVIEW_POS, OVERVIEW_LOOK)
        self._home()

    # ---- scene -------------------------------------------------------------

    def reset(self, scene: TaskScene, settle: float = 0.2) -> None:
        """Arm home, blocks on the table as the scene says, then settle (s)."""
        mujoco.mj_resetData(self.model, self.data)
        self._home()
        self.block_colors = [b.color for b in scene.blocks]
        if len(scene.blocks) > N_BLOCKS:
            raise ValueError(f"at most {N_BLOCKS} blocks")
        for i in range(N_BLOCKS):
            a = self._qadr[i]
            if i < len(scene.blocks):
                b = scene.blocks[i]
                self.data.qpos[a:a + 3] = [b.xy[0], b.xy[1], BLOCK_HALF]
                self.model.geom_rgba[self._block_geom[i]] = COLOR_RGBA[b.color]
            else:
                self.data.qpos[a:a + 3] = [*_park_xy(i), PARK_Z]
            self.data.qpos[a + 3:a + 7] = [1, 0, 0, 0]
        occ = self.model.body("occluder").id
        if scene.occluder is None:
            self.model.body_pos[occ] = OFF_TABLE
        else:
            self.model.body_pos[occ] = [*scene.occluder, 0.045]
            yaw = scene.occluder_yaw
            self.model.body_quat[occ] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self._set_human(scene.human_eye, scene.human_look)
        mujoco.mj_forward(self.model, self.data)
        self.step(self.data.ctrl.copy(), round(settle / TIMESTEP))

    def _home(self) -> None:
        self.data.qpos[: len(HOME_QPOS)] = HOME_QPOS
        self.data.ctrl[:] = HOME_QPOS
        mujoco.mj_forward(self.model, self.data)

    def _set_human(self, eye: np.ndarray, look: np.ndarray) -> None:
        self._set_camera(HUMAN_CAM, eye, look)
        back = np.asarray(eye, float) - look
        back[2] = 0
        back /= np.linalg.norm(back)
        self.model.body_pos[self.model.body("human_head").id] = eye + 0.04 * back

    def _set_camera(self, name: str, eye: np.ndarray, look: np.ndarray) -> None:
        cam = self.model.camera(name)
        cam.pos[:] = eye
        cam.quat[:] = _quat_from_rotation(look_at(np.asarray(eye, float), look))

    # ---- stepping and state ------------------------------------------------

    @property
    def time(self) -> float:
        return float(self.data.time)

    def step(self, ctrl: np.ndarray, n: int = 1) -> None:
        """Hold the actuator targets ctrl for n physics steps."""
        lo, hi = self.model.actuator_ctrlrange.T
        self.data.ctrl[:] = np.clip(np.asarray(ctrl, float), lo, hi)
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    def arm_qpos(self) -> np.ndarray:
        """The six arm joint angles, ARM_JOINTS order."""
        return self.data.qpos[: len(ARM_JOINTS)].copy()

    def ctrl(self) -> np.ndarray:
        return self.data.ctrl.copy()

    def block_position(self, i: int) -> np.ndarray:
        """Centre of block i (world, m)."""
        return self.data.xpos[self._block_body[i]].copy()

    def block_yaw(self, i: int) -> float:
        r = self.data.xmat[self._block_body[i]].reshape(3, 3)
        return float(np.arctan2(r[1, 0], r[0, 0]))

    def lifted(self, i: int, height: float = 0.03) -> bool:
        """True when block i's bottom is at least height above the table."""
        return bool(self.block_position(i)[2] - BLOCK_HALF >= height)

    # ---- rendering ---------------------------------------------------------

    def render_camera(self, name: str, *, segmentation: bool = False) -> CameraFrame:
        """A frame from a named camera, as SimWorld.render_camera makes it.
        Segmentation ids are geom ids (block i is geom block_ids()[i])."""
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, self.height, self.width)
        r = self._renderer
        option = mujoco.MjvOption()
        option.geomgroup[HUMAN_GROUP] = name != HUMAN_CAM
        r.update_scene(self.data, name, scene_option=option)
        rgb = r.render().copy()
        seg = None
        if segmentation:
            r.enable_segmentation_rendering()
            r.update_scene(self.data, name, scene_option=option)
            raw = r.render()
            r.disable_segmentation_rendering()
            seg = np.where(raw[..., 1] == mujoco.mjtObj.mjOBJ_GEOM, raw[..., 0], -1)
        cam_id = self.model.camera(name).id
        pose = CameraPose(
            position=self.data.cam_xpos[cam_id].copy(),
            rotation=self.data.cam_xmat[cam_id].reshape(3, 3).copy(),
        )
        k = intrinsics_from_fovy(self.model.cam_fovy[cam_id], r.width, r.height)
        return CameraFrame(name, rgb, k, pose, self.time, seg)

    def block_ids(self) -> list[int]:
        return list(self._block_geom)

    def body_names(self) -> tuple[str, ...]:
        """Bodies added around the arm (not part of the SO-101)."""
        return (*(f"block_{i}" for i in range(N_BLOCKS)), "occluder", "human_head")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
