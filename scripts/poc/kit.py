"""Shared pieces for the T5-18 proof-of-concept video: the poseable capsule humanoid, the
MJCF scene shell (lights, sky, materials), min-jerk paths and the head-frame maths.

Everything is built from MJCF primitives written here; nothing is downloaded."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from curio_robot.geometry import look_at
from curio_robot.types import CameraPose

HEAD_FOVY = 80.0
CINE_FOVY = 42.0


def rgba(c: tuple[float, ...], a: float = 1.0) -> str:
    return " ".join(f"{v:.3f}" for v in (*c[:3], a))


def v3(p) -> str:
    return " ".join(f"{float(v):.4f}" for v in p)


# ---- humanoid ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Palette:
    suit: tuple[float, float, float]
    skin: tuple[float, float, float]
    visor: tuple[float, float, float]
    accent: tuple[float, float, float]
    boots: tuple[float, float, float] = (0.12, 0.12, 0.14)


PERSON = Palette((0.20, 0.36, 0.62), (0.86, 0.68, 0.55), (0.08, 0.09, 0.11), (0.95, 0.75, 0.2))
ROBOT_HUMANOID = Palette((0.88, 0.89, 0.91), (0.70, 0.72, 0.76), (0.10, 0.55, 0.85),
                         (0.10, 0.55, 0.85), (0.25, 0.27, 0.30))

JOINTS = ("neck_yaw", "neck_pitch",
          "l_sh_pitch", "l_sh_abd", "l_elbow", "r_sh_pitch", "r_sh_abd", "r_elbow",
          "l_hip_pitch", "l_hip_abd", "l_knee", "r_hip_pitch", "r_hip_abd", "r_knee")


def humanoid(p: str, pal: Palette, *, head_cam: bool = True, extra_torso: str = "") -> str:
    """One capsule humanoid named with prefix p, facing +x, root (pelvis) on a free joint.
    The dark visor marks where the head faces; head_cam looks out through it."""
    s, k, vz, a, b = (rgba(pal.suit), rgba(pal.skin), rgba(pal.visor), rgba(pal.accent),
                      rgba(pal.boots))
    cam = (f'<camera name="{p}head_cam" pos="0.115 0 0.18" xyaxes="0 -1 0 0 0 1" '
           f'fovy="{HEAD_FOVY}"/>') if head_cam else ""

    def arm(side: str, y: float) -> str:
        return f"""
      <body name="{p}{side}_upper" pos="0 {y} 0.42">
        <joint name="{p}{side}_sh_pitch" axis="0 1 0" range="-3.2 3.2"/>
        <joint name="{p}{side}_sh_abd" axis="1 0 0" range="-3.2 3.2"/>
        <geom type="sphere" size="0.065" rgba="{s}"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.27" size="0.047" rgba="{s}"/>
        <body name="{p}{side}_fore" pos="0 0 -0.27">
          <joint name="{p}{side}_elbow" axis="0 1 0" range="-2.8 0.2"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.24" size="0.04" rgba="{s}"/>
          <geom type="sphere" pos="0 0 -0.285" size="0.046" rgba="{k}"/>
        </body>
      </body>"""

    def leg(side: str, y: float) -> str:
        return f"""
    <body name="{p}{side}_thigh" pos="0 {y} -0.04">
      <joint name="{p}{side}_hip_pitch" axis="0 1 0" range="-2.6 1.0"/>
      <joint name="{p}{side}_hip_abd" axis="1 0 0" range="-1 1"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.42" size="0.072" rgba="{s}"/>
      <body name="{p}{side}_shin" pos="0 0 -0.42">
        <joint name="{p}{side}_knee" axis="0 1 0" range="-0.1 2.6"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.40" size="0.056" rgba="{s}"/>
        <geom type="box" pos="0.05 0 -0.445" size="0.11 0.055 0.035" rgba="{b}"/>
      </body>
    </body>"""

    return f"""
  <body name="{p}root" pos="0 0 0.94">
    <freejoint name="{p}free"/>
    <geom type="capsule" fromto="0 -0.1 0 0 0.1 0" size="0.095" rgba="{s}"/>
    {leg("l", 0.1)}{leg("r", -0.1)}
    <body name="{p}torso" pos="0 0 0.06">
      <geom type="capsule" fromto="0 0 0.02 0 0 0.36" size="0.135" rgba="{s}"/>
      <geom type="capsule" fromto="0 -0.17 0.42 0 0.17 0.42" size="0.07" rgba="{s}"/>
      <geom type="box" pos="0.13 0 0.30" size="0.012 0.07 0.05" rgba="{a}"/>
      {extra_torso}
      {arm("l", 0.23)}{arm("r", -0.23)}
      <body name="{p}head" pos="0 0 0.48">
        <joint name="{p}neck_yaw" axis="0 0 1" range="-1.6 1.6"/>
        <joint name="{p}neck_pitch" axis="0 1 0" range="-1 1"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.08" size="0.05" rgba="{k}"/>
        <geom type="sphere" pos="0 0 0.17" size="0.112" rgba="{k}"/>
        <geom type="box" pos="0.085 0 0.185" size="0.04 0.085 0.034" rgba="{vz}"
              material="gloss"/>
        <geom type="cylinder" pos="0.0 0.0 0.26" size="0.08 0.025" rgba="{s}"/>
        <geom type="sphere" pos="0.105 0.06 0.235" size="0.014" rgba="0.2 0.9 0.5 1"/>
        {cam}
      </body>
    </body>
  </body>"""


def euler_quat(roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0) -> np.ndarray:
    """MuJoCo quaternion (w, x, y, z) for intrinsic z-y-x (yaw, pitch, roll), radians."""
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, np.array([roll, pitch, yaw]), "XYZ")
    return q


def mat_quat(rot: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(rot, float).ravel())
    return q


def set_free(model: mujoco.MjModel, data: mujoco.MjData, joint: str, pos, quat) -> None:
    adr = model.joint(joint).qposadr[0]
    data.qpos[adr:adr + 3] = pos
    data.qpos[adr + 3:adr + 7] = quat


def set_joints(model: mujoco.MjModel, data: mujoco.MjData, prefix: str, angles: dict) -> None:
    for name, value in angles.items():
        data.qpos[model.joint(prefix + name).qposadr[0]] = value


def pose_humanoid(model, data, p: str, pos, quat, angles: dict | None = None) -> None:
    set_free(model, data, p + "free", pos, quat)
    base = dict.fromkeys(JOINTS, 0.0)
    base.update({"l_elbow": -0.15, "r_elbow": -0.15, "l_sh_abd": 0.08, "r_sh_abd": -0.08})
    base.update(angles or {})
    set_joints(model, data, p, base)


SITTING = {"l_hip_pitch": -1.5, "r_hip_pitch": -1.5, "l_knee": 1.5, "r_knee": 1.5,
           "l_sh_pitch": -0.3, "r_sh_pitch": -0.3, "l_elbow": -1.0, "r_elbow": -1.0}


def walk_angles(phase: float, amp: float = 0.45) -> dict:
    s = np.sin(phase)
    return {"l_hip_pitch": -amp * s, "r_hip_pitch": amp * s,
            "l_knee": 0.5 * amp * (1 + np.sin(phase + np.pi / 2)) * 0.9,
            "r_knee": 0.5 * amp * (1 - np.sin(phase + np.pi / 2)) * 0.9,
            "l_sh_pitch": 0.6 * amp * s, "r_sh_pitch": -0.6 * amp * s}


# ---- scene shell ---------------------------------------------------------------------------


def shell(body: str, *, sky=((0.62, 0.70, 0.80), (0.16, 0.19, 0.24)), floor=None,
          ambient=(0.32, 0.32, 0.32), headlight=0.35, extra_visual: str = "", assets: str = "",
          lights: str | None = None) -> str:
    """Full MJCF: offscreen 1920x1080, 4096 shadow maps, a gradient sky, a few shared
    materials, a 'cine' camera on a mocap rig, and the caller's world body."""
    floor_geom = ""
    if floor is not None:
        floor_geom = f'<geom name="floor" type="plane" size="30 30 0.1" material="{floor}"/>'
    if lights is None:
        lights = ('<light name="key" pos="3 -4 9" dir="-0.3 0.45 -1" castshadow="true" '
                  'diffuse="0.75 0.73 0.7" specular="0.2 0.2 0.2"/>'
                  '<light name="fill" pos="-4 4 6" dir="0.4 -0.4 -1" castshadow="false" '
                  'diffuse="0.3 0.32 0.36"/>')
    return f"""<mujoco model="poc">
  <compiler angle="radian"/>
  <option gravity="0 0 0"/>
  <visual>
    <global offwidth="1920" offheight="1080" fovy="{CINE_FOVY}"/>
    <quality shadowsize="4096" offsamples="8"/>
    <headlight ambient="{v3(ambient)}" diffuse="{headlight} {headlight} {headlight}"
               specular="0.1 0.1 0.1"/>
    <map znear="0.01" zfar="200" shadowclip="12"/>
    {extra_visual}
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="{v3(sky[0])}" rgb2="{v3(sky[1])}"
             width="512" height="3072"/>
    <texture name="concrete" type="2d" builtin="flat" mark="random" random="0.03"
             rgb1="0.55 0.56 0.57" rgb2="0.5 0.5 0.5" markrgb="0.45 0.46 0.47" width="512"
             height="512"/>
    <material name="concrete" texture="concrete" texrepeat="6 6" reflectance="0.08"/>
    <texture name="tiles" type="2d" builtin="checker" rgb1="0.78 0.74 0.68"
             rgb2="0.72 0.68 0.62" width="512" height="512" mark="edge" markrgb="0.6 0.57 0.52"/>
    <material name="tiles" texture="tiles" texrepeat="8 8" reflectance="0.12"/>
    <texture name="wood" type="2d" builtin="flat" mark="random" random="0.15"
             rgb1="0.55 0.38 0.24" markrgb="0.48 0.32 0.2" width="256" height="256"/>
    <material name="wood" texture="wood" texrepeat="3 3"/>
    <texture name="sand" type="2d" builtin="flat" mark="random" random="0.25"
             rgb1="0.62 0.58 0.45" markrgb="0.52 0.48 0.36" width="512" height="512"/>
    <material name="sand" texture="sand" texrepeat="10 10"/>
    <texture name="panel" type="2d" builtin="checker" rgb1="0.82 0.83 0.85"
             rgb2="0.78 0.79 0.81" width="256" height="256" mark="edge" markrgb="0.55 0.57 0.6"/>
    <material name="panel" texture="panel" texrepeat="4 4" reflectance="0.05"/>
    <material name="glow" emission="1"/>
    <material name="gloss" specular="0.9" shininess="0.9" reflectance="0.2"/>
    <material name="metal" specular="0.7" shininess="0.7" reflectance="0.1"
              rgba="0.62 0.64 0.67 1"/>
    {assets}
  </asset>
  <worldbody>
    {lights}
    {floor_geom}
    <body name="cine_rig" mocap="true" pos="0 0 5">
      <camera name="cine" fovy="{CINE_FOVY}"/>
    </body>
    {body}
  </worldbody>
</mujoco>"""


def set_cine(model, data, eye, target, up=(0.0, 0.0, 1.0)) -> None:
    """Point the mocap-mounted cinematic camera from eye at target."""
    mid = model.body("cine_rig").mocapid[0]
    data.mocap_pos[mid] = eye
    data.mocap_quat[mid] = mat_quat(look_at(np.asarray(eye), np.asarray(target), up))


def cam_pose(model, data, cam: str) -> CameraPose:
    cid = model.camera(cam).id
    return CameraPose(data.cam_xpos[cid].copy(), data.cam_xmat[cid].reshape(3, 3).copy())


# ---- head frame ----------------------------------------------------------------------------


def head_coords(point, pose: CameraPose) -> np.ndarray:
    """A world point in the viewer's head frame: (right, up, forward)."""
    d = np.asarray(point, float) - pose.position
    return np.array([d @ pose.rotation[:, 0], d @ pose.rotation[:, 1], -d @ pose.rotation[:, 2]])


def side_of(point, pose: CameraPose) -> str:
    return "left" if head_coords(point, pose)[0] < 0 else "right"


def depth_of(point, pose: CameraPose) -> str:
    return "ahead" if head_coords(point, pose)[2] > 0 else "behind"


# ---- motion --------------------------------------------------------------------------------


def min_jerk(tau: float) -> float:
    t = float(np.clip(tau, 0.0, 1.0))
    return t ** 3 * (10 - 15 * t + 6 * t * t)


class Path:
    """A Catmull-Rom curve through waypoints, followed with min-jerk timing over arc length,
    so the robot starts and stops smoothly and its heading follows the tangent."""

    def __init__(self, points, samples: int = 400) -> None:
        pts = np.asarray(points, float)
        ext = np.vstack([2 * pts[0] - pts[1], pts, 2 * pts[-1] - pts[-2]])
        curve = []
        for i in range(1, len(ext) - 2):
            p0, p1, p2, p3 = ext[i - 1:i + 3]
            for u in np.linspace(0, 1, samples // max(1, len(pts) - 1), endpoint=False):
                curve.append(0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3)
                                    * u * u + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3))
        curve.append(pts[-1])
        self.curve = np.asarray(curve)
        seg = np.linalg.norm(np.diff(self.curve, axis=0), axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(seg)])
        self.length = float(self.s[-1])

    def at(self, tau: float) -> tuple[np.ndarray, np.ndarray]:
        """(position, unit tangent) at normalised time tau in [0, 1]."""
        target = min_jerk(tau) * self.length
        i = int(np.clip(np.searchsorted(self.s, target) - 1, 0, len(self.curve) - 2))
        f = (target - self.s[i]) / max(1e-9, self.s[i + 1] - self.s[i])
        pos = self.curve[i] + f * (self.curve[i + 1] - self.curve[i])
        tan = self.curve[i + 1] - self.curve[i]
        return pos, tan / max(1e-9, np.linalg.norm(tan))


def yaw_of(direction) -> float:
    return float(np.arctan2(direction[1], direction[0]))


def lerp_angle(a: float, b: float, f: float) -> float:
    d = (b - a + np.pi) % (2 * np.pi) - np.pi
    return a + d * f
