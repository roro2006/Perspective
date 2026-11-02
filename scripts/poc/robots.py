"""The robots of the PoC video, each an MJCF fragment of primitives on a free joint, facing +x,
with a forward 'cam' camera (the frame a robot-only system would read directions in)."""

from __future__ import annotations

import numpy as np
from kit import ROBOT_HUMANOID, humanoid, set_joints

ORANGE = "0.95 0.45 0.12 1"
DARK = "0.14 0.15 0.17 1"
WHITE = "0.9 0.91 0.92 1"
TEAL = "0.1 0.62 0.68 1"


def amr(p: str = "amr_") -> str:
    """Warehouse AMR: low box chassis, four wheels, lidar puck, a lift deck for the tote."""
    wheels = "".join(
        f'<geom type="cylinder" pos="{x} {y} 0.09" euler="1.5708 0 0" size="0.09 0.04" '
        f'rgba="{DARK}"/>' for x in (0.28, -0.28) for y in (0.27, -0.27))
    return f"""
  <body name="{p}root" pos="0 0 0">
    <freejoint name="{p}free"/>
    <geom type="box" pos="0 0 0.2" size="0.42 0.3 0.11" rgba="{ORANGE}" material="gloss"/>
    <geom type="box" pos="0 0 0.33" size="0.36 0.26 0.02" rgba="{DARK}"/>
    <geom type="box" pos="0.425 0 0.2" size="0.01 0.22 0.05" rgba="0.15 0.15 0.15 1"/>
    <geom type="box" pos="0.432 0.14 0.24" size="0.004 0.04 0.012" rgba="1 1 0.8 1"/>
    <geom type="box" pos="0.432 -0.14 0.24" size="0.004 0.04 0.012" rgba="1 1 0.8 1"/>
    <geom type="cylinder" pos="0.3 0 0.39" size="0.06 0.04" rgba="{DARK}"/>
    <geom type="cylinder" pos="0.3 0 0.44" size="0.045 0.015" rgba="0.2 0.8 0.9 1"/>
    {wheels}
    <camera name="{p}cam" pos="0.44 0 0.3" xyaxes="0 -1 0 0 0 1" fovy="70"/>
  </body>"""


def tote_geoms(rgba: str = "0.2 0.42 0.75 1") -> str:
    return (f'<geom type="box" pos="0 0 0.01" size="0.2 0.15 0.01" rgba="{rgba}"/>'
            f'<geom type="box" pos="0.19 0 0.09" size="0.01 0.15 0.09" rgba="{rgba}"/>'
            f'<geom type="box" pos="-0.19 0 0.09" size="0.01 0.15 0.09" rgba="{rgba}"/>'
            f'<geom type="box" pos="0 0.14 0.09" size="0.2 0.01 0.09" rgba="{rgba}"/>'
            f'<geom type="box" pos="0 -0.14 0.09" size="0.2 0.01 0.09" rgba="{rgba}"/>')


def quadruped(p: str = "dog_") -> str:
    """Rescue dog: box body, head with a lamp, four two-link legs with hip and knee hinges."""
    def leg(n: str, x: float, y: float) -> str:
        return f"""
    <body name="{p}{n}_thigh" pos="{x} {y} -0.03">
      <joint name="{p}{n}_hip" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.2" size="0.035" rgba="{DARK}"/>
      <body name="{p}{n}_shin" pos="0 0 -0.2">
        <joint name="{p}{n}_knee" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.2" size="0.025" rgba="0.3 0.31 0.33 1"/>
        <geom type="sphere" pos="0 0 -0.205" size="0.033" rgba="{DARK}"/>
      </body>
    </body>"""
    legs = "".join(leg(n, x, y) for n, x, y in
                   (("fl", 0.26, 0.13), ("fr", 0.26, -0.13), ("hl", -0.26, 0.13),
                    ("hr", -0.26, -0.13)))
    return f"""
  <body name="{p}root" pos="0 0 0.45">
    <freejoint name="{p}free"/>
    <geom type="box" size="0.34 0.13 0.075" rgba="1 0.78 0.1 1" material="gloss"/>
    <geom type="box" pos="0 0 0.08" size="0.2 0.09 0.01" rgba="{DARK}"/>
    <geom type="box" pos="0.38 0 0.03" size="0.06 0.08 0.055" rgba="{DARK}"/>
    <geom type="cylinder" pos="0.44 0.04 0.04" euler="0 1.5708 0" size="0.018 0.006"
          rgba="1 1 0.85 1"/>
    <geom type="cylinder" pos="0.44 -0.04 0.04" euler="0 1.5708 0" size="0.018 0.006"
          rgba="1 1 0.85 1"/>
    <geom type="capsule" fromto="-0.1 0.06 0.09 -0.1 0.06 0.2" size="0.008" rgba="{DARK}"/>
    <geom type="sphere" pos="-0.1 0.06 0.2" size="0.016" rgba="1 0.2 0.15 1"/>
    {legs}
    <camera name="{p}cam" pos="0.45 0 0.05" xyaxes="0 -1 0 0 0 1" fovy="75"/>
    <light name="{p}lamp" pos="0.45 0 0.05" dir="1 0 -0.15" mode="track" cutoff="40"
           exponent="8" diffuse="0.6 0.6 0.5" castshadow="false"/>
  </body>"""


def dog_gait(model, data, p: str, phase: float, amp: float = 1.0) -> None:
    """Trot: diagonal pairs in phase. amp 0 stands still."""
    angles = {}
    for n, off in (("fl", 0.0), ("hr", 0.0), ("fr", np.pi), ("hl", np.pi)):
        s = np.sin(phase + off)
        angles[f"{n}_hip"] = 0.35 * amp * s - 0.25
        angles[f"{n}_knee"] = 0.5 + 0.35 * amp * max(0.0, np.cos(phase + off))
    set_joints(model, data, p, angles)


def quadcopter(p: str = "drone_") -> str:
    """Delivery drone: body, four arms with rotor discs (spinning hinge), skids."""
    arms, rotors = [], []
    for i, (x, y) in enumerate(((0.22, 0.22), (0.22, -0.22), (-0.22, 0.22), (-0.22, -0.22))):
        arms.append(f'<geom type="capsule" fromto="0 0 0 {x} {y} 0.02" size="0.018" '
                    f'rgba="{DARK}"/>')
        rotors.append(f"""
    <body name="{p}rotor{i}" pos="{x} {y} 0.045">
      <joint name="{p}rotor{i}" axis="0 0 1"/>
      <geom type="cylinder" size="0.012 0.02" rgba="{DARK}"/>
      <geom type="box" size="0.13 0.012 0.003" rgba="0.75 0.77 0.8 0.85"/>
    </body>""")
    return f"""
  <body name="{p}root" pos="0 0 2">
    <freejoint name="{p}free"/>
    <geom type="box" size="0.11 0.09 0.05" rgba="{WHITE}" material="gloss"/>
    <geom type="box" pos="0.1 0 0" size="0.02 0.06 0.03" rgba="{TEAL}"/>
    {"".join(arms)}{"".join(rotors)}
    <geom type="capsule" fromto="0.08 0.1 -0.12 -0.08 0.1 -0.12" size="0.01" rgba="{DARK}"/>
    <geom type="capsule" fromto="0.08 -0.1 -0.12 -0.08 -0.1 -0.12" size="0.01" rgba="{DARK}"/>
    <geom type="capsule" fromto="0 0.1 -0.12 0 0.08 -0.04" size="0.008" rgba="{DARK}"/>
    <geom type="capsule" fromto="0 -0.1 -0.12 0 -0.08 -0.04" size="0.008" rgba="{DARK}"/>
    <camera name="{p}cam" pos="0.13 0 -0.02" xyaxes="0 -1 0 0.5 0 0.866" fovy="80"/>
  </body>"""


def parcel(p: str = "parcel_") -> str:
    return f"""
  <body name="{p}root" pos="0 0 1">
    <freejoint name="{p}free"/>
    <geom type="box" size="0.09 0.09 0.07" rgba="0.72 0.55 0.36 1"/>
    <geom type="box" pos="0 0 0.0705" size="0.025 0.09 0.001" rgba="0.85 0.8 0.65 1"/>
  </body>"""


def robot_humanoid(p: str = "bot_") -> str:
    return humanoid(p, ROBOT_HUMANOID, head_cam=False,
                    extra_torso=f'<geom type="box" pos="-0.13 0 0.25" size="0.04 0.11 0.12" '
                                f'rgba="{DARK}"/>').replace(
        '<geom type="sphere" pos="0.105 0.06 0.235" size="0.014" rgba="0.2 0.9 0.5 1"/>',
        f'<camera name="{p}cam" pos="0.13 0 0.18" xyaxes="0 -1 0 0 0 1" fovy="75"/>')


def rov(p: str = "rov_") -> str:
    """Work-class ROV: open frame of rails, buoyancy block, four thrusters, two lights,
    a sampling arm."""
    rails = []
    for y in (0.3, -0.3):
        for z in (0.0, 0.4):
            rails.append(f'<geom type="capsule" fromto="0.45 {y} {z} -0.45 {y} {z}" '
                         f'size="0.02" rgba="{DARK}"/>')
        for x in (0.45, -0.45):
            rails.append(f'<geom type="capsule" fromto="{x} {y} 0 {x} {y} 0.4" size="0.02" '
                         f'rgba="{DARK}"/>')
    for x in (0.45, -0.45):
        for z in (0.0, 0.4):
            rails.append(f'<geom type="capsule" fromto="{x} 0.3 {z} {x} -0.3 {z}" '
                         f'size="0.02" rgba="{DARK}"/>')
    thr = "".join(
        f'<geom type="cylinder" pos="{x} {y} 0.2" euler="0 1.5708 0" size="0.07 0.09" '
        f'rgba="0.2 0.2 0.22 1"/>' for x in (-0.38,) for y in (0.36, -0.36))
    thr += "".join(
        f'<geom type="cylinder" pos="{x} 0 0.46" size="0.07 0.05" rgba="0.2 0.2 0.22 1"/>'
        for x in (0.25, -0.25))
    return f"""
  <body name="{p}root" pos="0 0 1">
    <freejoint name="{p}free"/>
    <geom type="box" pos="0 0 0.33" size="0.42 0.27 0.08" rgba="1 0.82 0.1 1" material="gloss"/>
    {"".join(rails)}{thr}
    <geom type="box" pos="0.2 0 0.15" size="0.15 0.12 0.1" rgba="0.3 0.32 0.35 1"/>
    <geom type="sphere" pos="0.48 0.2 0.25" size="0.045" rgba="1 1 0.85 1"/>
    <geom type="sphere" pos="0.48 -0.2 0.25" size="0.045" rgba="1 1 0.85 1"/>
    <geom type="capsule" fromto="0.4 0 0.02 0.62 0 -0.08" size="0.025" rgba="0.6 0.62 0.65 1"/>
    <geom type="sphere" pos="0.64 0 -0.09" size="0.035" rgba="{ORANGE}"/>
    <camera name="{p}cam" pos="0.5 0 0.2" xyaxes="0 -1 0 0 0 1" fovy="75"/>
    <light name="{p}l1" pos="0.5 0.2 0.25" dir="1 0 -0.3" mode="track" cutoff="35"
           exponent="6" diffuse="0.8 0.85 0.8" castshadow="false"/>
    <light name="{p}l2" pos="0.5 -0.2 0.25" dir="1 0 -0.3" mode="track" cutoff="35"
           exponent="6" diffuse="0.8 0.85 0.8" castshadow="false"/>
  </body>"""


def free_flyer(p: str = "bee_") -> str:
    """Astrobee-style cube: blue body, side propulsion modules, nozzle grilles, a status
    light and a forward camera."""
    return f"""
  <body name="{p}root" pos="0 0 1">
    <freejoint name="{p}free"/>
    <geom type="box" size="0.14 0.1 0.14" rgba="0.22 0.42 0.82 1" material="gloss"/>
    <geom type="box" pos="0 0.13 0" size="0.15 0.04 0.15" rgba="{WHITE}"/>
    <geom type="box" pos="0 -0.13 0" size="0.15 0.04 0.15" rgba="{WHITE}"/>
    <geom type="box" pos="0 0.172 0" size="0.1 0.002 0.1" rgba="0.3 0.3 0.33 1"/>
    <geom type="box" pos="0 -0.172 0" size="0.1 0.002 0.1" rgba="0.3 0.3 0.33 1"/>
    <geom type="box" pos="0.145 0 0" size="0.005 0.08 0.1" rgba="{DARK}"/>
    <geom type="sphere" pos="0.15 0.04 0.06" size="0.018" rgba="0.1 0.1 0.1 1"
          material="gloss"/>
    <geom type="box" pos="0.15 -0.05 -0.07" size="0.004 0.03 0.012" rgba="0.2 1 0.4 1"/>
    <camera name="{p}cam" pos="0.16 0.04 0.06" xyaxes="0 -1 0 0 0 1" fovy="75"/>
  </body>"""


def table_arm(p: str = "arm_") -> str:
    """A simple primitive arm for the face-to-face shot: base, shoulder, upper arm, forearm,
    gripper, all fixed (posed with euler)."""
    return f"""
  <body name="{p}root" pos="0 0 0.75">
    <freejoint name="{p}free"/>
    <geom type="cylinder" size="0.1 0.03" pos="0 0 0.03" rgba="{DARK}"/>
    <geom type="cylinder" size="0.06 0.08" pos="0 0 0.12" rgba="{ORANGE}"/>
    <geom type="capsule" fromto="0 0 0.2 0.05 0 0.48" size="0.045" rgba="{ORANGE}"/>
    <geom type="capsule" fromto="0.05 0 0.48 0.3 0 0.36" size="0.038" rgba="{ORANGE}"/>
    <geom type="box" pos="0.33 0 0.33" size="0.025 0.05 0.02" rgba="{DARK}"/>
    <geom type="box" pos="0.36 0.035 0.29" size="0.025 0.008 0.04" rgba="{DARK}"/>
    <geom type="box" pos="0.36 -0.035 0.29" size="0.025 0.008 0.04" rgba="{DARK}"/>
    <geom type="sphere" pos="0.05 0 0.48" size="0.05" rgba="{DARK}"/>
    <camera name="{p}cam" pos="0.06 0 0.56" xyaxes="0 -1 0 0.5 0 0.866" fovy="70"/>
  </body>"""
