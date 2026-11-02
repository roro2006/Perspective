"""Scenes 4-6 of the PoC video (home humanoid, ROV, space free-flyer) and the face-to-face
explainer shot."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from kit import (
    PERSON,
    SITTING,
    Palette,
    Path,
    euler_quat,
    humanoid,
    mat_quat,
    pose_humanoid,
    set_free,
    set_joints,
    shell,
    walk_angles,
)
from robots import free_flyer, robot_humanoid, rov, table_arm
from scene_base import Candidate, Event, Query, Scene, drive, window
from scenes_a import box, verdicts

# ---- 4. home humanoid ----------------------------------------------------------------------


def bookcase(x: float, y: float, facing: float, keys: bool) -> str:
    """Identical bookcases; facing is the sign of y the open side points to."""
    wood = "0.42 0.28 0.18 1"
    g = [box((x - 0.45, y, 0.8), (0.02, 0.16, 0.8), wood), box((x + 0.45, y, 0.8),
                                                               (0.02, 0.16, 0.8), wood),
         box((x, y - 0.15 * facing, 0.8), (0.45, 0.01, 0.8), "0.36 0.24 0.15 1")]
    for z in (0.05, 0.5, 0.95, 1.4, 1.58):
        g.append(box((x, y, z), (0.45, 0.16, 0.015), wood))
    books = [(-0.35, 0.5, "0.6 0.15 0.15 1"), (-0.29, 0.5, "0.15 0.3 0.6 1"),
             (-0.23, 0.5, "0.75 0.65 0.2 1"), (0.3, 1.4, "0.2 0.5 0.3 1"),
             (0.36, 1.4, "0.6 0.15 0.15 1")]
    for dx, z, c in books:
        g.append(box((x + dx, y, z + 0.13), (0.025, 0.11, 0.12), c))
    g.append(f'<geom type="sphere" pos="{x + 0.3} {y} 0.62" size="0.1" '
             f'rgba="0.85 0.85 0.8 1"/>')
    g.append(box((x - 0.3, y, 1.05), (0.09, 0.09, 0.09), "0.3 0.55 0.6 1"))
    if keys:
        g.append(f'<geom type="cylinder" pos="{x + 0.05} {y + 0.02 * facing} 0.975" '
                 f'size="0.025 0.004" rgba="0.95 0.8 0.25 1" material="metal"/>')
        for k, a in enumerate((0.3, 1.1, 2.0)):
            ex, ey = x + 0.05 + 0.07 * np.cos(a), y + 0.07 * np.sin(a)
            g.append(f'<geom type="capsule" fromto="{x + 0.05} {y} 0.975 {ex:.3f} {ey:.3f} '
                     f'0.975" size="0.008" rgba="{0.8 - 0.1 * k} 0.75 0.6 1" '
                     f'material="metal"/>')
        g.append(box((x + 0.13, y - 0.04, 0.98), (0.03, 0.015, 0.012), "0.1 0.1 0.12 1"))
    return "".join(g)


class Home(Scene):
    START = (3.7, 0.0)
    T: ClassVar[dict[str, float]] = {"m0": 1.0, "m1": 4.7, "r0": 4.5, "r1": 5.3}

    def __init__(self) -> None:
        super().__init__("humanoid", "Directions from the sofa, carried out across the room.",
                         6.2, robot_cam="bot_cam")
        self.candidates = [Candidate("shelf_n", "shelf", (0.6, 2.42, 1.0)),
                           Candidate("shelf_s", "shelf", (0.6, -2.42, 1.0))]
        line = "Keys, shelf to my left."
        self.queries = [Query("geo", line, "speaker", ("shelf_n", "shelf_s"), "shelf_n"),
                        Query("naive", line, "robot", ("shelf_n", "shelf_s"), "shelf_n")]
        self._paths: dict = {}

    def xml(self) -> str:
        wc = "0.88 0.85 0.78 1"
        g = [box((-3.05, 0, 1.3), (0.05, 2.8, 1.3), wc), box((0.3, 2.75, 1.3), (3.4, 0.05, 1.3),
                                                             wc),
             box((0.3, -2.75, 1.3), (3.4, 0.05, 1.3), wc),
             box((3.75, 1.75, 1.3), (0.05, 1.05, 1.3), wc),
             box((3.75, -1.75, 1.3), (0.05, 1.05, 1.3), wc),
             box((3.75, 0, 2.4), (0.05, 0.7, 0.2), wc),
             box((0.3, 0, 0.004), (3.4, 2.7, 0.004), "0.6 0.45 0.3 1", 'material="wood"'),
             box((-1.0, 0, 0.009), (1.2, 1.4, 0.003), "0.55 0.25 0.22 1"),
             # sofa
             box((-2.5, 0, 0.22), (0.45, 1.15, 0.22), "0.3 0.36 0.45 1"),
             box((-2.9, 0, 0.6), (0.12, 1.15, 0.42), "0.3 0.36 0.45 1"),
             box((-2.5, 1.2, 0.35), (0.45, 0.1, 0.35), "0.27 0.32 0.4 1"),
             box((-2.5, -1.2, 0.35), (0.45, 0.1, 0.35), "0.27 0.32 0.4 1"),
             box((-2.45, 0.55, 0.49), (0.38, 0.5, 0.06), "0.34 0.4 0.5 1"),
             box((-2.45, -0.55, 0.49), (0.38, 0.5, 0.06), "0.34 0.4 0.5 1"),
             # coffee table, lamp, picture
             box((-1.1, 0, 0.4), (0.4, 0.6, 0.025), "0.4 0.27 0.17 1"),
             '<geom type="cylinder" pos="-1.1 0 0.2" size="0.05 0.2" rgba="0.2 0.2 0.2 1"/>',
             ('<geom type="cylinder" pos="-1.0 0.3 0.48" size="0.06 0.05" '
              'rgba="0.9 0.9 0.88 1"/>'),
             box((-3.0, 0, 1.75), (0.01, 0.6, 0.35), "0.25 0.45 0.55 1"),
             '<geom type="cylinder" pos="-2.7 2.3 0.8" size="0.02 0.8" material="metal"/>',
             ('<geom type="cylinder" pos="-2.7 2.3 1.65" size="0.17 0.12" '
              'rgba="1 0.95 0.8 1" material="glow"/>'),
             bookcase(0.6, 2.53, -1, keys=True), bookcase(0.6, -2.53, 1, keys=False)]
        person = humanoid("p_", Palette((0.55, 0.35, 0.55), PERSON.skin, PERSON.visor,
                                        (0.95, 0.85, 0.3)))
        lights = ('<light name="key" pos="1.5 -1.5 6" dir="-0.2 0.2 -1" castshadow="true" '
                  'diffuse="0.6 0.58 0.52"/>'
                  '<light name="fill" pos="-1 1 2.4" dir="0 0 -1" castshadow="false" '
                  'diffuse="0.35 0.33 0.3"/>')
        return shell("".join(g) + person + robot_humanoid(), lights=lights,
                     sky=((0.8, 0.82, 0.85), (0.6, 0.62, 0.66)), ambient=(0.34, 0.33, 0.31))

    def setup(self, model, data) -> None:
        pose_humanoid(model, data, "p_", (-2.4, 0.0, 0.62), euler_quat(yaw=0.0),
                      {**SITTING, "l_sh_pitch": -1.0, "l_sh_abd": 0.9, "l_elbow": -0.3})
        pose_humanoid(model, data, "bot_", (*self.START, 0.94), euler_quat(yaw=np.pi))

    def animate(self, model, data, t, answers) -> None:
        a = answers.get("geo")
        if a is None:
            return
        s = np.sign(self.cand(a).pos[1])
        if "p" not in self._paths:
            self._paths["p"] = Path([self.START, (2.6, 0.3 * s), (1.2, 1.2 * s),
                                     (0.65, 1.78 * s)])
        p = self._paths["p"]
        tau = window(t, self.T["m0"], self.T["m1"])
        pos, yaw = drive(model, data, "bot_free", p, tau, 0.0, np.pi, yaw_end=s * np.pi / 2)
        moving = 0.0 < tau < 1.0
        bob = 0.015 * abs(np.sin(p.length * tau * 4.5)) if moving else 0.0
        q = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        set_free(model, data, "bot_free", (pos[0], pos[1], 0.94 + bob), q)
        ang = walk_angles(p.length * tau * 4.5, 0.45 if moving else 0.0)
        r = window(t, self.T["r0"], self.T["r1"])
        ang.update({"r_sh_pitch": -1.25 * r, "r_elbow": -0.15 - 0.3 * r,
                    "neck_pitch": 0.25 * r})
        set_joints(model, data, "bot_", ang)

    def camera(self, t):
        f = t / self.duration
        return np.array([2.9, -2.35 + 0.3 * f, 3.0 - 0.2 * f]), np.array([-0.4, 0.35, 0.6])

    def events(self, answers, info) -> list[Event]:
        return ([Event(0.0, 3.0, "say", "Keys, shelf to my left.")]
                + verdicts(answers, info, self, "geo", self.T["m0"], self.T["m1"]))


# ---- 5. underwater ROV ---------------------------------------------------------------------


def coral(x: float, y: float) -> str:
    """One coral cluster: branching capsules and a brain-coral mound (identical copies)."""
    rng = np.random.default_rng(11)
    g = [(f'<geom type="ellipsoid" pos="{x} {y} 0.08" size="0.45 0.4 0.18" '
          'rgba="0.45 0.38 0.32 1"/>'),
         (f'<geom type="ellipsoid" pos="{x + 0.25} {y - 0.2} 0.2" size="0.18 0.16 0.13" '
          'rgba="0.85 0.55 0.35 1"/>')]
    for _ in range(9):
        bx, by = x + rng.uniform(-0.3, 0.3), y + rng.uniform(-0.3, 0.3)
        h = rng.uniform(0.35, 0.7)
        tx, ty = bx + rng.uniform(-0.15, 0.15), by + rng.uniform(-0.15, 0.15)
        c = ("0.95 0.42 0.55 1", "0.98 0.6 0.35 1", "0.75 0.35 0.75 1")[int(rng.integers(3))]
        g.append(f'<geom type="capsule" fromto="{bx:.3f} {by:.3f} 0.1 {tx:.3f} {ty:.3f} '
                 f'{h:.3f}" size="0.035" rgba="{c}"/>')
        g.append(f'<geom type="sphere" pos="{tx:.3f} {ty:.3f} {h:.3f}" size="0.05" '
                 f'rgba="{c}"/>')
    return "".join(g)


class Rov(Scene):
    START = (6.6, -0.4, 1.7)
    T: ClassVar[dict[str, float]] = {"m0": 1.0, "m1": 4.6, "a0": 4.6, "a1": 5.4}
    ROLL = np.radians(40)

    def __init__(self) -> None:
        super().__init__("rov", "No horizon down here. Your body is the map.", 6.2,
                         robot_cam="rov_cam", fog=True, small="diver comms")
        self.candidates = [Candidate("coral_n", "coral", (2.7, 1.5, 0.35)),
                           Candidate("coral_s", "coral", (2.7, -1.5, 0.35))]
        line = "Sample the coral on my left."
        self.queries = [Query("geo", line, "speaker", ("coral_n", "coral_s"), "coral_n"),
                        Query("naive", line, "robot", ("coral_n", "coral_s"), "coral_n")]
        self._paths: dict = {}

    def xml(self) -> str:
        rng = np.random.default_rng(5)
        g = [coral(2.7, 1.5), coral(2.7, -1.5)]
        for _ in range(28):
            x, y = rng.uniform(-6, 12), rng.uniform(-9, 9)
            if abs(x - 2.7) < 1.2 and abs(abs(y) - 1.5) < 1.0:
                continue
            s = rng.uniform(0.15, 0.6)
            g.append(f'<geom type="ellipsoid" pos="{x:.2f} {y:.2f} {s * 0.2:.2f}" '
                     f'size="{s:.2f} {s * 0.8:.2f} {s * 0.5:.2f}" rgba="0.36 0.38 0.36 1"/>')
        for k in range(8):
            g.append(f'<body name="bubble{k}" mocap="true" pos="0 0 -5"><geom type="sphere" '
                     f'size="{0.02 + 0.01 * (k % 3)}" rgba="0.85 0.95 1 0.5" '
                     f'contype="0" conaffinity="0"/></body>')
        tank = ('<geom type="capsule" fromto="-0.2 0 0.05 -0.2 0 0.45" size="0.085" '
                'rgba="0.85 0.75 0.15 1" material="gloss"/>')
        diver = humanoid("p_", Palette((0.08, 0.09, 0.12), (0.12, 0.13, 0.16),
                                       (0.6, 0.75, 0.85), (0.9, 0.5, 0.1)), extra_torso=tank)
        diver = diver.replace('<geom type="box" pos="0.05 0 -0.445" size="0.11 0.055 0.035"',
                              '<geom type="box" pos="0.1 0 -0.445" size="0.22 0.09 0.012"')
        extra_visual = ('<rgba fog="0.04 0.22 0.32 1" haze="0.1 0.35 0.45 1"/>'
                        '<map fogstart="1.0" fogend="13" haze="0.6" znear="0.01" zfar="60"/>')
        lights = ('<light name="sun" pos="0 0 14" dir="0.1 0.1 -1" castshadow="true" '
                  'diffuse="0.35 0.55 0.65"/>')
        return shell("".join(g) + diver + rov(), floor="sand", lights=lights,
                     sky=((0.05, 0.3, 0.42), (0.02, 0.12, 0.2)), ambient=(0.12, 0.25, 0.3),
                     headlight=0.2, extra_visual=extra_visual).replace(
                         '<map znear="0.01" zfar="200" shadowclip="12"/>', "")

    def setup(self, model, data) -> None:
        pose_humanoid(model, data, "p_", (0.0, 0.0, 1.35),
                      euler_quat(roll=-self.ROLL, pitch=0.25),
                      {"l_hip_pitch": 0.3, "r_hip_pitch": -0.1, "l_knee": 0.6, "r_knee": 0.3,
                       "l_sh_pitch": -0.6, "l_sh_abd": 0.7, "r_sh_pitch": 0.3,
                       "neck_pitch": 0.2})
        self._place(model, data, np.array(self.START), np.pi)

    def _place(self, model, data, pos, yaw) -> None:
        set_free(model, data, "rov_free", pos, (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)))

    def animate(self, model, data, t, answers) -> None:
        for k in range(8):
            ph = (t * 0.5 + k / 8) % 1.0
            mid = model.body(f"bubble{k}").mocapid[0]
            data.mocap_pos[mid] = (0.1 + 0.05 * np.sin(k + 3 * ph), -0.25 + 0.04 * k % 0.1,
                                   2.0 + 2.5 * ph)
        a = answers.get("geo")
        bob = np.array([0, 0, 0.03 * np.sin(t * 1.3)])
        if a is None:
            self._place(model, data, np.array(self.START) + bob, np.pi)
            return
        c = np.array(self.cand(a).pos)
        if "p" not in self._paths:
            self._paths["p"] = Path([self.START, (4.8, 0.7 * np.sign(c[1]), 1.3),
                                     (c[0] + 1.2, c[1], 0.7)])
        pos, _ = self._paths["p"].at(window(t, self.T["m0"], self.T["m1"]))
        self._place(model, data, pos + bob, np.pi)

    def camera(self, t):
        f = t / self.duration
        return np.array([4.2 + 0.6 * f, -5.0 + 0.5 * f, 2.7]), np.array([2.6, 0.2, 0.7])

    def events(self, answers, info) -> list[Event]:
        return ([Event(0.0, 3.0, "say", "Sample the coral on my left.")]
                + verdicts(answers, info, self, "geo", self.T["m0"], self.T["m1"]))


# ---- 6. space free-flyer -------------------------------------------------------------------

R_MOD = 1.3


def module() -> str:
    """Octagonal station module interior along x: panels, frame rings, handrails, lights,
    racks and an end hatch."""
    g = []
    half = R_MOD * np.tan(np.pi / 8)
    for i in range(8):
        th = i * np.pi / 4
        c, s = np.cos(th), np.sin(th)
        q = mat_quat(np.column_stack([[1, 0, 0], [0, s, -c], [0, c, s]]))
        quat = " ".join(f"{v:.5f}" for v in q)
        col = "0.86 0.87 0.88 1" if i % 2 else "0.8 0.81 0.83 1"
        g.append(f'<geom type="box" pos="0.5 {c * R_MOD:.4f} {s * R_MOD:.4f}" quat="{quat}" '
                 f'size="4.6 {half:.4f} 0.03" rgba="{col}" material="panel"/>')
        for x in np.arange(-3.6, 5.0, 1.5):
            g.append(f'<geom type="box" pos="{x:.2f} {c * (R_MOD - 0.04):.4f} '
                     f'{s * (R_MOD - 0.04):.4f}" quat="{quat}" size="0.04 {half:.4f} 0.03" '
                     f'rgba="0.55 0.57 0.6 1"/>')
        if i in (1, 5):
            r = R_MOD - 0.12
            g.append(f'<geom type="capsule" fromto="-3 {c * r:.3f} {s * r:.3f} 4 {c * r:.3f} '
                     f'{s * r:.3f}" size="0.018" rgba="0.95 0.75 0.15 1"/>')
        if i in (2, 6):
            r = R_MOD - 0.05
            g.append(f'<geom type="box" pos="0.5 {c * r:.3f} {s * r:.3f}" quat="{quat}" '
                     f'size="4.0 0.08 0.01" rgba="1 1 0.96 1" material="glow"/>')
        if i in (0, 4):
            r = R_MOD - 0.12
            for x in (-2.2, -1.0, 3.2):
                g.append(f'<geom type="box" pos="{x} {c * r:.3f} {s * r:.3f}" quat="{quat}" '
                         f'size="0.35 0.32 0.09" rgba="0.62 0.66 0.7 1"/>')
                g.append(f'<geom type="box" pos="{x + 0.2} {c * (r - 0.1):.3f} '
                         f'{s * (r - 0.1):.3f}" quat="{quat}" size="0.04 0.04 0.01" '
                         f'rgba="0.2 0.9 0.4 1" material="glow"/>')
    g.append(f'<geom type="cylinder" pos="-4.1 0 0" euler="0 1.5708 0" size="{R_MOD} 0.03" '
             f'rgba="0.7 0.72 0.75 1"/>')
    g.append('<geom type="cylinder" pos="-4.05 0 0" euler="0 1.5708 0" size="0.55 0.03" '
             'rgba="0.35 0.37 0.42 1"/>')
    g.append(f'<geom type="cylinder" pos="5.1 0 0" euler="0 1.5708 0" size="{R_MOD} 0.03" '
             f'rgba="0.7 0.72 0.75 1"/>')
    return "".join(g)


def wrench(p: str, pos) -> str:
    metal = 'rgba="0.7 0.72 0.76 1" material="metal"'
    return (f'<body name="{p}" pos="{pos[0]} {pos[1]} {pos[2]}"><freejoint name="{p}free"/>'
            f'<geom type="capsule" fromto="-0.11 0 0 0.11 0 0" size="0.014" {metal}/>'
            f'<geom type="box" pos="0.14 0 0" size="0.03 0.035 0.01" {metal}/>'
            f'<geom type="box" pos="0.17 0.022 0" size="0.02 0.01 0.01" {metal}/>'
            f'<geom type="box" pos="0.17 -0.022 0" size="0.02 0.01 0.01" {metal}/>'
            f'<geom type="cylinder" pos="-0.13 0 0" size="0.03 0.01" {metal}/>'
            f'<geom type="capsule" fromto="-0.06 0 0 0.06 0 0" size="0.018" '
            f'rgba="0.85 0.2 0.15 1"/></body>')


class Space(Scene):
    START = (3.3, 0.0, 0.0)
    ROLL = np.radians(120)
    T: ClassVar[dict[str, float]] = {"m0": 1.0, "m1": 4.6}

    def __init__(self) -> None:
        super().__init__("space", "No up, no down. Just your left.", 6.2, robot_cam="bee_cam")
        self.candidates = [Candidate("wrench_u", "wrench", (1.4, 0.35, 0.7)),
                           Candidate("wrench_d", "wrench", (1.4, -0.35, -0.7))]
        line = "Wrench on my left."
        self.queries = [Query("geo", line, "speaker", ("wrench_u", "wrench_d"), "wrench_u"),
                        Query("naive", line, "robot", ("wrench_u", "wrench_d"), "wrench_u")]
        self._paths: dict = {}

    def xml(self) -> str:
        suit = Palette((0.93, 0.93, 0.92), (0.93, 0.93, 0.92), (0.75, 0.55, 0.15),
                       (0.2, 0.35, 0.8), (0.8, 0.8, 0.8))
        pack = ('<geom type="box" pos="-0.2 0 0.22" size="0.08 0.15 0.2" '
                'rgba="0.85 0.85 0.84 1"/>')
        lights = ('<light name="l1" pos="-2 0 1.1" dir="0 0 -1" castshadow="true" '
                  'diffuse="0.5 0.5 0.5"/><light name="l2" pos="2.5 0 1.1" dir="0 0 -1" '
                  'castshadow="false" diffuse="0.45 0.45 0.45"/>')
        body = (module() + wrench("wrench_u", self.cand("wrench_u").pos)
                + wrench("wrench_d", self.cand("wrench_d").pos)
                + humanoid("p_", suit, extra_torso=pack) + free_flyer())
        return shell(body, lights=lights, sky=((0.02, 0.02, 0.05), (0.0, 0.0, 0.0)),
                     ambient=(0.36, 0.36, 0.38))

    def setup(self, model, data) -> None:
        model.cam_fovy[model.camera("cine").id] = 62.0
        pose_humanoid(model, data, "p_", (-0.3, 0.0, 0.05), euler_quat(roll=self.ROLL),
                      {"l_knee": 0.5, "r_knee": 0.3, "l_hip_pitch": -0.3, "r_hip_pitch": -0.1,
                       "l_sh_abd": 0.5, "r_sh_abd": -0.6, "l_elbow": -0.6, "r_elbow": -0.8,
                       "l_sh_pitch": -0.5, "neck_pitch": 0.35})
        set_free(model, data, "bee_free", self.START, (0, 0, 0, 1))

    def animate(self, model, data, t, answers) -> None:
        for name in ("wrench_u", "wrench_d"):
            c = np.array(self.cand(name).pos)
            set_free(model, data, name + "free", c + [0, 0, 0.02 * np.sin(t + c[2])],
                     euler_quat(roll=0.3 * t, yaw=0.6 + 0.1 * t))
        a = answers.get("geo")
        if a is None:
            return
        c = np.array(self.cand(a).pos)
        if "p" not in self._paths:
            self._paths["p"] = Path([self.START, (2.6, 0.15 * np.sign(c[1]), 0.3 * np.sign(c[2])),
                                     (c[0] + 0.42, c[1], c[2])])
        pos, _ = self._paths["p"].at(window(t, self.T["m0"], self.T["m1"]))
        set_free(model, data, "bee_free", pos, (0, 0, 0, 1))

    def camera(self, t):
        f = t / self.duration
        return np.array([4.6 - 0.3 * f, -0.45, -0.35 + 0.1 * f]), np.array([0.8, 0.1, 0.0])

    def events(self, answers, info) -> list[Event]:
        return ([Event(0.0, 3.0, "say", "Wrench on my left.")]
                + verdicts(answers, info, self, "geo", self.T["m0"], self.T["m1"]))


# ---- face-to-face explainer ----------------------------------------------------------------


class FaceToFace(Scene):
    def __init__(self) -> None:
        super().__init__("table", "Every robot that takes directions gets this backwards.", 4.0,
                         robot_cam="arm_cam")
        self.candidates = [Candidate("cube_l", "cube", (0.15, -0.2, 0.8)),
                           Candidate("cube_r", "cube", (0.15, 0.2, 0.8))]

    def xml(self) -> str:
        g = [box((0.15, 0, 0.72), (0.55, 0.45, 0.03), "0.8 0.68 0.52 1", 'material="wood"')]
        for x in (-0.35, 0.65):
            for y in (-0.38, 0.38):
                g.append(box((x, y, 0.35), (0.03, 0.03, 0.35), "0.35 0.27 0.2 1"))
        for c in self.candidates:
            g.append(box(c.pos, (0.04, 0.04, 0.04), "0.85 0.2 0.18 1"))
        g.append(box((2.3, 0, 1.5), (0.05, 4, 1.5), "0.86 0.85 0.82 1"))
        return shell("".join(g) + humanoid("p_", PERSON) + table_arm(), floor="tiles",
                     sky=((0.85, 0.85, 0.85), (0.6, 0.6, 0.62)))

    def setup(self, model, data) -> None:
        pose_humanoid(model, data, "p_", (1.0, 0.0, 0.94), euler_quat(yaw=np.pi),
                      {"neck_pitch": 0.45})
        set_free(model, data, "arm_free", (-0.28, 0.0, 0.75), (1, 0, 0, 0))

    def animate(self, model, data, t, answers) -> None:
        pass

    def camera(self, t):
        f = t / self.duration
        return np.array([-2.5 + 0.2 * f, -1.5, 2.4]), np.array([0.35, 0.0, 1.0])

    def events(self, answers, info) -> list[Event]:
        return []


def all_scenes() -> list[Scene]:
    from scenes_a import Drone, RescueDog, Warehouse
    return [Warehouse(), RescueDog(), Drone(), Home(), Rov(), Space()]
