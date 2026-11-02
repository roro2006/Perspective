"""Reel scenes 1-5: visibility (live grounder), reach, gaze, pointing, memory."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from kit import PERSON, Palette, euler_quat, humanoid, pose_humanoid, set_free, set_joints, shell
from kit import walk_angles as walk
from reel_kit import (
    Beat,
    ReelScene,
    answer_or,
    in_reach,
    person_point,
    pointed_door,
    ray_hit_plane_y,
    reach_arm,
    set_arm,
    track,
    two_stage,
)
from robots import amr, dog_gait, quadruped, robot_humanoid, tote_geoms
from scene_base import Candidate, Query, drive, window
from scenes_a import box, rack
from scenes_b import Home, bookcase

CARD = "0.68 0.52 0.34 1"


def yq(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])


def body_point(model, data, body: str, local) -> np.ndarray:
    return data.xpos[model.body(body).id] + data.xmat[model.body(body).id].reshape(3, 3) @ local


# ---- 1. visibility (live grounder) ---------------------------------------------------------


class Visibility(ReelScene):
    START = (4.8, -0.3)
    APPROACH: ClassVar[dict[str, tuple[float, float]]] = {"tote_h": (2.95, 0.3),
                                                          "tote_v": (2.25, -1.1)}
    T = (1.0, 2.3, 2.7, 5.0)

    def __init__(self) -> None:
        super().__init__("visibility", "It knows what you can see.", 6.6, robot_cam="amr_cam",
                         word="see", real=("geo", "naive"))
        self.candidates = [Candidate("tote_h", "tote", (2.3, 0.3, 0.1)),
                           Candidate("tote_v", "tote", (1.6, -1.1, 0.1))]
        line = "Bring the tote I can't see."
        self.queries = [Query("geo", line, "speaker", ("tote_h", "tote_v"), "tote_h"),
                        Query("naive", line, "robot", ("tote_h", "tote_v"), "tote_h")]
        self._c: dict = {}

    def xml(self) -> str:
        rng = np.random.default_rng(4)
        g = []
        for i, (x, y) in enumerate(((1.25, 0.0), (1.25, 0.42), (1.62, 0.0), (1.62, 0.42),
                                    (1.25, 0.84), (1.62, 0.84))):
            z = 0.0
            for _ in range(3 if i < 4 else 2):
                h = rng.uniform(0.17, 0.22)
                c = rng.uniform(0.9, 1.05)
                g.append(box((x + rng.uniform(-0.02, 0.02), y, z + h), (0.18, 0.2, h),
                             f"{0.68 * c:.2f} {0.52 * c:.2f} {0.34 * c:.2f} 1"))
                g.append(box((x, y, z + 2 * h - 0.003), (0.185, 0.03, 0.004),
                             "0.82 0.76 0.6 1"))
                z += 2 * h
        for c in self.candidates:
            g.append(f'<body name="{c.oid}"><freejoint name="{c.oid}free"/>'
                     f'{tote_geoms("0.16 0.45 0.78 1")}</body>')
        lines = "".join(box((x, -0.7, 0.002), (0.5, 0.04, 0.002), "0.95 0.8 0.1 1")
                        for x in np.arange(-1, 7, 1.4))
        body = (rack(-3.0, 7.0, 3.0, []) + "".join(g) + lines
                + humanoid("p_", Palette((0.22, 0.26, 0.34), PERSON.skin, PERSON.visor,
                                         (0.85, 0.95, 0.15))) + amr())
        return shell(body, floor="concrete", sky=((0.5, 0.52, 0.55), (0.25, 0.26, 0.28)),
                     ambient=(0.3, 0.3, 0.32))

    def setup(self, model, data) -> None:
        pose_humanoid(model, data, "p_", (0.0, 0.0, 0.94), euler_quat(yaw=0.0),
                      {"neck_pitch": 0.3})
        for c in self.candidates:
            set_free(model, data, c.oid + "free", (c.pos[0], c.pos[1], 0.0), yq(0))
        set_free(model, data, "amr_free", (*self.START, 0.0), yq(np.pi))

    def animate(self, model, data, t, answers) -> None:
        wrong = self.APPROACH[answer_or(answers, "naive", "tote_v")]
        pick = answer_or(answers, "geo", "tote_h")
        path, tau = two_stage(self._c, self.START, wrong, self.APPROACH[pick], t, self.T)
        pos, _ = drive(model, data, "amr_free", path, tau, 0.0, np.pi, yaw_end=np.pi,
                         turn=0.3)
        lift = window(t, self.T[3] + 0.1, self.T[3] + 0.7)
        if lift > 0:
            c = self.cand(pick).pos
            dst = np.array([pos[0], pos[1], 0.35])
            src = np.array([c[0], c[1], 0.0])
            set_free(model, data, pick + "free", src + lift * (dst - src), yq(0))

    def camera(self, t):
        f = t / self.duration
        return np.array([0.6 + 0.5 * f, -4.1, 2.9]), np.array([2.1 + 0.2 * f, -0.1, 0.45])

    def beats(self, answers) -> list[Beat]:
        b = [Beat(0.1, 2.9, "say", "Bring the tote I can't see.")]
        naive, geo = answers.get("naive"), answers.get("geo")
        if naive is not None:
            p = self.cand(naive).pos
            b.append(Beat(2.0, 3.2, "x" if naive != "tote_h" else "ok", pos=(p[0], p[1], 0.5)))
        if geo is not None:
            p = self.cand(geo).pos
            b.append(Beat(5.2, 6.6, "ok" if geo == "tote_h" else "x", pos=(p[0], p[1], 0.8)))
        return b


# ---- 2. reach ------------------------------------------------------------------------------


class Reach(ReelScene):
    PERSON_POS = (0.0, 0.1, 1.19)  # standing on a 0.25 m step stool
    PERSON_YAW = np.pi / 2
    START = (3.3, -0.7)
    APPROACH: ClassVar[dict[str, tuple[float, float]]] = {"jar_near": (0.75, -0.05),
                                                          "jar_far": (1.87, 0.02)}
    T = (0.9, 2.2, 2.6, 4.6)

    def __init__(self) -> None:
        super().__init__("reach", "It knows what you can reach.", 6.6, robot_cam="bot_cam",
                         word="reach")
        self.candidates = [Candidate("jar_near", "jar", (0.3, 0.66, 1.62)),
                           Candidate("jar_far", "jar", (2.1, 0.66, 1.62))]
        self._c: dict = {}

    def shoulders(self) -> list[np.ndarray]:
        return [person_point(self.PERSON_POS, self.PERSON_YAW, (0, s * 0.23, 0.48))
                for s in (1, -1)]

    def target(self) -> str:
        """The jar no shoulder can reach (the scripted right answer)."""
        out = [c.oid for c in self.candidates
               if not any(in_reach(s, c.pos) for s in self.shoulders())]
        return out[0]

    def xml(self) -> str:
        wood, white = "0.62 0.45 0.3 1", "0.92 0.92 0.9 1"
        g = [box((0.9, 1.05, 1.4), (3.2, 0.05, 1.4), "0.86 0.84 0.78 1"),
             box((0.9, 0.72, 0.43), (2.4, 0.3, 0.43), white),
             box((0.9, 0.72, 0.89), (2.42, 0.32, 0.025), "0.3 0.3 0.32 1"),
             box((0.9, 0.8, 1.49), (2.0, 0.22, 0.02), wood),
             box((0.9, 0.8, 2.24), (2.0, 0.22, 0.02), wood),
             box((-1.1, 0.8, 1.86), (0.02, 0.22, 0.38), wood),
             box((2.9, 0.8, 1.86), (0.02, 0.22, 0.38), wood),
             box((3.75, 0.62, 0.95), (0.42, 0.4, 0.95), "0.78 0.8 0.82 1", 'material="metal"'),
             box((0.0, 0.1, 0.125), (0.24, 0.2, 0.125), "0.85 0.3 0.25 1"),
             box((0.9, 0.72, 0.6), (0.3, 0.005, 0.2), "0.8 0.8 0.78 1")]
        for x in np.arange(-1.3, 3.2, 0.6):
            g.append(box((x, 0.415, 0.43), (0.28, 0.005, 0.38), "0.85 0.86 0.84 1"))
            g.append(box((x + 0.2, 0.405, 0.7), (0.05, 0.008, 0.01), "0.4 0.4 0.42 1"))
        for x, c in ((0.7, "0.25 0.5 0.7 1"), (1.0, "0.7 0.3 0.3 1"), (1.55, "0.9 0.75 0.3 1")):
            g.append(f'<geom type="cylinder" pos="{x} 0.8 1.6" size="0.05 0.09" rgba="{c}"/>')
        for c in self.candidates:
            g.append(f'<body name="{c.oid}"><freejoint name="{c.oid}free"/>'
                     f'<geom type="cylinder" size="0.07 0.1" rgba="0.75 0.88 0.92 0.75"/>'
                     f'<geom type="cylinder" pos="0 0 0.03" size="0.064 0.06" '
                     f'rgba="0.85 0.55 0.2 1"/>'
                     f'<geom type="cylinder" pos="0 0 0.11" size="0.073 0.015" '
                     f'rgba="0.2 0.2 0.22 1"/></body>')
        person = humanoid("p_", Palette((0.3, 0.55, 0.45), PERSON.skin, PERSON.visor,
                                        (0.95, 0.85, 0.3)))
        lights = ('<light name="key" pos="1 -2.5 5" dir="0 0.4 -1" castshadow="true" '
                  'diffuse="0.6 0.58 0.54"/><light name="fill" pos="-2 -2 3" dir="0.5 0.5 -1" '
                  'castshadow="false" diffuse="0.3 0.3 0.32"/>')
        return shell("".join(g) + person + robot_humanoid(), floor="tiles", lights=lights,
                     sky=((0.8, 0.82, 0.85), (0.6, 0.62, 0.66)), ambient=(0.34, 0.33, 0.31))

    def setup(self, model, data) -> None:
        pose_humanoid(model, data, "p_", self.PERSON_POS, euler_quat(yaw=self.PERSON_YAW),
                      {"neck_pitch": 0.2, "neck_yaw": -0.85, "r_sh_pitch": -1.2,
                       "r_sh_abd": -1.35, "r_elbow": -0.1})
        for c in self.candidates:
            set_free(model, data, c.oid + "free", c.pos, yq(0))
        pose_humanoid(model, data, "bot_", (*self.START, 0.94), euler_quat(yaw=2.9))

    def animate(self, model, data, t, answers) -> None:
        pick = self.target()
        wrong = "jar_near" if pick == "jar_far" else "jar_far"
        path, tau = two_stage(self._c, self.START, self.APPROACH[wrong], self.APPROACH[pick],
                              t, self.T)
        pos, yaw = drive(model, data, "bot_free", path, tau, 0.0, 2.9,
                         yaw_end=np.pi / 2 if t >= self.T[2] else None, turn=0.3)
        moving = 0.0 < tau < 1.0
        phase = path.length * tau * 4.5 + (0 if t < self.T[2] else 7)
        bob = 0.015 * abs(np.sin(phase)) if moving else 0.0
        set_free(model, data, "bot_free", (pos[0], pos[1], 0.94 + bob), yq(yaw))
        ang = walk(phase, 0.45 if moving else 0.0)
        up = window(t, self.T[3] + 0.05, self.T[3] + 0.6)
        down = window(t, self.T[3] + 0.9, self.T[3] + 1.6)
        ang.update({"r_sh_pitch": -1.85 * up + 0.8 * down, "r_elbow": -0.1,
                    "neck_pitch": -0.35 * up})
        set_joints(model, data, "bot_", ang)
        if up >= 1.0:
            import mujoco
            mujoco.mj_forward(model, data)
            hand = body_point(model, data, "bot_r_fore", np.array([0, 0, -0.3]))
            snap = window(t, self.T[3] + 0.6, self.T[3] + 0.85)
            c = np.asarray(self.cand(pick).pos)
            set_free(model, data, pick + "free", c + snap * (hand - c), yq(0))

    def camera(self, t):
        f = t / self.duration
        return np.array([3.7 - 0.3 * f, -3.3, 2.3]), np.array([1.05, 0.45, 1.25])

    def beats(self, answers) -> list[Beat]:
        pick = self.target()
        wrong = "jar_near" if pick == "jar_far" else "jar_far"
        pw, pp = self.cand(wrong).pos, self.cand(pick).pos
        return [Beat(0.1, 2.9, "say", "Pass me the one I can't reach."),
                Beat(1.8, 3.1, "x", pos=(pw[0], pw[1], pw[2] + 0.35)),
                Beat(5.2, 6.6, "ok", pos=(pp[0], pp[1], pp[2] + 0.4))]


# ---- 3. gaze -------------------------------------------------------------------------------


def drill(p: str) -> str:
    return (f'<body name="{p}"><freejoint name="{p}free"/>'
            '<geom type="box" pos="0 0 0.05" size="0.1 0.035 0.04" rgba="0.95 0.7 0.1 1"/>'
            '<geom type="box" pos="-0.03 0 -0.02" size="0.03 0.03 0.07" '
            'rgba="0.15 0.15 0.17 1"/>'
            '<geom type="box" pos="-0.03 0 -0.1" size="0.05 0.04 0.025" '
            'rgba="0.15 0.15 0.17 1"/>'
            '<geom type="cylinder" pos="0.13 0 0.05" euler="0 1.5708 0" size="0.022 0.03" '
            'rgba="0.3 0.3 0.32 1"/>'
            '<geom type="capsule" fromto="0.16 0 0.05 0.24 0 0.05" size="0.005" '
            'material="metal"/></body>')


def hammer(p: str) -> str:
    return (f'<body name="{p}"><freejoint name="{p}free"/>'
            '<geom type="capsule" fromto="-0.16 0 0 0.12 0 0" size="0.016" '
            'rgba="0.55 0.35 0.2 1"/>'
            '<geom type="box" pos="0.14 0 0" size="0.025 0.07 0.025" material="metal"/></body>')


def tape(p: str) -> str:
    return (f'<body name="{p}"><freejoint name="{p}free"/>'
            '<geom type="box" size="0.05 0.05 0.035" rgba="0.95 0.85 0.15 1"/>'
            '<geom type="cylinder" euler="1.5708 0 0" size="0.03 0.052" '
            'rgba="0.15 0.15 0.17 1"/></body>')


class Gaze(ReelScene):
    BASE = (0.15, 0.0, 0.92)
    PERSON_POS = (1.5, 0.0, 0.94)
    TOOLS: ClassVar[dict[str, tuple[float, float, float]]] = {
        "drill": (0.65, -0.42, 1.04), "hammer": (0.78, 0.05, 0.945), "tape": (0.62, 0.45, 0.965)}
    HAND = (1.08, -0.15, 1.1)

    def __init__(self) -> None:
        super().__init__("gaze", "It follows where you look.", 6.6, robot_cam="", word="look")
        self.candidates = [Candidate(k, k, v) for k, v in self.TOOLS.items()]

    def xml(self) -> str:
        g = [box((0.55, 0, 0.88), (0.75, 0.75, 0.04), "0.55 0.4 0.26 1", 'material="wood"'),
             box((-1.0, 0, 1.4), (0.04, 2.5, 1.4), "0.42 0.46 0.5 1"),
             box((-0.94, 0, 1.6), (0.02, 1.4, 0.55), "0.62 0.5 0.36 1")]
        for x in (-0.15, 1.25):
            for y in (-0.7, 0.7):
                g.append(box((x, y, 0.42), (0.04, 0.04, 0.42), "0.25 0.25 0.27 1"))
        for k, y in enumerate(np.arange(-1.1, 1.2, 0.3)):
            g.append(f'<geom type="capsule" fromto="-0.9 {y:.2f} 1.85 -0.9 {y:.2f} 1.55" '
                     f'size="0.012" rgba="{0.3 + 0.08 * (k % 3)} 0.32 0.36 1"/>')
        person = humanoid("p_", Palette((0.3, 0.3, 0.33), PERSON.skin, PERSON.visor,
                                        (0.95, 0.55, 0.15)))
        return shell("".join(g) + drill("drill") + hammer("hammer") + tape("tape") + person
                     + reach_arm("arm_", self.BASE), floor="concrete",
                     sky=((0.6, 0.62, 0.64), (0.3, 0.31, 0.33)), ambient=(0.32, 0.32, 0.33))

    def setup(self, model, data) -> None:
        import mujoco
        for k, v in self.TOOLS.items():
            set_free(model, data, k + "free", v, euler_quat(yaw=0.6 if k == "drill" else 0.3))
        best, target = None, np.array(self.TOOLS["drill"][:2])
        for yaw in np.arange(0.0, 0.9, 0.02):  # aim the head so its gaze lands on the drill
            for pitch in np.arange(0.2, 1.0, 0.02):
                self.look = (yaw, pitch)
                self._pose(model, data, 9.0)
                mujoco.mj_forward(model, data)
                err = np.linalg.norm(self.gaze_hit(self.head_pose(model, data))[:2] - target)
                if best is None or err < best[0]:
                    best = (err, yaw, pitch)
        self.look = best[1:]
        self._pose(model, data, 0.0)

    def _pose(self, model, data, t) -> None:
        f = window(t, 0.2, 1.1)
        reach = window(t, 4.2, 5.0)
        pose_humanoid(model, data, "p_", self.PERSON_POS, euler_quat(yaw=np.pi),
                      {"neck_yaw": self.look[0] * f, "neck_pitch": 0.15 + (self.look[1] - 0.15)
                       * f, "r_sh_pitch": -1.0 * reach, "r_sh_abd": 0.25 * reach,
                       "r_elbow": -0.4})

    def grip(self, t) -> np.ndarray:
        d, h = self.TOOLS["drill"], self.TOOLS["hammer"]
        return track(t, [(0.0, (0.3, 0.0, 1.35)), (1.2, (h[0], h[1], 1.12)),
                         (2.1, (h[0], h[1], 1.08)), (3.0, (d[0], d[1], 1.2)),
                         (3.35, (d[0], d[1], d[2] + 0.08)), (3.6, (d[0], d[1], d[2] + 0.08)),
                         (3.9, (d[0], d[1], 1.25)), (5.0, self.HAND)])

    def animate(self, model, data, t, answers) -> None:
        self._pose(model, data, t)
        g = self.grip(t)
        set_arm(model, data, "arm_", self.BASE, g)
        if t >= 3.6:
            set_free(model, data, "drillfree", g - [0, 0, 0.08], euler_quat(yaw=0.6))

    def gaze_hit(self, pose) -> np.ndarray:
        o, d = pose.position, pose.forward
        k = (0.95 - o[2]) / d[2]
        return o + k * d

    def camera(self, t):
        f = t / self.duration
        return np.array([2.0 - 0.3 * f, -2.3, 2.0]), np.array([0.8, -0.05, 1.0])

    def beats(self, answers) -> list[Beat]:
        h = self.TOOLS["hammer"]
        return [Beat(0.1, 2.6, "say", "Bring me that."),
                Beat(0.9, 6.6, "gaze"),
                Beat(1.7, 3.0, "x", pos=(h[0], h[1], 1.3)),
                Beat(5.0, 6.6, "ok", pos=(self.HAND[0], self.HAND[1], self.HAND[2] + 0.3))]


# ---- 4. pointing ---------------------------------------------------------------------------

WALL_Y = 1.3
FOG_GREY = ('<rgba fog="0.36 0.35 0.34 1" haze="0.45 0.44 0.42 1"/>'
            '<map fogstart="0.2" fogend="1.3" haze="0.6" znear="0.01" zfar="80" '
            'shadowclip="12"/>')
POINT_POSE = {"l_sh_pitch": -1.7, "l_sh_abd": 0.28, "l_elbow": -0.05, "neck_yaw": 0.3,
              "neck_pitch": 0.08}


class Pointing(ReelScene):
    PERSON_POS = (0.0, -0.5, 0.94)
    DOORS: ClassVar[dict[str, float]] = {"door_near": 2.3, "door_far": 5.4}
    START = (-1.1, 0.25)
    T = (1.0, 2.4, 2.8, 5.2)

    def __init__(self) -> None:
        super().__init__("pointing", "It sees your point from your eyes.", 6.6,
                         robot_cam="dog_cam", word="point", fog=True)
        self.candidates = [Candidate(k, "door", (x, WALL_Y, 1.0)) for k, x in self.DOORS.items()]
        self._c: dict = {}
        self.rays: dict = {}

    def xml(self) -> str:
        wall = "0.6 0.58 0.55 1"
        g = [box((3.0, WALL_Y + 0.06, 1.25), (5.5, 0.06, 1.25), wall),
             box((3.0, -1.36, 0.15), (5.5, 0.06, 0.15), wall),
             box((8.6, 0, 1.25), (0.06, 1.4, 1.25), wall)]
        for x in self.DOORS.values():
            g.append(box((x, WALL_Y - 0.005, 1.05), (0.48, 0.008, 1.05), "0.28 0.2 0.15 1"))
            g.append(box((x, WALL_Y - 0.012, 1.05), (0.42, 0.006, 0.99), "0.5 0.36 0.24 1"))
            g.append(box((x + 0.3, WALL_Y - 0.03, 1.0), (0.04, 0.02, 0.012), "0.8 0.75 0.5 1"))
        g.append(box((4.1, WALL_Y - 0.01, 2.0), (0.18, 0.005, 0.07), "0.1 0.75 0.3 1"))
        g.append(box((1.5, -0.9, 0.08), (0.25, 0.15, 0.08), "0.35 0.3 0.25 1"))
        g.append(box((6.8, -0.4, 0.1), (0.3, 0.2, 0.1), "0.3 0.3 0.32 1"))
        ff = humanoid("p_", Palette((0.45, 0.36, 0.2), (0.82, 0.64, 0.52), (0.1, 0.1, 0.12),
                                    (0.95, 0.95, 0.2)))
        ff = ff.replace('<geom type="cylinder" pos="0.0 0.0 0.26" size="0.08 0.025"',
                        '<geom type="cylinder" pos="0.0 0.0 0.24" size="0.15 0.03" '
                        'rgba="0.8 0.1 0.08 1"/><geom type="cylinder" pos="0.0 0.0 0.26" '
                        'size="0.08 0.025"')
        lights = ('<light name="key" pos="2 -3 6" dir="0.1 0.5 -1" castshadow="true" '
                  'diffuse="0.5 0.48 0.45"/>')
        return shell("".join(g) + ff + quadruped(), floor="concrete", lights=lights,
                     sky=((0.32, 0.3, 0.3), (0.12, 0.11, 0.11)), ambient=(0.3, 0.29, 0.28),
                     extra_visual=FOG_GREY).replace(
                         '<map znear="0.01" zfar="200" shadowclip="12"/>', "")

    def setup(self, model, data) -> None:
        import mujoco
        pose_humanoid(model, data, "p_", self.PERSON_POS, euler_quat(yaw=0.0), POINT_POSE)
        set_free(model, data, "dog_free", (*self.START, 0.45), yq(0.0))
        dog_gait(model, data, "dog_", 0.0, 0.0)
        mujoco.mj_forward(model, data)
        eye = data.cam_xpos[model.camera("p_head_cam").id].copy()
        shoulder = data.xpos[model.body("p_l_upper").id].copy()
        tip = body_point(model, data, "p_l_fore", np.array([0, 0, -0.33]))
        self.rays = {"eye": eye, "shoulder": shoulder, "tip": tip,
                     "eye_hit": ray_hit_plane_y(eye, tip, WALL_Y),
                     "arm_hit": ray_hit_plane_y(shoulder, tip, WALL_Y),
                     "pick": pointed_door(eye, tip, self.DOORS, WALL_Y),
                     "naive": pointed_door(shoulder, tip, self.DOORS, WALL_Y)}

    def animate(self, model, data, t, answers) -> None:
        pick, wrong = self.rays["pick"], self.rays["naive"]
        def goal(k):
            return (self.DOORS[k], WALL_Y - 0.55)

        path, tau = two_stage(self._c, self.START, goal(wrong), goal(pick), t, self.T,
                              frac=0.5)
        drive(model, data, "dog_free", path, tau, 0.45, 0.0, yaw_end=np.pi / 2, turn=0.25)
        moving = 0.0 < tau < 1.0
        dog_gait(model, data, "dog_", (path.length * tau + (0 if t < self.T[2] else 3)) * 9.0,
                 1.0 if moving else 0.0)

    def camera(self, t):
        f = t / self.duration
        return np.array([2.2 + 0.3 * f, -5.6, 2.9]), np.array([2.3 + 0.2 * f, 0.5, 1.0])

    def beats(self, answers) -> list[Beat]:
        r = self.rays
        pn, pp = self.cand(r["naive"]).pos, self.cand(r["pick"]).pos
        return [Beat(0.1, 2.6, "say", "Check that door."),
                Beat(1.0, 3.0, "badray", pos=tuple(r["shoulder"]), pos2=tuple(r["arm_hit"])),
                Beat(2.0, 3.2, "x", pos=(pn[0], pn[1] - 0.1, 1.9)),
                Beat(3.0, 6.6, "ray", pos=tuple(r["eye"]), pos2=tuple(r["eye_hit"])),
                Beat(5.2, 6.6, "ok", pos=(pp[0], pp[1] - 0.1, 1.9))]


# ---- 5. memory -----------------------------------------------------------------------------

FLASH = 1.5


class Memory(ReelScene):
    START = (2.3, 0.2)
    KEYS = (0.65, 2.51, 0.975)
    GOALS: ClassVar[dict[str, tuple[float, float]]] = {"shelf_n": (0.65, 1.78),
                                                       "shelf_s": (0.65, -1.78)}
    T = (FLASH + 0.6, FLASH + 1.9, FLASH + 2.2, FLASH + 4.3)

    def __init__(self) -> None:
        super().__init__("memory", "It remembers what you saw.", 7.0, robot_cam="bot_cam",
                         word="remember")
        self.candidates = [Candidate("shelf_n", "shelf", (0.6, 2.42, 1.0)),
                           Candidate("shelf_s", "shelf", (0.6, -2.42, 1.0))]
        self._c: dict = {}

    def xml(self) -> str:
        xml = Home.xml(self).replace(bookcase(0.6, 2.53, -1, keys=True),
                                     bookcase(0.6, 2.53, -1, keys=False))
        x, y, z = self.KEYS
        keys = (f'<body name="keys" mocap="true" pos="{x} {y} {z}">'
                '<geom type="cylinder" size="0.025 0.004" rgba="0.95 0.8 0.25 1" '
                'material="metal"/>')
        for k, a in enumerate((0.3, 1.1, 2.0)):
            keys += (f'<geom type="capsule" fromto="0 0 0 {0.07 * np.cos(a):.3f} '
                     f'{0.07 * np.sin(a):.3f} 0" size="0.008" rgba="{0.8 - 0.1 * k} 0.75 0.6 1" '
                     'material="metal"/>')
        keys += '<geom type="box" pos="0.08 -0.04 0.005" size="0.03 0.015 0.012" '
        keys += 'rgba="0.1 0.1 0.12 1"/></body>'
        return xml.replace("</worldbody>", keys + "</worldbody>")

    def setup(self, model, data) -> None:
        model.cam_fovy[model.camera("cine").id] = 60.0
        Home.setup(self, model, data)

    def animate(self, model, data, t, answers) -> None:
        mid = model.body("keys").mocapid[0]
        if t < FLASH:
            pose_humanoid(model, data, "p_", (0.6, 1.72, 0.94), euler_quat(yaw=np.pi / 2),
                          {"neck_pitch": 0.45, "r_sh_pitch": -1.1, "r_elbow": -0.5,
                           "l_sh_pitch": -0.2})
            drop = window(t, 0.45, 0.85)
            data.mocap_pos[mid] = np.array(self.KEYS) + [0, -0.12 * (1 - drop),
                                                         0.3 * (1 - drop ** 2)]
            return
        data.mocap_pos[mid] = self.KEYS
        Home.setup(self, model, data)
        path, tau = two_stage(self._c, self.START, self.GOALS["shelf_s"], self.GOALS["shelf_n"],
                              t, self.T, frac=0.45, mid=(1.9, 0.6))
        pos, yaw = drive(model, data, "bot_free", path, tau, 0.0, np.pi,
                         yaw_end=np.pi / 2 if t >= self.T[2] else None)
        moving = 0.0 < tau < 1.0
        phase = path.length * tau * 4.5
        bob = 0.015 * abs(np.sin(phase)) if moving else 0.0
        set_free(model, data, "bot_free", (pos[0], pos[1], 0.94 + bob), yq(yaw))
        ang = walk(phase, 0.45 if moving else 0.0)
        r = window(t, self.T[3], self.T[3] + 0.7)
        ang.update({"r_sh_pitch": -1.35 * r, "r_elbow": -0.15 - 0.3 * r, "neck_pitch": 0.3 * r})
        set_joints(model, data, "bot_", ang)

    def camera(self, t):
        f = t / self.duration
        return np.array([3.6, -1.3 + 0.2 * f, 3.5]), np.array([-0.4, 0.3, 0.45])

    def beats(self, answers) -> list[Beat]:
        s, n = self.cand("shelf_s").pos, self.cand("shelf_n").pos
        return [Beat(0.0, FLASH, "flash"),
                Beat(FLASH + 0.15, FLASH + 2.6, "say", "Where did I put my keys?"),
                Beat(FLASH + 1.5, FLASH + 2.7, "x", pos=(s[0], s[1], 1.9)),
                Beat(FLASH + 4.4, 7.0, "ok", pos=(n[0], n[1], 1.9))]


def scenes() -> list[ReelScene]:
    return [Visibility(), Reach(), Gaze(), Pointing(), Memory()]
