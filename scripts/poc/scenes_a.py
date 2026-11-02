"""Scenes 1-3 of the PoC video: warehouse AMR, rescue dog, delivery drone."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from kit import (
    PERSON,
    Palette,
    Path,
    euler_quat,
    humanoid,
    pose_humanoid,
    set_free,
    set_joints,
    shell,
)
from robots import amr, dog_gait, parcel, quadcopter, quadruped, tote_geoms
from scene_base import Candidate, Event, Query, Scene, drive, window


def box(pos, size, rgba="0.5 0.5 0.5 1", extra="") -> str:
    return (f'<geom type="box" pos="{pos[0]} {pos[1]} {pos[2]}" '
            f'size="{size[0]} {size[1]} {size[2]}" rgba="{rgba}" {extra}/>')


def route(a, b, lane: float) -> list:
    """Ground route from a to b that runs along y = lane in between."""
    dx = b[0] - a[0]
    return [a, (a[0] + 0.3 * dx, lane), (b[0] - 0.3 * dx, lane), b]


def verdicts(answers, info, scene, key, t0, t1) -> list[Event]:
    """Chip with the grounder's answer, then a check or an X on the object it chose."""
    q = next(q for q in scene.queries if q.key == key)
    a = answers.get(key)
    meta = info.get(key, {})
    frame = "speaker's frame" if q.frame == "speaker" else "robot's own frame"
    if a is None:
        return [Event(t0 - 0.2, scene.duration, "chip",
                      f"Grounder ({frame}): no answer ({meta.get('error', 'error')})", False,
                      key)]
    chip = (f"Grounder, {frame}: {meta.get('letter', '?')} = {scene.cand(a).label} "
            f"({meta.get('p', 0):.2f}) · {meta.get('latency_ms', 0):.0f} ms")
    return [Event(t0 - 0.2, t1, "chip", chip, a == q.correct, key),
            Event(t1, scene.duration, "verdict", "", a == q.correct, a)]


# ---- 1. warehouse --------------------------------------------------------------------------

RACK_Y = 1.68
UPRIGHT = "0.16 0.3 0.55 1"
BEAM = "0.95 0.48 0.08 1"


def rack(x0: float, x1: float, yc: float, skip: list[tuple[float, float]]) -> str:
    """Pallet racking along x at y = yc: uprights, orange beams, decks, cardboard boxes
    everywhere except the slots in skip (x, level)."""
    g = []
    xs = np.arange(x0, x1 + 1e-6, 1.8)
    for x in xs:
        for dy in (-0.25, 0.25):
            g.append(box((x, yc + dy, 1.25), (0.04, 0.04, 1.25), UPRIGHT))
    levels = (0.42, 1.12, 1.82)
    rng = np.random.default_rng(int(abs(yc) * 10 + x0))
    for z in levels:
        for dy in (-0.25, 0.25):
            g.append(box(((x0 + x1) / 2, yc + dy, z - 0.04), ((x1 - x0) / 2, 0.03, 0.05), BEAM))
        g.append(box(((x0 + x1) / 2, yc, z - 0.01), ((x1 - x0) / 2, 0.26, 0.01),
                     "0.45 0.47 0.5 1"))
        for x in np.arange(x0 + 0.45, x1 - 0.3, 0.6):
            if any(abs(x - sx) < 0.5 and abs(z - sz) < 0.1 for sx, sz in skip):
                continue
            if rng.random() < 0.2:
                continue
            h = rng.uniform(0.12, 0.25)
            w = rng.uniform(0.18, 0.26)
            c = rng.uniform(0.85, 1.05)
            g.append(box((x, yc + rng.uniform(-0.05, 0.05), z + h), (w, 0.2, h),
                         f"{0.68 * c:.2f} {0.52 * c:.2f} {0.34 * c:.2f} 1"))
    return "\n".join(g)


class Warehouse(Scene):
    START = (-2.6, 0.0)
    GOALS: ClassVar[dict[str, tuple[float, float]]] = {"tote_a": (-4.4, -0.88), "tote_b": (3.4, -0.88), "bin_c": (2.0, 0.88),
             "bin_d": (2.0, -0.88)}
    T: ClassVar[dict[str, float]] = {"n0": 1.0, "n1": 2.9, "x": 2.7, "r0": 3.9, "r1": 6.2, "s2": 6.6, "m0": 7.6, "m1": 9.8}

    def __init__(self) -> None:
        super().__init__("warehouse", "It learns how you talk.", 10.8, robot_cam="amr_cam")
        y = RACK_Y - 0.18
        self.candidates = [Candidate("tote_a", "tote", (-4.4, -y, 0.52)),
                           Candidate("tote_b", "tote", (3.4, -y, 0.52)),
                           Candidate("bin_c", "bin", (2.0, y, 0.52)),
                           Candidate("bin_d", "bin", (2.0, -y, 0.52))]
        line1 = "Bring the tote from the rack behind me."
        self.queries = [
            Query("naive", line1, "robot", ("tote_a", "tote_b"), "tote_a", depth=True),
            Query("geo", line1, "speaker", ("tote_a", "tote_b"), "tote_a", depth=True),
            Query("geo2", "Now the bin from the rack on my left.", "speaker",
                  ("tote_b", "bin_c", "bin_d"), "bin_c"),
        ]
        self._paths: dict = {}

    def xml(self) -> str:
        props = []
        for c in self.candidates:
            col = "0.16 0.45 0.78 1" if c.label == "tote" else "0.85 0.72 0.18 1"
            props.append(f'<body name="{c.oid}" pos="{c.pos[0]} {c.pos[1]} 0.42">'
                         f'{tote_geoms(col)}</body>')
        skip_n = [(2.0, 0.42)]
        skip_s = [(-4.4, 0.42), (3.4, 0.42), (2.0, 0.42)]
        floor_lines = "".join(box((x, s * 0.95, 0.002), (0.6 if abs(x) < 9 else 0, 0.04, 0.002),
                                  "0.95 0.8 0.1 1") for x in np.arange(-6, 7, 1.6)
                              for s in (-1, 1))
        body = (rack(-6.2, 4.6, RACK_Y, skip_n) + rack(-6.2, 4.6, -RACK_Y, skip_s)
                + rack(-6.2, 4.6, RACK_Y + 2.6, []) + rack(-6.2, 4.6, -RACK_Y - 2.6, [])
                + floor_lines + "".join(props)
                + humanoid("p_", Palette((0.22, 0.26, 0.34), PERSON.skin, PERSON.visor,
                                         (0.85, 0.95, 0.15)))
                + amr())
        return shell(body, floor="concrete", sky=((0.5, 0.52, 0.55), (0.25, 0.26, 0.28)),
                     ambient=(0.3, 0.3, 0.32))

    def setup(self, model, data) -> None:
        pose_humanoid(model, data, "p_", (0.0, -0.45, 0.94), euler_quat(yaw=0.0),
                      {"neck_pitch": 0.3})
        self._place(model, data, self.START, np.pi)

    def _place(self, model, data, xy, yaw) -> None:
        set_free(model, data, "amr_free", (xy[0], xy[1], 0.0),
                 (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)))

    def _path(self, key, pts) -> Path:
        if key not in self._paths:
            self._paths[key] = Path(pts)
        return self._paths[key]

    def animate(self, model, data, t, answers) -> None:
        T = self.T
        naive, geo, geo2 = answers.get("naive"), answers.get("geo"), answers.get("geo2")
        pos, yaw = np.array(self.START), np.pi
        if naive is not None:
            g = self.GOALS[naive]
            # toward the robot-frame pick; it stops partway when corrected
            stop = (self.START[0] + 0.45 * (g[0] - self.START[0]), 0.6)
            p = self._path("n", [self.START, (self.START[0] + 0.6, 0.45), stop])
            pos, yaw = drive(model, data, "amr_free", p, window(t, T["n0"], T["n1"]), 0.0,
                             np.pi, turn=0.35)
        if geo is not None and t >= T["r0"]:
            g = self.GOALS[geo]
            p = self._path("r", route(tuple(pos), g, 0.62))
            pos, yaw = drive(model, data, "amr_free", p, window(t, T["r0"], T["r1"]), 0.0,
                             yaw, yaw_end=np.sign(g[1]) * np.pi / 2)
        if geo2 is not None and t >= T["m0"]:
            g = self.GOALS[geo2]
            p = self._path("m", route(tuple(pos), g, 0.62))
            drive(model, data, "amr_free", p, window(t, T["m0"], T["m1"]), 0.0, yaw,
                  yaw_end=np.sign(g[1]) * np.pi / 2)

    def camera(self, t):
        f = t / self.duration
        return np.array([-0.6 + 0.6 * f, -3.0, 11.5]), np.array([-0.5 + 0.4 * f, 0.0, 0.0])

    def events(self, answers, info) -> list[Event]:
        T = self.T

        def clip(evs, chip_end, verdict_end):
            for e in evs:
                e.t1 = chip_end if e.kind == "chip" else verdict_end
            return evs

        ev = [Event(0.0, T["x"] + 1.3, "say", "Bring the tote from the rack behind me.")]
        naive = verdicts(answers, info, self, "naive", T["n0"], T["x"])
        ev += clip(naive, T["r0"], T["r0"] + 0.4)
        if answers.get("naive") not in (None, "tote_a"):
            ev.append(Event(T["x"], T["r0"] + 0.6, "caption2", "No, behind ME"))
        ev += clip(verdicts(answers, info, self, "geo", T["r0"], T["r1"]), T["s2"],
                   T["m0"] + 0.3)
        ev.append(Event(T["s2"], self.duration, "say", "Now the bin from the rack on my left."))
        ev += verdicts(answers, info, self, "geo2", T["m0"], T["m1"])
        return ev


# ---- 2. rescue dog -------------------------------------------------------------------------

WALL = "0.72 0.70 0.66 1"
WALL_H = 0.8  # half height (cutaway walls, 1.6 m)


def wall(x0, y0, x1, y1) -> str:
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return box((cx, cy, WALL_H), (max(0.06, abs(x1 - x0) / 2), max(0.06, abs(y1 - y0) / 2),
                                  WALL_H), WALL)


class RescueDog(Scene):
    START = (8.9, 0.0)
    DOOR_X = 2.0
    T: ClassVar[dict[str, float]] = {"m0": 1.0, "m1": 4.8}

    def __init__(self) -> None:
        super().__init__("dog", "It goes where you mean, even when it can't see you.", 6.2,
                         robot_cam="dog_cam")
        self.candidates = [Candidate("room_n", "room", (2.0, 2.4, 1.0)),
                           Candidate("room_s", "room", (2.0, -2.4, 1.0))]
        self.queries = [Query("geo", "Survivor's in the room on my left.", "speaker",
                              ("room_n", "room_s"), "room_n"),
                        Query("naive", "Survivor's in the room on my left.", "robot",
                              ("room_n", "room_s"), "room_n")]
        self._paths: dict = {}

    def xml(self) -> str:
        d0, d1 = self.DOOR_X - 0.55, self.DOOR_X + 0.55
        w = []
        for s in (1, -1):
            y = 0.9 * s
            w += [wall(-3, y, d0, y), wall(d1, y, 8.0, y)]
            w += [wall(0.5, y, 0.5, 3.9 * s), wall(3.5, y, 3.5, 3.9 * s),
                  wall(0.5, 3.9 * s, 3.5, 3.9 * s)]
            w += [wall(8.0, y, 8.0, 5 * s), wall(9.9, -5.0, 9.9, 5.0)]
            # neighbouring rooms along corridor A
            w += [wall(-3, y, -3, 3.9 * s), wall(-3, 3.9 * s, 0.5, 3.9 * s),
                  wall(3.5, 3.9 * s, 8.0, 3.9 * s), wall(5.8, y, 5.8, 3.9 * s)]
        w.append(wall(-3, -0.9, -3, 0.9))
        smoke = "".join(box((x, 0, 0.9), (0.5, 0.85, 0.9), "0.55 0.55 0.57 0.28",
                            'contype="0" conaffinity="0"') for x in (5.2, 6.4))
        debris = (box((4.4, -0.6, 0.06), (0.2, 0.12, 0.06), "0.4 0.3 0.2 1")
                  + box((0.95, 3.45, 0.25), (0.35, 0.3, 0.25), "0.4 0.35 0.3 1")
                  + box((1.2, -3.2, 0.4), (0.4, 0.4, 0.4), "0.35 0.36 0.4 1"))
        survivor = humanoid("s_", Palette((0.55, 0.15, 0.15), (0.8, 0.62, 0.5),
                                          (0.8, 0.62, 0.5), (0.9, 0.9, 0.9)), head_cam=False)
        ff = humanoid("p_", Palette((0.45, 0.36, 0.2), (0.82, 0.64, 0.52),
                                    (0.1, 0.1, 0.12), (0.95, 0.95, 0.2)))
        ff = ff.replace('<geom type="cylinder" pos="0.0 0.0 0.26" size="0.08 0.025"',
                        '<geom type="cylinder" pos="0.0 0.0 0.24" size="0.15 0.03" '
                        'rgba="0.8 0.1 0.08 1"/><geom type="cylinder" pos="0.0 0.0 0.26" '
                        'size="0.08 0.025"')
        body = "".join(w) + smoke + debris + survivor + ff + quadruped()
        return shell(body, floor="concrete", sky=((0.32, 0.3, 0.3), (0.12, 0.11, 0.11)),
                     ambient=(0.28, 0.27, 0.26))

    def setup(self, model, data) -> None:
        pose_humanoid(model, data, "p_", (-1.2, 0.0, 0.94), euler_quat(yaw=0.0),
                      {"neck_pitch": 0.05, "l_sh_pitch": -0.5, "l_sh_abd": 1.45, "l_elbow": -0.1})
        pose_humanoid(model, data, "s_", (2.4, 2.9, 0.15), euler_quat(pitch=-np.pi / 2,
                                                                       yaw=np.pi * 0.85),
                      {"l_sh_abd": 0.5, "r_sh_abd": -0.9, "l_knee": 0.4})
        set_free(model, data, "dog_free", (*self.START, 0.45), (0, 0, 0, 1))
        dog_gait(model, data, "dog_", 0.0, 0.0)

    def animate(self, model, data, t, answers) -> None:
        a = answers.get("geo")
        if a is None:
            return
        s = np.sign(self.cand(a).pos[1])
        if "p" not in self._paths:
            self._paths["p"] = Path([self.START, (5.0, 0.0), (2.8, 0.0), (2.0, 0.6 * s),
                                     (2.05, 1.7 * s), (2.25, 2.3 * s)])
        p = self._paths["p"]
        tau = window(t, self.T["m0"], self.T["m1"])
        drive(model, data, "dog_free", p, tau, 0.45, np.pi, turn=0.05)
        moving = 0.0 < tau < 1.0
        dog_gait(model, data, "dog_", p.length * (tau if moving else 1.0) * 9.0,
                 1.0 if moving else 0.0)

    def camera(self, t):
        f = t / self.duration
        return np.array([6.8 - 1.6 * f, -6.6, 7.4]), np.array([3.8 - 1.0 * f, 0.4, 0.0])

    def events(self, answers, info) -> list[Event]:
        return ([Event(0.0, 3.2, "say", "Survivor's in the room on my left.")]
                + verdicts(answers, info, self, "geo", self.T["m0"], self.T["m1"]))


# ---- 3. delivery drone ---------------------------------------------------------------------


class Drone(Scene):
    START = (7.5, 0.8, 3.6)
    T: ClassVar[dict[str, float]] = {"m0": 0.9, "m1": 3.9, "d0": 4.0, "d1": 4.8, "u0": 4.9, "u1": 6.0}

    def __init__(self) -> None:
        super().__init__("drone", "Looking down, it still sees it your way.", 6.2,
                         robot_cam="drone_cam")
        self.candidates = [Candidate("mat_n", "drop mat", (2.0, 1.25, 0.02)),
                           Candidate("mat_s", "drop mat", (2.0, -1.25, 0.02))]
        line = "Drop it on my right, not the plants."
        self.queries = [Query("geo", line, "speaker", ("mat_n", "mat_s"), "mat_s"),
                        Query("naive", line, "robot", ("mat_n", "mat_s"), "mat_s")]
        self._paths: dict = {}

    def xml(self) -> str:
        g = [box((-0.45, 0, -2), (0.25, 7, 10), "0.86 0.84 0.8 1"),
             box((-0.19, 0.0, 1.1), (0.02, 1.0, 1.1), "0.3 0.38 0.45 0.85", 'material="gloss"'),
             box((1.4, 0, -0.12), (1.6, 2.45, 0.12), "0.72 0.7 0.67 1"),
             box((1.4, 0, 0.003), (1.55, 2.4, 0.003), "0.6 0.45 0.3 1", 'material="wood"')]
        for y0, y1, x0, x1 in ((-2.45, 2.45, 3.0, 3.0), (2.45, 2.45, -0.2, 3.0),
                               (-2.45, -2.45, -0.2, 3.0)):
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            hx, hy = max(0.012, (x1 - x0) / 2), max(0.012, (y1 - y0) / 2)
            g.append(box((cx, cy, 0.5), (hx, hy, 0.45), "0.7 0.85 0.9 0.25"))
            g.append(f'<geom type="capsule" fromto="{x0} {y0} 1.0 {x1} {y1} 1.0" size="0.025" '
                     f'material="metal"/>')
        for x, y, r, h in ((2.6, 1.9, 0.2, 0.32), (1.45, 1.95, 0.17, 0.28),
                           (2.65, 0.65, 0.16, 0.26), (1.3, 1.1, 0.13, 0.22)):
            g.append(f'<geom type="cylinder" pos="{x} {y} {h / 2}" size="{r} {h / 2}" '
                     f'rgba="0.72 0.38 0.22 1"/>')
            for k, (dx, dy, dz) in enumerate(((0, 0, 0.3), (0.1, 0.06, 0.45), (-0.08, -0.05, 0.4),
                                              (0.03, -0.1, 0.6))):
                s = r * (1.3 - 0.2 * k)
                g.append(f'<geom type="ellipsoid" pos="{x + dx} {y + dy} {h + dz}" '
                         f'size="{s} {s} {s * 1.4}" rgba="{0.18 + 0.04 * k} {0.45 + 0.05 * k} '
                         f'0.2 1"/>')
        g.append(box((1.0, -2.0, 0.22), (0.3, 0.3, 0.02), "0.25 0.25 0.27 1"))
        g.append('<geom type="cylinder" pos="1.0 -2.0 0.11" size="0.03 0.11" material="metal"/>')
        for c in self.candidates:
            g.append(box((c.pos[0], c.pos[1], 0.008), (0.32, 0.32, 0.006), "0.22 0.24 0.27 1"))
            g.append(box((c.pos[0], c.pos[1], 0.015), (0.22, 0.22, 0.002), "0.9 0.9 0.9 1"))
            g.append(box((c.pos[0], c.pos[1], 0.016), (0.18, 0.18, 0.002), "0.22 0.24 0.27 1"))
        rng = np.random.default_rng(3)
        for _ in range(40):
            x, y = rng.uniform(6, 60), rng.uniform(-40, 40)
            h = rng.uniform(3, 30)
            c = rng.uniform(0.55, 0.8)
            g.append(box((x, y, -12 + h / 2), (rng.uniform(2, 5), rng.uniform(2, 5), h / 2),
                         f"{c:.2f} {c * 0.98:.2f} {c * 0.95:.2f} 1"))
        g.append('<geom type="plane" pos="0 0 -12" size="80 80 0.1" rgba="0.42 0.45 0.4 1"/>')
        person = humanoid("p_", Palette((0.62, 0.3, 0.32), PERSON.skin, PERSON.visor,
                                        (0.95, 0.95, 0.95)))
        body = "".join(g) + person + quadcopter() + parcel()
        return shell(body, sky=((0.62, 0.76, 0.92), (0.95, 0.93, 0.88)),
                     ambient=(0.38, 0.38, 0.4),
                     lights='<light name="sun" pos="6 -6 12" dir="-0.4 0.45 -1" '
                            'castshadow="true" diffuse="0.8 0.78 0.72"/>'
                            '<light name="fill" pos="-2 4 6" dir="0.4 -0.3 -1" '
                            'castshadow="false" diffuse="0.25 0.27 0.3"/>')

    def setup(self, model, data) -> None:
        pose_humanoid(model, data, "p_", (0.4, 0.0, 0.94), euler_quat(yaw=0.0),
                      {"neck_pitch": 0.45, "r_sh_pitch": -0.9, "r_sh_abd": -0.7,
                       "r_elbow": -0.2})
        self._fly(model, data, np.array(self.START), np.pi, 0.0, None)

    def _fly(self, model, data, pos, yaw, t, parcel_z) -> None:
        set_free(model, data, "drone_free", pos, (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)))
        set_joints(model, data, "drone_", {f"rotor{i}": (1 if i % 3 else -1) * t * 40
                                           for i in range(4)})
        pz = pos[2] - 0.21 if parcel_z is None else parcel_z
        px = pos[:2] if parcel_z is None else self._drop_xy
        set_free(model, data, "parcel_free", (px[0], px[1], pz),
                 (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)))

    def animate(self, model, data, t, answers) -> None:
        a = answers.get("geo")
        if a is None:
            self._fly(model, data, np.array(self.START), np.pi, t, None)
            return
        c = self.cand(a).pos
        if "p" not in self._paths:
            self._paths["p"] = Path([self.START, (4.6, 0.6 * np.sign(c[1]), 2.6),
                                     (c[0], c[1], 1.25)])
            self._drop_xy = np.array(c[:2])
        T = self.T
        pos, _ = self._paths["p"].at(window(t, T["m0"], T["m1"]))
        pz = None
        if t >= T["d0"]:
            pz = 1.25 - 0.21 - window(t, T["d0"], T["d1"]) ** 2 * (1.25 - 0.21 - 0.095)
        if t >= T["u0"]:
            pos = pos + np.array([0.6, 0, 0.9]) * (window(t, T["u0"], T["u1"]) ** 2)
        bob = 0.015 * np.sin(t * 5)
        self._fly(model, data, pos + [0, 0, bob], np.pi, t, pz)

    def camera(self, t):
        f = t / self.duration
        return (np.array([6.6 - 0.3 * f, -2.6 + 0.6 * f, 4.4]), np.array([1.4, 0.0, 0.6]))

    def events(self, answers, info) -> list[Event]:
        return ([Event(0.0, 3.2, "say", "Drop it on my right, not the plants.")]
                + verdicts(answers, info, self, "geo", self.T["m0"], self.T["d1"]))
