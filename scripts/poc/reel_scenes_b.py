"""Reel scenes 6-9: placement, your frame (live grounder, ROV), moving frame, habits."""

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
    min_jerk,
    pose_humanoid,
    set_free,
    shell,
    walk_angles,
)
from reel_kit import (
    Beat,
    ReelScene,
    answer_or,
    coral_tree,
    follow_bot,
    ghost,
    person_point,
    reach_arm,
    set_arm,
    track,
    two_stage,
)
from scene_base import Candidate, Query, window
from scenes_a import box
from scenes_b import Rov, coral, wrench

HOLD = 0.06  # grip point above an object's origin when the arm holds it


def yq(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])


def held(t: float, spans: list[tuple[float, float]]) -> bool:
    return any(a <= t < b for a, b in spans)


# ---- 6. placement --------------------------------------------------------------------------

TABLE_Z = 0.75


def mug(p: str) -> str:
    return (f'<body name="{p}"><freejoint name="{p}free"/>'
            '<geom type="cylinder" size="0.045 0.05" rgba="0.95 0.95 0.93 1"/>'
            '<geom type="cylinder" pos="0 0 0.045" size="0.038 0.006" rgba="0.35 0.2 0.1 1"/>'
            '<geom type="capsule" fromto="0.045 0 0.03 0.07 0 0.0" size="0.009" '
            'rgba="0.95 0.95 0.93 1"/>'
            '<geom type="capsule" fromto="0.07 0 0.0 0.045 0 -0.03" size="0.009" '
            'rgba="0.95 0.95 0.93 1"/></body>')


class Placement(ReelScene):
    PERSON_POS = (0.86, 0.52, 0.6)
    PERSON_YAW = float(np.arctan2(-0.52, -0.86))
    BASE = (-0.5, 0.0, TABLE_Z)
    CUP0 = (-0.12, 0.12)

    def __init__(self) -> None:
        super().__init__("placement", "It puts things where you want them.", 6.8,
                         robot_cam="", word="place")
        self.candidates = []

    def right_hand_target(self) -> np.ndarray:
        """Where 'by my right hand' is: a point in the person's own body frame (forward 0.45,
        0.24 to their right), on the table top."""
        p = person_point(self.PERSON_POS, self.PERSON_YAW, (0.45, -0.24, 0.0))
        return np.array([p[0], p[1], TABLE_Z + 0.05])

    def robot_right_target(self) -> np.ndarray:
        """The robot-only reading: the robot's right (the arm faces +x, so its right is -y)."""
        return np.array([self.BASE[0] + 0.45, self.BASE[1] - 0.36, TABLE_Z + 0.05])

    def xml(self) -> str:
        g = [box((0, 0, TABLE_Z - 0.02), (0.6, 0.6, 0.02), "0.8 0.7 0.55 1", 'material="wood"')]
        for x in (-0.52, 0.52):
            for y in (-0.52, 0.52):
                g.append(box((x, y, (TABLE_Z - 0.04) / 2), (0.03, 0.03, (TABLE_Z - 0.04) / 2),
                             "0.4 0.3 0.2 1"))
        cx, cy = person_point(self.PERSON_POS, self.PERSON_YAW, (-0.05, 0, 0))[:2]
        yaw = self.PERSON_YAW
        g.append(f'<body pos="{cx:.3f} {cy:.3f} 0" euler="0 0 {yaw:.4f}">'
                 + box((0, 0, 0.22), (0.24, 0.24, 0.22), "0.3 0.4 0.6 1")
                 + box((-0.26, 0, 0.62), (0.03, 0.24, 0.32), "0.3 0.4 0.6 1")
                 + '<geom type="cylinder" pos="-0.05 0.33 0.12" euler="1.5708 0 0" '
                   'size="0.12 0.02" rgba="0.15 0.15 0.17 1"/>'
                 + '<geom type="cylinder" pos="-0.05 -0.33 0.12" euler="1.5708 0 0" '
                   'size="0.12 0.02" rgba="0.15 0.15 0.17 1"/></body>')
        g.append(box((0.15, -0.3, TABLE_Z + 0.004), (0.12, 0.09, 0.004), "0.9 0.9 0.85 1"))
        g.append('<geom type="cylinder" pos="0.3 -0.05 0.8" size="0.035 0.05" '
                 'rgba="0.3 0.6 0.4 0.7"/>')
        g.append(box((-1.6, 0, 1.4), (0.05, 3, 1.4), "0.86 0.84 0.8 1"))
        g.append(box((-1.54, 0.6, 1.5), (0.01, 0.5, 0.35), "0.4 0.55 0.6 1"))
        person = humanoid("p_", Palette((0.55, 0.42, 0.62), PERSON.skin, PERSON.visor,
                                        (0.95, 0.85, 0.3)))
        return shell("".join(g) + mug("cup") + person + reach_arm("arm_", self.BASE),
                     floor="tiles", sky=((0.85, 0.85, 0.85), (0.6, 0.6, 0.62)))

    def setup(self, model, data) -> None:
        pose_humanoid(model, data, "p_", self.PERSON_POS, euler_quat(yaw=self.PERSON_YAW),
                      {**SITTING, "neck_pitch": 0.45, "l_sh_pitch": -0.55, "r_sh_pitch": -0.55,
                       "l_elbow": -0.9, "r_elbow": -0.9})
        set_free(model, data, "cupfree", (*self.CUP0, TABLE_Z + 0.05), yq(0))

    def grip(self, t: float) -> np.ndarray:
        c0 = np.array([*self.CUP0, TABLE_Z + 0.05 + HOLD])
        n = self.robot_right_target() + [0, 0, HOLD]
        r = self.right_hand_target() + [0, 0, HOLD]
        up = np.array([0, 0, 0.16])
        return track(t, [(0.0, (-0.25, 0.0, 1.2)), (0.8, c0 + up), (1.1, c0), (1.3, c0 + up),
                         (2.1, n + up), (2.4, n), (2.6, n + up), (3.1, n + up), (3.3, n),
                         (3.5, n + up), (4.6, r + up), (4.9, r), (5.15, r + up),
                         (5.9, (-0.25, 0.0, 1.2))])

    def animate(self, model, data, t, answers) -> None:
        g = self.grip(t)
        set_arm(model, data, "arm_", self.BASE, g)
        if held(t, [(1.1, 2.4), (3.3, 4.9)]):
            set_free(model, data, "cupfree", g - [0, 0, HOLD], yq(0))
        elif t >= 4.9:
            set_free(model, data, "cupfree", self.right_hand_target(), yq(0))
        elif t >= 2.4:
            set_free(model, data, "cupfree", self.robot_right_target(), yq(0))

    def camera(self, t):
        f = t / self.duration
        return np.array([0.9 - 0.3 * f, -2.3, 1.9]), np.array([0.05, 0.1, 0.8])

    def beats(self, answers) -> list[Beat]:
        n, r = self.robot_right_target(), self.right_hand_target()
        return [Beat(0.1, 2.6, "say", "Put the cup by my right hand."),
                Beat(2.45, 3.4, "x", pos=(n[0], n[1], n[2] + 0.25)),
                Beat(5.1, 6.8, "ok", pos=(r[0], r[1], r[2] + 0.25))]


# ---- 7. your frame (live grounder, ROV) ----------------------------------------------------


class YourFrame(ReelScene):
    START = (7.4, 0.8, 1.9)
    ROLL = Rov.ROLL
    T = (1.0, 2.3, 2.7, 5.0)

    def __init__(self) -> None:
        super().__init__("yourframe", "Left means your left.", 6.6, robot_cam="rov_cam",
                         fog=True, word="face", real=("geo", "naive"))
        self.candidates = [Candidate("coral_n", "coral", (2.7, 1.5, 0.35)),
                           Candidate("coral_s", "coral", (2.7, -1.5, 0.35))]
        line = "Sample the coral on my left."
        self.queries = [Query("geo", line, "speaker", ("coral_n", "coral_s"), "coral_n"),
                        Query("naive", line, "robot", ("coral_n", "coral_s"), "coral_n")]
        self._c: dict = {}

    def xml(self) -> str:
        return (Rov.xml(self).replace(coral(2.7, 1.5), coral_tree(2.7, 1.5))
                .replace(coral(2.7, -1.5), coral_tree(2.7, -1.5)))

    def _place(self, model, data, pos, yaw) -> None:
        Rov._place(self, model, data, pos, yaw)

    def setup(self, model, data) -> None:
        Rov.setup(self, model, data)

    def goal(self, oid: str) -> np.ndarray:
        c = self.cand(oid).pos
        return np.array([c[0] + 1.25, c[1], 0.75])

    def animate(self, model, data, t, answers) -> None:
        Rov.animate(self, model, data, t, {})  # bubbles and the idle bob only
        wrong, pick = answer_or(answers, "naive", "coral_s"), answer_or(answers, "geo", "coral_n")
        path, tau = two_stage(self._c, self.START, self.goal(wrong), self.goal(pick), t, self.T,
                              frac=0.4, mid=(4.6, 0.0, 1.3))
        pos, _ = path.at(tau)
        self._place(model, data, pos + [0, 0, 0.03 * np.sin(t * 1.3)], np.pi)

    def camera(self, t):
        f = t / self.duration
        return np.array([4.5 + 0.4 * f, -4.9 + 0.4 * f, 2.8]), np.array([2.7, 0.3, 0.6])

    def beats(self, answers) -> list[Beat]:
        b = [Beat(0.1, 2.8, "say", "Sample the coral on my left.")]
        naive, geo = answers.get("naive"), answers.get("geo")
        if naive is not None:
            p = self.cand(naive).pos
            b.append(Beat(2.0, 3.2, "x" if naive != "coral_n" else "ok",
                          pos=(p[0], p[1], 1.3)))
        if geo is not None:
            p = self.cand(geo).pos
            b.append(Beat(5.2, 6.6, "ok" if geo == "coral_n" else "x", pos=(p[0], p[1], 1.4)))
        return b


# ---- 8. moving frame -----------------------------------------------------------------------

LEFT = 0.65  # how far to the person's left the follow robot keeps, m
WALK = (0.3, 6.1)
WALL_C = "0.78 0.76 0.72 1"


class MovingFrame(ReelScene):
    ROUTE: ClassVar[list] = [(-4.4, -0.2), (-1.2, -0.2), (-0.05, 0.0), (0.25, 0.9), (0.25, 4.2)]

    def __init__(self) -> None:
        super().__init__("moving", "It keeps up with you.", 6.8, robot_cam="fb_cam",
                         word="move")
        self.path = Path(self.ROUTE)

    def walker(self, t: float) -> tuple[np.ndarray, float, float]:
        """(position, yaw, arc length walked) of the person at time t."""
        tau = window(t, *WALK)
        pos, tan = self.path.at(tau)
        return pos, float(np.arctan2(tan[1], tan[0])), min_jerk(tau) * self.path.length

    def follower(self, t: float) -> tuple[np.ndarray, float]:
        """The robot's spot: LEFT to the person's left in the person's frame."""
        pos, yaw, _ = self.walker(t)
        return person_point((*pos, 0.0), yaw, (-0.15, LEFT, 0.0)), yaw

    def naive_follower(self, t: float) -> np.ndarray:
        """A robot keeping to a fixed side in its own (starting) frame: +y, whatever the
        person does."""
        pos, _, _ = self.walker(t)
        return np.array([pos[0] - 0.15, pos[1] + LEFT, 0.0])

    def xml(self) -> str:
        g = [box((-1.9, -1.0, 0.5), (3.1, 0.05, 0.5), WALL_C),
             box((1.2, 2.0, 0.5), (0.05, 3.0, 0.5), WALL_C),
             box((-3.0, 1.0, 0.5), (2.0, 0.05, 0.5), WALL_C),
             box((-1.0, 3.0, 0.5), (0.05, 2.0, 0.5), WALL_C),
             box((-1.9, 0.0, 0.003), (3.1, 0.95, 0.003), "0.42 0.48 0.55 1"),
             box((0.1, 2.975, 0.003), (1.05, 2.025, 0.003), "0.42 0.48 0.55 1"),
             box((0.1, 5.0, 0.5), (1.15, 0.05, 0.5), WALL_C),
             box((-5.0, 0.0, 0.5), (0.05, 1.0, 0.5), WALL_C)]
        for x in (-4.0, -2.5):
            g.append(box((x, -0.95, 0.7), (0.4, 0.01, 0.2), "0.35 0.5 0.62 1"))
        g.append(box((1.15, 3.2, 0.7), (0.01, 0.4, 0.2), "0.62 0.45 0.3 1"))
        person = humanoid("p_", Palette((0.2, 0.45, 0.4), PERSON.skin, PERSON.visor,
                                        (0.95, 0.8, 0.3)))
        return shell("".join(g) + person + follow_bot()
                     + ghost(follow_bot("gh_"), 0.3, "0.95 0.35 0.3"), floor="concrete",
                     sky=((0.72, 0.74, 0.78), (0.4, 0.42, 0.46)))

    def setup(self, model, data) -> None:
        self.animate(model, data, 0.0, {})

    def animate(self, model, data, t, answers) -> None:
        pos, yaw, s = self.walker(t)
        moving = WALK[0] < t < WALK[1]
        bob = 0.02 * abs(np.sin(s * 4.5)) if moving else 0.0
        pose_humanoid(model, data, "p_", (pos[0], pos[1], 0.94 + bob), euler_quat(yaw=yaw),
                      {**walk_angles(s * 4.5, 0.45 if moving else 0.0), "neck_pitch": 0.15})
        fp, fyaw = self.follower(t)
        set_free(model, data, "fb_free", fp, yq(fyaw))
        g = self.naive_follower(t) if 3.3 <= t <= 5.2 else np.array([0, 0, -5.0])
        set_free(model, data, "gh_free", g, yq(0.0))

    def trail_points(self, t: float) -> list[np.ndarray]:
        return [self.follower(t - 0.12 * k)[0] + [0, 0, 0.05] for k in range(1, 11)
                if t - 0.12 * k > WALK[0]]

    def camera(self, t):
        pos, _, _ = self.walker(t)
        tgt = np.array([pos[0], pos[1], 0.6]) * 0.75 + np.array([-0.8, 0.8, 0.6]) * 0.25
        return tgt + np.array([-2.0, -3.6, 3.6]), tgt

    def beats(self, answers) -> list[Beat]:
        g = self.naive_follower(4.6)
        f = self.follower(5.8)[0]
        return [Beat(0.1, 2.6, "say", "Stay on my left."),
                Beat(0.4, 6.8, "trail"),
                Beat(4.2, 5.2, "x", pos=(g[0], g[1], 1.0)),
                Beat(5.4, 6.8, "ok", pos=(f[0], f[1], 1.0))]


# ---- 9. habits -----------------------------------------------------------------------------


def torque_wrench(p: str, pos) -> str:
    metal = 'rgba="0.7 0.72 0.76 1" material="metal"'
    return (f'<body name="{p}" pos="{pos[0]} {pos[1]} {pos[2]}"><freejoint name="{p}free"/>'
            f'<geom type="capsule" fromto="-0.2 0 0 0.14 0 0" size="0.014" {metal}/>'
            '<geom type="capsule" fromto="-0.2 0 0 -0.08 0 0" size="0.02" '
            'rgba="0.12 0.12 0.14 1"/>'
            f'<geom type="cylinder" pos="0.17 0 0" size="0.034 0.014" {metal}/>'
            '<geom type="cylinder" pos="0.0 0 0.012" size="0.03 0.006" '
            'rgba="0.95 0.95 0.9 1"/>'
            '<geom type="box" pos="0.0 0 0.019" size="0.025 0.002 0.001" '
            'rgba="0.85 0.15 0.1 1"/></body>')


class Habits(ReelScene):
    BASE = (0.1, 0.0, 0.92)
    PERSON_POS = (1.47, 0.0, 0.94)
    SPOTS: ClassVar[dict[str, tuple[float, float, float]]] = {"combo": (0.62, 0.32, 0.94),
                                                              "torque": (0.62, -0.32, 0.94)}
    HAND = (1.06, 0.0, 1.1)
    RESET = 5.3

    def __init__(self) -> None:
        super().__init__("habits", "It learns how you talk.", 7.8, robot_cam="", word="learn")
        self.candidates = [Candidate(k, "wrench", v) for k, v in self.SPOTS.items()]

    def xml(self) -> str:
        g = [box((0.55, 0, 0.88), (0.75, 0.75, 0.04), "0.62 0.64 0.66 1"),
             box((-1.0, 0, 1.4), (0.04, 2.5, 1.4), "0.85 0.86 0.88 1"),
             box((-0.92, 0.0, 1.55), (0.04, 0.9, 0.35), "0.3 0.4 0.5 1")]
        for x in (-0.15, 1.25):
            for y in (-0.7, 0.7):
                g.append(box((x, y, 0.42), (0.04, 0.04, 0.42), "0.25 0.25 0.27 1"))
        g.append(box((0.3, 0.55, 0.96), (0.12, 0.08, 0.04), "0.2 0.5 0.8 1"))
        g.append('<geom type="cylinder" pos="0.95 0.5 0.98" size="0.04 0.06" '
                 'rgba="0.8 0.8 0.3 0.6"/>')
        person = humanoid("p_", Palette((0.85, 0.86, 0.88), PERSON.skin, PERSON.visor,
                                        (0.2, 0.5, 0.85)))
        return shell("".join(g) + wrench("combo", self.SPOTS["combo"])
                     + torque_wrench("torque", self.SPOTS["torque"]) + person
                     + reach_arm("arm_", self.BASE, "0.2 0.55 0.85 1"), floor="tiles",
                     sky=((0.85, 0.86, 0.88), (0.6, 0.62, 0.65)))

    def setup(self, model, data) -> None:
        self._person(model, data, 0.0)
        for k, v in self.SPOTS.items():
            set_free(model, data, k + "free", v, yq(0.4))

    def _person(self, model, data, t: float) -> None:
        r = self.RESET
        reach = max(window(t, 1.5, 1.9) - window(t, 2.3, 2.7), window(t, 4.5, 4.9)
                    - window(t, r - 0.2, r), window(t, r + 1.5, r + 1.9))
        pose_humanoid(model, data, "p_", self.PERSON_POS, euler_quat(yaw=np.pi),
                      {"neck_pitch": 0.5, "r_sh_pitch": -1.0 * reach, "r_sh_abd": 0.2 * reach,
                       "r_elbow": -0.4})

    def grip(self, t: float) -> np.ndarray:
        c = np.array(self.SPOTS["combo"]) + [0, 0, HOLD]
        q = np.array(self.SPOTS["torque"]) + [0, 0, HOLD]
        up, rest = np.array([0, 0, 0.18]), (0.25, 0.0, 1.3)
        if t < self.RESET:
            return track(t, [(0.0, rest), (0.6, c + up), (0.85, c), (1.05, c + up),
                             (1.9, self.HAND), (2.6, self.HAND), (3.2, c + up), (3.4, c),
                             (3.6, c + up), (4.0, q + up), (4.2, q), (4.4, q + up),
                             (4.9, self.HAND)])
        return track(t, [(self.RESET + 0.25, rest), (self.RESET + 0.75, q + up),
                         (self.RESET + 0.95, q), (self.RESET + 1.15, q + up),
                         (self.RESET + 1.9, self.HAND)])

    def animate(self, model, data, t, answers) -> None:
        self._person(model, data, t)
        g = self.grip(t)
        set_arm(model, data, "arm_", self.BASE, g)
        for k, v in self.SPOTS.items():
            set_free(model, data, k + "free", v, yq(0.4))
        if held(t, [(0.85, 3.4)]):
            set_free(model, data, "combofree", g - [0, 0, HOLD], yq(0.4))
        if held(t, [(4.2, self.RESET), (self.RESET + 0.95, 99)]):
            set_free(model, data, "torquefree", g - [0, 0, HOLD], yq(0.4))

    def camera(self, t):
        f = t / self.duration
        return np.array([1.75 - 0.2 * f, -1.95, 1.8]), np.array([0.7, 0.0, 1.0])

    def beats(self, answers) -> list[Beat]:
        h, q = self.HAND, self.SPOTS["torque"]
        r = self.RESET
        return [Beat(0.1, 1.8, "say", "The usual tool."),
                Beat(2.0, 3.5, "say", "No, the torque wrench."),
                Beat(2.1, 3.2, "x", pos=(h[0], h[1], h[2] + 0.3)),
                Beat(3.9, 4.95, "pulse", "remembered", pos=(q[0], q[1], q[2] + 0.35)),
                Beat(r - 0.2, r + 0.2, "dip"),
                Beat(r + 0.3, r + 1.6, "say", "The usual."),
                Beat(r + 1.9, 7.8, "ok", pos=(h[0], h[1], h[2] + 0.3))]


def scenes() -> list[ReelScene]:
    return [Placement(), YourFrame(), MovingFrame(), Habits()]
