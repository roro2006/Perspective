"""T5-20: the reel's scenes with physical robots. Visibility (wheeled AMR) and your frame (ROV)
ask the live grounder; reach and memory (G1) and pointing (Go2) keep their scripted targets but
walk there under physics. Beat times follow the recorded legs."""

from __future__ import annotations

import mujoco
import numpy as np
from kit import euler_quat, pose_humanoid, set_free
from phys_core import attach
from phys_g1 import G1Walker, g1_spec
from phys_go2 import Go2Walker, go2_spec
from phys_robots import AmrDrive, add_amr_actuators, amr_phys
from phys_run import Leg
from phys_scenes import (
    G1_COLOR,
    G1_FALL,
    GO2_FALL,
    RESCUE_YELLOW,
    Phys,
    PhysHome,
    PhysRov,
    g1_shelf_legs,
    home_floor,
    paint,
    with_gravity,
)
from reel_kit import Beat
from reel_scenes_a import (
    FLASH,
    POINT_POSE,
    WALL_Y,
    Memory,
    Pointing,
    Reach,
    Visibility,
    body_point,
    pointed_door,
    ray_hit_plane_y,
    yq,
)
from reel_scenes_b import YourFrame
from robots import amr, quadruped, robot_humanoid

ATTACH_RADIUS = 0.9  # m from the AMR's centre to a tote it may lift


def toward(start, goal, frac: float) -> tuple:
    s, g = np.asarray(start, float), np.asarray(goal, float)
    return tuple(s + frac * (g - s))


class _Beats:
    """Beat times from the recorded legs."""

    def t_of(self, key: str, end: bool, default: float) -> float:
        return self.leg_time(key, end, default)

    def retime(self, traj) -> None:
        self.traj = traj
        last = max((v[1] for v in traj.legs.values() if v[1] is not None), default=5.0)
        self.duration = float(last) + 1.4


# ---- 1. visibility (live grounder, wheeled AMR) --------------------------------------------


class PhysVisibility(_Beats, Phys, Visibility):
    kind, rprefix, props = "amr", "amr_", ("tote_hfree", "tote_vfree")

    def phys_xml(self) -> str:
        return with_gravity(paint(Visibility.xml(self).replace(amr(), amr_phys())))

    def add_robot(self, spec) -> None:
        add_amr_actuators(spec)

    def make_robot(self, model, data):
        return AmrDrive(model, data)

    def start(self):
        return (*self.START, 0.0, np.pi)

    def legs(self, answers) -> list[Leg]:
        naive, pick = answers.get("naive"), answers.get("geo")
        out = []
        if naive is not None:
            out.append(Leg("naive", [toward(self.START, self.APPROACH[naive], 0.42)], start=1.0))
        if pick is not None:
            goal = self.APPROACH[pick]

            def lift(run, pick=pick):
                tote = run.data.qpos[run.model.joint(pick + "free").qposadr[0]:][:2]
                x, y, _ = run.robot.pose()
                if np.hypot(tote[0] - x, tote[1] - y) <= ATTACH_RADIUS:
                    run.hold(pick + "free", "amr_root", (0.0, 0.0, 0.35), blend=0.6)
            out += [Leg("geo", [goal], np.pi, goal, pause=0.4, start=1.0),
                    Leg("lift", None, wait=0.9, pause=0.1, on_start=lift)]
        return out

    def beats(self, answers) -> list[Beat]:
        b = [Beat(0.1, 2.9, "say", "Bring the tote I can't see.")]
        naive, geo = answers.get("naive"), answers.get("geo")
        n1 = self.t_of("naive", True, 2.3)
        if naive is not None:
            p = self.cand(naive).pos
            b.append(Beat(n1 - 0.3, n1 + 0.9, "x" if naive != "tote_h" else "ok",
                          pos=(p[0], p[1], 0.5)))
        if geo is not None:
            p = self.cand(geo).pos
            t = self.t_of("lift", True, 5.0)
            b.append(Beat(t, self.duration, "ok" if geo == "tote_h" else "x",
                          pos=(p[0], p[1], 0.8)))
        return b


# ---- 7. your frame (live grounder, ROV) ----------------------------------------------------


class PhysYourFrame(_Beats, Phys, YourFrame):
    kind, rprefix = "flyer", "rov_"

    def phys_xml(self) -> str:
        return with_gravity(YourFrame.xml(self))

    def make_robot(self, model, data):
        return PhysRov.make_robot(self, model, data)

    def start(self):
        return (*self.START, np.pi)

    def scenery(self, model, data, t) -> None:
        PhysRov.scenery(self, model, data, t)

    def legs(self, answers) -> list[Leg]:
        naive, pick = answers.get("naive"), answers.get("geo")
        out = []
        if naive is not None:
            out.append(Leg("naive", [toward(self.START, self.goal(naive), 0.4)], start=1.0))
        if pick is not None:
            g = tuple(self.goal(pick))
            out.append(Leg("geo", [(4.6, 0.0, 1.3), g], None, g, dims=3, pause=0.4, start=1.0))
        return out

    def beats(self, answers) -> list[Beat]:
        b = [Beat(0.1, 2.8, "say", "Sample the coral on my left.")]
        naive, geo = answers.get("naive"), answers.get("geo")
        n1 = self.t_of("naive", True, 2.3)
        if naive is not None:
            p = self.cand(naive).pos
            b.append(Beat(n1 - 0.3, n1 + 0.9, "x" if naive != "coral_n" else "ok",
                          pos=(p[0], p[1], 1.3)))
        if geo is not None:
            p = self.cand(geo).pos
            t = self.t_of("geo", True, 5.0)
            b.append(Beat(t, self.duration, "ok" if geo == "coral_n" else "x",
                          pos=(p[0], p[1], 1.4)))
        return b


# ---- 2. reach (scripted target, G1) --------------------------------------------------------


class PhysReach(_Beats, Phys, Reach):
    kind, rprefix, fall_z = "g1", "g1_", G1_FALL

    def __init__(self) -> None:
        super().__init__()
        self.robot_cam = "g1_cam"

    def phys_xml(self) -> str:
        return with_gravity(Reach.xml(self).replace(robot_humanoid(), ""))

    def add_robot(self, spec) -> None:
        attach(spec, g1_spec(G1_COLOR), "pelvis", "g1_")

    def setup(self, model, data) -> None:
        pose_humanoid(model, data, "p_", self.PERSON_POS, euler_quat(yaw=self.PERSON_YAW),
                      {"neck_pitch": 0.2, "neck_yaw": -0.85, "r_sh_pitch": -1.2,
                       "r_sh_abd": -1.35, "r_elbow": -0.1})
        for c in self.candidates:
            set_free(model, data, c.oid + "free", c.pos, yq(0))
        G1Walker(model, data).place(*self.START, 2.9)

    def make_robot(self, model, data):
        return G1Walker(model, data)

    def start(self):
        return (*self.START, 0.0, 2.9)

    def legs(self, answers) -> list[Leg]:
        pick = self.target()
        wrong = "jar_near" if pick == "jar_far" else "jar_far"
        jar = self.cand(pick).pos
        return [Leg("naive", [toward(self.START, self.APPROACH[wrong], 0.42)], start=0.9),
                Leg("geo", [self.APPROACH[pick]], np.pi / 2, self.APPROACH[pick], pause=0.3),
                Leg("reach", None, wait=1.6, pause=0.1,
                    on_start=lambda run: run.robot.reach(jar)),  # reaches toward it, can't hold
                Leg("lower", None, wait=0.8, on_start=lambda run: run.robot.lower_arm())]

    def beats(self, answers) -> list[Beat]:
        pick = self.target()
        wrong = "jar_near" if pick == "jar_far" else "jar_far"
        pw, pp = self.cand(wrong).pos, self.cand(pick).pos
        n1 = self.t_of("naive", True, 2.2)
        return [Beat(0.1, 2.9, "say", "Pass me the one I can't reach."),
                Beat(n1 - 0.4, n1 + 0.9, "x", pos=(pw[0], pw[1], pw[2] + 0.35)),
                Beat(self.t_of("reach", False, 5.2), self.duration, "ok",
                     pos=(pp[0], pp[1], pp[2] + 0.4))]


# ---- 4. pointing (scripted target, Go2) ----------------------------------------------------


class PhysPointing(_Beats, Phys, Pointing):
    kind, rprefix, fall_z = "go2", "go2_", GO2_FALL

    def __init__(self) -> None:
        super().__init__()
        self.robot_cam = "go2_cam"

    def phys_xml(self) -> str:
        return with_gravity(Pointing.xml(self).replace(quadruped(), ""))

    def add_robot(self, spec) -> None:
        attach(spec, go2_spec(color=RESCUE_YELLOW), "base", "go2_")

    def setup(self, model, data) -> None:
        pose_humanoid(model, data, "p_", self.PERSON_POS, euler_quat(yaw=0.0), POINT_POSE)
        Go2Walker(model, data).place(*self.START, 0.0)
        mujoco.mj_forward(model, data)
        eye = data.cam_xpos[model.camera("p_head_cam").id].copy()
        shoulder = data.xpos[model.body("p_l_upper").id].copy()
        tip = body_point(model, data, "p_l_fore", np.array([0, 0, -0.33]))
        self.rays = {"eye": eye, "shoulder": shoulder, "tip": tip,
                     "eye_hit": ray_hit_plane_y(eye, tip, WALL_Y),
                     "arm_hit": ray_hit_plane_y(shoulder, tip, WALL_Y),
                     "pick": pointed_door(eye, tip, self.DOORS, WALL_Y),
                     "naive": pointed_door(shoulder, tip, self.DOORS, WALL_Y)}

    def make_robot(self, model, data):
        return Go2Walker(model, data)

    def start(self):
        return (*self.START, 0.0, 0.0)

    def legs(self, answers) -> list[Leg]:
        pick, wrong = self.rays["pick"], self.rays["naive"]
        gw, gp = (self.DOORS[wrong], WALL_Y - 0.55), (self.DOORS[pick], WALL_Y - 0.55)
        return [Leg("naive", [toward(self.START, gw, 0.5)], start=1.0),
                Leg("geo", [(gp[0], 0.2), gp], None, gp, pause=0.3)]

    def beats(self, answers) -> list[Beat]:
        r = self.rays
        pn, pp = self.cand(r["naive"]).pos, self.cand(r["pick"]).pos
        n1 = self.t_of("naive", True, 2.4)
        return [Beat(0.1, 2.6, "say", "Check that door."),
                Beat(1.0, n1 + 0.6, "badray", pos=tuple(r["shoulder"]), pos2=tuple(r["arm_hit"])),
                Beat(n1 - 0.4, n1 + 0.8, "x", pos=(pn[0], pn[1] - 0.1, 1.9)),
                Beat(n1 + 0.6, self.duration, "ray", pos=tuple(r["eye"]),
                     pos2=tuple(r["eye_hit"])),
                Beat(self.t_of("geo", True, 5.2), self.duration, "ok",
                     pos=(pp[0], pp[1] - 0.1, 1.9))]


# ---- 5. memory (scripted target, G1) -------------------------------------------------------


class PhysMemory(_Beats, Phys, Memory):
    kind, rprefix, fall_z = "g1", "g1_", G1_FALL

    def __init__(self) -> None:
        super().__init__()
        self.robot_cam = "g1_cam"

    def phys_xml(self) -> str:
        return home_floor(Memory.xml(self).replace(robot_humanoid(), ""))

    def add_robot(self, spec) -> None:
        attach(spec, g1_spec(G1_COLOR), "pelvis", "g1_")

    def setup(self, model, data) -> None:
        model.cam_fovy[model.camera("cine").id] = 60.0
        PhysHome.setup(self, model, data)

    def make_robot(self, model, data):
        return G1Walker(model, data)

    def start(self):
        return (*self.START, 0.0, np.pi)

    def scenery(self, model, data, t) -> None:
        mid = model.body("keys").mocapid[0]
        if t < FLASH:
            pose_humanoid(model, data, "p_", (0.6, 1.72, 0.94), euler_quat(yaw=np.pi / 2),
                          {"neck_pitch": 0.45, "r_sh_pitch": -1.1, "r_elbow": -0.5,
                           "l_sh_pitch": -0.2})
            drop = float(np.clip((t - 0.45) / 0.4, 0, 1))
            data.mocap_pos[mid] = np.array(self.KEYS) + [0, -0.12 * (1 - drop),
                                                         0.3 * (1 - drop ** 2)]
            return
        data.mocap_pos[mid] = self.KEYS
        from kit import SITTING
        pose_humanoid(model, data, "p_", (-2.4, 0.0, 0.62), euler_quat(yaw=0.0),
                      {**SITTING, "l_sh_pitch": -1.0, "l_sh_abd": 0.9, "l_elbow": -0.3})

    def legs(self, answers) -> list[Leg]:
        naive = Leg("naive", [toward(self.START, self.GOALS["shelf_s"], 0.45)], start=self.T[0])
        return [naive, *g1_shelf_legs(1.0, [(1.9, 0.6), (0.65, 1.78)], start=0.0)]

    def beats(self, answers) -> list[Beat]:
        s, n = self.cand("shelf_s").pos, self.cand("shelf_n").pos
        n1 = self.t_of("naive", True, FLASH + 1.9)
        return [Beat(0.0, FLASH, "flash"),
                Beat(FLASH + 0.15, FLASH + 2.6, "say", "Where did I put my keys?"),
                Beat(n1 - 0.4, n1 + 0.9, "x", pos=(s[0], s[1], 1.9)),
                Beat(self.t_of("reach", False, FLASH + 4.4), self.duration, "ok",
                     pos=(n[0], n[1], 1.9))]


def reel_live() -> list:
    return [PhysVisibility(), PhysYourFrame()]


def reel_walkers() -> list:
    return [PhysReach(), PhysPointing(), PhysMemory()]
