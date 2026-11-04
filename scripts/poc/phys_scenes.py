"""T5-20: the PoC video's six live-grounder scenes with physically simulated robots.

Each class keeps its T5-18 scene (geometry, people, camera, overlays, grounder queries) and
swaps the scripted robot for a physical one: the home humanoid is the G1, the rescue dog the
Go2, the warehouse AMR has real wheels, and the drone, ROV and space flyer are free bodies
under force PD. legs() turns the grounder's answers into the robot's plan; retime() moves the
scene's beat times onto when the physical robot really got there."""

from __future__ import annotations

import mujoco
import numpy as np
from kit import SITTING, euler_quat, pose_humanoid, set_free
from phys_core import GRAVITY, attach
from phys_g1 import FALL_Z as G1_FALL
from phys_g1 import G1Walker, g1_spec
from phys_go2 import FALL_Z as GO2_FALL
from phys_go2 import Go2Walker, go2_spec
from phys_robots import AmrDrive, Flyer, add_amr_actuators, amr_phys
from phys_run import Leg, Run, Traj
from robots import amr, quadruped, robot_humanoid
from scenes_a import Drone, RescueDog, Warehouse, route
from scenes_b import Home, Rov, Space

NO_GRAVITY = '<option gravity="0 0 0"/>'
# A collision-only ground plane (geom group 3 is not drawn) for scenes whose visible floor is
# a finite box the walker could step off.
HIDDEN_FLOOR = ('<geom name="phys_floor" type="plane" pos="0 0 {z}" size="40 40 0.1" '
                'group="3" rgba="0 0 0 0"/>')
HOME_FLOOR_Z = 0.008  # top of the wooden floor box; the box and rug become visual only
G1_COLOR = (0.82, 0.83, 0.85, 1.0)
RESCUE_YELLOW = (1.0, 0.78, 0.1, 1.0)


def with_gravity(xml: str, g: float = GRAVITY, floor: float | None = None) -> str:
    assert NO_GRAVITY in xml
    xml = xml.replace(NO_GRAVITY, f'<option gravity="0 0 {-g}" timestep="0.002"/>')
    if floor is None:
        return xml
    return xml.replace("</worldbody>", HIDDEN_FLOOR.format(z=floor) + "</worldbody>")


def home_floor(xml: str) -> str:
    """The home's wooden floor and rug are thin boxes with edges a foot can catch; walk on a
    hidden plane level with the wood instead."""
    for rgba in ('rgba="0.6 0.45 0.3 1" material="wood"/>', 'rgba="0.55 0.25 0.22 1" />'):
        assert rgba in xml
        xml = xml.replace(rgba, rgba[:-2] + ' contype="0" conaffinity="0"/>')
    return with_gravity(xml, floor=HOME_FLOOR_Z)


def paint(xml: str) -> str:
    """Floor markings are paint: no collision (they are 4 mm boxes in the T5-18 MJCF)."""
    return xml.replace('rgba="0.95 0.8 0.1 1" />', 'rgba="0.95 0.8 0.1 1" contype="0" '
                       'conaffinity="0"/>')


class Phys:
    """Mixin for a scene with a physical robot; listed first so its methods win."""

    kind = ""  # g1, go2, amr or flyer
    rprefix = ""
    props: tuple[str, ...] = ()  # free joints left to physics (not scripted)
    fall_z = 0.0
    traj: Traj | None = None

    def phys_xml(self) -> str:
        raise NotImplementedError

    def add_robot(self, spec: mujoco.MjSpec) -> None:
        """Attach a robot that isn't an MJCF fragment (G1, Go2) or its actuators."""

    def spec(self) -> mujoco.MjSpec:
        s = mujoco.MjSpec.from_string(self.phys_xml())
        self.add_robot(s)
        return s

    def make_robot(self, model, data):
        raise NotImplementedError

    def start(self) -> tuple[float, float, float, float]:
        raise NotImplementedError

    def legs(self, answers: dict) -> list[Leg]:
        raise NotImplementedError

    def begin(self, run: Run) -> None:
        """Called once the robot is placed (attach props it starts holding)."""

    def scenery(self, model, data, t: float) -> None:
        """Scripted, time-varying scenery (bubbles, floating tools)."""

    def leg_error(self, run: Run, leg: Leg) -> float:
        if leg.key == "reach":  # the G1's right palm against the reach point
            return float(np.linalg.norm(run.robot.palm_pos() - np.asarray(leg.goal, float)))
        if self.kind == "flyer":
            return float(np.linalg.norm(run.robot.position() - np.asarray(leg.goal, float)))
        x, y, _ = run.robot.pose()
        return float(np.hypot(x - leg.goal[0], y - leg.goal[1]))

    def animate(self, model, data, t, answers) -> None:
        if self.traj is None:
            self.scenery(model, data, t)
            return
        i = self.traj.frame(t)
        data.qpos[:] = self.traj.qpos[i]
        data.mocap_pos[:] = self.traj.mocap[i]

    def retime(self, traj: Traj) -> None:
        """Move T and duration onto the recorded leg times (default: nothing)."""

    def leg_time(self, key: str, end: bool, default: float) -> float:
        span = self.traj.legs.get(key) if self.traj else None
        if not span:
            return default
        v = span[1 if end else 0]
        return default if v is None else float(v)


# ---- 1. warehouse (wheeled AMR) ------------------------------------------------------------


class PhysWarehouse(Phys, Warehouse):
    kind, rprefix = "amr", "amr_"

    def phys_xml(self) -> str:
        return with_gravity(paint(Warehouse.xml(self).replace(amr(), amr_phys())))

    def add_robot(self, spec) -> None:
        add_amr_actuators(spec)

    def make_robot(self, model, data):
        return AmrDrive(model, data)

    def start(self):
        return (*self.START, 0.0, np.pi)

    def legs(self, answers) -> list[Leg]:
        out = []
        naive, geo, geo2 = answers.get("naive"), answers.get("geo"), answers.get("geo2")
        sx = self.START[0]
        if naive is not None:  # toward the robot-frame pick; it stops partway when corrected
            g = self.GOALS[naive]
            stop = (sx + 0.45 * (g[0] - sx), 0.6)
            out.append(Leg("naive", [stop], start=1.0))
        for key, ans, pause in (("geo", geo, 1.0), ("geo2", geo2, 1.4)):
            if ans is None:
                continue
            g = self.GOALS[ans]

            def lane(robot, g=g):
                x, y, _ = robot.pose()
                return route((x, y), g, 0.62)[1:]
            out.append(Leg(key, lane, np.sign(g[1]) * np.pi / 2, g, pause=pause, start=1.0))
        return out

    def retime(self, traj) -> None:
        self.traj = traj
        T = dict(self.T)
        T["n0"], T["n1"] = self.leg_time("naive", False, 1.0), self.leg_time("naive", True, 2.9)
        T["x"] = T["n1"] - 0.2
        T["r0"], T["r1"] = self.leg_time("geo", False, 3.9), self.leg_time("geo", True, 6.2)
        T["s2"] = T["r1"] + 0.4
        T["m0"], T["m1"] = self.leg_time("geo2", False, 7.6), self.leg_time("geo2", True, 9.8)
        self.T = T
        self.duration = T["m1"] + 1.2


# ---- 2. rescue dog (Go2) -------------------------------------------------------------------


class PhysDog(Phys, RescueDog):
    kind, rprefix, fall_z = "go2", "go2_", GO2_FALL

    def __init__(self) -> None:
        super().__init__()
        self.robot_cam = "go2_cam"

    def phys_xml(self) -> str:
        return with_gravity(RescueDog.xml(self).replace(quadruped(), ""))

    def add_robot(self, spec) -> None:
        attach(spec, go2_spec(color=RESCUE_YELLOW), "base", "go2_")

    def camera(self, t):
        """Closer than T5-18's (the Go2 is small): follow the dog, half-way to the rooms."""
        if self.traj is None:
            return RescueDog.camera(self, t)
        tgt = 0.55 * self.traj.smooth_base(t) + 0.45 * np.array([3.0, 0.8, 0.0])
        tgt[2] = 0.2
        return tgt + np.array([1.4, -3.6, 7.2]), tgt

    def setup(self, model, data) -> None:
        pose_humanoid(model, data, "p_", (-1.2, 0.0, 0.94), euler_quat(yaw=0.0),
                      {"neck_pitch": 0.05, "l_sh_pitch": -0.5, "l_sh_abd": 1.45, "l_elbow": -0.1})
        pose_humanoid(model, data, "s_", (2.4, 2.9, 0.15), euler_quat(pitch=-np.pi / 2,
                                                                       yaw=np.pi * 0.85),
                      {"l_sh_abd": 0.5, "r_sh_abd": -0.9, "l_knee": 0.4})
        Go2Walker(model, data).place(*self.START, np.pi)

    def make_robot(self, model, data):
        return Go2Walker(model, data)

    def start(self):
        return (*self.START, 0.0, np.pi)

    def legs(self, answers) -> list[Leg]:
        a = answers.get("geo")
        if a is None:
            return []
        s = np.sign(self.cand(a).pos[1])
        goal = (2.25, 2.3 * s)
        pts = [(5.0, 0.0), (2.8, 0.0), (2.0, 0.6 * s), (2.05, 1.7 * s), goal]
        return [Leg("geo", pts, None, goal, start=1.0)]

    def retime(self, traj) -> None:
        self.traj = traj
        self.T = {"m0": self.leg_time("geo", False, 1.0), "m1": self.leg_time("geo", True, 4.8)}
        self.duration = self.T["m1"] + 1.4


# ---- 3. delivery drone (thrust, gravity) ---------------------------------------------------


class PhysDrone(Phys, Drone):
    kind, rprefix, props = "flyer", "drone_", ("parcel_free",)

    def phys_xml(self) -> str:
        return with_gravity(Drone.xml(self))

    def make_robot(self, model, data):
        return Flyer(model, data, "drone_", "thrust", yaw=np.pi, speed=1.8, wn=2.2)

    def start(self):
        return (*self.START, np.pi)

    def begin(self, run) -> None:
        run.hold("parcel_free", "drone_root", (0.0, 0.0, -0.21))

    def legs(self, answers) -> list[Leg]:
        a = answers.get("geo")
        if a is None:
            return []
        c = self.cand(a).pos
        self._drop_xy = np.array(c[:2])
        goal = (c[0], c[1], 1.25)

        def drop(run):
            v = run.robot.free.dofadr[0]
            run.release("parcel_free", run.data.qvel[v:v + 3].copy())
        return [Leg("geo", [(4.6, 0.6 * np.sign(c[1]), 2.6), goal], None, goal, dims=3,
                    start=0.9),
                Leg("drop", None, wait=1.0, pause=0.3, on_start=drop),
                Leg("rise", [(c[0] + 0.6, c[1], 2.15)], pause=0.1)]

    def retime(self, traj) -> None:
        self.traj = traj
        T = {"m0": self.leg_time("geo", False, 0.9), "m1": self.leg_time("geo", True, 3.9)}
        T["d0"] = self.leg_time("drop", False, 4.0)
        T["d1"] = T["d0"] + 0.8
        T["u0"], T["u1"] = self.leg_time("rise", False, 4.9), self.leg_time("rise", True, 6.0)
        self.T = T
        self.duration = max(T["u1"], T["d1"] + 1.0) + 0.4


# ---- 4. home humanoid (G1) -----------------------------------------------------------------

REACH_FWD, REACH_SIDE = 0.27, 0.13  # palm target relative to the pelvis when reaching
HOME_REACH = {1: (0.65, 2.30, 1.02), -1: (0.65, -2.30, 1.02)}  # in front of each shelf


def creep_point(reach, heading: float) -> tuple[float, float]:
    """Where the G1's pelvis should stand so its right hand can reach `reach` ahead."""
    fwd = np.array([np.cos(heading), np.sin(heading)])
    right = np.array([np.sin(heading), -np.cos(heading)])
    p = np.asarray(reach[:2]) - REACH_FWD * fwd - REACH_SIDE * right
    return float(p[0]), float(p[1])


class PhysHome(Phys, Home):
    kind, rprefix, fall_z = "g1", "g1_", G1_FALL

    def __init__(self) -> None:
        super().__init__()
        self.robot_cam = "g1_cam"

    def phys_xml(self) -> str:
        return home_floor(Home.xml(self).replace(robot_humanoid(), ""))

    def add_robot(self, spec) -> None:
        attach(spec, g1_spec(G1_COLOR), "pelvis", "g1_")

    def setup(self, model, data) -> None:
        pose_humanoid(model, data, "p_", (-2.4, 0.0, 0.62), euler_quat(yaw=0.0),
                      {**SITTING, "l_sh_pitch": -1.0, "l_sh_abd": 0.9, "l_elbow": -0.3})
        G1Walker(model, data).place(*self.START, np.pi)

    def make_robot(self, model, data):
        return G1Walker(model, data)

    def start(self):
        return (*self.START, 0.0, np.pi)

    def legs(self, answers) -> list[Leg]:
        a = answers.get("geo")
        if a is None:
            return []
        return g1_shelf_legs(np.sign(self.cand(a).pos[1]),
                             [(2.6, 0.3), (1.2, 1.2), (0.65, 1.78)], start=1.0)

    def retime(self, traj) -> None:
        self.traj = traj
        T = {"m0": self.leg_time("geo", False, 1.0), "m1": self.leg_time("geo", True, 4.7)}
        T["r0"], T["r1"] = self.leg_time("reach", False, 4.5), self.leg_time("reach", True, 5.3)
        self.T = T
        self.duration = T["r1"] + 0.8


def g1_shelf_legs(s: float, via: list, start: float, key: str = "geo") -> list[Leg]:
    """Walk to the shelf's stand-off point (scored), step in to arm's length, reach in front
    of the keys (scored to 0.10 m), lower the arm."""
    heading = s * np.pi / 2
    pts = [(x, y * s) for x, y in via]
    reach = HOME_REACH[int(s)]
    return [Leg(key, pts, heading, pts[-1], start=start),
            Leg("step_in", [creep_point(reach, heading)], heading, pause=0.2),
            Leg("reach", None, wait=1.8, pause=0.1, goal=reach, dims=3,
                on_start=lambda run: run.robot.reach(reach)),
            Leg("lower", None, wait=0.8, pause=0.0, on_start=lambda run: run.robot.lower_arm())]


# ---- 5. underwater ROV (buoyant, drag) -----------------------------------------------------


class PhysRov(Phys, Rov):
    kind, rprefix = "flyer", "rov_"

    def phys_xml(self) -> str:
        return with_gravity(Rov.xml(self))

    def make_robot(self, model, data):
        return Flyer(model, data, "rov_", "holonomic", yaw=np.pi, speed=1.2, wn=1.8,
                     amax=2.5, buoyant=True, drag_lin=0.6, drag_quad=0.4)

    def start(self):
        return (*self.START, np.pi)

    def scenery(self, model, data, t) -> None:
        for k in range(8):
            ph = (t * 0.5 + k / 8) % 1.0
            mid = model.body(f"bubble{k}").mocapid[0]
            data.mocap_pos[mid] = (0.1 + 0.05 * np.sin(k + 3 * ph), -0.25 + 0.04 * k % 0.1,
                                   2.0 + 2.5 * ph)

    def legs(self, answers) -> list[Leg]:
        a = answers.get("geo")
        if a is None:
            return []
        c = np.array(self.cand(a).pos)
        goal = (c[0] + 1.2, c[1], 0.7)
        return [Leg("geo", [(4.8, 0.7 * np.sign(c[1]), 1.3), goal], None, goal, dims=3,
                    start=1.0)]

    def retime(self, traj) -> None:
        self.traj = traj
        self.T = {"m0": self.leg_time("geo", False, 1.0), "m1": self.leg_time("geo", True, 4.6),
                  "a0": 0.0, "a1": 0.0}
        self.duration = self.T["m1"] + 1.2


# ---- 6. space free-flyer (no gravity, no drag) ---------------------------------------------


class PhysSpace(Phys, Space):
    kind, rprefix = "flyer", "bee_"

    def phys_xml(self) -> str:
        return Space.xml(self).replace(NO_GRAVITY, '<option gravity="0 0 0" timestep="0.002"/>')

    def make_robot(self, model, data):
        return Flyer(model, data, "bee_", "holonomic", yaw=np.pi, speed=0.9, wn=2.0, amax=1.5)

    def start(self):
        return (*self.START, np.pi)

    def scenery(self, model, data, t) -> None:
        for name in ("wrench_u", "wrench_d"):
            c = np.array(self.cand(name).pos)
            set_free(model, data, name + "free", c + [0, 0, 0.02 * np.sin(t + c[2])],
                     euler_quat(roll=0.3 * t, yaw=0.6 + 0.1 * t))

    def legs(self, answers) -> list[Leg]:
        a = answers.get("geo")
        if a is None:
            return []
        c = np.array(self.cand(a).pos)
        goal = (c[0] + 0.42, c[1], c[2])
        return [Leg("geo", [(2.6, 0.15 * np.sign(c[1]), 0.3 * np.sign(c[2])), goal], None, goal,
                    dims=3, start=1.0)]

    def retime(self, traj) -> None:
        self.traj = traj
        self.T = {"m0": self.leg_time("geo", False, 1.0), "m1": self.leg_time("geo", True, 4.6)}
        self.duration = self.T["m1"] + 1.2


def poc_scenes() -> list:
    return [PhysWarehouse(), PhysDog(), PhysDrone(), PhysHome(), PhysRov(), PhysSpace()]
