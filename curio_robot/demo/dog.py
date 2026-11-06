"""The rescue-dog demo (T5-34): a firefighter's words send the physical Go2 to a room.

    uv run python -m curio_robot.demo.viewer --robot dog

The scene is T5-20's rescue corridor (scripts/poc/phys_scenes.PhysDog). The firefighter stands
at the corridor's far end facing down it; the dog comes in from the other end, facing them, so
"the room on my left" is the dog's right. The dog can't see the firefighter through the smoke.

Curio's part is only the choice of room: the two rooms come from the building's floor map (not
from a detector: rooms aren't things a camera boxes), and `resolver.resolve` ranks them from the
firefighter's head-camera pose (Curio on) or the dog's own camera pose (Curio off). There is no
VLM on this laptop, and the overlay says so. The motion is T5-20's: our pure-pursuit path
follower through the scene's door waypoints, commanding the downloaded cagataydev Isaac Lab
walking policy (CC-BY-4.0). T5-32's navigation policy is not used; it failed its port.

Physics runs live: the people and scenery are kinematic (put back after every step, as in
T5-20's rollouts) and the dog is plain MuJoCo dynamics under its policy."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from curio_robot.resolver import parse_request, resolve
from curio_robot.types import CameraPose, ObjectMap, Target, WorldObject

POC = Path(__file__).resolve().parents[2] / "scripts" / "poc"
HEAD_CAM, DOG_CAM = "p_head_cam", "go2_cam"
FF_Z = 0.94
ARRIVE_TOL = 0.5  # the gate's end-to-end success radius
CREDIT = "walking policy: cagataydev / strands-robots Isaac Lab showcase (CC-BY-4.0)"
HIDDEN = (-30.0, -30.0, 0.15)  # where the survivor waits when the words have no truth


def _poc() -> None:
    """The Go2 port lives with the other T5-20 scripts; import it the way their tests do."""
    for p in (str(POC), str(POC.parent)):
        if p not in sys.path:
            sys.path.insert(0, p)


def have_assets() -> bool:
    _poc()
    from phys_core import GO2_NPZ, GO2_XML

    return GO2_XML.exists() and GO2_NPZ.exists()


def left_of(yaw: float) -> np.ndarray:
    return np.array([-np.sin(yaw), np.cos(yaw)])


def cam_pose(model, data, name: str) -> CameraPose:
    i = model.camera(name).id
    return CameraPose(data.cam_xpos[i].copy(), data.cam_xmat[i].reshape(3, 3).copy())


class DogWorld:
    """One building layout (dog_scenes), its Go2 and the kinematic people. One owner only."""

    def __init__(self, kind: str = "rescue", seed: int = 0) -> None:
        _poc()
        from kit import euler_quat, pose_humanoid
        from phys_go2 import FALL_Z, Go2Walker

        from curio_robot.demo import dog_scenes

        self._euler, self._pose = euler_quat, pose_humanoid
        self.layout = dog_scenes.make(kind, seed)
        self.kind, self.seed = kind, seed
        self.model = self.layout.spec().compile()
        self.data = mujoco.MjData(self.model)
        self.dog = Go2Walker(self.model, self.data)
        self.fall_z = FALL_Z
        self.rooms = self.layout.room_xyz()
        self._kinematic()
        self.facing = float(self.layout.ff[2])
        self.fell = False
        self.min_z = 9.0
        self.sim_t = 0.0
        self.goal: np.ndarray | None = None
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        self.home()

    @property
    def request(self) -> str:
        return self.layout.request

    @property
    def ff_xy(self) -> np.ndarray:
        return np.array(self.layout.ff[:2], float)

    def _kinematic(self) -> None:
        """qpos/qvel indices of every joint that isn't the dog's (phys_run.Run._kin)."""
        m = self.model
        size = {mujoco.mjtJoint.mjJNT_FREE: (7, 6), mujoco.mjtJoint.mjJNT_BALL: (4, 3)}
        qa, va = [], []
        for j in range(m.njnt):
            if m.joint(j).name.startswith("go2_"):
                continue
            nq, nv = size.get(m.jnt_type[j], (1, 1))
            qa += range(m.jnt_qposadr[j], m.jnt_qposadr[j] + nq)
            va += range(m.jnt_dofadr[j], m.jnt_dofadr[j] + nv)
        self.kq, self.kv = np.array(qa, int), np.array(va, int)

    # ---- layout ------------------------------------------------------------------------------

    def home(self, seed: int | None = None) -> None:
        """The layout's start: the dog at the corridor's east end (jittered as T5-20's gate
        runs when a seed is given), the firefighter as they now face."""
        from phys_run import JOINT_JITTER, START_XY, START_YAW

        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        x, y, yaw = self.layout.start
        rng = None
        if seed is not None:
            rng = np.random.default_rng(seed)
            x += rng.uniform(-START_XY, START_XY)
            y += rng.uniform(-START_XY, START_XY)
            yaw += rng.uniform(-START_YAW, START_YAW)
        self.dog.reset()
        self.dog.place(x, y, yaw, rng=rng, jitter=JOINT_JITTER if rng is not None else 0.0)
        self._place_people()
        mujoco.mj_forward(m, d)
        self._saved = d.qpos[self.kq].copy()
        self.fell, self.min_z, self.sim_t, self.goal = False, 9.0, 0.0, None

    def _place_people(self) -> None:
        m, d = self.model, self.data
        fx, fy, _ = self.layout.ff
        self._pose(m, d, "p_", (fx, fy, FF_Z), self._euler(yaw=self.facing),
                   self.layout.ff_angles)
        # The survivor is in the room the scene's own words mean (the story's truth).
        rid, _ = oracle_room(self.request, self)
        pos, yaw = self.layout.survivor(rid) if rid else (HIDDEN, 0.0)
        self._pose(m, d, "s_", pos, self._euler(pitch=-np.pi / 2, yaw=yaw),
                   {"l_sh_abd": 0.5, "r_sh_abd": -0.9, "l_knee": 0.4})

    def turn_firefighter(self) -> None:
        """Face the other way: left and right swap for the same words."""
        self.facing = float((self.facing + 2 * np.pi) % (2 * np.pi) - np.pi)
        self.home()

    @property
    def turned(self) -> bool:
        return bool(np.cos(self.facing - self.layout.ff[2]) < 0)

    # ---- live physics ------------------------------------------------------------------------

    def go(self, room: str) -> np.ndarray:
        """Start the walk to `room` along the layout's route (door waypoints); returns its
        goal."""
        pts, goal = self.layout.route(room)
        self.dog.go(pts, None)
        self.goal = np.asarray(goal, float)
        return self.goal

    @property
    def walking(self) -> bool:
        return self.goal is not None and not self.dog.arrived and not self.fell

    def step(self, seconds: float) -> None:
        m, d = self.model, self.data
        for _ in range(max(1, round(seconds / m.opt.timestep))):
            self.dog.control()
            mujoco.mj_step(m, d)
            d.qpos[self.kq] = self._saved
            d.qvel[self.kv] = 0
            self.dog.after_step()
            z = self.dog.height()
            self.min_z = min(self.min_z, z)
            if z < self.fall_z:
                self.fell = True
            self.sim_t += m.opt.timestep

    def dog_xy(self) -> np.ndarray:
        x, y, _ = self.dog.pose()
        return np.array([x, y])

    def room_goal(self, room: str) -> np.ndarray:
        return np.asarray(self.layout.route(room)[1], float)

    # ---- cameras -----------------------------------------------------------------------------

    def render(self, cam: str, width: int = 480, height: int = 360) -> Image.Image:
        r = self._renderers.get((width, height))
        if r is None:
            r = self._renderers[(width, height)] = mujoco.Renderer(self.model, height, width)
        r.update_scene(self.data, camera=cam)
        return Image.fromarray(r.render())

    def close(self) -> None:
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()


# ---- the decision ------------------------------------------------------------------------------


@dataclass
class DogDecision:
    text: str
    curio: bool
    object_map: ObjectMap
    target: Target
    oracle_id: str | None
    oracle_why: str
    verdict: str
    resolve_ms: float
    parsed: dict = field(default_factory=dict)


def room_map(world: DogWorld, curio: bool) -> ObjectMap:
    """The floor map's rooms, with the dog's camera pose and (Curio on) the firefighter's."""
    m, d = world.model, world.data
    rooms = [WorldObject(rid, world.layout.color_of(rid), pos.copy(), {"map"})
             for rid, pos in world.rooms.items()]
    return ObjectMap(rooms, cam_pose(m, d, DOG_CAM), cam_pose(m, d, HEAD_CAM) if curio else None)


def oracle_room(text: str, world: DogWorld) -> tuple[str | None, str]:
    """The room the words mean, from the simulator's truth: the rooms of the named colour,
    ranked from the speaker's (or for "your", the dog's) true position and body yaw. Words with
    no relation, or naming no room that exists, have no truth here ("unknown")."""
    req = parse_request(text)
    rooms = {rid: p for rid, p in world.rooms.items()
             if req.color is None or world.layout.color_of(rid) == req.color}
    if not rooms or req.relation is None:
        return None, "unknown (no colour match or no relation in the words)"
    if req.speaker_frame:
        origin, yaw, whose = world.ff_xy, world.facing, "firefighter's"
    else:
        x, y, yaw = world.dog.pose()
        origin, whose = np.array([x, y]), "dog's"
    if req.relation in ("left", "right"):
        score = {r: float((p[:2] - origin) @ left_of(yaw)) for r, p in rooms.items()}
        what = f"true {req.relation}"
    else:
        score = {r: -float(np.linalg.norm(p[:2] - origin)) for r, p in rooms.items()}
        what = req.relation
    best = max if req.relation in ("left", "nearest") else min
    pick = best(score, key=score.get)
    return pick, f"{pick} (the {whose} {what})"


def decide(world: DogWorld, text: str, curio: bool) -> DogDecision:
    om = room_map(world, curio)
    t0 = time.perf_counter()
    target = resolve(text, om)
    ms = (time.perf_counter() - t0) * 1000
    oid, why = oracle_room(text, world)
    if target.object_id is None:
        verdict = "not_found"
    elif oid is None:
        verdict = "unknown"
    else:
        verdict = "correct" if target.object_id == oid else "wrong"
    req = parse_request(text)
    parsed = {"color": req.color, "relation": req.relation,
              "frame": "speaker" if req.speaker_frame else "listener"}
    return DogDecision(text, curio, om, target, oid, why, verdict, round(ms, 3), parsed)


def arrived_at(world: DogWorld, room: str) -> bool:
    """The gates' end-to-end test: within ARRIVE_TOL of the room's goal and nearer to it than
    to any other room's."""
    xy = world.dog_xy()
    dists = {n: float(np.linalg.norm(xy - world.room_goal(n))) for n in world.rooms}
    return dists[room] <= ARRIVE_TOL and dists[room] == min(dists.values())


# ---- the viewer --------------------------------------------------------------------------------

INSET_W, INSET_H = 400, 300
DOG_INSET_PERIOD = 1.0  # s between dog-camera refreshes while it walks (a render is ~0.2 s)
CATCH_UP_S = 0.4  # most simulated time made up in one frame after a slow one (a render)
PAUSE_S = 1.5
# Showcase scenes, picked from the T5-35 gate by its rule (Curio off wrong, on right, walked).
SHOWCASE: tuple[tuple[str, int], ...] = (
    ("rescue", 0), ("corridor", 0), ("doorway", 0), ("closest", 0),
    ("corridor", 1), ("doorway", 1), ("closest", 1))
ROOM_MARK = 0.28  # m, the floor discs at the chosen and the true room's goal

HELP = """Type a request and press Enter, e.g.  The red room on my left
  (Enter alone)   the scene's own request
  on / off        Curio on (dog + firefighter's head camera) or off (dog's camera only)
  c               compare: the same words with Curio off, then on
  turn            the firefighter turns round: left and right swap
  n               next showcase scene
  scene K S       load scene kind K (corridor, doorway, closest, rescue) with seed S
                  (the window reopens on the new building)
  r               reset        h  this help        q  quit"""


def setup_camera(v, world: DogWorld) -> None:
    ov = world.layout.overview()
    v.cam.lookat[:] = ov["lookat"]
    v.cam.distance = ov["distance"]
    v.cam.azimuth, v.cam.elevation = ov["azimuth"], ov["elevation"]


class DogDemo:
    """The rescue demo in MuJoCo's viewer; same interface as viewer.ViewerDemo."""

    HELP = HELP
    VOICE_EXAMPLE = "Curio, the red room on my left"

    def __init__(self, viewer, world: DogWorld) -> None:
        from curio_robot.demo.viewer import VERDICT

        self.v, self.world = viewer, world
        self.verdict_names = VERDICT
        self.curio = True
        self.last_text = ""
        self.decision: DogDecision | None = None
        self.compare: dict[bool, DogDecision] = {}
        self.pending: list[tuple[str, bool]] = []
        self.pause_until = 0.0
        self.status = "ready"
        self.result_line = "-"
        self.say = lambda text: None
        self.voice_line = "off (terminal only)"
        self._head: Image.Image | None = None
        self._dog: Image.Image | None = None
        self._full = True
        self._next_inset = 0.0
        self._placed_for: tuple[int, int] | None = None
        self._texts_shown: list | None = None
        self._walk_wall = 0.0
        self.rate: float | None = None  # simulated / wall time over the last walk
        self.paced = True  # False only in the gate's capacity run
        self.switch_to: tuple[str, int] | None = None  # a new building: the viewer reopens
        self.showcase_i = -1

    # ---- scene and runs -------------------------------------------------------------------

    def load(self) -> None:
        with self.v.lock():
            self.world.home()
        self.decision, self.compare, self.pending = None, {}, []
        self.status, self.result_line = "ready", "-"
        self._full = True
        w = self.world
        print(f"\nScene {w.kind} {w.seed}: {len(w.rooms)} rooms. The firefighter "
              f'{self.facing_name()}. Its request: "{w.request}"', flush=True)

    def facing_name(self) -> str:
        yaw = self.world.facing
        way = {0: "faces east", 1: "faces north", 2: "faces west", 3: "faces south"}
        name = way[round(yaw / (np.pi / 2)) % 4]
        return name + (" (turned round)" if self.world.turned else "")

    @property
    def busy(self) -> bool:
        return self.world.walking or bool(self.pending)

    def run(self, text: str, curio: bool) -> None:
        text = text.strip()[:200] or self.world.request
        self.last_text, self.curio = text, curio
        with self.v.lock():
            self.world.home()  # every request starts the dog from the corridor's end
            d = decide(self.world, text, curio)
            if d.target.object_id is not None:
                self.world.go(d.target.object_id)
        self.decision = d
        t = d.target
        print(f"\n[{'Curio ON ' if curio else 'Curio OFF'}] \"{text}\"\n"
              f"  parsed   relation={d.parsed['relation']} frame={d.parsed['frame']}\n"
              f"  target   {t.object_id}  ({t.reason}; rooms from the floor map)\n"
              f"  truth    {d.oracle_why}\n"
              f"  decision {self.verdict_names[d.verdict]}   resolve {d.resolve_ms} ms",
              flush=True)
        self._full = True
        self._walk_wall = time.perf_counter()
        if t.object_id is None:
            self.status = "no room chosen"
            self.say("I don't know which room.")
            return
        self.status = f"walking to {t.object_id}"
        color = self.world.layout.color_of(t.object_id)
        room = f"the {color} room" if color != "grey" else (
            "the north room" if self.world.rooms[t.object_id][1] > 0 else "the south room")
        self.say(f"Going to {room}." if curio else f"Dog's view only. {room.capitalize()}.")

    def start_compare(self) -> None:
        text = self.last_text or self.world.request
        self.compare = {}
        self.pending = [(text, False), (text, True)]

    def tick(self, dt: float) -> None:
        """The dog keeps to the wall clock: step whatever simulated time it owes since the walk
        began (up to CATCH_UP_S after a slow frame, so a long stall shows as slower, not as a
        jump). `dt` is the loop's own and is not needed."""
        if self.world.walking:
            owed = (time.perf_counter() - self._walk_wall) - self.world.sim_t
            if not self.paced:  # the gate's capacity test: as fast as the loop allows
                owed = CATCH_UP_S
            if owed >= self.world.model.opt.timestep:
                with self.v.lock():
                    self.world.step(min(owed, CATCH_UP_S))
            if not self.world.walking:
                self._finish()
        elif self.pending and time.perf_counter() >= self.pause_until:
            text, curio = self.pending.pop(0)
            self.run(text, curio)

    def _finish(self) -> None:
        d, w = self.decision, self.world
        wall = time.perf_counter() - self._walk_wall
        self.rate = w.sim_t / wall if wall > 0 else None
        if w.fell:
            self.status, self.result_line = "fell", "the dog fell (walking policy)"
        else:
            err = float(np.linalg.norm(w.dog_xy() - w.room_goal(d.target.object_id)))
            self.status = f"stopped {err:.2f} m from {d.target.object_id}'s goal"
            if d.oracle_id is None:
                self.result_line = "truth unknown for these words"
                self.say("In the room.")
            elif arrived_at(w, d.oracle_id):
                self.result_line = "in the room the firefighter meant"
                self.say("In the room. Searching.")
            else:
                self.result_line = "in the wrong room"
                self.say("In the room. Nobody here.")
        self.compare[d.curio] = d
        print(f"  walk     {self.status} in {w.sim_t:.1f} s simulated, "
              f"{self.rate or 0:.1f}x real time; {self.result_line}", flush=True)
        self.pause_until = time.perf_counter() + PAUSE_S if self.pending else 0.0
        self._full = True

    def turn(self) -> None:
        with self.v.lock():
            self.world.turn_firefighter()
        self.decision, self.compare = None, {}
        self.status, self.result_line = "ready", "-"
        self._full = True
        print(f"\nThe firefighter now {self.facing_name()}. Same words, their left and right "
              "have swapped: press Enter.", flush=True)
        self.say("Turned around.")

    # ---- drawing --------------------------------------------------------------------------

    def draw(self, now: float) -> None:
        from curio_robot.demo.viewer import GREEN, MAGENTA, _marker

        scn = self.v.user_scn
        scn.ngeom = 0
        d = self.decision
        if d is not None and d.oracle_id is not None:
            _marker(scn, self.world.room_goal(d.oracle_id), ROOM_MARK, GREEN)
        if d is not None and d.target.object_id is not None:
            _marker(scn, self.world.room_goal(d.target.object_id), ROOM_MARK * 0.62, MAGENTA)
        if self.world.walking:  # follow the dog, half-way to the rooms (T5-20's camera)
            mid = np.asarray(self.world.layout.overview()["lookat"], float)
            tgt = 0.55 * np.array([*self.world.dog_xy(), 0.0]) + 0.45 * mid
            self.v.cam.lookat[:] = 0.9 * np.asarray(self.v.cam.lookat) + 0.1 * tgt
        if self._full:
            self._full = False
            self._next_inset = now + DOG_INSET_PERIOD
            self._insets(head=True)
        elif self.world.walking and now >= self._next_inset:
            self._next_inset = now + DOG_INSET_PERIOD
            self._insets(head=False)
        vp = self.v.viewport
        if vp is not None and (vp.width, vp.height) != self._placed_for:
            self._place()
        self._texts()

    def _insets(self, head: bool) -> None:
        from curio_robot.demo.viewer import _label

        used = self.curio or self.decision is None
        if head or self._head is None:
            img = self.world.render(HEAD_CAM, INSET_W, INSET_H)
            _label(img, "FIREFIGHTER  head camera", (80, 200, 235), "" if used else "NOT USED")
            self._head = img
        img = self.world.render(DOG_CAM, INSET_W, INSET_H)
        _label(img, "DOG  its camera", (255, 170, 60))
        self._dog = img
        self._place()

    def _place(self) -> None:
        from curio_robot.demo.viewer import INSET_FRACTION, INSET_MAX, INSET_MIN, MARGIN

        vp = self.v.viewport
        if vp is None or self._head is None or self._dog is None:
            return
        self._placed_for = (vp.width, vp.height)
        w = int(min(max(vp.width * INSET_FRACTION, INSET_MIN), INSET_MAX))
        w = min(w, max(1, (vp.height - 3 * MARGIN) * 2 // 3))
        h = w * 3 // 4
        imgs = []
        for i, img in enumerate((self._head, self._dog)):
            rect = mujoco.MjrRect(vp.width - w - MARGIN, vp.height - (h + MARGIN) * (i + 1), w, h)
            small = img.resize((w, h), Image.Resampling.LANCZOS)
            imgs.append((rect, np.ascontiguousarray(np.asarray(small))))
        self.v.set_images(imgs)

    def _texts(self) -> None:
        big = int(mujoco.mjtFontScale.mjFONTSCALE_150)
        mode = ("CURIO ON   dog + firefighter's view" if self.curio
                else "CURIO OFF   dog's view only")
        d = self.decision
        rate = f",  {self.rate:.1f}x real time" if self.rate and not self.world.walking else ""
        left = ["Mode", "Voice", "Scene", "Firefighter", "Request", "Room", "Truth", "Decision", "Dog",
                "Arrived"]
        right = [mode, self.voice_line,
                 f"{self.world.kind} {self.world.seed}  ({len(self.world.rooms)} rooms)",
                 self.facing_name(),
                 f'"{d.text}"' if d else f'"{self.world.request}" (press Enter)',
                 f"{d.target.object_id or 'none'}  ({d.target.reason})" if d else "-",
                 d.oracle_why if d else "-",
                 self.verdict_names[d.verdict] if d else "-",
                 f"{self.status}{rate}  (live physics)",
                 self.result_line]
        texts = [(big, int(mujoco.mjtGridPos.mjGRID_TOPLEFT), "\n".join(left), "\n".join(right))]
        if len(self.compare) == 2:
            off, on = self.compare[False], self.compare[True]
            texts.append((big, int(mujoco.mjtGridPos.mjGRID_BOTTOMLEFT), "Curio OFF\nCurio ON",
                          (f"{self.verdict_names[off.verdict]}  {off.target.object_id}\n"
                          f"{self.verdict_names[on.verdict]}  {on.target.object_id}")))
        texts.append((int(mujoco.mjtFontScale.mjFONTSCALE_100),
                      int(mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT), "CURIO",
                      "simulation  |  rooms from the floor map, geometry, no VLM  |  motion: "
                      "our path follower + pretrained walking policy\n" + CREDIT
                      + '  |  say "Curio, ..." or type  |  h for help'))
        if texts != self._texts_shown:
            self._texts_shown = texts
            self.v.set_texts(texts)

    # ---- commands -------------------------------------------------------------------------

    def command(self, line: str) -> bool:
        """Handle one terminal line; False to quit."""
        words = line.strip().split()
        cmd = words[0].lower() if words else ""
        if cmd in ("q", "quit", "exit"):
            return False
        if cmd in ("h", "help", "?"):
            print(HELP, flush=True)
        elif cmd in ("on", "off"):
            self.curio = cmd == "on"
            print(f"Curio {cmd}", flush=True)
            self.say(f"Curio {cmd}.")
        elif self.busy:
            print("busy: the dog is walking; wait for it to stop", flush=True)
            self.say("One moment.")
        elif cmd == "c":
            self.start_compare()
        elif cmd == "turn":
            self.turn()
        elif cmd == "r":
            self.load()
        elif cmd == "n":
            self.showcase_i = (self.showcase_i + 1) % len(SHOWCASE)
            self.switch_to = SHOWCASE[self.showcase_i]
        elif cmd == "scene":
            from curio_robot.demo.dog_scenes import KINDS

            if len(words) != 3 or words[1] not in KINDS or not words[2].isdigit():
                print(f"usage: scene {'|'.join(KINDS)} SEED", flush=True)
            else:
                self.switch_to = (words[1], int(words[2]))
        else:
            self.run(line, self.curio)
        return True
