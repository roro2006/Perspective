"""The demo in MuJoCo's own viewer window (T5-33): no browser.

    uv run python -m curio_robot.demo.viewer [--scene perspective:0]
    uv run python -m curio_robot.demo.viewer --robot dog      (the rescue dog, demo/dog.py)

The main view is the live SO-101 scene. Two insets show what the cameras see right now: the
person's head camera and the robot's camera, rings on what each detected, a thick magenta ring
on Curio's choice. On the table, a magenta disc marks Curio's target and a green disc the
ground truth for the words. Requests are spoken ("Curio, ...", demo/voice.py) or typed in
the terminal that started it. A seated person sits across the table and can walk round it.

Same decision code as the browser console (`session.decide`: `pipeline.step` on fresh renders,
scored against the simulator's ground truth). The arm then picks the chosen block up and sets
it down in front of the person (demo/handover.py): scripted motion, not a learned policy, and
the overlay says so. Each request starts from the scene's own layout.

All world changes happen on this thread under the viewer's lock; the viewer only draws."""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from collections.abc import Callable

import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image, ImageDraw

from curio_robot.demo import voice
from curio_robot.demo.handover import HandOver
from curio_robot.demo.session import (
    MAX_TEXT,
    PANE,
    Scene,
    decide,
    draw_camera,
    seat_angle,
    seat_name,
    seat_person,
)
from curio_robot.sim.scenarios import SCENARIOS
from curio_robot.sim.shots import _font
from curio_robot.sim.world import HUMAN_GROUP, SimWorld
from curio_robot.types import HUMAN_CAM, ROBOT_CAM

INSET_FRACTION = 0.24  # inset width as a share of the window, kept 4:3
INSET_MIN, INSET_MAX = 260, 560  # pixels
MARGIN = 14
INSET_PERIOD = 0.5  # robot-inset refresh while the arm moves; each render costs ~70 ms here
LOOP_HZ = 30
VIEWER_SHADOW = 2048
PAUSE_S = 1.5  # between the two halves of a comparison
WALK_DEG_S = 60.0  # how fast the person walks around the table
WALK_STEP = 45.0  # `walk` with no angle
SHOWCASE = (("perspective", 0), ("occlusion", 1), ("perspective", 1), ("cluttered", 2))

MAGENTA = (0.93, 0.28, 0.6, 0.75)
GREEN = (0.35, 0.78, 0.47, 0.45)
VERDICT = {"correct": "CORRECT", "wrong": "WRONG", "not_found": "NOT FOUND",
           "unknown": "UNKNOWN (words outside the truth rule)"}

HELP = """Type a request and press Enter, e.g.  Hand me the red one on my left
  (Enter alone)   the scene's own request
  on / off        Curio on (robot + head camera) or off (robot camera only)
  c               compare: the same words with Curio off, then on
  n               next showcase scene
  scene K S       load scene kind K (perspective, occlusion, cluttered) with seed S
  seat D          walk the person to D degrees around the table: 0 across from the
                  robot, 90 / -90 the ends, 180 beside the robot
  walk [D]        walk D degrees further round (default 45)
  r               reset the scene        h  this help        q  quit"""


def _label(img: Image.Image, title: str, color: tuple[int, int, int], note: str = "") -> None:
    d = ImageDraw.Draw(img)
    f = _font(20)
    d.rectangle([0, 0, img.width, 30], fill=(15, 17, 20))
    d.text((10, 4), title, font=f, fill=color)
    if note:
        d.text((img.width - 10, 4), note, font=f, fill=(235, 90, 80), anchor="ra")


def _marker(scn: mujoco.MjvScene, xy: np.ndarray, radius: float, rgba) -> None:
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], int(mujoco.mjtGeom.mjGEOM_CYLINDER),
                        np.array([radius, 0.0015, 0.0]), np.array([xy[0], xy[1], 0.0016]),
                        np.eye(3).flatten(), np.asarray(rgba, np.float32))
    scn.ngeom += 1


class ViewerDemo:
    def __init__(self, viewer, world: SimWorld) -> None:
        self.v, self.world = viewer, world
        self.scene: Scene | None = None
        self.curio = True
        self.last_text = ""
        self.motion: HandOver | None = None  # the pick and hand-over in progress
        self.result: dict | None = None
        self.decision = None
        self.pending: list[tuple[str, bool]] = []  # queued (text, curio) runs
        self.pause_until = 0.0
        self.showcase_i = -1
        self.status = "ready"
        self.t_move = 0.0
        self._full = True  # re-render both insets on the next frame
        self._next_inset = 0.0
        self._head: Image.Image | None = None
        self._robot: Image.Image | None = None
        self._placed_for: tuple[int, int] | None = None
        self._texts_shown: list | None = None
        self.compare: dict[bool, dict] = {}
        self.say: Callable[[str], None] = lambda text: None  # voice reply, if voice is on
        self.voice_line = "off (terminal only)"
        self.walk_to: float | None = None  # seat angle the person is walking to

    # ---- scene ------------------------------------------------------------------------------

    def load(self, kind: str, seed: int) -> None:
        with self.v.lock():
            self.world.home()
            sc = SCENARIOS[kind](self.world, np.random.default_rng(seed))
        self.scene = Scene(kind, seed, sc.request, sc.expected_xy, sc.human_eye, sc.human_look)
        self.motion = self.result = self.decision = self.walk_to = None
        self.pending, self.compare = [], {}
        self.status = "ready"
        self._full = True
        print(f"\nScene {kind} {seed}. Its request: \"{sc.request}\"", flush=True)

    # ---- one run ----------------------------------------------------------------------------

    def run(self, text: str, curio: bool) -> None:
        text = text.strip()[:MAX_TEXT] or self.scene.request
        self.last_text, self.curio = text, curio
        self._restore_layout()
        d = decide(self.world, self.scene, text, curio)
        self.decision, self.result = d, d.result
        r = d.result
        p = r["parsed"]
        print(f"\n[{'Curio ON ' if curio else 'Curio OFF'}] \"{text}\"\n"
              f"  parsed   colour={p['color']} relation={p['relation']} frame={p['frame']}\n"
              f"  target   {r['target_id']}  ({r['reason']})\n"
              f"  truth    {r['oracle']['why']}\n"
              f"  result   {VERDICT[r['verdict']]}   "
              f"detect {r['timings_ms']['detect']} / fuse {r['timings_ms']['fuse']} / "
              f"resolve {r['timings_ms']['resolve']} ms", flush=True)
        if d.chosen is not None:
            self.motion = HandOver(self.world, d.chosen, self.scene.eye, self.scene.look)
        self.status = "reaching"
        self.t_move = time.perf_counter()
        self._full = True
        color = r["parsed"]["color"]
        it = f"the {color} one" if color else "that one"
        if d.chosen is None:
            self.say(f"I can't see {'a ' + color + ' one' if color else 'that'}.")
            self._finish("nothing to reach")
        else:
            self.say(f"Got it, {it}." if curio else f"Robot view only. {it.capitalize()}.")

    def _restore_layout(self) -> None:
        """Every request starts from the scene's own layout, with the person where they are,
        so a block handed over last time is back on the table and off and on see the same."""
        seat = seat_angle(self.scene)
        with self.v.lock():
            self.world.home()
            sc = SCENARIOS[self.scene.kind](self.world, np.random.default_rng(self.scene.seed))
            base = Scene(self.scene.kind, self.scene.seed, sc.request, sc.expected_xy,
                         sc.human_eye, sc.human_look)
            self.scene = seat_person(self.world, base, seat)

    def start_compare(self) -> None:
        text = self.last_text or self.scene.request
        self.compare = {}
        self.pending = [(text, False), (text, True)]

    # ---- per frame --------------------------------------------------------------------------

    def tick(self, dt: float) -> None:
        if self.walk_to is not None:
            self._walk(dt)
        if self.motion is not None:
            with self.v.lock():
                event = self.motion.step(dt)
            self.status = self.motion.label
            if event == "release":
                self.say("Here you go.")
                self._full = True  # the person's view of the hand-over
            if self.motion.done:
                self._finish(self.motion.result)
        if self.motion is None and self.pending and time.perf_counter() >= self.pause_until:
            text, curio = self.pending.pop(0)
            self.pause_until = 0.0
            self.run(text, curio)

    def _walk(self, dt: float) -> None:
        here = seat_angle(self.scene)
        diff = (self.walk_to - here + 180.0) % 360.0 - 180.0  # the short way round
        stride = WALK_DEG_S * dt
        done = abs(diff) <= stride
        with self.v.lock():
            self.scene = seat_person(self.world, self.scene,
                                     self.walk_to if done else here + np.sign(diff) * stride)
        if done:
            self.walk_to = None
            self.status = "ready"
            self._full = True
            deg = seat_angle(self.scene)
            print(f"\nPerson now at {deg:+.0f} degrees, {seat_name(deg)}. "
                  "Same words, new seat: press Enter.", flush=True)

    def start_walk(self, deg: float) -> None:
        """The answer depends on where the person sits, so the last one no longer applies."""
        self.walk_to = (deg + 180.0) % 360.0 - 180.0
        self.decision = self.result = None
        self.compare = {}
        self.status = "person walking"
        with self.v.lock():
            self.world.home()

    def _finish(self, status: str) -> None:
        self.motion = None
        self.status = status
        if self.result is not None:
            self.compare[self.result["curio"]] = self.result
            print(f"  motion   {status} in {time.perf_counter() - self.t_move:.1f} s", flush=True)
        self.pause_until = time.perf_counter() + PAUSE_S if self.pending else 0.0
        self._full = True

    # ---- drawing ----------------------------------------------------------------------------

    def draw(self, now: float) -> None:
        scn = self.v.user_scn
        scn.ngeom = 0
        d = self.decision
        if d is not None and d.oracle is not None:
            _marker(scn, d.oracle, 0.03, GREEN)
        if d is not None and d.chosen is not None:
            _marker(scn, d.chosen[:2], 0.019, MAGENTA)
        # Offscreen renders block this loop, so render only when something changed: both
        # insets after a load, a decision or an arrival; the robot inset alone while moving.
        if self._full:
            self._full = False
            self._next_inset = now + INSET_PERIOD
            self._insets(head=True)
        elif self.walk_to is not None and now >= self._next_inset:
            self._next_inset = now + INSET_PERIOD
            self._insets(head=True, robot=False)
        elif self.motion is not None and now >= self._next_inset:
            self._next_inset = now + INSET_PERIOD
            self._insets(head=False)
        vp = self.v.viewport
        if vp is not None and (vp.width, vp.height) != self._placed_for:
            self._place()  # the window was resized (e.g. full screen): re-anchor, no re-render
        self._texts()

    def _insets(self, head: bool, robot: bool = True) -> None:
        d = self.decision
        om = d.object_map if d is not None else None
        cid = d.chosen_id if d is not None else None
        if head or self._head is None:
            used = self.curio or d is None
            img = draw_camera(self.world.render_camera(HUMAN_CAM), om, cid, used)
            _label(img, "HUMAN VIEW  head camera", (80, 200, 235), "" if used else "NOT USED")
            self._head = img
        if robot or self._robot is None:
            img = draw_camera(self.world.render_camera(ROBOT_CAM), om, cid, True)
            _label(img, "ROBOT VIEW  its camera", (255, 170, 60))
            self._robot = img
        self._place()

    def _place(self) -> None:
        """Stack the two insets in the window's top-right corner, sized to the window."""
        vp = self.v.viewport
        if vp is None or self._head is None or self._robot is None:
            return
        self._placed_for = (vp.width, vp.height)
        w = int(min(max(vp.width * INSET_FRACTION, INSET_MIN), INSET_MAX))
        w = min(w, max(1, (vp.height - 3 * MARGIN) * 2 // 3))  # both must fit vertically
        h = w * 3 // 4
        imgs = []
        for i, img in enumerate((self._head, self._robot)):
            rect = mujoco.MjrRect(vp.width - w - MARGIN, vp.height - (h + MARGIN) * (i + 1), w, h)
            small = img.resize((w, h), Image.Resampling.LANCZOS)
            imgs.append((rect, np.ascontiguousarray(np.asarray(small))))
        self.v.set_images(imgs)

    def _texts(self) -> None:
        big = int(mujoco.mjtFontScale.mjFONTSCALE_150)
        mode = "CURIO ON   robot + your view" if self.curio else "CURIO OFF   robot's view only"
        r = self.result
        seat = seat_angle(self.scene) if self.scene else 0.0
        left = ["Mode", "Voice", "Scene", "Person", "Request", "Target", "Truth", "Result",
                "Arm"]
        right = [mode,
                 self.voice_line,
                 f"{self.scene.kind} {self.scene.seed}" if self.scene else "-",
                 (f"walking to {self.walk_to:+.0f} deg" if self.walk_to is not None  # fixed
                  else f"seat {seat:+.0f} deg, {seat_name(seat)}"),  # text while moving
                 f'"{r["text"]}"' if r else f'"{self.scene.request}" (press Enter)',
                 f'{r["target_id"] or "none"}  ({r["reason"]})' if r else "-",
                 r["oracle"]["why"] if r else "-",
                 VERDICT[r["verdict"]] if r else "-",
                 f"{self.status}  (scripted pick and hand-over, not a learned policy)"]
        texts = [(big, int(mujoco.mjtGridPos.mjGRID_TOPLEFT), "\n".join(left), "\n".join(right))]
        if len(self.compare) == 2:
            off, on = self.compare[False], self.compare[True]
            texts.append((big, int(mujoco.mjtGridPos.mjGRID_BOTTOMLEFT),
                          "Curio OFF\nCurio ON",
                          (f'{VERDICT[off["verdict"]]}  {off["target_id"] or "none"}\n'
                           f'{VERDICT[on["verdict"]]}  {on["target_id"] or "none"}')))
        texts.append((int(mujoco.mjtFontScale.mjFONTSCALE_100),
                      int(mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT), "CURIO",
                      "simulation  |  say \"Curio, ...\" or type  |  h for help"))
        if texts != self._texts_shown:  # set_texts waits on the render thread (26-97 ms here)
            self._texts_shown = texts
            self.v.set_texts(texts)

    # ---- commands ---------------------------------------------------------------------------

    def command(self, line: str) -> bool:
        """Handle one terminal line; False to quit."""
        words = line.strip().split()
        cmd = words[0].lower() if words else ""
        busy = self.motion is not None or bool(self.pending) or self.walk_to is not None
        if cmd in ("q", "quit", "exit"):
            return False
        if cmd in ("h", "help", "?"):
            print(HELP, flush=True)
        elif cmd in ("on", "off"):
            self.curio = cmd == "on"
            print(f"Curio {cmd}", flush=True)
            self.say(f"Curio {cmd}.")
        elif busy:
            print("busy: the arm is moving; wait for it to finish", flush=True)
            self.say("One moment.")
        elif cmd == "c":
            self.start_compare()
        elif cmd == "n":
            self.showcase_i = (self.showcase_i + 1) % len(SHOWCASE)
            self.load(*SHOWCASE[self.showcase_i])
        elif cmd == "r":
            self.load(self.scene.kind, self.scene.seed)
        elif cmd in ("seat", "walk"):
            try:
                deg = float(words[1]) if len(words) > 1 else WALK_STEP
            except ValueError:
                print("usage: seat DEGREES  |  walk [DEGREES]", flush=True)
                return True
            if cmd == "seat" and len(words) < 2:
                print("usage: seat DEGREES  (0 across, 90 / -90 the ends, 180 beside the robot)",
                      flush=True)
                return True
            self.start_walk(deg if cmd == "seat" else seat_angle(self.scene) + deg)
        elif cmd == "scene":
            if len(words) != 3 or words[1] not in SCENARIOS or not words[2].isdigit():
                print("usage: scene perspective|occlusion|cluttered SEED", flush=True)
            else:
                self.load(words[1], int(words[2]))
        else:
            self.run(line, self.curio)
        return True


def _read_stdin(lines: queue.Queue[str | None]) -> None:
    for line in sys.stdin:
        lines.put(line.rstrip("\n"))
    lines.put(None)


def _start_voice(demo, lines: queue.Queue[str | None], robot: str = "arm"):
    """Say "Curio, ..." instead of typing. Commands join the terminal's queue."""
    if not voice.available():
        demo.voice_line = "unavailable on this system (terminal only)"
        print("Voice: not available here; use the terminal.", flush=True)
        return None

    def heard(command: str, text: str, conf: float) -> None:
        demo.voice_line = f'heard "{text}" ({conf:.2f})'
        print(f'\nHeard: "{text}" ({conf:.2f})', flush=True)
        lines.put(command)

    def status(s: str) -> None:
        demo.voice_line = 'listening: say "Curio, ..."' if s == "listening" else s

    listener = voice.Listener(heard, status, robot=robot)
    demo.voice_line = "starting the microphone..."
    listener.start()
    demo.say = listener.say
    example = getattr(demo, "VOICE_EXAMPLE", "Curio, hand me the red one on my left")
    print(f'Voice: say "{example}". The terminal still works.', flush=True)
    return listener


def main() -> None:
    p = argparse.ArgumentParser(prog="curio_robot.demo.viewer", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", default="perspective:0",
                   help="first scene, kind:seed (dog: corridor, doorway, closest or rescue)")
    p.add_argument("--no-voice", action="store_true", help="terminal only, no microphone")
    p.add_argument("--robot", choices=("arm", "dog"), default="arm",
                   help="arm: the SO-101 table demo (T5-33); dog: the rescue Go2 (T5-34)")
    args = p.parse_args()
    if args.robot == "dog":
        _main_dog(args.no_voice, args.scene if args.scene != "perspective:0" else "rescue:0")
        return
    kind, _, seed = args.scene.partition(":")
    if kind not in SCENARIOS:
        p.error(f"scene kind must be one of {sorted(SCENARIOS)}")

    world = SimWorld(*PANE, person_body=True)  # detection at the console's resolution
    # The viewer builds its own GL context from these settings at launch; the offscreen
    # renderer above already exists, so detection renders are unchanged. 8192 is costly on
    # integrated graphics.
    world.model.vis.quality.shadowsize = VIEWER_SHADOW
    lines: queue.Queue[str | None] = queue.Queue()
    listener = None
    threading.Thread(target=_read_stdin, args=(lines,), daemon=True).start()
    try:
        with mujoco.viewer.launch_passive(world.model, world.data, show_left_ui=False,
                                          show_right_ui=False) as v:
            v.opt.geomgroup[HUMAN_GROUP] = 1  # show the person's head in the main view
            v.cam.lookat[:] = (0.28, 0.0, 0.0)  # robot and person face to face
            v.cam.distance, v.cam.azimuth, v.cam.elevation = 1.5, 60.0, -30.0
            demo = ViewerDemo(v, world)
            demo.load(kind, int(seed or 0))
            print(HELP, flush=True)
            listener = _start_voice(demo, lines) if not args.no_voice else None
            _loop(v, demo, lines)
    finally:
        if listener is not None:
            listener.close()
        world.close()


def _main_dog(no_voice: bool, scene: str = "rescue:0") -> None:
    """The dog demo. A new building is a new MuJoCo model, which the passive viewer can't
    swap, so `scene` and `n` close the window and open one on the new layout (about 2 s);
    the terminal, the voice and the Curio switch carry over."""
    from curio_robot.demo import dog
    from curio_robot.demo.dog_scenes import KINDS

    if not dog.have_assets():
        sys.exit("The Go2 model and walking policy aren't here: see third_party/README.md "
                 "(fetch them, then uv run python scripts/poc/export_policies.py --go2).")
    kind, _, seed = scene.partition(":")
    if kind not in KINDS:
        sys.exit(f"dog scene kind must be one of {', '.join(KINDS)}")
    lines: queue.Queue[str | None] = queue.Queue()
    listener = None
    threading.Thread(target=_read_stdin, args=(lines,), daemon=True).start()
    carry: dict = {}
    nxt: tuple[str, int] | None = (kind, int(seed or 0))
    try:
        while nxt is not None:
            world = dog.DogWorld(*nxt)
            world.model.vis.quality.shadowsize = VIEWER_SHADOW
            nxt = None
            try:
                with mujoco.viewer.launch_passive(world.model, world.data, show_left_ui=False,
                                                  show_right_ui=False) as v:
                    dog.setup_camera(v, world)
                    demo = dog.DogDemo(v, world)
                    for k, val in carry.items():
                        setattr(demo, k, val)
                    demo.load()
                    if not carry:
                        print(dog.HELP, flush=True)
                        listener = _start_voice(demo, lines, "dog") if not no_voice else None
                    _loop(v, demo, lines)
                    nxt = demo.switch_to
                    carry = {"curio": demo.curio, "say": demo.say,
                             "voice_line": demo.voice_line, "showcase_i": demo.showcase_i}
            finally:
                world.close()
    finally:
        if listener is not None:
            listener.close()


def _loop(v, demo, lines: queue.Queue[str | None]) -> None:
    """Commands, then the demo's tick and drawing, at LOOP_HZ until the window closes."""
    period = 1.0 / LOOP_HZ
    last = time.perf_counter()
    while v.is_running():
        now = time.perf_counter()
        try:
            while True:
                line = lines.get_nowait()
                if line is None or not demo.command(line):
                    v.close()
                    break
        except queue.Empty:
            pass
        if getattr(demo, "switch_to", None) is not None:
            v.close()  # the dog demo reopens on a new building
        if not v.is_running():
            break
        demo.tick(min(now - last, 0.25))
        last = now
        demo.draw(now)
        v.sync()
        time.sleep(max(0.0, period - (time.perf_counter() - now)))


if __name__ == "__main__":
    main()
