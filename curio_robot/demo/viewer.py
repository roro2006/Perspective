"""The demo in MuJoCo's own viewer window (T5-33): no browser.

    uv run python -m curio_robot.demo.viewer [--scene perspective:0]

The main view is the live SO-101 scene. Two insets show what the cameras see right now: the
person's head camera and the robot's camera, rings on what each detected, a thick magenta ring
on Curio's choice. On the table, a magenta disc marks Curio's target and a green disc the
ground truth for the words. Requests are typed in the terminal that started it.

Same decision code as the browser console (`session.decide`: `pipeline.step` on fresh renders,
scored against the simulator's ground truth). The arm moves by position-only IK steps toward
the chosen block: scripted motion, not a learned policy, and the overlay says so.

All world changes happen on this thread under the viewer's lock; the viewer only draws."""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image, ImageDraw

from curio_robot.demo.session import (
    ARRIVE,
    MAX_TEXT,
    PANE,
    SPEED,
    Scene,
    decide,
    draw_camera,
)
from curio_robot.sim.scenarios import SCENARIOS
from curio_robot.sim.shots import _font
from curio_robot.sim.world import HUMAN_GROUP, SimWorld
from curio_robot.types import HUMAN_CAM, ROBOT_CAM

INSET = (384, 288)  # insets are drawn at PANE (the detection resolution) and scaled to this
INSET_PERIOD = 0.25  # seconds between inset refreshes (two CPU renders each)
LOOP_HZ = 30
PAUSE_S = 1.5  # between the two halves of a comparison
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
        self.goal: np.ndarray | None = None
        self.result: dict | None = None
        self.decision = None
        self.pending: list[tuple[str, bool]] = []  # queued (text, curio) runs
        self.pause_until = 0.0
        self.showcase_i = -1
        self.status = "ready"
        self.t_move = 0.0
        self._next_inset = 0.0
        self.compare: dict[bool, dict] = {}

    # ---- scene ------------------------------------------------------------------------------

    def load(self, kind: str, seed: int) -> None:
        with self.v.lock():
            self.world.home()
            sc = SCENARIOS[kind](self.world, np.random.default_rng(seed))
        self.scene = Scene(kind, seed, sc.request, sc.expected_xy, sc.human_eye, sc.human_look)
        self.goal = self.result = self.decision = None
        self.pending, self.compare = [], {}
        self.status = "ready"
        self._next_inset = 0.0
        print(f"\nScene {kind} {seed}. Its request: \"{sc.request}\"", flush=True)

    # ---- one run ----------------------------------------------------------------------------

    def run(self, text: str, curio: bool) -> None:
        text = text.strip()[:MAX_TEXT] or self.scene.request
        self.last_text, self.curio = text, curio
        with self.v.lock():
            self.world.home()
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
        self.goal = d.chosen
        self.status = "moving"
        self.t_move = time.perf_counter()
        self._next_inset = 0.0
        if d.chosen is None:
            self._finish("nothing to reach")

    def start_compare(self) -> None:
        text = self.last_text or self.scene.request
        self.compare = {}
        self.pending = [(text, False), (text, True)]

    # ---- per frame --------------------------------------------------------------------------

    def tick(self, dt: float) -> None:
        if self.goal is not None:
            here = np.asarray(self.world.gripper_position(), float)
            delta = self.goal - here
            dist = float(np.linalg.norm(delta))
            if dist < ARRIVE:
                self._finish("arrived")
            else:
                step_m = SPEED * dt
                with self.v.lock():
                    self.world.move_gripper_to(self.goal if dist <= step_m
                                               else here + delta / dist * step_m)
                if time.perf_counter() - self.t_move > 15:
                    self._finish("unreachable")
        if self.goal is None and self.pending and time.perf_counter() >= self.pause_until:
            text, curio = self.pending.pop(0)
            self.pause_until = 0.0
            self.run(text, curio)

    def _finish(self, status: str) -> None:
        self.goal = None
        self.status = status
        if self.result is not None:
            self.compare[self.result["curio"]] = self.result
            print(f"  motion   {status} in {time.perf_counter() - self.t_move:.1f} s", flush=True)
        self.pause_until = time.perf_counter() + PAUSE_S if self.pending else 0.0
        self._next_inset = 0.0

    # ---- drawing ----------------------------------------------------------------------------

    def draw(self, now: float) -> None:
        scn = self.v.user_scn
        scn.ngeom = 0
        d = self.decision
        if d is not None and d.oracle is not None:
            _marker(scn, d.oracle, 0.03, GREEN)
        if d is not None and d.chosen is not None:
            _marker(scn, d.chosen[:2], 0.019, MAGENTA)
        if now >= self._next_inset:
            self._next_inset = now + INSET_PERIOD
            self._insets()
        self._texts()

    def _insets(self) -> None:
        d = self.decision
        om = d.object_map if d is not None else None
        cid = d.chosen_id if d is not None else None
        head = draw_camera(self.world.render_camera(HUMAN_CAM), om, cid, self.curio or d is None)
        robot = draw_camera(self.world.render_camera(ROBOT_CAM), om, cid, True)
        _label(head, "HUMAN VIEW  head camera", (80, 200, 235),
               "" if self.curio or d is None else "NOT USED")
        _label(robot, "ROBOT VIEW  its camera", (255, 170, 60))
        vp = self.v.viewport
        if vp is None:
            return
        w, h = INSET
        imgs = []
        for i, img in enumerate((head, robot)):
            rect = mujoco.MjrRect(vp.width - w - 12, vp.height - (h + 12) * (i + 1), w, h)
            imgs.append((rect, np.ascontiguousarray(np.asarray(img.resize(INSET)))))
        self.v.set_images(imgs)

    def _texts(self) -> None:
        big = int(mujoco.mjtFontScale.mjFONTSCALE_150)
        mode = "CURIO ON   robot + your view" if self.curio else "CURIO OFF   robot's view only"
        r = self.result
        left = ["Mode", "Scene", "Request", "Target", "Truth", "Result", "Arm"]
        right = [mode,
                 f"{self.scene.kind} {self.scene.seed}" if self.scene else "-",
                 f'"{r["text"]}"' if r else f'"{self.scene.request}" (press Enter)',
                 f'{r["target_id"] or "none"}  ({r["reason"]})' if r else "-",
                 r["oracle"]["why"] if r else "-",
                 VERDICT[r["verdict"]] if r else "-",
                 f"{self.status}  (scripted IK, not a learned policy)"]
        texts = [(big, int(mujoco.mjtGridPos.mjGRID_TOPLEFT), "\n".join(left), "\n".join(right))]
        if len(self.compare) == 2:
            off, on = self.compare[False], self.compare[True]
            texts.append((big, int(mujoco.mjtGridPos.mjGRID_BOTTOMLEFT),
                          "Curio OFF\nCurio ON",
                          (f'{VERDICT[off["verdict"]]}  {off["target_id"] or "none"}\n'
                           f'{VERDICT[on["verdict"]]}  {on["target_id"] or "none"}')))
        texts.append((int(mujoco.mjtFontScale.mjFONTSCALE_100),
                      int(mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT), "CURIO",
                      "simulation  ·  type in the terminal  ·  h for help"))
        self.v.set_texts(texts)

    # ---- commands ---------------------------------------------------------------------------

    def command(self, line: str) -> bool:
        """Handle one terminal line; False to quit."""
        words = line.strip().split()
        cmd = words[0].lower() if words else ""
        busy = self.goal is not None or bool(self.pending)
        if cmd in ("q", "quit", "exit"):
            return False
        if cmd in ("h", "help", "?"):
            print(HELP, flush=True)
        elif cmd in ("on", "off"):
            self.curio = cmd == "on"
            print(f"Curio {cmd}", flush=True)
        elif busy:
            print("busy: the arm is moving; wait for it to finish", flush=True)
        elif cmd == "c":
            self.start_compare()
        elif cmd == "n":
            self.showcase_i = (self.showcase_i + 1) % len(SHOWCASE)
            self.load(*SHOWCASE[self.showcase_i])
        elif cmd == "r":
            self.load(self.scene.kind, self.scene.seed)
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


def main() -> None:
    p = argparse.ArgumentParser(prog="curio_robot.demo.viewer", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", default="perspective:0", help="first scene, kind:seed")
    args = p.parse_args()
    kind, _, seed = args.scene.partition(":")
    if kind not in SCENARIOS:
        p.error(f"scene kind must be one of {sorted(SCENARIOS)}")

    world = SimWorld(*PANE)  # detection runs at the console's resolution
    lines: queue.Queue[str | None] = queue.Queue()
    threading.Thread(target=_read_stdin, args=(lines,), daemon=True).start()
    try:
        with mujoco.viewer.launch_passive(world.model, world.data, show_left_ui=False,
                                          show_right_ui=False) as v:
            v.opt.geomgroup[HUMAN_GROUP] = 1  # show the person's head in the main view
            v.cam.lookat[:] = (0.24, 0.0, 0.03)
            v.cam.distance, v.cam.azimuth, v.cam.elevation = 1.05, 90.0, -38.0
            demo = ViewerDemo(v, world)
            demo.load(kind, int(seed or 0))
            print(HELP, flush=True)
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
                if not v.is_running():
                    break
                demo.tick(min(now - last, 0.1))
                last = now
                demo.draw(now)
                v.sync()
                time.sleep(max(0.0, period - (time.perf_counter() - now)))
    finally:
        world.close()


if __name__ == "__main__":
    main()
