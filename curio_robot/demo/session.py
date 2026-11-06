"""The demo's engine (T5-33): one SimWorld, driven by one worker thread, streaming frames.

MuJoCo's renderer belongs to the thread that made its GL context, so the world is built and
used only on the worker; HTTP threads queue jobs. A job that arrives while another runs is
refused (Busy), so two commands never drive the same arm. Cancel is checked between steps.

The target is whatever `pipeline.step` returns: the detector on live renders, fusion with
`seen_by`, the perspective transform and the rule resolver. Curio on fuses the head camera;
off uses the robot camera only, as in `sim.ab`. Nothing here picks a target.

The oracle is the simulator's ground truth: the true block positions and the person's true
eye and gaze, read with the scenarios' own left/right rule (lateral offset along the
viewer's right axis) or distance for nearest/farthest. It covers only requests that rule
can read; anything else is reported as unknown rather than guessed.

The arm moves to the chosen block in small position-only IK steps (`SimWorld.move_gripper_to`).
That is scripted motion, not a learned policy, and the console says so."""

from __future__ import annotations

import base64
import io
import json
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from curio_robot.geometry import look_at, project
from curio_robot.perception import SegmentationDetector
from curio_robot.pipeline import step
from curio_robot.resolver import parse_request
from curio_robot.sim import ab
from curio_robot.sim.ab import SUCCESS_RADIUS
from curio_robot.sim.scenarios import SCENARIOS
from curio_robot.sim.world import ROBOT_CAM_LOOK, ROBOT_CAM_POS, SimWorld
from curio_robot.types import HUMAN_CAM, ROBOT_CAM, CameraFrame, ObjectMap

PANE = (480, 360)
SPEED = 0.25  # m/s at the gripper, as runtime.motion
FPS = 8.0  # stream rate during motion; CPU renders cost about 70 ms each
ARRIVE = 0.006  # metres, as runtime.motion
MAX_STEPS = 400
MAX_TEXT = 200
HISTORY = 30
JPEG_QUALITY = 80

ROBOT_RING = (255, 170, 60)
HEAD_RING = (80, 200, 235)
CHOSEN = (236, 72, 153)
ORACLE = (90, 200, 120)
DIM = (18, 20, 24)

Emit = Callable[[dict], None]


class Busy(RuntimeError):
    """Another job is driving the robot."""


@dataclass(slots=True)
class Scene:
    kind: str
    seed: int
    request: str
    expected_xy: np.ndarray
    eye: np.ndarray
    look: np.ndarray

    def to_dict(self) -> dict:
        return {"kind": self.kind, "seed": self.seed, "request": self.request}


# ---- ground truth ---------------------------------------------------------------------------


def oracle_xy(text: str, blocks: list[tuple[str, np.ndarray]], eye: np.ndarray,
              look: np.ndarray) -> tuple[np.ndarray | None, str]:
    """The block the words mean, from ground truth, or (None, why) when the rule can't say.

    Same reading as the scenarios: "my"/no owner is the person's frame, "your" the robot's;
    left/right by the lateral offset along the viewer's right axis, nearest/farthest by
    horizontal distance from the viewer."""
    req = parse_request(text)
    cands = [p for c, p in blocks if req.color is None or c == req.color]
    if not cands:
        return None, f"there is no {req.color or 'block'} on the table"
    if len(cands) == 1:
        return cands[0][:2].copy(), f"the only {req.color or 'block'}"
    if req.relation is None:
        return None, "several blocks match and the words give no relation"
    if req.speaker_frame:
        viewer, gaze, whose = eye, look, "the person's"
    else:
        viewer, gaze, whose = ROBOT_CAM_POS, ROBOT_CAM_LOOK, "the robot's"
    right = look_at(viewer, gaze)[:, 0]
    if req.relation in ("left", "right"):
        key = [float(np.dot(p - viewer, right)) for p in cands]
        i = int(np.argmin(key)) if req.relation == "left" else int(np.argmax(key))
    else:
        key = [float(np.linalg.norm((p - viewer)[:2])) for p in cands]
        i = int(np.argmin(key)) if req.relation == "nearest" else int(np.argmax(key))
    return cands[i][:2].copy(), f"{req.relation} from {whose} view (ground truth)"


def verdict(chosen: np.ndarray | None, oracle: np.ndarray | None, oracle_why: str,
            known: bool) -> str:
    """correct / wrong / not_found / unknown, against the oracle within SUCCESS_RADIUS."""
    if not known:
        return "unknown"
    if oracle is None:  # ground truth: nothing matches
        return "correct" if chosen is None else "wrong"
    if chosen is None:
        return "not_found"
    return "correct" if np.linalg.norm(chosen[:2] - oracle) < SUCCESS_RADIUS else "wrong"


# ---- drawing --------------------------------------------------------------------------------


def jpeg(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _ring(d: ImageDraw.ImageDraw, uv, r: float, color, width: int) -> None:
    u, v = uv
    d.ellipse([u - r, v - r, u + r, v + r], outline=color, width=width)


def draw_camera(frame: CameraFrame, om: ObjectMap | None, chosen_id: str | None,
                used: bool) -> Image.Image:
    """A camera pane: a ring on every object this camera saw, a thick ring on the chosen one,
    dimmed when the pipeline didn't use this camera (Curio off, head camera)."""
    img = Image.fromarray(frame.rgb)
    d = ImageDraw.Draw(img)
    color = ROBOT_RING if frame.camera == ROBOT_CAM else HEAD_RING
    if om is not None:
        for o in om.objects:
            if frame.camera not in o.seen_by:
                continue
            uv = project(o.position, frame.intrinsics, frame.pose)
            if uv is None:
                continue
            if o.object_id == chosen_id:
                _ring(d, uv, 18, CHOSEN, 4)
            else:
                _ring(d, uv, 14, color, 2)
    if not used:
        img = Image.blend(img, Image.new("RGB", img.size, DIM), 0.65)
    return img


def draw_twin(frame: CameraFrame, chosen: np.ndarray | None,
              oracle: np.ndarray | None) -> Image.Image:
    img = Image.fromarray(frame.rgb)
    d = ImageDraw.Draw(img)
    if oracle is not None:
        uv = project(np.array([*oracle, 0.0125]), frame.intrinsics, frame.pose)
        if uv:
            _ring(d, uv, 22, ORACLE, 2)
    if chosen is not None:
        uv = project(np.asarray(chosen, float), frame.intrinsics, frame.pose)
        if uv:
            _ring(d, uv, 15, CHOSEN, 4)
    return img


# ---- one decision (shared by the browser engine and the MuJoCo viewer) -----------------------


@dataclass(slots=True)
class Decision:
    result: dict  # JSON-ready: words, parse, map, target, reason, timings, truth, verdict
    object_map: ObjectMap
    chosen_id: str | None
    chosen: np.ndarray | None  # the target's world position
    oracle: np.ndarray | None  # ground-truth xy, or None


def decide(world: SimWorld, scene: Scene, text: str, curio: bool) -> Decision:
    """`pipeline.step` on fresh renders of the world as it stands, scored against ground truth.
    Curio on fuses the head camera; off is the robot camera only. Doesn't move anything."""
    frames = world.capture()
    res = step(frames, text, SegmentationDetector(world.block_colors), use_human_view=curio)
    target = res.target
    oracle, why = oracle_xy(text, world.block_positions(), scene.eye, scene.look)
    known = oracle is not None or why.startswith("there is no")
    chosen = None if target.position is None else np.asarray(target.position, float)
    req = parse_request(text)
    result = {
        "text": text,
        "curio": curio,
        "scene": scene.to_dict(),
        "parsed": {"color": req.color, "relation": req.relation,
                   "frame": "speaker" if req.speaker_frame else "robot"},
        "objects": [{"id": o.object_id, "color": o.color, "seen_by": sorted(o.seen_by)}
                    for o in res.object_map.objects],
        "target_id": target.object_id,
        "reason": target.reason,
        "timings_ms": {k: round(v, 2) for k, v in res.timings_ms.items()},
        "oracle": {"known": known, "why": why},
        "verdict": verdict(chosen, oracle, why, known),
    }
    return Decision(result, res.object_map, target.object_id, chosen, oracle)


# ---- the engine (worker thread only) --------------------------------------------------------


class Engine:
    """Everything that touches the world. Call it from one thread only."""

    def __init__(self, world: SimWorld) -> None:
        self.world = world
        self.scene: Scene | None = None
        self._om: ObjectMap | None = None
        self._chosen_id: str | None = None
        self._chosen: np.ndarray | None = None
        self._oracle: np.ndarray | None = None
        self._curio = True

    def load(self, kind: str, seed: int, emit: Emit) -> dict:
        if kind not in SCENARIOS:
            raise ValueError(f"unknown scene {kind!r}")
        self.world.home()
        sc = SCENARIOS[kind](self.world, np.random.default_rng(seed))
        self.scene = Scene(kind, seed, sc.request, sc.expected_xy, sc.human_eye, sc.human_look)
        self._om = self._chosen_id = self._chosen = None
        self._oracle = None
        emit(self.frame(head=True))
        return self.scene.to_dict()

    def reset(self, emit: Emit) -> dict:
        if self.scene is None:
            raise ValueError("no scene loaded")
        return self.load(self.scene.kind, self.scene.seed, emit)

    def frame(self, head: bool) -> dict:
        w = self.world
        panes = {
            "robot": jpeg(draw_camera(w.render_camera(ROBOT_CAM), self._om, self._chosen_id,
                                      True)),
            "twin": jpeg(draw_twin(w.render_camera("overview"), self._chosen, self._oracle)),
        }
        if head:
            panes["head"] = jpeg(draw_camera(w.render_camera(HUMAN_CAM), self._om,
                                             self._chosen_id, self._curio))
        return {"type": "frame", "panes": panes, "replay": False}

    def run(self, text: str, curio: bool, emit: Emit, cancelled: Callable[[], bool],
            fps: float = FPS, speed: float = SPEED) -> dict:
        """The request through pipeline.step on fresh renders, then the arm to its choice."""
        if self.scene is None:
            raise ValueError("no scene loaded")
        text = text.strip()[:MAX_TEXT]
        if not text:
            raise ValueError("empty request")
        w = self.world
        w.home()
        self._om = self._chosen_id = self._chosen = None
        self._curio = curio
        d = decide(w, self.scene, text, curio)
        self._om, self._chosen_id, self._chosen, self._oracle = (d.object_map, d.chosen_id,
                                                                 d.chosen, d.oracle)
        out = d.result
        oracle = d.oracle
        emit({"type": "interpretation", **out})
        emit(self.frame(head=True))
        t_move = time.perf_counter()
        status = "no_target"
        if self._chosen is not None:
            status = self._move(self._chosen, emit, cancelled, fps, speed)
        out["motion"] = {"status": status, "s": round(time.perf_counter() - t_move, 2)}
        if self._chosen is not None and oracle is not None:
            err = float(np.linalg.norm(w.gripper_position()[:2] - oracle))
            out["motion"]["error_to_oracle_m"] = round(err, 4)
        emit(self.frame(head=True))
        return out

    def _move(self, goal: np.ndarray, emit: Emit, cancelled: Callable[[], bool],
              fps: float, speed: float) -> str:
        period = 1.0 / fps
        step_m = speed * period
        for k in range(MAX_STEPS):
            if cancelled():
                return "cancelled"
            t0 = time.perf_counter()
            here = np.asarray(self.world.gripper_position(), float)
            delta = goal - here
            dist = float(np.linalg.norm(delta))
            if dist < ARRIVE:
                return "arrived"
            self.world.move_gripper_to(goal if dist <= step_m else here + delta / dist * step_m)
            emit(self.frame(head=k % 3 == 2))
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))
        return "unreachable"


# ---- the session: thread, queue, busy, recording, replay -----------------------------------


class DemoSession:
    """Owns the worker thread. `submit` queues a job or raises Busy; events go to `publish`."""

    def __init__(self, publish: Emit, *, pane: tuple[int, int] = PANE,
                 record_dir: Path | None = None, fps: float = FPS, speed: float = SPEED,
                 world_factory: Callable[[], SimWorld] | None = None) -> None:
        self._out = publish
        self.last_frame: dict | None = None  # all three panes, for a client that joins late
        self._pane = pane
        self._fps, self._speed = fps, speed
        self._factory = world_factory or (lambda: SimWorld(*pane))
        self.record_dir = record_dir
        self._jobs: queue.Queue[tuple[str, dict] | None] = queue.Queue()
        self._lock = threading.Lock()
        self._busy: str | None = None
        self._cancel = threading.Event()
        self._ready = threading.Event()
        self.history: list[dict] = []
        self.ab: dict | None = None
        self.scene: dict | None = None
        self.render_ms: float | None = None
        self.error: str | None = None
        self._thread = threading.Thread(target=self._loop, name="demo-world", daemon=True)

    def _publish(self, event: dict) -> None:
        if event.get("type") == "frame":
            panes = dict(self.last_frame["panes"]) if self.last_frame else {}
            panes.update(event["panes"])
            self.last_frame = {**event, "panes": panes}
        self._out(event)

    # any thread
    def start(self, timeout: float = 30.0) -> None:
        self._thread.start()
        self._ready.wait(timeout)

    def close(self) -> None:
        self._cancel.set()
        self._jobs.put(None)
        self._thread.join(timeout=5)

    def submit(self, job: str, **args) -> None:
        with self._lock:
            if self._busy is not None:
                raise Busy(f"busy: {self._busy}")
            self._busy = job
        self._cancel.clear()
        self._jobs.put((job, args))
        self._publish(self.state())

    def cancel(self) -> None:
        self._cancel.set()

    def state(self) -> dict:
        return {"type": "state", "busy": self._busy, "scene": self.scene,
                "history": self.history[-HISTORY:], "ab": self.ab,
                "recordings": self.recordings(), "render_ms": self.render_ms,
                "error": self.error}

    def wait_idle(self, timeout: float = 60.0) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            if self._busy is None and self._jobs.empty():
                return True
            time.sleep(0.02)
        return False

    def recordings(self) -> list[dict]:
        if self.record_dir is None or not self.record_dir.exists():
            return []
        out = []
        for p in sorted(self.record_dir.glob("*.jsonl"), reverse=True)[:HISTORY]:
            meta = p.with_suffix(".json")
            if meta.exists():
                out.append({"id": p.stem, **json.loads(meta.read_text(encoding="utf-8"))})
        return out

    # worker thread
    def _loop(self) -> None:
        try:
            world = self._factory()
        except Exception as e:  # noqa: BLE001 - no GL context or MuJoCo: say so, don't pretend
            self.error = f"simulator failed to start: {e}"
            self._ready.set()
            self._publish(self.state())
            return
        try:
            engine = Engine(world)
            t0 = time.perf_counter()
            world.render_camera("overview")
            self.render_ms = round((time.perf_counter() - t0) * 1000, 1)
            self._ready.set()
            while True:
                item = self._jobs.get()
                if item is None:
                    return
                job, args = item
                try:
                    self._do(engine, job, args)
                except Exception as e:  # noqa: BLE001 - any job failure goes to the page
                    self._publish({"type": "error", "job": job, "message": str(e)})
                finally:
                    with self._lock:
                        self._busy = None
                    self._publish(self.state())
        finally:
            world.close()

    def _do(self, engine: Engine, job: str, args: dict) -> None:
        if job == "load":
            self.scene = engine.load(args["kind"], int(args["seed"]), self._publish)
        elif job == "reset":
            self.scene = engine.reset(self._publish)
        elif job == "run":
            self._record_run(engine, args["text"], bool(args["curio"]))
        elif job == "compare":
            for curio in (False, True):
                if self._cancel.is_set():
                    break
                self._record_run(engine, args["text"], curio, compare=True)
                time.sleep(0.6)
        elif job == "ab":
            trials = int(args["trials"])
            t0 = time.perf_counter()
            res = ab.run(trials, seed=int(args.get("seed", 0)))
            res["seconds"] = round(time.perf_counter() - t0, 1)
            res["measured_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self.ab = res
        elif job == "replay":
            self._replay(args["id"])
        else:
            raise ValueError(f"unknown job {job!r}")

    def _record_run(self, engine: Engine, text: str, curio: bool, compare: bool = False) -> None:
        frames: list[tuple[float, dict]] = []
        t0 = time.perf_counter()

        def emit(event: dict) -> None:
            frames.append((round(time.perf_counter() - t0, 3), event))
            self._publish(event)

        res = engine.run(text, curio, emit, self._cancel.is_set, self._fps, self._speed)
        res["compare"] = compare
        res["at"] = time.strftime("%H:%M:%S")
        self.history.append(res)
        self._publish({"type": "result", **res})
        if self.record_dir is not None:
            self.record_dir.mkdir(parents=True, exist_ok=True)
            rid = time.strftime("%Y%m%d-%H%M%S") + f"-{len(self.history):03d}"
            with (self.record_dir / f"{rid}.jsonl").open("w", encoding="utf-8") as f:
                for t, event in frames:
                    f.write(json.dumps({"t": t, "event": event}) + "\n")
            meta = {"text": res["text"], "curio": curio, "scene": res["scene"],
                    "verdict": res["verdict"], "target_id": res["target_id"], "at": res["at"]}
            (self.record_dir / f"{rid}.json").write_text(json.dumps(meta), encoding="utf-8")

    def _replay(self, rid: str) -> None:
        if self.record_dir is None:
            raise ValueError("recording is off")
        path = (self.record_dir / f"{rid}.jsonl").resolve()
        if path.parent != self.record_dir.resolve() or not path.exists():
            raise ValueError(f"no recording {rid!r}")
        self._publish({"type": "replay", "id": rid, "phase": "start"})
        start = time.perf_counter()
        for line in path.read_text(encoding="utf-8").splitlines():
            if self._cancel.is_set():
                break
            row = json.loads(line)
            time.sleep(max(0.0, row["t"] - (time.perf_counter() - start)))
            event = row["event"]
            event["replay"] = True
            self._publish(event)
        self._publish({"type": "replay", "id": rid, "phase": "end"})
