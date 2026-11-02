"""The scene interface the renderer drives, plus the grounding glue: object maps in the
speaker's head frame (the geometry condition) or the robot's frame (a robot-only reading)."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field

import mujoco
import numpy as np
from kit import Path, cam_pose, depth_of, lerp_angle, set_free, side_of, yaw_of
from PIL import Image, ImageDraw

from curio_robot.geometry import intrinsics_from_fovy
from curio_robot.grounding.marks import LETTERS, Mark, place_marks, render_marks
from curio_robot.grounding.prompt import GroundingRequest, _encode_png, build_request
from curio_robot.types import CameraFrame, CameraPose, ObjectMap, WorldObject

GROUND_W, GROUND_H = 640, 480
INTRO = "You control a robot arm sharing a table with a person."


@dataclass(frozen=True)
class Candidate:
    oid: str
    label: str  # the "colour" the grounder reads, identical for look-alikes
    pos: tuple[float, float, float]


@dataclass(frozen=True)
class Query:
    key: str
    text: str
    frame: str  # "speaker" (geometry condition) or "robot" (robot-only reading)
    ids: tuple[str, ...]
    correct: str
    depth: bool = False  # also say ahead/behind (only needed for "behind me")


@dataclass
class Event:
    t0: float
    t1: float
    kind: str  # say, caption, chip, verdict, small
    text: str = ""
    good: bool = True
    oid: str | None = None


@dataclass
class Scene:
    name: str
    caption: str
    duration: float
    person: str = "p_"
    robot_cam: str = ""
    candidates: list[Candidate] = field(default_factory=list)
    queries: list[Query] = field(default_factory=list)
    fog: bool = False
    small: str = ""

    def xml(self) -> str:
        raise NotImplementedError

    def setup(self, model, data) -> None:
        """Static poses (person, props); called once before the first frame."""

    def animate(self, model, data, t: float, answers: dict[str, str | None]) -> None:
        raise NotImplementedError

    def camera(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def events(self, answers: dict[str, str | None], info: dict[str, dict]) -> list[Event]:
        raise NotImplementedError

    def cand(self, oid: str) -> Candidate:
        return next(c for c in self.candidates if c.oid == oid)

    # -- grounding -------------------------------------------------------------------------

    def head_pose(self, model, data) -> CameraPose:
        return cam_pose(model, data, self.person + "head_cam")

    def robot_pose(self, model, data) -> CameraPose:
        return cam_pose(model, data, self.robot_cam)


def camera_frame(renderer: mujoco.Renderer, model, data, cam: str, name: str,
                 fog: bool = False) -> CameraFrame:
    renderer.update_scene(data, camera=cam)
    if fog:
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = True
    rgb = renderer.render().copy()
    k = intrinsics_from_fovy(float(model.cam_fovy[model.camera(cam).id]), GROUND_W, GROUND_H)
    return CameraFrame(name, rgb, k, cam_pose(model, data, cam), 0.0)


def object_map(scene: Scene, ids, human: CameraPose | None, robot: CameraPose) -> ObjectMap:
    objs = [WorldObject(c.oid, c.label, np.asarray(c.pos, float), set())
            for c in scene.candidates if c.oid in ids]
    return ObjectMap(objs, robot, human)


def depth_request(text: str, omap: ObjectMap, frame: CameraFrame) -> GroundingRequest:
    """build_request's layout with ahead/behind added for each viewer. Used only for the
    warehouse's "behind me": the shipped prompt names left/right only, and an object behind
    the speaker can't be in their camera, so every object gets a letter here and only the
    ones in view are drawn."""
    visible = {m.object_id: m for m in place_marks(omap, frame)}
    marks = [visible.get(o.object_id, Mark(LETTERS[i], o.object_id, (-1.0, -1.0)))
             for i, o in enumerate(sorted(omap.objects, key=lambda o: o.object_id))]
    marks = [Mark(LETTERS[i], m.object_id, m.pixel) for i, m in enumerate(marks)]
    image, _ = render_marks(frame, [m for m in marks if m.pixel[0] >= 0])
    lines = [INTRO, ("The image is what the person saw when they spoke. Objects are marked "
                     "with letters; some are out of view.")]
    for m in marks:
        o = next(o for o in omap.objects if o.object_id == m.object_id)
        rob = f"{depth_of(o.position, omap.robot_pose)} the robot, robot's " \
              f"{side_of(o.position, omap.robot_pose)}"
        if omap.human_pose is not None:
            sp = f"{depth_of(o.position, omap.human_pose)} the speaker, speaker's " \
                 f"{side_of(o.position, omap.human_pose)}; "
        else:
            sp = ""
        lines.append(f"{m.letter}: {o.color} ({sp}{rob})".replace("ahead the", "in front of the"))
    lines += [f'The person said: "{text}"',
              "Answer with the single letter of the object they mean."]
    msg = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_encode_png(image)}"}},
        {"type": "text", "text": "\n".join(lines)}]}]
    return GroundingRequest(messages=msg, letters={m.letter: m.object_id for m in marks})


def make_request(scene: Scene, q: Query, renderer, model, data):
    """(request, the marked image the grounder sees) for one query."""
    human = scene.head_pose(model, data)
    robot = scene.robot_pose(model, data)
    if q.frame == "speaker":
        frame = camera_frame(renderer, model, data, scene.person + "head_cam", "human", scene.fog)
        omap = object_map(scene, q.ids, human, robot)
    else:
        frame = camera_frame(renderer, model, data, scene.robot_cam, "robot", scene.fog)
        omap = object_map(scene, q.ids, None, robot)
    if q.depth:
        req = depth_request(q.text, omap, frame)
    else:
        marks = place_marks(omap, frame)
        image, _ = render_marks(frame, marks)
        req = build_request(q.text, omap, marks, image)
    url = req.messages[0]["content"][0]["image_url"]["url"]
    sent = np.asarray(Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))))
    letter_of = {v: k for k, v in req.letters.items()}
    shown = [Mark(letter_of[m.object_id], m.object_id, m.pixel)
             for m in place_marks(omap, frame) if m.object_id in letter_of]
    view = np.asarray(Image.fromarray(render_marks(frame, [])[0]).resize(
        (frame.intrinsics.width, frame.intrinsics.height)))
    view = _draw_marks(view, shown)
    return req, sent, view


def _draw_marks(rgb: np.ndarray, marks: list[Mark]) -> np.ndarray:
    """The full head-camera frame with the same letter discs, for the video inset."""
    img = Image.fromarray(rgb.copy())
    d = ImageDraw.Draw(img)
    for m in marks:
        u, v = m.pixel
        d.ellipse((u - 14, v - 14, u + 14, v + 14), fill="white", outline="black", width=2)
        d.text((u, v), m.letter, fill="black", anchor="mm", font_size=18)
    return np.asarray(img)


# ---- motion helpers ------------------------------------------------------------------------


def window(t: float, t0: float, t1: float) -> float:
    return float(np.clip((t - t0) / max(1e-6, t1 - t0), 0.0, 1.0))


def drive(model, data, joint: str, path: Path, tau: float, z: float, yaw0: float,
          yaw_end: float | None = None, turn: float = 0.2) -> tuple[np.ndarray, float]:
    """Place a ground robot on the path: heading turns from yaw0 to the path tangent over the
    first `turn` of the move, and to yaw_end (if given) over the last part."""
    pos, tan = path.at(tau)
    yaw = yaw_of(tan) if np.linalg.norm(tan[:2]) > 1e-6 else yaw0
    yaw = lerp_angle(yaw0, yaw, min(1.0, tau / turn)) if tau < turn else yaw
    if yaw_end is not None and tau > 1 - turn:
        yaw = lerp_angle(yaw, yaw_end, (tau - (1 - turn)) / turn)
    q = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    set_free(model, data, joint, [pos[0], pos[1], z if path.curve.shape[1] == 2 else pos[2]], q)
    return pos, yaw
