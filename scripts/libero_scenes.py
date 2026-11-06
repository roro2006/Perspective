"""The person in LIBERO-Spatial, and the scene as Curio's object map. Task T5-31.

The person is a head camera only: a MuJoCo free camera rendered with its own mujoco.Renderer
from the live model and data, so nothing is added to the scene and the policy's agentview and
wrist images are exactly the ones it was trained on. The camera sits at one of SEATS around the
far side and the ends of the table, looking at the objects, so the person's left, right and
"closest to me" are not the robot's.

The robot's frame is its base (robot0_base, facing +x; its left is +y), not the agentview
camera, which faces the robot from the front.

Each Curio episode needs words that are true in the person's frame and point at the other bowl
when read in the robot's frame. choose_request tries the seats in a rotating order and, for
each, "on my left/right" then "closest to me / furthest from me", and returns the first that
works with SIDE_GAP / DISTANCE_GAP to spare in both frames and keeps both bowls in view. None
when no seat works (that init state is skipped and reported). Rules fixed before any rollout.

Only HeadCamera imports mujoco; the rest is plain numpy so the CPU tests can run it."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:  # run as a plain script from the libero venv
    sys.path.insert(0, str(REPO))

from curio_robot.geometry import intrinsics_from_fovy, look_at, project
from curio_robot.layer.goal import Base
from curio_robot.layer.libero_words import BOWLS
from curio_robot.layer.robot_words import DISTANCE_GAP, SIDE_GAP
from curio_robot.perspective import horizontal_distance, lateral
from curio_robot.types import HUMAN_CAM, CameraFrame, CameraPose, ObjectMap, WorldObject

HEAD_W, HEAD_H = 640, 480  # the grounding resolution (flywheel.scenes WIDTH, HEIGHT)
HEAD_FOVY = 70.0  # the sim's human camera (sim.physics HUMAN_CAM)
HEAD_Z = 1.45  # m: a seated head, 0.55 m above LIBERO's table top (z = 0.90)
LOOK_AT = np.array([-0.15, 0.0, 0.90])  # middle of where LIBERO-Spatial puts its objects
EDGE_PX = 20  # both bowls must project at least this far inside the head image
ROBOT_BASE = np.array([-0.66, 0.0, 0.912])  # robot0_base in every LIBERO-Spatial scene

# Seat name -> head position. "across" faces the robot; the others sit at the far corners
# and at the two ends of the table. All look at LOOK_AT.
SEATS: dict[str, tuple[float, float, float]] = {
    "across": (0.60, 0.00, HEAD_Z),
    "far_left_corner": (0.55, -0.55, HEAD_Z),  # the robot's right-hand far corner
    "far_right_corner": (0.55, 0.55, HEAD_Z),
    "end_robot_left": (-0.15, 0.90, HEAD_Z),  # the table end on the robot's left (+y)
    "end_robot_right": (-0.15, -0.90, HEAD_Z),
}

# phrase -> (key, sign): the meant bowl minimises sign * key, as robot_words.RELATIONS.
PHRASES = {
    "on my left": ("lateral", 1.0),
    "on my right": ("lateral", -1.0),
    "closest to me": ("distance", 1.0),
    "furthest from me": ("distance", -1.0),
}
TEMPLATE = "pick up the black bowl {phrase} and place it on the plate"

# LIBERO object -> (Curio id, colour word shown to the grounder). The bowls' ids are given by
# position (bowl_1 is the one further to the robot's left), never by which one is the target,
# so the grounder's letters (sorted by id) say nothing about the answer.
OTHERS: dict[str, tuple[str, str]] = {
    "plate_1": ("plate", "white"),
    "glazed_rim_porcelain_ramekin_1": ("ramekin", "white"),
    "cookies_1": ("cookie_box", "red"),
    "wooden_cabinet_1": ("cabinet", "grey"),
    "flat_stove_1": ("stove", "grey"),
}
BOWL_COLOUR = "black"


def head_pose(seat: str) -> CameraPose:
    eye = np.asarray(SEATS[seat], float)
    return CameraPose(eye, look_at(eye, LOOK_AT))


def robot_pose(base: np.ndarray = ROBOT_BASE) -> CameraPose:
    """The robot's own frame as a CameraPose at its base, looking along +x (left is +y)."""
    base = np.asarray(base, float)
    return CameraPose(base, look_at(base, base + np.array([1.0, 0.0, 0.0])))


def robot_base(base: np.ndarray = ROBOT_BASE) -> Base:
    return Base(tuple(float(v) for v in base), 0.0)


def head_intrinsics():
    return intrinsics_from_fovy(HEAD_FOVY, HEAD_W, HEAD_H)


def _key(xyz: np.ndarray, viewer: CameraPose, kind: str) -> float:
    obj = WorldObject("x", "x", np.asarray(xyz, float))
    return lateral(obj, viewer) if kind == "lateral" else horizontal_distance(obj, viewer)


def margin(meant: np.ndarray, other: np.ndarray, viewer: CameraPose, phrase: str) -> float:
    """How far ahead of the other bowl the meant one is on the phrase, from viewer (m)."""
    kind, sign = PHRASES[phrase]
    return sign * _key(other, viewer, kind) - sign * _key(meant, viewer, kind)


def _gap(phrase: str) -> float:
    return SIDE_GAP if PHRASES[phrase][0] == "lateral" else DISTANCE_GAP


def in_view(xyz: np.ndarray, pose: CameraPose, edge: float = EDGE_PX) -> bool:
    uv = project(np.asarray(xyz, float), head_intrinsics(), pose)
    if uv is None:
        return False
    u, v = uv
    return edge <= u < HEAD_W - edge and edge <= v < HEAD_H - edge


@dataclass(frozen=True, slots=True)
class Request:
    seat: str
    phrase: str
    words: str
    person_margin: float  # m: the target ahead of the other bowl in the person's frame
    robot_margin: float  # m: the other bowl ahead of the target in the robot's frame


def choose_request(target: np.ndarray, other: np.ndarray, rotate: int,
                   base: np.ndarray = ROBOT_BASE) -> Request | None:
    """The first (seat, phrase) that singles out the target in the person's frame and the
    other bowl in the robot's, both by the gap, with both bowls in the head image. Seats are
    tried starting at index rotate % len(SEATS), phrases in PHRASES order."""
    names = list(SEATS)
    robot = robot_pose(base)
    for i in range(len(names)):
        seat = names[(rotate + i) % len(names)]
        pose = head_pose(seat)
        if not (in_view(target, pose) and in_view(other, pose)):
            continue
        for phrase in PHRASES:
            mine = margin(target, other, pose, phrase)
            robots = margin(other, target, robot, phrase)
            if mine >= _gap(phrase) and robots >= _gap(phrase):
                return Request(seat, phrase, TEMPLATE.format(phrase=phrase), round(mine, 4),
                               round(robots, 4))
    return None


def bowl_ids(positions: dict[str, np.ndarray]) -> dict[str, str]:
    """LIBERO bowl name -> Curio id: bowl_1 is the one with the larger world y (further to the
    robot's left), ties broken by smaller x."""
    order = sorted(BOWLS, key=lambda b: (-float(positions[b][1]), float(positions[b][0])))
    return {b: f"bowl_{i + 1}" for i, b in enumerate(order)}


def object_map(positions: dict[str, np.ndarray], head: CameraPose | None,
               base: np.ndarray = ROBOT_BASE) -> tuple[ObjectMap, dict[str, str]]:
    """(Curio's object map of the LIBERO scene, Curio id -> LIBERO name). positions: LIBERO
    object name -> world xyz for the two bowls and every name in OTHERS that is present."""
    ids = bowl_ids(positions)
    objects, back = [], {}
    for name in BOWLS:
        objects.append(WorldObject(ids[name], BOWL_COLOUR, np.asarray(positions[name], float)))
        back[ids[name]] = name
    for name, (cid, colour) in OTHERS.items():
        if name in positions:
            objects.append(WorldObject(cid, colour, np.asarray(positions[name], float)))
            back[cid] = name
    objects.sort(key=lambda o: o.object_id)
    return ObjectMap(objects, robot_pose(base), head), back


class HeadCamera:
    """Renders the person's head camera from a LIBERO env's MuJoCo model and data. The free
    camera reads the model's global fovy, so it is set for each render and put back."""

    def __init__(self, model, width: int = HEAD_W, height: int = HEAD_H) -> None:
        import mujoco

        self._mujoco = mujoco
        self.model = model
        self.renderer = mujoco.Renderer(model, height, width)

    def render(self, data, pose: CameraPose) -> CameraFrame:
        mj = self._mujoco
        eye = pose.position
        forward = pose.forward
        cam = mj.MjvCamera()
        cam.type = mj.mjtCamera.mjCAMERA_FREE
        dist = float(np.linalg.norm(LOOK_AT - eye))
        cam.lookat[:] = eye + dist * forward
        cam.distance = dist
        cam.azimuth = math.degrees(math.atan2(forward[1], forward[0]))
        cam.elevation = math.degrees(math.asin(float(forward[2])))
        old = float(self.model.vis.global_.fovy)
        self.model.vis.global_.fovy = HEAD_FOVY
        try:
            self.renderer.update_scene(data, camera=cam)
            rgb = self.renderer.render().copy()
        finally:
            self.model.vis.global_.fovy = old
        return CameraFrame(HUMAN_CAM, rgb, head_intrinsics(), pose, 0.0)

    def close(self) -> None:
        self.renderer.close()
