"""Benchmark items: one rendered scene with the person view and the robot view,
the words, and the exact answer. Task T5-01."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from curio_robot import perspective
from curio_robot.collab.tasks import TaskScene, apply_scene, fetch_scene, true_map
from curio_robot.conventions.readings import Frame
from curio_robot.geometry import project
from curio_robot.sim.world import ROBOT_CAM_POS, SimWorld
from curio_robot.types import (
    HUMAN_CAM,
    ROBOT_CAM,
    CameraFrame,
    CameraPose,
    Intrinsics,
    ObjectMap,
    WorldObject,
)

# New kinds go at the end: generate() draws kind-major from one rng, so the
# items of earlier kinds stay identical for the same seed.
KINDS = ("color", "marked", "habit", "hidden", "hidden-pair", "free")
HABITS = (0.15, 0.85)  # p_own_frame shown for "habit" items, as in sim/grounding_data.py
MIN_VISIBLE_PX = 20  # a block with fewer robot-camera pixels counts as hidden from the robot
MAX_ATTEMPTS = 200  # redraws allowed while looking for an ambiguous pair

SINGLE_INDEX = 4  # the single block in fetch_scene's default layout (after two pairs)
PAIR_BLOCKS = 4  # blocks 0..3 are the two same-colour pairs

BOX_OFFSET = 0.045  # box centre from the hidden block toward the robot camera, m (fetch_scene)
BOX_HALF = (0.012, 0.055)  # box half extents: thin side toward the robot, width (sim.world)
BLOCK_HALF = 0.0125  # block half size, m (sim.world.BLOCK_HALF)
MIN_DISTANCE_GAP = 0.02  # "closest"/"farthest" need the pair this much apart in distance, m

FREE_LEADS = ("Hand me", "Pass me", "Could you get me", "Grab")
FREE_NOUNS = ("one", "block", "cube", "thing")
# (relation, frame, template) with {lead} {color} {noun} {side} filled in. Half
# the side templates name the frame without the rule's " my "/" your " cue;
# the distance templates have no left/right at all. Distance templates
# outnumber side ones because more distance draws are rejected (the two
# viewers must disagree), which keeps the accepted mix near half and half.
FREE_TEMPLATES: tuple[tuple[str, Frame, str], ...] = (
    ("side", "speaker", "{lead} the {color} {noun} to my {side}"),
    ("side", "speaker", "{lead} the {color} {noun} on my {side}-hand side"),
    ("side", "speaker", "{lead} the {side} {color} {noun} from where I sit"),
    ("side", "speaker", "{lead} the {color} {noun} on the {side} as I see it"),
    ("side", "robot", "{lead} the {color} {noun} to your {side}"),
    ("side", "robot", "{lead} the {color} {noun} on your {side}-hand side"),
    ("side", "robot", "{lead} the {side} {color} {noun} from where you are"),
    ("side", "robot", "{lead} the {color} {noun} on the {side} as you see it"),
    ("nearest", "speaker", "{lead} the {color} {noun} closest to me"),
    ("nearest", "speaker", "{lead} the {color} {noun} nearest me"),
    ("farthest", "speaker", "{lead} the {color} {noun} farthest from me"),
    ("farthest", "speaker", "{lead} the {color} {noun} further away from me"),
    ("nearest", "robot", "{lead} the {color} {noun} closest to you"),
    ("nearest", "robot", "{lead} the {color} {noun} nearest you"),
    ("farthest", "robot", "{lead} the {color} {noun} farthest from you"),
    ("farthest", "robot", "{lead} the {color} {noun} further away from you"),
    ("nearest", "speaker", "{lead} the {color} {noun} nearer to me"),
    ("nearest", "speaker", "{lead} the {color} {noun} at my end of the table"),
    ("farthest", "speaker", "{lead} the {color} {noun} on the far side from me"),
    ("nearest", "robot", "{lead} the {color} {noun} nearer to you"),
    ("nearest", "robot", "{lead} the {color} {noun} at your end of the table"),
    ("farthest", "robot", "{lead} the {color} {noun} on the far side from you"),
)


@dataclass(frozen=True, slots=True)
class BenchItem:
    id: str
    kind: str
    text: str
    frame: Frame | None  # whose left/right the words mean; None: no side word
    p_own_frame: float | None  # the habit shown to grounders ("habit" items only)
    target_id: str
    object_map: ObjectMap  # exact block positions; robot_pose and human_pose from the renders
    robot_visible: tuple[str, ...]  # object ids with >= MIN_VISIBLE_PX robot-camera pixels
    human: CameraFrame  # rendered person view, segmentation None
    robot: CameraFrame  # rendered robot view, segmentation None

    @property
    def target_xyz(self) -> np.ndarray:
        return next(o.position for o in self.object_map.objects if o.object_id == self.target_id)

    @property
    def target_pixel_robot(self) -> tuple[float, float] | None:
        """The target projected into the robot image (what a policy is told)."""
        return project(self.target_xyz, self.robot.intrinsics, self.robot.pose)


def make_item(world: SimWorld, rng: np.random.Generator, kind: str, item_id: str) -> BenchItem:
    """One item of `kind` (ValueError for an unknown kind).

    Scene: scene = collab.tasks.fetch_scene(rng, occlude=(kind == "hidden"));
    apply_scene(world, scene); human = world.render_camera(HUMAN_CAM); robot =
    world.render_camera(ROBOT_CAM, segmentation=True). object_map =
    ObjectMap(true_map(scene).objects, robot.pose, human.pose). Block i of the
    scene is object i and geom world._block_geoms[i]; robot_visible = ids of
    objects whose geom fills >= MIN_VISIBLE_PX pixels of robot.segmentation.
    Store both frames with segmentation=None.

    - color, hidden: target = object 4 (the single block, a unique colour;
      hidden by the box for "hidden"); text f"Hand me the {color} one";
      frame None, p_own_frame None.
    - marked, habit: after the scene, draw in this order: idx =
      int(rng.integers(0, 4)) (a pair block); for "habit" p =
      HABITS[int(rng.integers(0, 2))], else p = 0.5; frame = "speaker" if
      rng.random() < p else "robot". The viewer is human.pose for "speaker",
      robot.pose for "robot"; side = "left" if perspective.pick(same-colour
      objects, "left", viewer) is the target, else "right". If both viewers
      pick the same block for that side, the item is not ambiguous: draw a new
      scene and repeat (at most MAX_ATTEMPTS, then RuntimeError).
      marked: text f"Hand me the {color} one on {m} {side}", m "my" for
      speaker, "your" for robot; p_own_frame None.
      habit: text f"Hand me the {color} one on the {side}"; p_own_frame = p.
    - hidden-pair: a "marked" request whose target is a pair block the box
      hides from the robot, while its partner stays visible: robot-only
      cannot answer and the frame still matters (_try_hidden_pair).
    - free: varied wording a keyword rule cannot parse, with an exact answer
      from perspective.pick (_try_free, FREE_TEMPLATES)."""
    if kind not in KINDS:
        raise ValueError(f"unknown kind: {kind}")
    attempt = {"hidden-pair": _try_hidden_pair, "free": _try_free}.get(kind, _try_item)
    for _ in range(MAX_ATTEMPTS):
        item = attempt(world, rng, kind, item_id)
        if item is not None:
            return item
    raise RuntimeError(f"no ambiguous {kind} item in {MAX_ATTEMPTS} attempts")


def _try_item(
    world: SimWorld, rng: np.random.Generator, kind: str, item_id: str
) -> BenchItem | None:
    """One draw of a scene and its words; None when a marked or habit pair is ambiguous."""
    scene = fetch_scene(rng, occlude=(kind == "hidden"))
    apply_scene(world, scene)
    human = world.render_camera(HUMAN_CAM)
    robot = world.render_camera(ROBOT_CAM, segmentation=True)
    object_map = ObjectMap(true_map(scene).objects, robot.pose, human.pose)
    visible = _robot_visible(world, object_map, robot)
    human = replace(human, segmentation=None)
    robot = replace(robot, segmentation=None)

    if kind in ("color", "hidden"):
        target = object_map.objects[SINGLE_INDEX]
        return BenchItem(
            id=item_id,
            kind=kind,
            text=f"Hand me the {target.color} one",
            frame=None,
            p_own_frame=None,
            target_id=target.object_id,
            object_map=object_map,
            robot_visible=visible,
            human=human,
            robot=robot,
        )

    target = object_map.objects[int(rng.integers(0, PAIR_BLOCKS))]
    if kind == "habit":
        p = HABITS[int(rng.integers(0, 2))]
    else:
        p = 0.5
    frame: Frame = "speaker" if rng.random() < p else "robot"

    cands = object_map.by_color(target.color)
    viewer = human.pose if frame == "speaker" else robot.pose
    side = "left" if perspective.pick(cands, "left", viewer) is target else "right"
    if _same_pick(cands, side, human.pose, robot.pose):
        return None

    if kind == "marked":
        m = "my" if frame == "speaker" else "your"
        text = f"Hand me the {target.color} one on {m} {side}"
        p_own_frame = None
    else:
        text = f"Hand me the {target.color} one on the {side}"
        p_own_frame = p
    return BenchItem(
        id=item_id,
        kind=kind,
        text=text,
        frame=frame,
        p_own_frame=p_own_frame,
        target_id=target.object_id,
        object_map=object_map,
        robot_visible=visible,
        human=human,
        robot=robot,
    )


def _render(
    world: SimWorld, scene: TaskScene
) -> tuple[ObjectMap, tuple[str, ...], CameraFrame, CameraFrame]:
    """Apply and render a scene: (object_map, robot_visible, human, robot), unsegmented."""
    apply_scene(world, scene)
    human = world.render_camera(HUMAN_CAM)
    robot = world.render_camera(ROBOT_CAM, segmentation=True)
    object_map = ObjectMap(true_map(scene).objects, robot.pose, human.pose)
    visible = _robot_visible(world, object_map, robot)
    return object_map, visible, replace(human, segmentation=None), replace(robot, segmentation=None)


def _box_in_front(scene: TaskScene, index: int) -> TaskScene | None:
    """The scene with the box BOX_OFFSET in front of block `index` toward the
    robot camera, placed as fetch_scene places it for the single block; None
    when the box would overlap another block."""
    target = np.asarray(scene.blocks[index].xy)
    toward = ROBOT_CAM_POS[:2] - target
    toward = toward / np.linalg.norm(toward)
    across = np.array([-toward[1], toward[0]])
    centre = target + BOX_OFFSET * toward
    for i, block in enumerate(scene.blocks):
        d = np.asarray(block.xy) - centre
        near_along = abs(d @ toward) < BOX_HALF[0] + BLOCK_HALF
        if i != index and near_along and abs(d @ across) < BOX_HALF[1] + BLOCK_HALF:
            return None
    yaw = float(np.arctan2(toward[1], toward[0]))
    return replace(scene, occluder=(float(centre[0]), float(centre[1])), occluder_yaw=yaw)


def _try_hidden_pair(
    world: SimWorld, rng: np.random.Generator, kind: str, item_id: str
) -> BenchItem | None:
    """scene = fetch_scene(rng) (no box); idx = int(rng.integers(0, 4)); frame
    = "speaker" if rng.random() < 0.5 else "robot"; the box goes in front of
    block idx. Words as "marked". None when the words are not ambiguous, the
    box overlaps a block, the target is robot-visible or its partner is not."""
    scene = fetch_scene(rng)
    idx = int(rng.integers(0, PAIR_BLOCKS))
    frame: Frame = "speaker" if rng.random() < 0.5 else "robot"
    boxed = _box_in_front(scene, idx)
    if boxed is None:
        return None
    exact = true_map(boxed)
    target = exact.objects[idx]
    cands = exact.by_color(target.color)
    viewer = exact.human_pose if frame == "speaker" else exact.robot_pose
    side = "left" if perspective.pick(cands, "left", viewer) is target else "right"
    if _same_pick(cands, side, exact.human_pose, exact.robot_pose):
        return None
    object_map, visible, human, robot = _render(world, boxed)
    partner = next(o for o in cands if o is not target)
    if target.object_id in visible or partner.object_id not in visible:
        return None
    m = "my" if frame == "speaker" else "your"
    return BenchItem(
        id=item_id,
        kind=kind,
        text=f"Hand me the {target.color} one on {m} {side}",
        frame=frame,
        p_own_frame=None,
        target_id=target.object_id,
        object_map=object_map,
        robot_visible=visible,
        human=human,
        robot=robot,
    )


def _try_free(
    world: SimWorld, rng: np.random.Generator, kind: str, item_id: str
) -> BenchItem | None:
    """scene = fetch_scene(rng) (no box); idx = int(rng.integers(0, 4)) picks
    a pair block; then a template, lead and noun, each by rng.integers over its
    tuple. Side relations: target = block idx, side = the one perspective.pick
    gives it from the frame's viewer. nearest/farthest: target = what pick
    gives in idx's pair from that viewer, whose distances must differ by >=
    MIN_DISTANCE_GAP. None when the other frame's viewer would pick the same
    block (the frame must matter)."""
    scene = fetch_scene(rng)
    idx = int(rng.integers(0, PAIR_BLOCKS))
    relation, frame, template = FREE_TEMPLATES[int(rng.integers(0, len(FREE_TEMPLATES)))]
    lead = FREE_LEADS[int(rng.integers(0, len(FREE_LEADS)))]
    noun = FREE_NOUNS[int(rng.integers(0, len(FREE_NOUNS)))]
    exact = true_map(scene)
    target = exact.objects[idx]
    cands = exact.by_color(target.color)
    viewer = exact.human_pose if frame == "speaker" else exact.robot_pose
    other = exact.robot_pose if frame == "speaker" else exact.human_pose
    side = ""
    if relation == "side":
        side = "left" if perspective.pick(cands, "left", viewer) is target else "right"
        if _same_pick(cands, side, exact.human_pose, exact.robot_pose):
            return None
    else:
        target = perspective.pick(cands, relation, viewer)
        a, b = (perspective.horizontal_distance(o, viewer) for o in cands)
        if abs(a - b) < MIN_DISTANCE_GAP or perspective.pick(cands, relation, other) is target:
            return None
    object_map, visible, human, robot = _render(world, scene)
    return BenchItem(
        id=item_id,
        kind=kind,
        text=template.format(lead=lead, color=target.color, noun=noun, side=side),
        frame=frame,
        p_own_frame=None,
        target_id=target.object_id,
        object_map=object_map,
        robot_visible=visible,
        human=human,
        robot=robot,
    )


def _robot_visible(world: SimWorld, object_map: ObjectMap, robot: CameraFrame) -> tuple[str, ...]:
    """Ids of the blocks that fill at least MIN_VISIBLE_PX robot-camera pixels."""
    return tuple(
        o.object_id
        for i, o in enumerate(object_map.objects)
        if int((robot.segmentation == world._block_geoms[i]).sum()) >= MIN_VISIBLE_PX
    )


def _same_pick(
    cands: list[WorldObject], side: str, human_pose: CameraPose, robot_pose: CameraPose
) -> bool:
    """True when both viewers pick the same block for `side` (the words are ambiguous)."""
    by_speaker = perspective.pick(cands, side, human_pose)
    by_robot = perspective.pick(cands, side, robot_pose)
    return by_speaker.object_id == by_robot.object_id


def generate(
    world: SimWorld, n_per_kind: int, seed: int, kinds: tuple[str, ...] = KINDS
) -> list[BenchItem]:
    """rng = np.random.default_rng(seed); for kind in kinds, for i in
    range(n_per_kind): make_item(world, rng, kind, f"{kind}-{i:04d}").
    Same arguments -> same items."""
    rng = np.random.default_rng(seed)
    return [
        make_item(world, rng, kind, f"{kind}-{i:04d}") for kind in kinds for i in range(n_per_kind)
    ]


def _cam_to_dict(frame: CameraFrame) -> dict[str, Any]:
    k = frame.intrinsics
    return {
        "position": [float(x) for x in frame.pose.position],
        "rotation": [[float(x) for x in row] for row in frame.pose.rotation],
        "fx": float(k.fx),
        "fy": float(k.fy),
        "cx": float(k.cx),
        "cy": float(k.cy),
        "width": int(k.width),
        "height": int(k.height),
        "timestamp": float(frame.timestamp),
    }


def _item_to_dict(item: BenchItem) -> dict[str, Any]:
    return {
        "id": item.id,
        "kind": item.kind,
        "text": item.text,
        "frame": item.frame,
        "p_own_frame": item.p_own_frame,
        "target_id": item.target_id,
        "robot_visible": list(item.robot_visible),
        "objects": [
            {
                "id": o.object_id,
                "color": o.color,
                "xyz": [float(x) for x in o.position],
            }
            for o in item.object_map.objects
        ],
        "cameras": {"human": _cam_to_dict(item.human), "robot": _cam_to_dict(item.robot)},
    }


def save(items: list[BenchItem], out_dir: Path) -> Path:
    """Write out_dir/images/{id}_human.png and {id}_robot.png (the rgb arrays)
    and out_dir/items.jsonl, one JSON object per item:
      {"id", "kind", "text", "frame", "p_own_frame", "target_id",
       "robot_visible": [...],
       "objects": [{"id", "color", "xyz": [x, y, z]}, ...] in map order,
       "cameras": {"human": cam, "robot": cam}} with
      cam = {"position": [3], "rotation": [[3] x 3], "fx", "fy", "cx", "cy",
             "width", "height", "timestamp"}.
    Creates folders. Returns the items.jsonl path."""
    images = out_dir / "images"
    images.mkdir(parents=True, exist_ok=True)
    lines = []
    for item in items:
        Image.fromarray(item.human.rgb).save(images / f"{item.id}_human.png")
        Image.fromarray(item.robot.rgb).save(images / f"{item.id}_robot.png")
        lines.append(json.dumps(_item_to_dict(item)))
    path = out_dir / "items.jsonl"
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


def _load_cam(data: dict[str, Any], camera: str, rgb: np.ndarray) -> CameraFrame:
    intrinsics = Intrinsics(
        fx=data["fx"],
        fy=data["fy"],
        cx=data["cx"],
        cy=data["cy"],
        width=data["width"],
        height=data["height"],
    )
    pose = CameraPose(
        position=np.asarray(data["position"], dtype=float),
        rotation=np.asarray(data["rotation"], dtype=float),
    )
    return CameraFrame(camera, rgb, intrinsics, pose, data["timestamp"], None)


def _item_from_dict(data: dict[str, Any], out_dir: Path) -> BenchItem:
    images = out_dir / "images"
    human_rgb = np.asarray(Image.open(images / f"{data['id']}_human.png").convert("RGB"))
    robot_rgb = np.asarray(Image.open(images / f"{data['id']}_robot.png").convert("RGB"))
    human = _load_cam(data["cameras"]["human"], HUMAN_CAM, human_rgb)
    robot = _load_cam(data["cameras"]["robot"], ROBOT_CAM, robot_rgb)
    objects = [
        WorldObject(o["id"], o["color"], np.asarray(o["xyz"], dtype=float), set())
        for o in data["objects"]
    ]
    object_map = ObjectMap(objects, robot.pose, human.pose)
    return BenchItem(
        id=data["id"],
        kind=data["kind"],
        text=data["text"],
        frame=data["frame"],
        p_own_frame=data["p_own_frame"],
        target_id=data["target_id"],
        object_map=object_map,
        robot_visible=tuple(data["robot_visible"]),
        human=human,
        robot=robot,
    )


def load(out_dir: Path) -> list[BenchItem]:
    """Inverse of save: CameraFrame(camera name, rgb from the PNG as uint8
    (H, W, 3), Intrinsics, CameraPose, timestamp, None); WorldObject seen_by
    empty sets; tuples where the dataclass has tuples."""
    with (out_dir / "items.jsonl").open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    return [_item_from_dict(r, out_dir) for r in records]
