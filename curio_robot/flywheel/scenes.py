"""Front/behind requests (kind "front-behind"): a person standing among look-alike objects,
one in front of them and one behind, says "the tote behind me". The robot stands so that its
own front/behind gives the other object, as in the T5-18 warehouse, so the frame matters.

The grounder sees the person's head camera (the behind object is out of view and gets a
letter without a mark) and the depth layout of flywheel.request. Deployment and the held-out
test draw from disjoint seed ranges, and the test also uses phrasings (HELDOUT_TEMPLATES,
HELDOUT_LEADS) that no training row may contain."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

import mujoco
import numpy as np
from PIL import Image

from curio_robot.flywheel.request import depth_of, depth_request, head_coords
from curio_robot.geometry import intrinsics_from_fovy
from curio_robot.grounding.prompt import GroundingRequest
from curio_robot.types import CameraFrame, CameraPose, Intrinsics, ObjectMap, WorldObject

KIND = "front-behind"
Relation = Literal["front", "behind"]
Frame = Literal["speaker", "robot"]

# Seeds: bench 0, v2 training 11-19, e2e 70000+. One rng per episode / item.
DEPLOY_SEEDS = range(524000, 527000)
HELDOUT_SEEDS = range(624000, 625000)

WIDTH, HEIGHT = 640, 480  # the PoC grounding resolution (scene_base.GROUND_W/H)
HEAD_FOVY = 80.0  # the PoC humanoid's head camera (kit.HEAD_FOVY)
P_SPEAKER = 0.75  # deployment: share of requests in the speaker's frame
P_DOT = 0.5  # share of utterances ending with a full stop
P_ONE = 0.25  # "the one ..." instead of the label, when that is unambiguous
MIN_DEPTH = 0.35  # every object at least this far ahead of / behind each viewer, m
MIN_GAP = 0.7  # objects and the robot at least this far apart, m
MAX_ATTEMPTS = 400

# label -> (MuJoCo geom type, size, rgba). Two slots per label (a look-alike pair).
LABELS: dict[str, tuple[str, tuple[float, ...], str]] = {
    "tote": ("box", (0.26, 0.18, 0.13), "0.16 0.45 0.78 1"),
    "bin": ("box", (0.22, 0.22, 0.18), "0.85 0.72 0.18 1"),
    "crate": ("box", (0.3, 0.3, 0.24), "0.55 0.38 0.2 1"),
    "barrel": ("cylinder", (0.24, 0.4), "0.3 0.5 0.32 1"),
    "case": ("box", (0.32, 0.12, 0.2), "0.12 0.12 0.14 1"),
    "drum": ("cylinder", (0.2, 0.3), "0.75 0.18 0.15 1"),
}
SLOTS = 2

# (relation, frame, template); {lead} {noun}. Training phrasings.
TRAIN_TEMPLATES: tuple[tuple[Relation, Frame, str], ...] = (
    ("behind", "speaker", "{lead} the {noun} behind me"),
    ("behind", "speaker", "{lead} the {noun} at my back"),
    ("behind", "speaker", "{lead} the {noun} that's behind me"),
    ("front", "speaker", "{lead} the {noun} in front of me"),
    ("front", "speaker", "{lead} the {noun} right in front of me"),
    ("behind", "robot", "{lead} the {noun} behind you"),
    ("behind", "robot", "{lead} the {noun} at your back"),
    ("front", "robot", "{lead} the {noun} in front of you"),
)
TRAIN_LEADS = ("Bring me", "Get me", "Grab", "Hand me", "Go get")
# Test-only phrasings: no training row may contain these templates or leads.
HELDOUT_TEMPLATES: tuple[tuple[Relation, Frame, str], ...] = (
    ("behind", "speaker", "{lead} the {noun} I have my back to"),
    ("behind", "speaker", "{lead} the {noun} over my shoulder"),
    ("behind", "speaker", "{lead} the {noun} back there behind where I'm standing"),
    ("front", "speaker", "{lead} the {noun} I'm facing"),
    ("front", "speaker", "{lead} the {noun} ahead of me"),
    ("behind", "robot", "{lead} the {noun} you have your back to"),
    ("front", "robot", "{lead} the {noun} you're facing"),
    ("front", "robot", "{lead} the {noun} ahead of you"),
)
HELDOUT_LEADS = ("Could you bring", "I need", "Go and fetch")
POOLS = {"train": (TRAIN_TEMPLATES, TRAIN_LEADS), "heldout": (HELDOUT_TEMPLATES, HELDOUT_LEADS)}


@dataclass(frozen=True, slots=True)
class FBItem:
    id: str
    seed: int
    text: str
    frame: Frame  # whose front/behind the words mean
    relation: Relation
    pool: str  # "train" or "heldout" phrasing
    target_id: str
    object_map: ObjectMap  # labels in WorldObject.color; human_pose = the speaker's head
    view: CameraFrame  # the speaker's head camera, unmarked

    @property
    def kind(self) -> str:
        return KIND


def request_for(item: FBItem) -> GroundingRequest:
    """The request deployment sends: the depth layout over the speaker's view."""
    return depth_request(item.text, item.object_map, item.view)


# ---------------------------------------------------------------- the world


def _xml() -> str:
    bodies = []
    for label, (gtype, size, rgba) in LABELS.items():
        for k in range(SLOTS):
            s = " ".join(str(v) for v in size)
            bodies.append(f'<body name="{label}{k}" mocap="true" pos="0 0 -50">'
                          f'<geom type="{gtype}" size="{s}" rgba="{rgba}"/></body>')
    return f"""<mujoco model="front-behind">
  <visual><global offwidth="{WIDTH}" offheight="{HEIGHT}"/>
    <headlight ambient="0.35 0.35 0.36" diffuse="0.45 0.45 0.45"/></visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.55 0.58 0.62"
             rgb2="0.25 0.26 0.28" width="256" height="1536"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.52 0.53 0.54"
             rgb2="0.46 0.47 0.48" width="256" height="256"/>
    <material name="grid" texture="grid" texrepeat="40 40"/>
  </asset>
  <worldbody>
    <light pos="2 -3 8" dir="-0.2 0.3 -1" diffuse="0.6 0.6 0.58"/>
    <geom name="floor" type="plane" size="30 30 0.1" material="grid"/>
    {"".join(bodies)}
    <body name="robot" mocap="true" pos="0 0 0">
      <geom type="cylinder" size="0.3 0.15" pos="0 0 0.15" rgba="0.85 0.45 0.1 1"/>
      <geom type="box" size="0.06 0.06 0.35" pos="0 0 0.55" rgba="0.3 0.3 0.32 1"/>
      <camera name="robot_cam" pos="0.1 0 0.85" xyaxes="0 -1 0 0 0 1" fovy="60"/>
    </body>
    <body name="head" mocap="true" pos="0 0 1.6">
      <camera name="head_cam" fovy="{HEAD_FOVY}"/>
    </body>
  </worldbody>
</mujoco>"""


class FBWorld:
    """One MuJoCo model with parked objects; each item moves what it needs and renders the
    head camera. Use as a context manager (it owns a renderer)."""

    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_string(_xml())
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, HEIGHT, WIDTH)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.renderer.close()

    def _mocap(self, body: str) -> int:
        return int(self.model.body_mocapid[self.model.body(body).id])

    def place(self, objects: list[tuple[str, str, np.ndarray]], robot_xy: np.ndarray,
              robot_yaw: float, head: np.ndarray, head_rot: np.ndarray) -> None:
        """objects: (slot body, label, centre). Unused slots go under the floor."""
        for label in LABELS:
            for k in range(SLOTS):
                self.data.mocap_pos[self._mocap(f"{label}{k}")] = (0.0, 0.0, -50.0)
        for body, _, centre in objects:
            self.data.mocap_pos[self._mocap(body)] = centre
        r = self._mocap("robot")
        self.data.mocap_pos[r] = (robot_xy[0], robot_xy[1], 0.0)
        self.data.mocap_quat[r] = (np.cos(robot_yaw / 2), 0, 0, np.sin(robot_yaw / 2))
        h = self._mocap("head")
        self.data.mocap_pos[h] = head
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, np.ascontiguousarray(head_rot).flatten())
        self.data.mocap_quat[h] = quat
        mujoco.mj_forward(self.model, self.data)

    def pose(self, cam: str) -> CameraPose:
        cid = self.model.camera(cam).id
        return CameraPose(self.data.cam_xpos[cid].copy(),
                          self.data.cam_xmat[cid].reshape(3, 3).copy())

    def render_head(self) -> CameraFrame:
        self.renderer.update_scene(self.data, camera="head_cam")
        rgb = self.renderer.render().copy()
        k = intrinsics_from_fovy(HEAD_FOVY, WIDTH, HEIGHT)
        return CameraFrame("human", rgb, k, self.pose("head_cam"), 0.0)


def look_rotation(yaw: float, pitch_down: float) -> np.ndarray:
    """Camera-to-world rotation (columns right, up, back) looking along yaw, tilted down."""
    f = np.array([np.cos(yaw) * np.cos(pitch_down), np.sin(yaw) * np.cos(pitch_down),
                  -np.sin(pitch_down)])
    r = np.cross(f, (0.0, 0.0, 1.0))
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return np.column_stack([r, u, -f])


# ---------------------------------------------------------------- items


def _centre(label: str, xy: np.ndarray) -> np.ndarray:
    gtype, size, _ = LABELS[label]
    half_h = size[2] if gtype == "box" else size[1]
    return np.array([xy[0], xy[1], half_h])


def _clear(xy: np.ndarray, taken: list[np.ndarray]) -> bool:
    return all(np.linalg.norm(xy - t) >= MIN_GAP for t in taken)


def _layout(rng: np.random.Generator):
    """One draw of speaker, pair, robot and distractors; None if the robot's front/behind
    does not swap the pair or anything sits too close to a viewer's ahead/behind line."""
    yaw = float(rng.uniform(-np.pi, np.pi))
    head = np.array([0.0, 0.0, float(rng.uniform(1.5, 1.75))])
    head_rot = look_rotation(yaw, float(rng.uniform(0.12, 0.35)))
    fwd = np.array([np.cos(yaw), np.sin(yaw)])
    right = np.array([np.sin(yaw), -np.cos(yaw)])
    df, db = rng.uniform(1.4, 4.5, size=2)
    front_xy = df * fwd + float(rng.uniform(-0.6, 0.6)) * df * right
    back_xy = -db * fwd + float(rng.uniform(-0.8, 0.8)) * db * right
    labels = list(LABELS)
    pair = labels[int(rng.integers(0, len(labels)))]
    rest = [lb for lb in labels if lb != pair]
    n_extra = int(rng.choice([0, 1, 2], p=[0.3, 0.4, 0.3]))
    extra = [rest[int(i)] for i in rng.permutation(len(rest))[:n_extra]]

    ang, dist = float(rng.uniform(-np.pi, np.pi)), float(rng.uniform(1.0, 5.0))
    robot_xy = dist * np.array([np.cos(ang), np.sin(ang)])
    robot_yaw = float(rng.uniform(-np.pi, np.pi))
    taken = [np.zeros(2), front_xy, back_xy]
    if not _clear(robot_xy, taken):
        return None
    taken.append(robot_xy)
    extras = []
    for label in extra:
        a, d = float(rng.uniform(-np.pi, np.pi)), float(rng.uniform(1.4, 5.0))
        xy = d * np.array([np.cos(a), np.sin(a)])
        if not _clear(xy, taken):
            return None
        taken.append(xy)
        extras.append((label, xy))
    return head, head_rot, pair, front_xy, back_xy, robot_xy, robot_yaw, extras


def _assign_ids(rng: np.random.Generator, n: int) -> list[str]:
    """Shuffled ids so the target's letter (ids sorted) is not tied to its role."""
    return [f"obj_{int(i)}" for i in rng.permutation(n)]


def make_item(world: FBWorld, seed: int, item_id: str, pool: str = "train",
              frame: Frame | None = None) -> FBItem:
    """One item from rng = default_rng(seed). frame None: speaker with P_SPEAKER.
    ValueError for an unknown pool; RuntimeError if no valid layout in MAX_ATTEMPTS."""
    if pool not in POOLS:
        raise ValueError(f"unknown pool: {pool}")
    templates, leads = POOLS[pool]
    rng = np.random.default_rng(seed)
    if frame is None:
        frame = "speaker" if rng.random() < P_SPEAKER else "robot"
    relation: Relation = "behind" if rng.random() < 0.5 else "front"
    for _ in range(MAX_ATTEMPTS):
        item = _try_item(world, rng, seed, item_id, pool, frame, relation, templates, leads)
        if item is not None:
            return item
    raise RuntimeError(f"{item_id}: no front/behind layout in {MAX_ATTEMPTS} attempts")


def _try_item(world, rng, seed, item_id, pool, frame, relation, templates, leads):
    layout = _layout(rng)
    if layout is None:
        return None
    head, head_rot, pair, front_xy, back_xy, robot_xy, robot_yaw, extras = layout
    placed = [(f"{pair}0", pair, _centre(pair, front_xy)),
              (f"{pair}1", pair, _centre(pair, back_xy))]
    placed += [(f"{label}0", label, _centre(label, xy)) for label, xy in extras]
    world.place(placed, robot_xy, robot_yaw, head, head_rot)
    human, robot = world.pose("head_cam"), world.pose("robot_cam")
    centres = [c for _, _, c in placed]
    for c in centres:  # nothing on a viewer's ahead/behind boundary
        if min(abs(head_coords(c, human)[2]), abs(head_coords(c, robot)[2])) < MIN_DEPTH:
            return None
    front_c, back_c = centres[0], centres[1]
    if not (depth_of(front_c, robot) == "behind" and depth_of(back_c, robot) == "ahead"):
        return None  # the robot's own reading must pick the other object

    ids = _assign_ids(rng, len(placed))
    objects = [WorldObject(i, label, c, set()) for i, (_, label, c) in zip(ids, placed)]
    want_ahead = relation == "front"
    viewer = human if frame == "speaker" else robot
    target = next(o for o in objects[:2]
                  if (depth_of(o.position, viewer) == "ahead") == want_ahead)
    choices = [t for t in templates if t[0] == relation and t[1] == frame]
    _, _, template = choices[int(rng.integers(0, len(choices)))]
    lead = leads[int(rng.integers(0, len(leads)))]
    same_side = [o for o in objects
                 if (depth_of(o.position, viewer) == "ahead") == want_ahead]
    noun = "one" if len(same_side) == 1 and rng.random() < P_ONE else target.color
    text = template.format(lead=lead, noun=noun) + ("." if rng.random() < P_DOT else "")
    view = world.render_head()
    return FBItem(id=item_id, seed=seed, text=text, frame=frame, relation=relation, pool=pool,
                  target_id=target.object_id, object_map=ObjectMap(objects, robot, human),
                  view=view)


def heldout_set(world: FBWorld, n: int = 200) -> list[FBItem]:
    """The gate's test set: seeds HELDOUT_SEEDS[0] + i; blocks of four items alternate
    between training and held-out phrasing, three speaker-frame items then one robot-frame."""
    items = []
    for i in range(n):
        pool = "train" if (i // 4) % 2 == 0 else "heldout"
        frame: Frame = "robot" if i % 4 == 3 else "speaker"
        items.append(make_item(world, HELDOUT_SEEDS[0] + i, f"fbt-{i:04d}", pool, frame))
    return items


# ---------------------------------------------------------------- save / load


def pose_dict(pose: CameraPose) -> dict:
    return {"position": [float(x) for x in pose.position],
            "rotation": [[float(x) for x in row] for row in pose.rotation]}


def _pose(d: dict) -> CameraPose:
    return CameraPose(np.asarray(d["position"], float), np.asarray(d["rotation"], float))


def objects_dict(omap: ObjectMap) -> list[dict]:
    return [{"id": o.object_id, "label": o.color, "xyz": [float(x) for x in o.position]}
            for o in omap.objects]


def save_items(items: list[FBItem], out_dir: Path) -> Path:
    """out_dir/images/<id>.png (head view) and out_dir/items.jsonl."""
    images = out_dir / "images"
    images.mkdir(parents=True, exist_ok=True)
    lines = []
    for it in items:
        Image.fromarray(it.view.rgb).save(images / f"{it.id}.png")
        k = it.view.intrinsics
        lines.append(json.dumps({
            "id": it.id, "seed": it.seed, "kind": KIND, "text": it.text, "frame": it.frame,
            "relation": it.relation, "pool": it.pool, "target_id": it.target_id,
            "objects": objects_dict(it.object_map),
            "human": pose_dict(it.object_map.human_pose),
            "robot": pose_dict(it.object_map.robot_pose),
            "intrinsics": [k.fx, k.fy, k.cx, k.cy, k.width, k.height]}))
    path = out_dir / "items.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def load_items(out_dir: Path) -> list[FBItem]:
    items = []
    for line in (out_dir / "items.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        human, robot = _pose(d["human"]), _pose(d["robot"])
        objects = [WorldObject(o["id"], o["label"], np.asarray(o["xyz"], float), set())
                   for o in d["objects"]]
        fx, fy, cx, cy, w, h = d["intrinsics"]
        rgb = np.asarray(Image.open(out_dir / "images" / f"{d['id']}.png").convert("RGB"))
        view = CameraFrame("human", rgb, Intrinsics(fx, fy, cx, cy, int(w), int(h)), human, 0.0)
        items.append(FBItem(id=d["id"], seed=d["seed"], text=d["text"], frame=d["frame"],
                            relation=d["relation"], pool=d["pool"], target_id=d["target_id"],
                            object_map=ObjectMap(objects, robot, human), view=view))
    return items


def _pattern(template: str) -> re.Pattern:
    body = re.escape(template).replace(re.escape("{lead}"), ".+").replace(
        re.escape("{noun}"), r"\w+")
    return re.compile(body, re.IGNORECASE)


_HELDOUT_PATTERNS = tuple(_pattern(t) for _, _, t in HELDOUT_TEMPLATES)


def heldout_phrasing(text: str) -> bool:
    """True if the text uses a test-only template or lead (it must never be trained on)."""
    starts = any(text.lower().startswith(lead.lower()) for lead in HELDOUT_LEADS)
    return starts or any(p.search(text) for p in _HELDOUT_PATTERNS)
