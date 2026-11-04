"""False-belief scenes for the belief tracker, and how the systems are scored on them.
Task T5-21.

Timeline of every item: the person sits and sees every block; they look behind them or
leave; the robot or a second person moves one block of the target's pair (false-belief
and present items) or nothing (controls); the person returns to the same seat and speaks
at once. The return frame is the current view; the tracker sees it only after the
request, so the request meets the belief from before the person left."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from curio_robot import perspective
from curio_robot.belief import Belief, Grounder, Resolution, resolve, visible_ids
from curio_robot.collab.tasks import TaskScene, apply_scene, fetch_scene, true_map
from curio_robot.conventions.readings import Frame
from curio_robot.sim.scenarios import MIN_GAP, _in_workspace, _random_xy
from curio_robot.sim.world import TABLE_CENTER, BlockSpec, SimWorld
from curio_robot.types import HUMAN_CAM, CameraFrame, CameraPose, ObjectMap

KINDS = ("false-belief", "control", "present")
SEEDS = {"false-belief": 2101, "control": 2102, "present": 2103}  # disjoint from training
COUNTS = {"false-belief": 120, "control": 120, "present": 60}
PAIR_BLOCKS = 4  # blocks 0..3 are fetch_scene's two same-colour pairs
MIN_MOVE = 0.06  # a moved block ends at least this far from where it was, m
RETURN_JITTER = 0.03  # the returning person's gaze point moves by up to this, m
MAX_ATTEMPTS = 200
NOUNS = ("one", "block", "cube")
LEAVE_DISTANCE = 0.8  # m the person walks away from their seat when they leave
LEAVE_MAX_X = 1.3  # stay in front of the wall (sim.world puts it at x = 1.6)
STAND_EYE_Z = 0.75  # m, eye height when standing

# (frame, template). Memory wording names where the person LEFT something.
MEMORY_TEMPLATES: tuple[tuple[Frame, str], ...] = (
    ("speaker", "Get the {color} {noun} I left on my {side}"),
    ("speaker", "Hand me the {color} {noun} I put on my {side}"),
    ("speaker", "Pass me the {color} {noun} that was on my {side}"),
    ("speaker", "Could you get the {color} {noun} I left to my {side} earlier"),
    ("speaker", "Bring me the {color} {noun} I set down on my {side}"),
    ("robot", "Get the {color} {noun} I left on your {side}"),
    ("robot", "Hand me the {color} {noun} I put on your {side}"),
    ("robot", "Pass me the {color} {noun} that was on your {side}"),
    ("robot", "Could you get the {color} {noun} I left to your {side} earlier"),
    ("robot", "Bring me the {color} {noun} I set down on your {side}"),
)
PRESENT_TEMPLATES: tuple[tuple[Frame, str], ...] = (
    ("speaker", "Get the {color} {noun} on my {side}"),
    ("speaker", "Hand me the {color} {noun} that is on my {side} now"),
    ("speaker", "Pass me the {color} {noun} on my {side}"),
    ("robot", "Get the {color} {noun} on your {side}"),
    ("robot", "Hand me the {color} {noun} that is on your {side} now"),
    ("robot", "Pass me the {color} {noun} on your {side}"),
)


@dataclass(frozen=True, slots=True)
class Observation:
    frame: CameraFrame  # head camera, no segmentation
    truth: ObjectMap  # the true table at that moment
    visible: tuple[str, ...]  # ids with >= MIN_VISIBLE_PX head-camera pixels


@dataclass(frozen=True, slots=True)
class BeliefItem:
    id: str
    kind: str  # one of KINDS
    text: str
    marked_text: str  # bench "marked" wording for the same target, the v2 reference
    frame: Frame
    target_id: str
    moved_id: str | None
    mover: str | None  # "robot" or "person"
    away: str  # "look-away" or "leave"
    observations: tuple[Observation, ...]  # what the tracker sees before the request
    truth: ObjectMap  # the table when the person speaks
    current: CameraFrame  # the person's head frame when they speak

    @property
    def target_moved(self) -> bool:
        """The spoken note is due: the target moved while the person couldn't see it."""
        return self.moved_id is not None and self.moved_id == self.target_id


def _head(
    world: SimWorld, eye: np.ndarray, look: np.ndarray, t: float, truth: ObjectMap
) -> tuple[CameraFrame, tuple[str, ...], int]:
    """Render the head camera at a pose: (frame without segmentation, visible ids,
    block pixels in total)."""
    world.set_human(eye, look)
    frame = world.render_camera(HUMAN_CAM, segmentation=True)
    geoms = {o.object_id: world._block_geoms[i] for i, o in enumerate(truth.objects)}
    seen = visible_ids(frame, truth, geoms)
    pixels = int(np.isin(frame.segmentation, list(geoms.values())).sum())
    return replace(frame, segmentation=None, timestamp=t), seen, pixels


def _with_pose(omap: ObjectMap, human: CameraPose) -> ObjectMap:
    return ObjectMap(omap.objects, omap.robot_pose, human)


def _away_pose(scene: TaskScene, away: str) -> tuple[np.ndarray, np.ndarray]:
    """Looking behind them from the seat, or standing up and walking away from the table."""
    eye = np.asarray(scene.human_eye, float)
    out = eye[:2] - TABLE_CENTER
    out /= np.linalg.norm(out)
    if away == "look-away":
        return eye, np.array([*(eye[:2] + out), eye[2]])
    stand = eye[:2] + LEAVE_DISTANCE * out
    stand[0] = min(stand[0], LEAVE_MAX_X)
    return np.array([*stand, STAND_EYE_Z]), np.array([*(stand + out), STAND_EYE_Z])


def _moved_scene(
    rng: np.random.Generator, scene: TaskScene, idx: int, check: Callable[[TaskScene], bool]
) -> TaskScene | None:
    """Block idx at a new free spot >= MIN_MOVE from its old one for which check holds."""
    old = np.asarray(scene.blocks[idx].xy)
    others = [np.asarray(b.xy) for i, b in enumerate(scene.blocks) if i != idx]
    for _ in range(MAX_ATTEMPTS):
        xy = _random_xy(rng)
        if not _in_workspace(xy) or np.linalg.norm(xy - old) < MIN_MOVE:
            continue
        if any(np.linalg.norm(xy - o) <= MIN_GAP for o in others):
            continue
        blocks = list(scene.blocks)
        blocks[idx] = BlockSpec(blocks[idx].color, (float(xy[0]), float(xy[1])))
        moved = replace(scene, blocks=tuple(blocks))
        if check(moved):
            return moved
    return None


def _pick_id(omap: ObjectMap, color: str, side: str, viewer: CameraPose) -> str:
    return perspective.pick(omap.by_color(color), side, viewer).object_id


def _ambiguous(omap: ObjectMap, color: str, side: str, human: CameraPose) -> bool:
    return _pick_id(omap, color, side, human) != _pick_id(omap, color, side, omap.robot_pose)


def _try_item(world: SimWorld, rng: np.random.Generator, kind: str, i: int) -> BeliefItem | None:
    scene = fetch_scene(rng)
    idx = int(rng.integers(0, PAIR_BLOCKS))
    frame: Frame = "speaker" if rng.random() < 0.5 else "robot"
    templates = PRESENT_TEMPLATES if kind == "present" else MEMORY_TEMPLATES
    choices = [t for f, t in templates if f == frame]
    template = choices[int(rng.integers(0, len(choices)))]
    noun = NOUNS[int(rng.integers(0, len(NOUNS)))]
    away = "look-away" if rng.random() < 0.5 else "leave"
    mover = "robot" if rng.random() < 0.5 else "person"
    move_target = i % 2 == 0  # false-belief items: target and partner moved half each
    jitter = rng.uniform(-RETURN_JITTER, RETURN_JITTER, 2)

    apply_scene(world, scene)
    before = true_map(scene)
    seen, seen_ids, _ = _head(world, scene.human_eye, scene.human_look, 0.0, before)
    if len(seen_ids) != len(before.objects):
        return None  # the person must see the whole table before they go
    before = _with_pose(before, seen.pose)
    target = before.objects[idx]
    color = target.color
    viewer0 = seen.pose if frame == "speaker" else before.robot_pose
    on_left = _pick_id(before, color, "left", viewer0) == target.object_id
    side = "left" if on_left else "right"
    if not _ambiguous(before, color, side, seen.pose):
        return None
    partner = next(j for j in range(PAIR_BLOCKS) if j != idx and scene.blocks[j].color == color)

    eye_a, look_a = _away_pose(scene, away)
    gone, gone_ids, gone_px = _head(world, eye_a, look_a, 1.0, before)
    if gone_px:
        return None  # away means not a single block pixel
    look_back = np.asarray(scene.human_look, float) + np.array([*jitter, 0.0])
    world.set_human(scene.human_eye, look_back)
    now_pose = world.render_camera(HUMAN_CAM).pose
    viewer_now = now_pose if frame == "speaker" else before.robot_pose

    def flips(s: TaskScene) -> bool:
        return _pick_id(true_map(s), color, side, viewer_now) != target.object_id

    moved_id = None
    after_scene = scene
    if kind != "control":
        j = idx if move_target else partner
        after_scene = _moved_scene(rng, scene, j, flips)
        if after_scene is None:
            return None
        moved_id = before.objects[j].object_id

    apply_scene(world, after_scene)
    after = true_map(after_scene)
    away_after, away_ids, away_px = _head(world, eye_a, look_a, 2.0, after)
    if away_px:
        return None
    current, now_ids, _ = _head(world, scene.human_eye, look_back, 3.0, after)
    if len(now_ids) != len(after.objects):
        return None
    after = _with_pose(after, current.pose)

    target_id = target.object_id
    if kind == "present":
        # Present tense: the words mean the block that is on that side now.
        target_id = _pick_id(after, color, side, viewer_now)
        if not _ambiguous(after, color, side, current.pose):
            return None
    m = "my" if frame == "speaker" else "your"
    return BeliefItem(
        id=f"{kind}-{i:04d}",
        kind=kind,
        text=template.format(color=color, noun=noun, side=side),
        marked_text=f"Hand me the {color} one on {m} {side}",
        frame=frame,
        target_id=target_id,
        moved_id=moved_id,
        mover=mover if moved_id else None,
        away=away,
        observations=(
            Observation(seen, before, seen_ids),
            Observation(gone, before, gone_ids),
            Observation(away_after, after, away_ids),
        ),
        truth=after,
        current=current,
    )


def make_item(world: SimWorld, rng: np.random.Generator, kind: str, i: int) -> BeliefItem:
    if kind not in KINDS:
        raise ValueError(f"unknown kind: {kind}")
    for _ in range(MAX_ATTEMPTS):
        item = _try_item(world, rng, kind, i)
        if item is not None:
            return item
    raise RuntimeError(f"no {kind} item in {MAX_ATTEMPTS} attempts")


def generate(
    world: SimWorld, counts: dict[str, int] | None = None, seeds: dict[str, int] | None = None
) -> list[BeliefItem]:
    """Items kind by kind, each kind from its own rng (SEEDS), COUNTS by default."""
    counts = COUNTS if counts is None else counts
    seeds = SEEDS if seeds is None else seeds
    items: list[BeliefItem] = []
    for kind, n in counts.items():
        rng = np.random.default_rng(seeds[kind])
        items += [make_item(world, rng, kind, i) for i in range(n)]
    return items


# ---- scoring ------------------------------------------------------------------------------

SYSTEMS = ("current", "belief", "reference", "belief-raw")


def tracked(item: BeliefItem) -> Belief:
    """The person's belief when they speak: every observation before the request."""
    belief = Belief("person")
    for obs in item.observations:
        belief = belief.observe(obs.frame, obs.truth, obs.visible)
    return belief


def run(item: BeliefItem, grounder: Grounder, system: str) -> Resolution:
    """current: the live pipeline on the current view; belief: belief-aware; reference:
    the current view with the bench "marked" wording; belief-raw: belief-aware without
    taking the memory clause out of the words (an ablation)."""
    if system not in SYSTEMS:
        raise ValueError(f"unknown system: {system}")
    belief = tracked(item)
    movers = {item.moved_id: item.mover} if item.moved_id and item.mover else {}
    text = item.marked_text if system == "reference" else item.text
    return resolve(
        text, grounder, item.current, item.truth, belief, movers,
        use_belief=system.startswith("belief"), rewrite=system != "belief-raw",
    )


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes in n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def paired_diff_ci(
    a: list[bool], b: list[bool], n_boot: int = 10000, seed: int = 0
) -> tuple[float, float, float]:
    """mean(a) - mean(b) over the same items, with a 95% paired-bootstrap interval."""
    x = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    boots = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def rate(hits: list[bool]) -> dict:
    k, n = int(sum(hits)), len(hits)
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "acc": k / n if n else 0.0, "ci95": [lo, hi]}
