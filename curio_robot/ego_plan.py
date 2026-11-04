"""Teach by doing (task T5-23): a task plan from the person's head camera alone.

The person does the task once; the glasses give RGB-D frames and their own head pose. Blocks
and the hand are found by colour and lifted to 3-D with depth (the project's detector reads
segmentation ground truth, which a real head camera doesn't have, so it isn't used here).
Blocks are tracked; a block that leaves its rest with the hand beside it and comes to rest
again once the hand has let go is one step. Each step's placement is written down the way the
person would say it, in their frame and relative to the other objects: "on blue_1", "left of
red_1 (your left)", "in your right zone". That is what makes the plan portable: on a new layout
the same words still mean something, where copied coordinates would not.

Object ids are colour_k, k counting same-coloured blocks from the person's left at the start."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from curio_robot.types import CameraPose, Intrinsics

BLOCK_HALF = 0.0125
TABLE_POINT = np.array([0.24, 0.0])  # C: the middle of the shared workspace (world xy)
ZONE_EDGE = 0.05  # m from C along the person's left axis: inside "my left/right zone"
ADJ_RANGE = (0.035, 0.08)  # m: a block this far from another, sideways, is "left/right of" it
ADJ_ANGLE = np.radians(30)  # off the person's left axis, for "left/right of"
ON_XY = 0.02  # m: stacked blocks are this well centred
ON_DZ = (0.015, 0.035)  # m: centre heights of a block on another
SAME_LEVEL = 0.012  # m between centre heights of side-by-side blocks

# Colours, as hue (deg) and least saturation of the rendered blocks; the hand by its own hue.
HUES = {"red": 0.0, "yellow": 48.0, "green": 140.0, "blue": 222.0}
HUE_TOL = 18.0
MIN_SAT = 0.55
HAND_HUE = (8.0, 28.0)
HAND_SAT = (0.3, 0.62)
HAND_MIN_Z = 0.03  # m above the table: the hand, not the table, which has a similar hue
MIN_POINTS = 25  # pixels of a block (a whole one is 250-450 at 640x480)
MIN_TOP = 0.015  # m: a block's highest point. Edge pixels whose depth hit the table behind
# a block carry its colour but lie flat on the table: REGION starts 4 mm up to drop them (they
# would join neighbouring blocks into one cluster), and a flat cluster is not a block
MAX_EXTENT = 0.045  # m: bigger same-coloured things (the yellow arm) aren't blocks
VOXEL = 0.01  # m, for clustering a colour's points
REGION = ((0.09, 0.55), (-0.32, 0.32), (0.004, 0.25))  # x, y, z where blocks can be

START_MAX_Z = 0.03  # m: block centres at the start are on the table (0.0125, give or take)
TRACK_GATE = 0.03  # m a block at rest can seem to move between frames
MOVING_GATE = 0.15  # m for a block in the hand, which can be hidden by it for a frame or two
MOVE_START = 0.025  # m off its rest: a block has left it
MOVE_MIN = 0.03  # m between rests for a step to count
GONE_FRAMES = 3  # frames the hand stays away from a moved block: it has been let go
HAND_NEAR = 0.08  # m: the hand is at the block when it starts to move
HAND_GONE = 0.05  # m: the hand has let go
HAND_OVER = 0.035  # m sideways: the hand is over a block (to pick it up)


@dataclass(frozen=True, slots=True)
class EgoFrame:
    """One head-camera frame. depth is metres along the camera axis."""

    rgb: np.ndarray  # (H, W, 3) uint8
    depth: np.ndarray  # (H, W) float
    intrinsics: Intrinsics
    pose: CameraPose
    t: float


@dataclass(frozen=True, slots=True)
class Blob:
    color: str
    position: np.ndarray  # block centre (world, m)
    points: int


@dataclass(frozen=True, slots=True)
class Step:
    """Pick obj, then place it relative to ref.

    relation: "on" (ref an object id), "left_of" / "right_of" (ref an object id, the person's
    left/right), "zone" (ref "left" / "right", the person's), "at" (ref "s,t": metres along
    the person's left and forward axes from TABLE_POINT; the fallback)."""

    obj: str
    relation: str
    ref: str

    def text(self) -> str:
        what = {"on": f"on {self.ref}", "left_of": f"left of {self.ref} (your left)",
                "right_of": f"right of {self.ref} (your right)",
                "zone": f"in your {self.ref} zone", "at": f"at {self.ref} (your frame)"}
        return f"pick {self.obj}, put it {what[self.relation]}"


@dataclass(frozen=True, slots=True)
class PersonFrame:
    """The person's table frame: left axis and forward (away from them) axis, horizontal."""

    left: np.ndarray  # (3,), z = 0
    forward: np.ndarray  # (3,), z = 0

    @staticmethod
    def from_left(left: np.ndarray) -> PersonFrame:
        lft = np.array([left[0], left[1], 0.0])
        lft /= np.linalg.norm(lft)
        return PersonFrame(lft, np.array([lft[1], -lft[0], 0.0]))  # left turned -90 deg

    @staticmethod
    def from_poses(poses: list[CameraPose]) -> PersonFrame:
        """Minus the mean head right axis, flattened: the seat's left, not a glance's."""
        return PersonFrame.from_left(-np.mean([p.rotation[:, 0] for p in poses], axis=0))

    def coords(self, p: np.ndarray) -> tuple[float, float]:
        """(s, t): metres along left and forward from TABLE_POINT."""
        d = np.array([p[0] - TABLE_POINT[0], p[1] - TABLE_POINT[1], 0.0])
        return float(d @ self.left), float(d @ self.forward)

    def point(self, s: float, t: float, z: float = BLOCK_HALF) -> np.ndarray:
        xy = TABLE_POINT + s * self.left[:2] + t * self.forward[:2]
        return np.array([xy[0], xy[1], z])


ROBOT_FRAME = PersonFrame.from_left(np.array([0.0, 1.0, 0.0]))  # the robot's own left (+y)


@dataclass(frozen=True, slots=True)
class Plan:
    steps: tuple[Step, ...]
    frame: PersonFrame
    start: dict[str, np.ndarray] = field(default_factory=dict)  # id -> start position

    def text(self) -> list[str]:
        return [f"{k + 1}. {s.text()}" for k, s in enumerate(self.steps)]


# ---- naming and relations (shared with the task generator and the retargeting) -----------


def name_objects(blobs: list[tuple[str, np.ndarray]], frame: PersonFrame) -> list[str]:
    """colour_k for each (colour, position), k from the person's left (1 = leftmost)."""
    names = [""] * len(blobs)
    for color in {c for c, _ in blobs}:
        idx = [i for i, (c, _) in enumerate(blobs) if c == color]
        idx.sort(key=lambda i: -frame.coords(blobs[i][1])[0])
        for k, i in enumerate(idx):
            names[i] = f"{color}_{k + 1}"
    return names


def classify(p: np.ndarray, others: dict[str, np.ndarray], frame: PersonFrame) -> Step:
    """How the person would say where a block at p was put, given the other blocks.

    First match wins: on another block; beside one (ADJ_RANGE, within ADJ_ANGLE of the left
    axis, same level), the nearest; in a zone; else a pose in the person's frame.
    Returns a Step with an empty obj."""
    on = [(np.linalg.norm((p - q)[:2]), k) for k, q in others.items()
          if np.linalg.norm((p - q)[:2]) <= ON_XY and ON_DZ[0] <= p[2] - q[2] <= ON_DZ[1]]
    if on:
        return Step("", "on", min(on)[1])
    beside = []
    for k, q in others.items():
        d = (p - q)[:2]
        dist = float(np.linalg.norm(d))
        if not ADJ_RANGE[0] <= dist <= ADJ_RANGE[1] or abs(p[2] - q[2]) > SAME_LEVEL:
            continue
        along = float(d @ frame.left[:2]) / dist
        if abs(along) >= np.cos(ADJ_ANGLE):
            beside.append((dist, k, "left_of" if along > 0 else "right_of"))
    if beside:
        _, k, rel = min(beside)
        return Step("", rel, k)
    s, t = frame.coords(p)
    if s >= ZONE_EDGE:
        return Step("", "zone", "left")
    if s <= -ZONE_EDGE:
        return Step("", "zone", "right")
    return Step("", "at", f"{s:.2f},{t:.2f}")


# ---- retargeting: a step's words -> a goal point on a new layout ---------------------------

REACH_R = (0.13, 0.31)  # m from the SO-101 base: where the expert picks and places reliably
REACH_MIN_X = 0.10
BESIDE = 0.06  # m from the reference block for "left/right of"
BESIDE_TRIES = [(d, a) for d in (0.06, 0.07, 0.05) for a in (0, 15, -15, 30, -30)]
ZONE_S = (0.10, 0.085, 0.115)  # m along the left axis: the middle of a zone, then either side
ZONE_T = (0.0, -0.05, 0.05, -0.10, 0.10)  # slots along the forward axis, tried in turn
FREE = 0.05  # m between block centres


def reachable(p: np.ndarray) -> bool:
    r = float(np.hypot(p[0], p[1]))
    return REACH_R[0] <= r <= REACH_R[1] and p[0] >= REACH_MIN_X


def free(p: np.ndarray, others: dict[str, np.ndarray], gap: float = FREE) -> bool:
    return all(np.linalg.norm((p - q)[:2]) >= gap for q in others.values())


def _turn(v: np.ndarray, deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([v[0] * np.cos(a) - v[1] * np.sin(a), v[0] * np.sin(a) + v[1] * np.cos(a), 0])


def goal_for(step: Step, positions: dict[str, np.ndarray], frame: PersonFrame,
             gap: float = FREE, beside: list[tuple[float, float]] | None = None
             ) -> np.ndarray | None:
    """Where step.obj's centre should rest, reading the step's words in frame (the person's
    frame for Curio; ROBOT_FRAME for the baseline that ignores perspective). positions holds
    every block's current centre by id. None if the reference is unknown or nothing fits.
    beside: (distance, degrees off the axis) tried for left/right of, BESIDE_TRIES if None."""
    others = {k: v for k, v in positions.items() if k != step.obj}
    if step.relation in ("on", "left_of", "right_of") and step.ref not in others:
        return None
    if step.relation == "on":
        return others[step.ref] + [0, 0, 2 * BLOCK_HALF]
    if step.relation in ("left_of", "right_of"):
        side = frame.left if step.relation == "left_of" else -frame.left
        base = others[step.ref].copy()
        base[2] = BLOCK_HALF
        for dist, ang in BESIDE_TRIES if beside is None else beside:
            g = base + dist * _turn(side, ang)
            if reachable(g) and free(g, others, gap):
                return g
        return None
    if step.relation == "zone":
        sign = 1.0 if step.ref == "left" else -1.0
        for s, t in ((s, t) for s in ZONE_S for t in ZONE_T):
            g = frame.point(sign * s, t)
            if reachable(g) and free(g, others, gap):
                return g
        return None
    s, t = (float(v) for v in step.ref.split(","))
    g = frame.point(s, t)
    return g if reachable(g) else None


# ---- perception --------------------------------------------------------------------------


def hue_sat(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hue (deg, 0-360) and saturation (0-1) of uint8 RGB pixels."""
    c = rgb.astype(np.float32) / 255.0
    mx, mn = c.max(-1), c.min(-1)
    span = mx - mn + 1e-6
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    h = np.where(mx == r, (g - b) / span % 6, np.where(mx == g, (b - r) / span + 2,
                                                       (r - g) / span + 4))
    return h * 60.0, span / (mx + 1e-6)


def back_project(depth: np.ndarray, k: Intrinsics, pose: CameraPose) -> np.ndarray:
    """(H, W, 3) world points of every pixel."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    cam = np.stack([(u - k.cx) / k.fx * depth, -(v - k.cy) / k.fy * depth, -depth], -1)
    return cam @ pose.rotation.T + pose.position


def _clusters(points: np.ndarray) -> list[np.ndarray]:
    """Groups of points whose VOXEL cells touch (26-neighbourhood)."""
    cells = np.floor(points / VOXEL).astype(np.int64)
    members: dict[tuple, list[int]] = {}
    for i, c in enumerate(map(tuple, cells)):
        members.setdefault(c, []).append(i)
    seen: set[tuple] = set()
    out = []
    offsets = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)]
    for start in members:
        if start in seen:
            continue
        seen.add(start)
        queue, idx = deque([start]), []
        while queue:
            cell = queue.popleft()
            idx += members[cell]
            for o in offsets:
                n = (cell[0] + o[0], cell[1] + o[1], cell[2] + o[2])
                if n in members and n not in seen:
                    seen.add(n)
                    queue.append(n)
        out.append(points[idx])
    return out


def _in_region(p: np.ndarray) -> np.ndarray:
    ok = np.ones(len(p), bool)
    for axis, (lo, hi) in enumerate(REGION):
        ok &= (p[:, axis] >= lo) & (p[:, axis] <= hi)
    return ok


def _block_centre(pts: np.ndarray) -> np.ndarray:
    """Centre from the visible top face: its mean xy, half a block under its height."""
    top = np.percentile(pts[:, 2], 90)
    face = pts[pts[:, 2] >= top - 0.004]
    return np.array([*face[:, :2].mean(0), max(top - BLOCK_HALF, BLOCK_HALF * 0.8)])


def detect(frame: EgoFrame) -> tuple[list[Blob], np.ndarray | None]:
    """Blocks (by colour) and the hand's position in one RGB-D frame."""
    hue, sat = hue_sat(frame.rgb)
    pts = back_project(frame.depth, frame.intrinsics, frame.pose)
    valid = np.isfinite(frame.depth) & (frame.depth > 0.02)
    blobs = []
    for color, h0 in HUES.items():
        dh = np.abs((hue - h0 + 180.0) % 360.0 - 180.0)
        mask = valid & (dh <= HUE_TOL) & (sat >= MIN_SAT)
        p = pts[mask]
        p = p[_in_region(p)]
        for c in _clusters(p) if len(p) else []:
            extent = c.max(0) - c.min(0)
            if (len(c) >= MIN_POINTS and extent[:2].max() <= MAX_EXTENT
                    and c[:, 2].max() >= MIN_TOP):
                blobs.append(Blob(color, _block_centre(c), len(c)))
    hand_mask = (valid & (hue >= HAND_HUE[0]) & (hue <= HAND_HUE[1])
                 & (sat >= HAND_SAT[0]) & (sat <= HAND_SAT[1]))
    hp = pts[hand_mask]
    hp = hp[(hp[:, 2] >= HAND_MIN_Z) & _in_region(hp)]
    hand = np.median(hp, axis=0) if len(hp) >= MIN_POINTS else None
    return blobs, hand


# ---- tracking and steps --------------------------------------------------------------------


@dataclass(slots=True)
class _Track:
    """A block over time. Between moves it is at rest, and its rest position is the median of
    every sighting since it came to rest (single frames jitter by about a centimetre: depth,
    and the glasses' pose)."""

    color: str
    position: np.ndarray  # last sighting
    sightings: list[np.ndarray] = field(default_factory=list)  # since the current rest began
    moving: bool = False
    grasp_t: float = 0.0
    gone: int = 0  # frames the hand has been away while moving
    start_rest: np.ndarray | None = None  # where the current move began
    rests: list[tuple[float, list[np.ndarray]]] = field(default_factory=list)  # (from t, seen)

    @property
    def rest(self) -> np.ndarray:
        return np.median(self.sightings, axis=0) if self.sightings else self.position

    def rest_at(self, t: float) -> np.ndarray:
        """Median position of the rest that started last at or before t."""
        seen = [r for r in self.rests if r[0] <= t]
        pts = seen[-1][1] if seen else self.rests[0][1]
        return np.median(pts, axis=0) if pts else self.position


@dataclass(frozen=True, slots=True)
class Move:
    obj: str
    track: int
    grasp_t: float
    release_t: float


def _assign(tracks: list[_Track], blobs: list[Blob]) -> list[Blob | None]:
    """Each track's detection: same colour, nearest first, within its gate. A block at rest
    keeps a tight gate, so a block carried past it is not taken for it."""
    pairs = sorted((float(np.linalg.norm(t.position - b.position)), ti, bi)
                   for ti, t in enumerate(tracks) for bi, b in enumerate(blobs)
                   if t.color == b.color)
    out: list[Blob | None] = [None] * len(tracks)
    used: set[int] = set()
    for dist, ti, bi in pairs:
        gate = MOVING_GATE if tracks[ti].moving else TRACK_GATE
        if dist > gate or out[ti] is not None or bi in used:
            continue
        out[ti] = blobs[bi]
        used.add(bi)
    return out


class Extractor:
    """Feed head-camera frames in order with add(); plan() gives the steps."""

    def __init__(self, init_frames: int = 3) -> None:
        self.init_frames = init_frames
        self._seed: list[list[Blob]] = []
        self.tracks: list[_Track] = []
        self.names: list[str] = []
        self.moves: list[Move] = []
        self.poses: list[CameraPose] = []
        self.hand_log: list[np.ndarray | None] = []
        self.fixtures: list[np.ndarray] = []

    def add(self, frame: EgoFrame) -> None:
        blobs, hand = detect(frame)
        self.poses.append(frame.pose)
        self.hand_log.append(hand)
        if len(self._seed) < self.init_frames:
            self._seed.append(blobs)
            if len(self._seed) == self.init_frames:
                self._start(frame.t)
            return
        blobs = [b for b in blobs
                 if all(np.linalg.norm(b.position - q) > 0.03 for q in self.fixtures)]
        matched = _assign(self.tracks, blobs)
        self._carried(matched, blobs, hand)
        for k, (track, blob) in enumerate(zip(self.tracks, matched)):
            if blob is not None:
                track.position = blob.position
            self._update(k, track, blob is not None, hand, frame.t)

    def _carried(self, matched: list[Blob | None], blobs: list[Blob],
                 hand: np.ndarray | None) -> None:
        """A block at rest went unseen under the hand and reappears in it, further off than
        a resting block's gate: an unclaimed same-coloured blob at the hand is that block."""
        if hand is None:
            return
        used = {id(b) for b in matched if b is not None}
        for k, track in enumerate(self.tracks):
            if matched[k] is not None or track.moving:
                continue
            if np.linalg.norm((hand - track.position)[:2]) > HAND_OVER:
                continue
            free_ = [b for b in blobs if b.color == track.color and id(b) not in used
                     and np.linalg.norm(b.position - hand) <= HAND_NEAR
                     and np.linalg.norm(b.position - track.position) <= MOVING_GATE]
            if free_:
                best = min(free_, key=lambda b: np.linalg.norm(b.position - hand))
                matched[k] = best
                used.add(id(best))

    def _start(self, t: float) -> None:
        """Tracks from the start frames: the frame with most blocks, positions averaged. Every
        block starts on the table; anything else block-coloured (a part of the robot arm) is
        a fixture, and detections there are ignored from then on."""
        base = max(self._seed, key=len)
        self.tracks = []
        self.fixtures = [b.position for b in base if b.position[2] > START_MAX_Z]
        for b in (b for b in base if b.position[2] <= START_MAX_Z):
            near = [o.position for f in self._seed for o in f
                    if o.color == b.color and np.linalg.norm(o.position - b.position) < 0.02]
            track = _Track(b.color, np.mean(near, axis=0), list(near))
            track.rests.append((0.0, track.sightings))
            self.tracks.append(track)
        frame = PersonFrame.from_poses(self.poses)
        self.names = name_objects([(tr.color, tr.rest) for tr in self.tracks], frame)

    def _update(self, k: int, track: _Track, seen: bool, hand: np.ndarray | None,
                t: float) -> None:
        if not track.moving:
            near = hand is not None and np.linalg.norm(hand - track.position) <= HAND_NEAR
            lifted = track.position[2] > START_MAX_Z  # people lift what they move
            off = np.linalg.norm(track.position - track.rest) > MOVE_START
            if seen and near and lifted and off:
                track.moving, track.grasp_t, track.gone = True, t, 0
                track.start_rest = track.rest
            elif seen:
                track.sightings.append(track.position.copy())
            return
        away = hand is None or np.linalg.norm(hand - track.position) > HAND_GONE
        track.gone = track.gone + 1 if away and self._supported(track) else 0
        if track.gone < GONE_FRAMES:
            return
        track.moving = False
        before = track.start_rest if track.start_rest is not None else track.position
        track.sightings = [track.position.copy()]
        track.rests.append((t, track.sightings))
        if np.linalg.norm(track.position - before) >= MOVE_MIN:
            self.moves.append(Move(self.names[k], k, track.grasp_t, t))

    def _supported(self, track: _Track) -> bool:
        """On the table or on another block: let go of, not just apart from the hand."""
        p = track.position
        if p[2] <= START_MAX_Z:
            return True
        return any(o is not track and np.linalg.norm((p - o.position)[:2]) <= ON_XY * 2
                   and ON_DZ[0] <= p[2] - o.position[2] <= ON_DZ[1] for o in self.tracks)

    def plan(self) -> Plan:
        """Ids are given here, from the start rests in the frame of the whole demo (the first
        few head poses alone can be a glance to one side)."""
        frame = PersonFrame.from_poses(self.poses)
        self.names = name_objects([(tr.color, tr.rest_at(0.0)) for tr in self.tracks], frame)
        steps = []
        for m in sorted(self.moves, key=lambda m: m.grasp_t):
            final = self.tracks[m.track].rest_at(m.release_t)
            if np.linalg.norm(final - self.tracks[m.track].rest_at(m.grasp_t - 1e-6)) < MOVE_MIN:
                continue  # a nudge, not a step, once the rests are averaged
            scene = {n: tr.rest_at(m.release_t) for n, tr in zip(self.names, self.tracks)
                     if tr is not self.tracks[m.track]}
            rel = classify(final, scene, frame)
            steps.append(Step(self.names[m.track], rel.relation, rel.ref))
        start = {n: tr.rests[0][1][0] for n, tr in zip(self.names, self.tracks) if tr.rests}
        return Plan(tuple(steps), frame, start)


def extract(frames: list[EgoFrame]) -> Plan:
    ex = Extractor()
    for f in frames:
        ex.add(f)
    return ex.plan()
