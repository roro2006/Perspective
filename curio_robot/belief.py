"""What each person believes about where things are (theory of mind). Task T5-21.

Not to be confused with conventions/: that module learns whose "left" a person means;
this one tracks the object map in a person's head. A person's belief changes only when
their head camera can actually see an object, so anything moved while they looked away
or were out of the room stays where they last saw it. The robot's own map is the truth.

Requests that refer to the person's memory ("the red one I left on my left") are grounded
on the belief: the stored head-camera frame from when they last saw the scene, marked at
the believed positions. The chosen id is then mapped to where it is now. Present-tense
requests are grounded on the current view, as the live pipeline does."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Protocol

import numpy as np

from curio_robot.geometry import project
from curio_robot.grounding.client import GroundingAnswer
from curio_robot.grounding.marks import place_marks, render_marks
from curio_robot.grounding.prompt import GroundingRequest, build_request
from curio_robot.perspective import horizontal_distance, lateral
from curio_robot.types import CameraFrame, CameraPose, ObjectMap, WorldObject

MOVED_THRESHOLD = 0.03  # m: believed vs true position beyond this is a false belief
MIN_VISIBLE_PX = 20  # head-camera pixels for an object to count as seen (as bench.items)
NEARER = 0.03  # m: a move this much closer to / further from the person is worth saying

# Words that point at the person's memory of the table rather than at what is there now.
_MEMORY = re.compile(
    r"\b(?:I|we)\s+(?:left|put|placed|set|had|dropped|kept|saw)\b"
    r"|\b(?:was|were|earlier|before|used to|last time)\b",
    re.IGNORECASE,
)


# The memory clause itself, removed before grounding on the belief: the stored frame
# already says "back then", and "I left on my right" puts a stray "left" in the words.
_CLAUSE = re.compile(
    r"\s+(?:that\s+(?:was|were)|(?:I|we)\s+(?:left|put|placed|set(?:\s+down)?|had|dropped|kept))"
    r"(?=\s+(?:on|to|at|in|by|over)\b)",
    re.IGNORECASE,
)
_WHEN = re.compile(r"\s+(?:earlier|before|last time)\b", re.IGNORECASE)


class Grounder(Protocol):
    def score(self, request: GroundingRequest) -> GroundingAnswer: ...


@dataclass(frozen=True, slots=True)
class Sighting:
    object_id: str
    color: str
    position: np.ndarray  # (3,) where the person last saw it
    time: float


@dataclass(frozen=True, slots=True)
class Belief:
    """One person's object map. Immutable: observe() returns the updated belief."""

    person: str
    sightings: Mapping[str, Sighting] = field(default_factory=dict)
    last_view: CameraFrame | None = None  # latest head frame that showed any object

    def observe(self, frame: CameraFrame, truth: ObjectMap, visible: Iterable[str]) -> Belief:
        """Every visible object is now believed where the truth has it, at frame.timestamp.
        The frame (without segmentation) becomes last_view when it showed something."""
        seen = set(visible)
        if not seen:
            return self
        updated = dict(self.sightings)
        for obj in truth.objects:
            if obj.object_id in seen:
                updated[obj.object_id] = Sighting(
                    obj.object_id, obj.color, obj.position.copy(), frame.timestamp
                )
        return replace(self, sightings=updated, last_view=replace(frame, segmentation=None))

    def believed_map(self, robot_pose: CameraPose) -> ObjectMap | None:
        """The table as the person remembers it, seen from where they last looked;
        None before they have seen anything."""
        if self.last_view is None:
            return None
        objects = [
            WorldObject(s.object_id, s.color, s.position.copy(), {self.person})
            for s in sorted(self.sightings.values(), key=lambda s: s.object_id)
        ]
        return ObjectMap(objects, robot_pose, self.last_view.pose)

    def false_beliefs(self, truth: ObjectMap, threshold: float = MOVED_THRESHOLD) -> set[str]:
        """Ids whose believed position is more than `threshold` from the true one."""
        true_pos = {o.object_id: o.position for o in truth.objects}
        return {
            oid
            for oid, s in self.sightings.items()
            if oid in true_pos and float(np.linalg.norm(true_pos[oid] - s.position)) > threshold
        }


def visible_ids(
    frame: CameraFrame,
    truth: ObjectMap,
    geoms: Mapping[str, int] | None = None,
    min_px: int = MIN_VISIBLE_PX,
) -> tuple[str, ...]:
    """Objects the camera can see. With a segmentation and each object's geom id (sim):
    at least `min_px` pixels, so occlusion counts. Otherwise: the centre projects in
    front of the camera and inside the image (no occlusion test)."""
    if frame.segmentation is not None and geoms is not None:
        return tuple(
            o.object_id
            for o in truth.objects
            if o.object_id in geoms and int((frame.segmentation == geoms[o.object_id]).sum()) >= min_px
        )
    k = frame.intrinsics
    out = []
    for o in truth.objects:
        px = project(o.position, k, frame.pose)
        if px is not None and 0 <= px[0] < k.width and 0 <= px[1] < k.height:
            out.append(o.object_id)
    return tuple(out)


def refers_to_memory(text: str) -> bool:
    """True for wording about where the person left or saw something."""
    return _MEMORY.search(text) is not None


def present_reading(text: str) -> str:
    """The words with the memory clause taken out, for grounding on the remembered view:
    "Get the red one I left on my right" -> "Get the red one on my right"."""
    return re.sub(r"\s+", " ", _WHEN.sub("", _CLAUSE.sub("", text))).strip()


def ground_request(text: str, omap: ObjectMap, frame: CameraFrame) -> GroundingRequest:
    """The live pipeline's request (the bench "geometry" condition) for any map and frame."""
    marks = place_marks(omap, frame)
    image, _ = render_marks(frame, marks)
    return build_request(text, omap, marks, image)


def moved_note(obj: WorldObject, believed: np.ndarray, person: CameraPose, mover: str) -> str:
    """What the robot says about an object moved while the person couldn't see it, in the
    person's own frame: "I moved it; it's now on your right, nearer you." """
    who = "I moved it" if mover == "robot" else "Someone moved it"
    side = "left" if lateral(obj, person) < 0 else "right"
    was = WorldObject(obj.object_id, obj.color, np.asarray(believed, float))
    change = horizontal_distance(obj, person) - horizontal_distance(was, person)
    depth = ""
    if change < -NEARER:
        depth = ", nearer you"
    elif change > NEARER:
        depth = ", further from you"
    return f"{who}; it's now on your {side}{depth}."


@dataclass(frozen=True, slots=True)
class Resolution:
    object_id: str | None
    goal: np.ndarray | None  # the chosen object's CURRENT true position
    used_belief: bool
    note: str | None  # spoken when the object moved while the person couldn't see it
    probs: dict[str, float]
    error: str | None = None


def resolve(
    text: str,
    grounder: Grounder,
    current: CameraFrame,
    truth: ObjectMap,
    belief: Belief,
    movers: Mapping[str, str] | None = None,
    *,
    use_belief: bool = True,
    rewrite: bool = True,
) -> Resolution:
    """Ground `text` and say where to go.

    Memory wording (refers_to_memory) with a stored view -> the belief: the request is
    built on belief.last_view with marks at the believed positions. Otherwise -> the
    current head frame and the true map. use_belief=False always takes the current
    view (the current-state-only baseline, which also never speaks a note). On the belief
    the grounder gets present_reading(text) (rewrite=False: the words as said). The answer's
    id is mapped to its true position. A note is added when that id is a false belief;
    `movers` (id -> "robot" or anyone else) says who moved it."""
    believed = belief.believed_map(truth.robot_pose) if use_belief else None
    on_belief = believed is not None and refers_to_memory(text)
    if on_belief:
        words = present_reading(text) if rewrite else text
        request = ground_request(words, believed, belief.last_view)
    else:
        request = ground_request(text, truth, current)
    answer = grounder.score(request)
    if not answer.probs:
        return Resolution(None, None, on_belief, None, {}, answer.error or "no answer")
    chosen = max(answer.probs, key=answer.probs.get)
    obj = next((o for o in truth.objects if o.object_id == chosen), None)
    if obj is None:
        return Resolution(chosen, None, on_belief, None, dict(answer.probs), "not in the map")
    note = None
    if use_belief and chosen in belief.false_beliefs(truth):
        mover = (movers or {}).get(chosen, "someone")
        note = moved_note(obj, belief.sightings[chosen].position, current.pose, mover)
    return Resolution(chosen, obj.position.copy(), on_belief, note, dict(answer.probs))
