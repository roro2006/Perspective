"""Readings: every way a request could be interpreted, one object each.
Task T1-01 implements the functions; the dataclass is a frozen contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from curio_robot.types import ObjectMap

Frame = Literal["speaker", "robot"]


@dataclass(frozen=True, slots=True)
class Reading:
    object_id: str
    frame: Frame | None  # whose viewpoint the relation is judged from; None: no relation word
    relation: str | None  # "left" / "right" / "nearest" / "farthest" / None
    explicit: bool  # the speaker named the frame ("my", "your", "to me", "to you")


def frame_marker(text: str) -> Frame | None:
    """The frame the speaker named, if any.

    "my" or "to me" -> "speaker"; "your" or "to you" -> "robot"; else None.
    "me" on its own ("hand me ...") is NOT a marker. If both kinds appear,
    the last one in the sentence wins ("not my left, your left" -> "robot").
    Case and punctuation are ignored."""
    raise NotImplementedError("T1-01")


def enumerate_readings(text: str, object_map: ObjectMap) -> list[Reading]:
    """All readings of `text` against the map.

    1. Candidates: objects of the requested colour (resolver.parse_request),
       or every object if no colour was said. None -> [].
    2. No relation word: one Reading per candidate, frame=None, explicit=False.
    3. Relation word and an explicit frame_marker: one Reading, that frame,
       explicit=True, object chosen with perspective.pick from that viewer.
       Viewer for "speaker" is object_map.human_pose; for "robot" robot_pose.
       If the marker is "speaker" but human_pose is None: fall back to the
       robot frame with explicit=False (the robot can't see from the speaker).
    4. Relation word, no marker: one Reading per frame, explicit=False,
       speaker first then robot. Only the robot frame if human_pose is None.
       Keep both readings even when they point at the same object.
    """
    raise NotImplementedError("T1-01")
