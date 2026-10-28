"""Readings: every way a request could be interpreted, one object each.
Task T1-01 implements the functions; the dataclass is a frozen contract."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from curio_robot.perspective import pick
from curio_robot.resolver import parse_request
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
    words = re.findall(r"[a-z]+", text.lower())

    # We need to find the last occurrence of a marker.
    # Markers:
    # "my" -> speaker
    # "your" -> robot
    # "to me" -> speaker
    # "to you" -> robot

    # We iterate through words to find the last valid marker.
    # A marker is either a single word ("my", "your") or a bigram ("to me", "to you").
    # Since "to me" and "to you" involve two words, we need to be careful.
    # If we see "to", we check the next word.

    last_marker: Frame | None = None
    i = 0
    while i < len(words):
        if words[i] == "my":
            last_marker = "speaker"
            i += 1
        elif words[i] == "your":
            last_marker = "robot"
            i += 1
        elif words[i] == "to":
            if i + 1 < len(words):
                if words[i + 1] == "me":
                    last_marker = "speaker"
                    i += 2
                elif words[i + 1] == "you":
                    last_marker = "robot"
                    i += 2
                else:
                    i += 1
            else:
                i += 1
        else:
            i += 1

    return last_marker


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
    parsed = parse_request(text)

    # 1. Determine candidates
    if parsed.color:
        candidates = object_map.by_color(parsed.color)
    else:
        candidates = object_map.objects

    if not candidates:
        return []

    # 2. No relation word
    if not parsed.relation:
        return [
            Reading(obj.object_id, None, None, False)
            for obj in candidates
        ]

    # 3 & 4. Relation word present
    marker = frame_marker(text)

    if marker:
        # Explicit frame marker found
        if marker == "speaker":
            if object_map.human_pose is None:
                # Fall back to robot frame
                viewer = object_map.robot_pose
                frame = "robot"
                explicit = False
            else:
                viewer = object_map.human_pose
                frame = "speaker"
                explicit = True
        else:  # marker == "robot"
            viewer = object_map.robot_pose
            frame = "robot"
            explicit = True

        obj = pick(candidates, parsed.relation, viewer)
        return [Reading(obj.object_id, frame, parsed.relation, explicit)]

    # No marker
    readings = []

    # Speaker frame first
    if object_map.human_pose is not None:
        obj = pick(candidates, parsed.relation, object_map.human_pose)
        readings.append(Reading(obj.object_id, "speaker", parsed.relation, False))

    # Robot frame
    obj = pick(candidates, parsed.relation, object_map.robot_pose)
    readings.append(Reading(obj.object_id, "robot", parsed.relation, False))

    return readings
