"""Turn a spoken request into a target object.

The rule-based resolver handles the demo vocabulary (colour + left/right/
nearest/farthest + whose perspective) with no network, so it doubles as the
booth fallback. The VLM resolver handles open-ended requests by sending the
object map as text and reading back a [TARGET:id] tag, the same pattern as
Curio's [POINT:x,y:label]."""

from __future__ import annotations

import re
from dataclasses import dataclass

from curio_robot.perspective import lateral, pick
from curio_robot.types import ObjectMap, Target

COLORS = ("red", "blue", "green", "yellow")
_RELATIONS = {
    "left": "left",
    "right": "right",
    "nearest": "nearest",
    "closest": "nearest",
    "farthest": "farthest",
    "furthest": "farthest",
}


@dataclass(frozen=True, slots=True)
class ParsedRequest:
    color: str | None
    relation: str | None
    speaker_frame: bool  # "my left" / "closest to me" vs "your left"


def parse_request(text: str) -> ParsedRequest:
    words = re.findall(r"[a-z]+", text.lower())
    color = next((w for w in words if w in COLORS), None)
    relation = next((_RELATIONS[w] for w in words if w in _RELATIONS), None)
    speaker_frame = "your" not in words  # people default to their own perspective
    return ParsedRequest(color=color, relation=relation, speaker_frame=speaker_frame)


def resolve(text: str, object_map: ObjectMap) -> Target:
    req = parse_request(text)
    candidates = object_map.by_color(req.color) if req.color else list(object_map.objects)
    if not candidates:
        return Target(None, None, f"no {req.color or 'object'} found in the map")
    if len(candidates) == 1 or req.relation is None:
        o = candidates[0]
        return Target(o.object_id, o.position, "only match")
    if req.speaker_frame and object_map.human_pose is not None:
        viewer, whose = object_map.human_pose, "speaker's"
    else:
        # Without the human view the robot can only interpret "left" its own way.
        viewer, whose = object_map.robot_pose, "robot's"
    o = pick(candidates, req.relation, viewer)
    return Target(o.object_id, o.position, f"{req.relation} from the {whose} view")


_TARGET_RE = re.compile(r"\[TARGET:(none|[a-z]+_\d+)\]\s*$")


def parse_target_tag(response: str) -> tuple[str, str | None]:
    """Split a VLM reply into (spoken_text, object_id or None)."""
    m = _TARGET_RE.search(response)
    if not m:
        return response, None
    spoken = response[: m.start()].rstrip()
    return spoken, None if m.group(1) == "none" else m.group(1)


def _side(lat: float) -> str:
    return "left" if lat < 0 else "right"


def frame_habit(p_own_frame: float) -> str:
    """One line telling the VLM what we've learned about this person's "left"."""
    if p_own_frame >= 0.6:
        whose = "their own left/right"
    elif p_own_frame <= 0.4:
        whose = "the robot's left/right"
    else:
        return "We don't yet know whose left/right this person means when they don't say."
    return (
        f"When this person says left or right without saying whose, they usually mean "
        f"{whose} (confidence {max(p_own_frame, 1 - p_own_frame):.2f})."
    )


def vlm_prompt(text: str, object_map: ObjectMap, p_own_frame: float | None = None) -> str:
    """The text prompt the VLM sees (sent by curio_robot.vlm.VlmResolver). Every
    object is labelled from both viewpoints, so the model never has to guess
    whose "left" the geometry meant; p_own_frame adds the learned habit."""
    lines = []
    for o in object_map.objects:
        x, y, _ = o.position
        sides = [f"robot's {_side(lateral(o, object_map.robot_pose))}"]
        if object_map.human_pose is not None:
            sides.insert(0, f"speaker's {_side(lateral(o, object_map.human_pose))}")
        lines.append(
            f"- {o.object_id}: {o.color} at x={x:.2f} y={y:.2f} ({', '.join(sides)}), "
            f"seen by {sorted(o.seen_by)}"
        )
    habit = f"{frame_habit(p_own_frame)}\n" if p_own_frame is not None else ""
    return (
        "You control a robot arm sharing a table with a person.\n"
        "Objects on the table:\n" + "\n".join(lines) + "\n"
        + habit
        + f'The person said: "{text}"\n'
        "Reply with one short sentence, then [TARGET:object_id] or [TARGET:none]."
    )
