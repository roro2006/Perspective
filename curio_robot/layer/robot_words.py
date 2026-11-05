"""Robot-frame words for language-conditioned policies (pi0 in T5-28). Task T5-27.

The person said "the red one on my left"; a policy that sees only the robot's cameras can't
know whose left that was. Curio has already resolved it, so it re-says the request in the
robot's own frame from the object map: the colour, plus the one relation that singles the
target out from its look-alikes as the robot sees them ("on your left", or "closest to you"
when left/right doesn't separate them). It never mentions the speaker. The rules are
deterministic and the text is short, because T5-28 trains a policy on it.

check() reads the words back with the same rules and returns the object they pick out, or
None when they don't pick out exactly one; the gate uses it to show the words are
unambiguous without any camera."""

from __future__ import annotations

import re

from curio_robot.layer.goal import Goal
from curio_robot.perspective import horizontal_distance, lateral
from curio_robot.types import ObjectMap, WorldObject

VERB = "pick up"
NOUN = "block"
SIDE_GAP = 0.02  # m: "on your left" needs the target this much further left than the next
DISTANCE_GAP = 0.02  # m: "closest to you" needs this much difference in distance
RELATIONS = {  # phrase -> (key, sign): the target minimises sign * key among its colour
    "on your left": ("lateral", 1.0),
    "on your right": ("lateral", -1.0),
    "closest to you": ("distance", 1.0),
    "furthest from you": ("distance", -1.0),
}
_ORDER = tuple(RELATIONS)  # tried in this order: left/right first, then distance


def _key(obj: WorldObject, om: ObjectMap, kind: str) -> float:
    if kind == "lateral":  # negative = robot's left
        return lateral(obj, om.robot_pose)
    return horizontal_distance(obj, om.robot_pose)


def _margin(target: WorldObject, group: list[WorldObject], om: ObjectMap, phrase: str) -> float:
    """How far ahead of the runner-up the target is on this relation (m; < 0: not first)."""
    kind, sign = RELATIONS[phrase]
    mine = sign * _key(target, om, kind)
    others = [sign * _key(o, om, kind) for o in group if o.object_id != target.object_id]
    return min(others) - mine


def relation(target_id: str, om: ObjectMap) -> str | None:
    """The phrase that singles the target out among its colour, "" when its colour is unique,
    None when no relation separates it by the required gap."""
    target = next(o for o in om.objects if o.object_id == target_id)
    group = om.by_color(target.color)
    if len(group) == 1:
        return ""
    for phrase in _ORDER:
        gap = SIDE_GAP if RELATIONS[phrase][0] == "lateral" else DISTANCE_GAP
        if _margin(target, group, om, phrase) >= gap:
            return phrase
    return None


def instruction(target_id: str, om: ObjectMap, verb: str = VERB, noun: str = NOUN) -> str:
    """"pick up the red block on your left". When nothing separates the target by the gap,
    the relation with the largest margin is used anyway (check() then returns None)."""
    target = next(o for o in om.objects if o.object_id == target_id)
    phrase = relation(target_id, om)
    if phrase is None:
        group = om.by_color(target.color)
        phrase = max(_ORDER, key=lambda p: _margin(target, group, om, p))
    thing = f"{target.color} {noun}" if noun else target.color
    return f"{verb} the {thing}" + (f" {phrase}" if phrase else "")


def robot_words(goal: Goal, om: ObjectMap, verb: str = VERB, noun: str = NOUN) -> str:
    """The words adapter: the Goal's target re-said in the robot's frame."""
    return instruction(goal.target_id, om, verb, noun)


def check(words: str, om: ObjectMap) -> str | None:
    """The object the words pick out under the same rules, or None if they don't pick out
    exactly one (no known colour, a colour shared with no relation, or a relation that
    doesn't separate the target by the gap)."""
    colours = sorted({o.color for o in om.objects}, key=len, reverse=True)
    found = [(m.start(), c) for c in colours
             if (m := re.search(rf"\b{re.escape(c)}\b", words)) is not None]
    if not found:
        return None
    group = om.by_color(min(found)[1])
    phrases = [p for p in RELATIONS if words.endswith(" " + p)]
    if not phrases:
        return group[0].object_id if len(group) == 1 else None
    phrase = phrases[0]
    kind, sign = RELATIONS[phrase]
    best = min(group, key=lambda o: sign * _key(o, om, kind))
    if len(group) == 1:
        return best.object_id
    gap = SIDE_GAP if kind == "lateral" else DISTANCE_GAP
    return best.object_id if _margin(best, group, om, phrase) >= gap else None
