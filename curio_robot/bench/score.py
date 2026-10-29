"""Scoring grounders on benchmark items, and the no-VLM rule that reads the
frame from the words and the habit. Task T5-03."""

from __future__ import annotations

from collections.abc import Callable

from curio_robot import perspective
from curio_robot.bench.conditions import build
from curio_robot.bench.items import BenchItem
from curio_robot.grounding.client import VllmGrounder

Predict = Callable[[BenchItem], str | None]  # item -> object_id (None: no answer)


def rule_predict(item: BenchItem) -> str | None:
    """No VLM. Colour = the first map colour named in the text; candidates =
    object_map.by_color(colour) (none -> None). One candidate -> it. Otherwise
    side = "left" or "right" from the text (neither -> None); frame =
    "speaker" if " my " in the text, "robot" if " your ", else "speaker" if
    p_own_frame is None or p_own_frame >= 0.5 else "robot"; viewer =
    object_map.human_pose for speaker, robot_pose for robot; return
    perspective.pick(candidates, side, viewer).object_id."""
    text, m = item.text, item.object_map
    named = [(text.find(c), c) for c in {o.color for o in m.objects} if c in text]
    if not named:
        return None
    candidates = m.by_color(min(named)[1])
    if len(candidates) == 1:
        return candidates[0].object_id
    side = "left" if "left" in text else "right" if "right" in text else None
    if side is None:
        return None
    if " my " in text:
        frame = "speaker"
    elif " your " in text:
        frame = "robot"
    else:
        frame = "speaker" if item.p_own_frame is None or item.p_own_frame >= 0.5 else "robot"
    viewer = m.human_pose if frame == "speaker" and m.human_pose is not None else m.robot_pose
    return perspective.pick(candidates, side, viewer).object_id


def vlm_predict(grounder: VllmGrounder, condition: str) -> Predict:
    """item -> the object_id with the highest probability in
    grounder.score(conditions.build(condition, item)).probs; None if empty."""

    def predict(item: BenchItem) -> str | None:
        probs = grounder.score(build(condition, item)).probs
        return max(probs, key=probs.get) if probs else None

    return predict


def score(items: list[BenchItem], predict: Predict) -> dict:
    """{"n", "acc", "by_kind": {kind: {"n", "acc"}}} with acc = fraction of
    items where predict(item) == item.target_id (kinds in first-seen order)."""
    hits: dict[str, list[bool]] = {}
    for item in items:
        hits.setdefault(item.kind, []).append(predict(item) == item.target_id)
    every = [h for kind_hits in hits.values() for h in kind_hits]
    return {
        "n": len(every),
        "acc": sum(every) / len(every) if every else 0.0,
        "by_kind": {k: {"n": len(v), "acc": sum(v) / len(v)} for k, v in hits.items()},
    }
