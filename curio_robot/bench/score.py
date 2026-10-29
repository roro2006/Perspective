"""Scoring grounders on benchmark items, and the no-VLM rule that reads the
frame from the words and the habit. Task T5-03."""

from __future__ import annotations

from collections.abc import Callable

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
    raise NotImplementedError("T5-03")


def vlm_predict(grounder: VllmGrounder, condition: str) -> Predict:
    """item -> the object_id with the highest probability in
    grounder.score(conditions.build(condition, item)).probs; None if empty."""
    raise NotImplementedError("T5-03")


def score(items: list[BenchItem], predict: Predict) -> dict:
    """{"n", "acc", "by_kind": {kind: {"n", "acc"}}} with acc = fraction of
    items where predict(item) == item.target_id (kinds in first-seen order)."""
    raise NotImplementedError("T5-03")
