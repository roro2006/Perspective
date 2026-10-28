"""Act, hedge or ask, from a posterior over objects. Task T1-04."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from curio_robot.resolver import parse_request
from curio_robot.types import ObjectMap


@dataclass(frozen=True, slots=True)
class Thresholds:
    act: float = 0.7  # top object at least this likely -> act on it
    hedge: float = 0.5  # at least this likely -> move toward the weighted centre


@dataclass(frozen=True, slots=True)
class Decision:
    kind: Literal["act", "hedge", "ask", "none"]
    object_id: str | None  # act/hedge: the most likely object; ask/none: None
    position: np.ndarray | None  # act: that object; hedge: weighted centre; else None
    confidence: float  # probability of the most likely object (0.0 for none)
    question: str | None = None  # ask only


def _filter_and_renormalize(
    posterior: dict[str, float], object_map: ObjectMap
) -> dict[str, float]:
    """Keep ids present in the map, then renormalize. Empty if nothing is left."""
    valid_ids = {obj.object_id for obj in object_map.objects}
    filtered = {k: v for k, v in posterior.items() if k in valid_ids}
    total = sum(filtered.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in filtered.items()}


def _weighted_mean(
    posterior: dict[str, float], object_map: ObjectMap
) -> np.ndarray:
    """Probability-weighted mean of the positions of objects with p > 0."""
    positions = {obj.object_id: obj.position for obj in object_map.objects}
    weighted_sum = np.zeros(3, dtype=float)
    total_weight = 0.0
    for obj_id, p in posterior.items():
        if p > 0:
            weighted_sum += p * np.asarray(positions[obj_id], dtype=float)
            total_weight += p
    return weighted_sum / total_weight


def _ask_question(text: str) -> str:
    relation = parse_request(text).relation
    if relation in ("left", "right"):
        return f"Your {relation} or mine?"
    return "Which one do you mean?"


def _none() -> Decision:
    return Decision(kind="none", object_id=None, position=None, confidence=0.0)


def decide(
    posterior: dict[str, float],
    object_map: ObjectMap,
    text: str,
    thresholds: Thresholds | None = None,
) -> Decision:
    """thresholds None means Thresholds() (the defaults).

    - Empty posterior, or no posterior id is in the map -> kind "none".
    - Ignore ids not in object_map; renormalize what's left.
    - Top probability >= thresholds.act -> "act" on it (ties: smallest object_id).
    - >= thresholds.hedge -> "hedge": position is the probability-weighted
      mean of the positions of objects with p > 0.
    - Otherwise "ask". Question: if the request has relation "left" or "right"
      (resolver.parse_request) -> f"Your {relation} or mine?"; else
      "Which one do you mean?"."""
    if thresholds is None:
        thresholds = Thresholds()
    filtered = _filter_and_renormalize(posterior, object_map)
    if not filtered:
        return _none()

    top_id = min(filtered, key=lambda k: (-filtered[k], k))
    top_p = filtered[top_id]

    if top_p >= thresholds.act:
        position = next(o.position for o in object_map.objects if o.object_id == top_id)
        return Decision(
            kind="act",
            object_id=top_id,
            position=np.array(position, dtype=float),
            confidence=top_p,
        )
    if top_p >= thresholds.hedge:
        return Decision(
            kind="hedge",
            object_id=top_id,
            position=_weighted_mean(filtered, object_map),
            confidence=top_p,
        )
    return Decision(
        kind="ask",
        object_id=None,
        position=None,
        confidence=top_p,
        question=_ask_question(text),
    )
