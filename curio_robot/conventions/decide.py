"""Act, hedge or ask, from a posterior over objects. Task T1-04."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

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
    raise NotImplementedError("T1-04")
