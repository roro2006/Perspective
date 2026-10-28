"""The grounding request: one marked image plus a short prompt whose answer
is a single letter. One function builds it for serving AND for training data,
so the two can't drift. Task T2-04."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from curio_robot.grounding.marks import Mark
from curio_robot.types import ObjectMap

PROMPT_VERSION = "1"  # bump when the wording or layout changes; logged with every decision


@dataclass(frozen=True, slots=True)
class GroundingRequest:
    messages: list[dict]  # OpenAI chat format, one user message: [image, text]
    letters: dict[str, str]  # letter -> object_id, only letters that are in the image
    prompt_version: str = PROMPT_VERSION


def build_request(
    text: str,
    object_map: ObjectMap,
    marks: list[Mark],
    image: np.ndarray,
    p_own_frame: float | None = None,
) -> GroundingRequest:
    """messages = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,<PNG of image>"}},
        {"type": "text", "text": <prompt>}]}]

    The prompt text, lines joined by "\\n":
      "You control a robot arm sharing a table with a person."
      "The image is what the person saw when they spoke. Objects are marked with letters."
      one line per mark, in mark order:
        "<letter>: <colour> (speaker's <left|right>, robot's <left|right>)"
        using perspective.lateral (< 0 is left); omit the "speaker's ..." part
        (and its comma) when object_map.human_pose is None; the colour comes
        from the object with that object_id in object_map.
      if p_own_frame is not None: resolver.frame_habit(p_own_frame)
      'The person said: "<text>"'
      "Answer with the single letter of the object they mean."
    letters maps each mark's letter to its object_id."""
    raise NotImplementedError("T2-04")
