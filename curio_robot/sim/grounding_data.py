"""Sim training and eval data for the grounder: rendered human-view frames
with marks, a request, and the exact answer letter. Task T2-07.

    uv run python -m curio_robot.sim.grounding_data out/grounding --n 300 --seed 0"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from curio_robot.grounding.prompt import GroundingRequest
from curio_robot.sim.world import SimWorld

KINDS = ("relation", "deictic", "color")
HABITS = (0.15, 0.85)  # the p_own_frame shown in relation prompts


@dataclass(frozen=True, slots=True)
class GroundingExample:
    id: str
    kind: str
    text: str
    image: np.ndarray  # the marked, cropped image (render_marks output)
    request: GroundingRequest  # built with build_request on that image
    answer: str  # the correct letter
    target_id: str  # the object the person meant


def make_example(
    world: SimWorld, rng: np.random.Generator, kind: str, example_id: str
) -> GroundingExample | None:
    """One example of `kind` (ValueError for an unknown kind).

    - "relation": scenarios.perspective sets the scene. Draw habit p from
      HABITS, frame "speaker" with probability p else "robot", side and
      whether to say a marker (50%: "my"/"your", else "the"). Text
      f"Hand me the {color} one on {my|your|the} {side}". Target: the pair
      block on that side from that frame's viewer (perspective.pick); the
      pair is the first two entries of world.block_positions(). The prompt
      gets p_own_frame=p.
    - "deictic": scenarios.cluttered sets the scene; pick a random block as the
      target and call world.set_human(eye, target position) so the person
      looks straight at it. Text "Hand me that one". p_own_frame None.
    - "color": scenarios.cluttered; target = a block whose colour is unique
      on the table (None if there isn't one). Text f"Hand me the {color} one".
      p_own_frame None.
    Then render the human camera (world.render_camera(HUMAN_CAM)), build the
    map from world.block_positions() with ids "<colour>_<n>" in that order,
    place_marks, render_marks, build_request. Return None if the target has
    no mark (out of the person's view)."""
    raise NotImplementedError("T2-07")


def write_dataset(out_dir: Path, n: int, seed: int = 0, kinds: tuple[str, ...] = KINDS) -> Path:
    """n examples cycling through kinds (example i has kind kinds[i % len]),
    retrying a kind until make_example returns one. Writes
    out_dir/images/<id>.png and out_dir/data.jsonl, one line per example:
    {"id", "kind", "text", "answer", "target_id", "letters", "prompt_version",
     "image": "images/<id>.png", "messages": request.messages with the image
     data URL replaced by the relative path, plus a final
     {"role": "assistant", "content": answer}}.
    ids are f"{seed}-{i:05d}". Returns the data.jsonl path."""
    raise NotImplementedError("T2-07")


def load_dataset(path: Path) -> list[dict]:
    """The data.jsonl lines as dicts, in order."""
    raise NotImplementedError("T2-07")
