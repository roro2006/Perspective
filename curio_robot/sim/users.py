"""Simulated people with a fixed habit about whose "left" they mean. They
test that the learning loop works; they are not evidence about real people.
Task T1-06."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from curio_robot.conventions.readings import Frame
from curio_robot.types import ObjectMap


@dataclass(frozen=True, slots=True)
class Utterance:
    text: str
    frame: Frame  # the frame this person actually meant
    color: str
    side: str  # "left" or "right"


@dataclass(slots=True)
class SimUser:
    name: str
    p_own_frame: float  # how often they mean their own left/right
    p_marker: float = 0.0  # how often they say "my"/"your" explicitly

    def utter(self, color: str, side: str, rng: np.random.Generator) -> Utterance:
        """Draw the frame (speaker with p_own_frame, else robot), then the text:
        with probability p_marker, f"Hand me the {color} one on {m} {side}"
        where m is "my" (speaker) or "your" (robot); otherwise
        f"Hand me the {color} one on the {side}". Draw the frame first, then
        the marker, each with one rng.random() call, in that order."""
        raise NotImplementedError("T1-06")

    def intended(self, utt: Utterance, object_map: ObjectMap) -> str:
        """The object this person meant: among objects of utt.color, pick
        utt.side from their frame's viewer (perspective.pick; speaker ->
        human_pose, robot -> robot_pose). ValueError if no such objects or
        the speaker frame is asked for with no human_pose."""
        raise NotImplementedError("T1-06")

    def answer(self, utt: Utterance) -> str:
        """Reply to "your left or mine?": "mine" (speaker frame) or "yours"."""
        raise NotImplementedError("T1-06")

    def react(self, intended_id: str, chosen_id: str | None) -> str | None:
        """None if chosen_id == intended_id (they accept), else
        "no, the other one"."""
        raise NotImplementedError("T1-06")


POPULATION = (
    SimUser("own-frame", p_own_frame=0.95),
    SimUser("robot-frame", p_own_frame=0.05),
    SimUser("mixed", p_own_frame=0.5),
)
