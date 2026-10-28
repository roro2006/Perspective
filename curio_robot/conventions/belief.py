"""Per-person belief about conventions, and learning from outcomes.
Task T1-02 implements the methods; field names and signatures are frozen."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from curio_robot.conventions.readings import Reading

ACCEPT_WEIGHT = 0.3  # an uncomplained arrival is weak evidence
CORRECT_WEIGHT = 1.0  # "no, the other one" is strong evidence

Outcome = Literal["accepted", "corrected", "answered"]


@dataclass(slots=True)
class ConventionBelief:
    """Beta(a, b) over P(the person means their own frame when they don't say).
    Prior Beta(1.6, 0.4): mean 0.8, but weak (worth two observations), so a
    person who means the robot's left is learned within a few requests."""

    own_frame_a: float = 1.6
    own_frame_b: float = 0.4

    def p_own_frame(self) -> float:
        """Posterior mean a / (a + b)."""
        raise NotImplementedError("T1-02")

    def posterior(self, readings: list[Reading]) -> dict[str, float]:
        """P(object_id) from readings, summing to 1 ({} for no readings).

        Weight per reading: explicit or frame None -> 1.0; frame "speaker" ->
        p_own_frame(); frame "robot" -> 1 - p_own_frame(). Sum weights per
        object_id, then normalize. If every weight is 0, spread uniformly
        over the distinct object_ids."""
        raise NotImplementedError("T1-02")

    def update(self, frame: str, correct: bool, weight: float) -> None:
        """Evidence that `frame` was (correct=True) or wasn't the right reading.
        speaker+correct or robot+wrong -> a += weight; otherwise b += weight.
        weight <= 0 is a no-op."""
        raise NotImplementedError("T1-02")

    def to_dict(self) -> dict:
        """JSON-safe. Must round-trip through from_dict; ignore unknown keys there."""
        raise NotImplementedError("T1-02")

    @classmethod
    def from_dict(cls, data: dict) -> ConventionBelief:
        raise NotImplementedError("T1-02")


def learn_from_outcome(
    belief: ConventionBelief,
    readings: list[Reading],
    chosen_id: str,
    outcome: Outcome,
    right_id: str | None = None,
    p_flip: float = 0.0,
) -> float:
    """Update `belief` from what happened to the chosen object; return the
    weight applied (0.0 when there was nothing to learn).

    Nothing to learn when: any reading is explicit; no reading has a frame;
    or the speaker-frame and robot-frame readings point at the same object.
    Otherwise there is one speaker reading S and one robot reading R:
    - "accepted": the frame whose reading has object_id == chosen_id was
      correct, weight ACCEPT_WEIGHT.
    - "corrected": the frame whose reading has object_id == chosen_id was
      wrong, weight CORRECT_WEIGHT. (right_id is for the log; ignore it here.)
    - "answered" (the robot asked "your left or mine?" and got an answer):
      the frame whose reading has object_id == right_id was correct, weight
      CORRECT_WEIGHT; chosen_id is ignored. right_id None: nothing to learn.
    - If neither S nor R points at the relevant object: nothing to learn.
    Then multiply the weight by max(0, 1 - 2 * p_flip) before belief.update."""
    raise NotImplementedError("T1-02")
