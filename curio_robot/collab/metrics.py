"""Fluency of one task, from its tracer marks: how much the person waited, how
much the arm moved while they talked, how many turns and corrections it took.
Idle and concurrency follow Hoffman's human-robot fluency metrics (IEEE THMS,
2019). Definitions: docs/target3-collaboration.md, section 3. Task T3-02."""

from __future__ import annotations

from dataclasses import dataclass

from curio_robot.collab.session import Session


@dataclass(frozen=True, slots=True)
class Fluency:
    completion_s: float
    success: float
    turns: int
    corrections: int
    questions: int
    human_idle_s: float
    robot_idle_frac: float
    concurrent_frac: float
    functional_delay_s: float | None
    reply_s: float | None


def fluency(session: Session) -> Fluency:
    """Marks are read from each step's `marks` (absolute seconds); a step
    lacking a mark simply doesn't contribute to measures that need it.

    - completion_s = t_end - t_start.
    - success = len(done) / len(goals); 0.0 with no goals.
    - turns = number of steps; corrections = steps of kind "correction";
      questions = steps with asked True.
    - Speaking intervals: [speech_start, speech_end] per step that has both.
      Moving intervals: [motion_start, arrived] per step that has both.
      Merge each set into a union of disjoint intervals.
    - human_idle_s = sum over steps with speech_end and arrived and
      arrived > speech_end of (arrived - speech_end).
    - robot_idle_frac = 1 - (total length of the moving union) / completion_s,
      clipped to [0, 1]; 0.0 when completion_s <= 0.
    - concurrent_frac = (length of the intersection of the speaking union and
      the moving union) / completion_s; 0.0 when completion_s <= 0.
    - functional_delay_s = statistics.median of (motion_start - speech_end)
      over steps with both; None if there are none.
    - reply_s = statistics.median of (tts_start - speech_end) over steps with
      both; None if there are none."""
    raise NotImplementedError("T3-02")


def summarize(sessions: list[Session]) -> dict[str, dict[str, float | None]]:
    """Per condition (keys sorted), the mean of each Fluency field over that
    condition's sessions, ignoring None (None if every value is None), plus
    "n": the number of sessions (as a float). Field order as in Fluency, then
    "n". [] -> {}."""
    raise NotImplementedError("T3-02")
