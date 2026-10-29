"""Fluency of one task, from its tracer marks: how much the person waited, how
much the arm moved while they talked, how many turns and corrections it took.
Idle and concurrency follow Hoffman's human-robot fluency metrics (IEEE THMS,
2019). Definitions: docs/target3-collaboration.md, section 3. Task T3-02."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, fields

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


def _union(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    sorted_ivs = sorted(intervals, key=lambda x: x[0])
    merged: list[tuple[float, float]] = [sorted_ivs[0]]
    for start, end in sorted_ivs[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _overlap(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    total = 0.0
    i, j = 0, 0
    while i < len(a) and j < len(b):
        a_start, a_end = a[i]
        b_start, b_end = b[j]
        lo = max(a_start, b_start)
        hi = min(a_end, b_end)
        if lo < hi:
            total += hi - lo
        if a_end < b_end:
            i += 1
        else:
            j += 1
    return total


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
    completion_s = session.t_end - session.t_start
    success = len(session.done) / len(session.goals) if session.goals else 0.0
    turns = len(session.steps)
    corrections = sum(1 for s in session.steps if s.kind == "correction")
    questions = sum(1 for s in session.steps if s.asked)

    speaking_ivs: list[tuple[float, float]] = []
    moving_ivs: list[tuple[float, float]] = []
    human_idle_s = 0.0
    functional_delays: list[float] = []
    reply_durations: list[float] = []

    for step in session.steps:
        marks = step.marks
        speech_start = marks.get("speech_start")
        speech_end = marks.get("speech_end")
        motion_start = marks.get("motion_start")
        arrived = marks.get("arrived")
        tts_start = marks.get("tts_start")

        if speech_start is not None and speech_end is not None:
            speaking_ivs.append((speech_start, speech_end))
        if motion_start is not None and arrived is not None:
            moving_ivs.append((motion_start, arrived))

        if speech_end is not None and arrived is not None and arrived > speech_end:
            human_idle_s += arrived - speech_end

        if motion_start is not None and speech_end is not None:
            functional_delays.append(motion_start - speech_end)

        if tts_start is not None and speech_end is not None:
            reply_durations.append(tts_start - speech_end)

    speaking_union = _union(speaking_ivs)
    moving_union = _union(moving_ivs)

    moving_total = sum(end - start for start, end in moving_union)
    if completion_s > 0:
        robot_idle_frac = max(0.0, min(1.0, 1.0 - moving_total / completion_s))
        concurrent_frac = _overlap(speaking_union, moving_union) / completion_s
    else:
        robot_idle_frac = 0.0
        concurrent_frac = 0.0

    functional_delay_s = statistics.median(functional_delays) if functional_delays else None
    reply_s = statistics.median(reply_durations) if reply_durations else None

    return Fluency(
        completion_s=completion_s,
        success=success,
        turns=turns,
        corrections=corrections,
        questions=questions,
        human_idle_s=human_idle_s,
        robot_idle_frac=robot_idle_frac,
        concurrent_frac=concurrent_frac,
        functional_delay_s=functional_delay_s,
        reply_s=reply_s,
    )


def summarize(sessions: list[Session]) -> dict[str, dict[str, float | None]]:
    """Per condition (keys sorted), the mean of each Fluency field over that
    condition's sessions, ignoring None (None if every value is None), plus
    "n": the number of sessions (as a float). Field order as in Fluency, then
    "n". [] -> {}."""
    if not sessions:
        return {}

    by_condition: dict[str, list[Fluency]] = {}
    for session in sessions:
        by_condition.setdefault(session.condition, []).append(fluency(session))

    result: dict[str, dict[str, float | None]] = {}
    for condition in sorted(by_condition):
        fluencies = by_condition[condition]
        row: dict[str, float | None] = {}
        for field in fields(Fluency):
            values = [getattr(f, field.name) for f in fluencies]
            non_none = [v for v in values if v is not None]
            if non_none:
                row[field.name] = float(sum(non_none) / len(non_none))
            else:
                row[field.name] = None
        row["n"] = float(len(fluencies))
        result[condition] = row

    return result
