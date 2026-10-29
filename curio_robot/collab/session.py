"""One task with one person, as recorded: the goals, what got done, and each
thing the person said with the tracer's marks. The task-level index over the
decision records and episode log. Task T3-01."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

StepKind = Literal["request", "correction", "answer"]


@dataclass(frozen=True, slots=True)
class Step:
    utterance: int  # the runtime's utterance number
    kind: StepKind
    text: str  # what the person said
    goal_id: str | None  # the object the person wanted at this step
    chosen_id: str | None  # where the arm ended up (None: it didn't move)
    asked: bool  # the robot asked a clarifying question in reply
    marks: dict[str, float]  # tracer stage -> absolute perf_counter seconds


@dataclass(frozen=True, slots=True)
class Session:
    session_id: str
    person: str  # the SimUser's name, or a pseudonym for a real person
    condition: str  # "view-off" | "view-on" | "view-on+gaze"
    task: str  # the task scene's kind, e.g. "fetch"
    seed: int
    t_start: float  # perf_counter seconds
    t_end: float
    goals: tuple[str, ...]  # object ids in the order the person asked for them
    done: tuple[str, ...]  # goals the arm reached and the person accepted
    steps: tuple[Step, ...]


def session_to_dict(session: Session) -> dict:
    """A JSON-safe dict: every field by name, tuples as lists, each step as a
    dict of its fields (marks copied as a plain dict). No rounding."""
    return {
        "session_id": session.session_id,
        "person": session.person,
        "condition": session.condition,
        "task": session.task,
        "seed": session.seed,
        "t_start": session.t_start,
        "t_end": session.t_end,
        "goals": list(session.goals),
        "done": list(session.done),
        "steps": [
            {
                "utterance": step.utterance,
                "kind": step.kind,
                "text": step.text,
                "goal_id": step.goal_id,
                "chosen_id": step.chosen_id,
                "asked": step.asked,
                "marks": dict(step.marks),
            }
            for step in session.steps
        ],
    }


def session_from_dict(data: dict) -> Session:
    """Inverse of session_to_dict: lists back to tuples, steps back to Step,
    marks values as float. session_from_dict(session_to_dict(s)) == s."""
    return Session(
        session_id=data["session_id"],
        person=data["person"],
        condition=data["condition"],
        task=data["task"],
        seed=data["seed"],
        t_start=data["t_start"],
        t_end=data["t_end"],
        goals=tuple(data["goals"]),
        done=tuple(data["done"]),
        steps=tuple(
            Step(
                utterance=step["utterance"],
                kind=step["kind"],
                text=step["text"],
                goal_id=step["goal_id"],
                chosen_id=step["chosen_id"],
                asked=step["asked"],
                marks={k: float(v) for k, v in step["marks"].items()},
            )
            for step in data["steps"]
        ),
    )


def save_sessions(sessions: list[Session], path: Path) -> None:
    """Write one json.dumps(session_to_dict(s)) per line, in order, UTF-8,
    replacing the file. Create parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in sessions:
            f.write(json.dumps(session_to_dict(s)))
            f.write("\n")


def load_sessions(path: Path) -> list[Session]:
    """Read a file written by save_sessions; blank lines are skipped."""
    sessions: list[Session] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            sessions.append(session_from_dict(json.loads(stripped)))
    return sessions
