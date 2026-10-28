"""Decision records: every choice the robot makes, with what it considered,
what it believed and which versions decided, then what happened. One JSON line
each. This is target 1's audit trail and target 2's training data. Task T1-07."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from curio_robot.conventions.decide import Decision
from curio_robot.conventions.readings import Reading

# Bump the matching entry whenever that component's behaviour changes.
VERSIONS: dict[str, str] = {
    "readings": "1",
    "belief": "1",
    "decide": "1",
    "vlm_prompt": "1",
}

Stage = Literal["partial", "final", "vlm", "correction"]


@dataclass(slots=True)
class DecisionRecord:
    utterance: int
    t: float  # perf_counter seconds, same clock as the tracer
    text: str
    stage: Stage
    readings: list[Reading]
    posterior: dict[str, float]
    decision: Decision
    belief: dict  # ConventionBelief.to_dict() at decision time
    person: str = "anonymous"
    versions: dict[str, str] = field(default_factory=lambda: dict(VERSIONS))


class DecisionRecorder:
    """Append-only JSONL. Line kinds: {"kind": "decision", ...record fields}
    and {"kind": "outcome", "utterance", "outcome", "right_id", "t"}.
    numpy arrays -> lists rounded to 4 decimals; dataclasses -> dicts."""

    def __init__(self, path: Path) -> None:
        raise NotImplementedError("T1-07")

    def record(self, rec: DecisionRecord) -> None:
        """Write one decision line and flush."""
        raise NotImplementedError("T1-07")

    def outcome(
        self, utterance: int, outcome: str, right_id: str | None, t: float
    ) -> None:
        """Write one outcome line and flush."""
        raise NotImplementedError("T1-07")

    def close(self) -> None:
        raise NotImplementedError("T1-07")


def load_records(path: Path) -> list[dict]:
    """Read a recorder file back as dicts, decisions joined with their outcome:
    each decision dict gains "outcome" and "right_id" (None if no outcome line
    for its utterance; the last outcome line for an utterance wins)."""
    raise NotImplementedError("T1-07")
