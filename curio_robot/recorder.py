"""Decision records: every choice the robot makes, with what it considered,
what it believed and which versions decided, then what happened. One JSON line
each. This is target 1's audit trail and target 2's training data. Task T1-07."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Literal

import numpy as np

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


def _plain(value):
    """JSON-safe copy: slotted dataclasses have no __dict__, so walk their fields."""
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, np.ndarray):
        return value.round(4).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, set):
        return sorted(_plain(v) for v in value)
    return value


class DecisionRecorder:
    """Append-only JSONL. Line kinds: {"kind": "decision", ...record fields}
    and {"kind": "outcome", "utterance", "outcome", "right_id", "t"}.
    numpy arrays -> lists rounded to 4 decimals; dataclasses -> dicts."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("a", encoding="utf-8")

    def _write(self, line: dict) -> None:
        self._file.write(json.dumps(line) + "\n")
        self._file.flush()

    def record(self, rec: DecisionRecord) -> None:
        """Write one decision line and flush."""
        self._write({"kind": "decision"} | _plain(rec))

    def outcome(
        self, utterance: int, outcome: str, right_id: str | None, t: float
    ) -> None:
        """Write one outcome line and flush."""
        self._write(
            {"kind": "outcome", "utterance": utterance, "outcome": outcome,
             "right_id": right_id, "t": t}
        )

    def close(self) -> None:
        self._file.close()


def load_records(path: Path) -> list[dict]:
    """Read a recorder file back as dicts, decisions joined with their outcome:
    each decision dict gains "outcome" and "right_id" (None if no outcome line
    for its utterance; the last outcome line for an utterance wins)."""
    decisions: list[dict] = []
    outcomes: dict[int, dict] = {}
    with path.open(encoding="utf-8") as f:
        for raw in f:
            if not raw.strip():
                continue
            line = json.loads(raw)
            if line.get("kind") == "outcome":
                outcomes[line["utterance"]] = line
            elif line.get("kind") == "decision":
                decisions.append(line)
    joined = []
    for d in decisions:
        out = outcomes.get(d["utterance"], {})
        joined.append(d | {"outcome": out.get("outcome"), "right_id": out.get("right_id")})
    return joined
