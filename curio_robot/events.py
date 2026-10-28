"""Everything that happens in the live runtime is one of these events, on one
queue, handled in order by the orchestrator. The three feedback channels:

- voice:  Heard (speech start / partial transcript / speech end / final)
- human:  Interrupt (a hand in the workspace, "no", "stop")
- robot:  ArmEvent (started, arrived, unreachable, stopped)

plus VlmReply, the slow model's verdict arriving after the arm is already moving."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass(frozen=True, slots=True)
class Heard:
    kind: Literal["start", "partial", "end", "final"]
    text: str
    t: float  # when the audio happened (perf_counter), not when the transcript arrived


@dataclass(frozen=True, slots=True)
class Interrupt:
    source: Literal["speech", "hand"]
    detail: str
    t: float


@dataclass(frozen=True, slots=True)
class ArmEvent:
    kind: Literal["started", "arrived", "unreachable", "stopped"]
    goal: np.ndarray | None
    t: float
    tag: str = ""  # who commanded the goal: "approach", "speculative", "rules", "vlm"


@dataclass(frozen=True, slots=True)
class VlmReply:
    utterance: int
    object_id: str | None
    spoken: str
    source: str  # which endpoint answered, or "timeout" / "error"
    t: float
    detail: dict = field(default_factory=dict)


Event = Heard | Interrupt | ArmEvent | VlmReply
