"""Score a grounder on a labelled set, and decide whether a new model may
replace the current one. Task T2-08."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Example:
    id: str
    kind: str  # "relation" | "deictic" | "color" | anything a dataset adds
    answer: str  # the correct letter


@dataclass(frozen=True, slots=True)
class Prediction:
    letter: str | None
    latency_ms: float


def evaluate(predict: Callable[[Example], Prediction], examples: list[Example]) -> dict:
    """Run predict on every example (in order). Returns
    {"n", "accuracy", "by_kind": {kind: {"n", "accuracy"}}, "p50_ms", "p95_ms",
     "failures": [example ids where letter != answer, in order]}.
    letter None counts as wrong. Percentiles with numpy (linear interpolation).
    Accuracies are fractions in [0, 1]. Empty examples -> n 0, accuracy 0.0,
    by_kind {}, p50/p95 0.0, failures []."""
    raise NotImplementedError("T2-08")


def passes_gate(
    candidate: dict, current: dict | None, *, max_p95_ms: float, tolerance: float = 0.0
) -> tuple[bool, str]:
    """May `candidate` replace `current`? Both are evaluate() results.
    - candidate p95_ms > max_p95_ms -> (False, "p95 <x> ms over budget <y> ms")
    - current None -> (True, "no current model")
    - any kind in current["by_kind"] where candidate's accuracy is lower than
      current's by more than tolerance, or the kind is missing from the
      candidate -> (False, "worse on <kind>: <cand> vs <cur>") for the first
      such kind in sorted order (a missing kind shows as "missing")
    - else (True, "accuracy <cand overall> vs <cur overall>")
    Numbers in messages: accuracies to 3 decimals, ms to 0 decimals."""
    raise NotImplementedError("T2-08")
