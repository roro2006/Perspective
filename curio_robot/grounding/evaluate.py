"""Score a grounder on a labelled set, and decide whether a new model may
replace the current one. Task T2-08."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


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
    if not examples:
        return {
            "n": 0,
            "accuracy": 0.0,
            "by_kind": {},
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "failures": [],
        }

    n = len(examples)
    correct = 0
    latencies = []
    failures = []
    by_kind: dict[str, dict] = {}

    for ex in examples:
        pred = predict(ex)
        latencies.append(pred.latency_ms)

        is_correct = pred.letter is not None and pred.letter == ex.answer
        if is_correct:
            correct += 1
        else:
            failures.append(ex.id)

        if ex.kind not in by_kind:
            by_kind[ex.kind] = {"n": 0, "accuracy": 0.0}
        by_kind[ex.kind]["n"] += 1
        if is_correct:
            by_kind[ex.kind]["accuracy"] += 1

    # Convert counts to fractions
    for stats in by_kind.values():
        if stats["n"] > 0:
            stats["accuracy"] /= stats["n"]

    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))

    return {
        "n": n,
        "accuracy": correct / n,
        "by_kind": by_kind,
        "p50_ms": p50,
        "p95_ms": p95,
        "failures": failures,
    }


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
    # 1. Check latency budget
    if candidate["p95_ms"] > max_p95_ms:
        return (
            False,
            f"p95 {candidate['p95_ms']:.0f} ms over budget {max_p95_ms:.0f} ms",
        )

    # 2. No current model to compare against
    if current is None:
        return True, "no current model"

    # 3. Check per-kind accuracy regression
    for kind in sorted(current["by_kind"].keys()):
        cur_acc = current["by_kind"][kind]["accuracy"]

        if kind not in candidate["by_kind"]:
            return False, f"worse on {kind}: missing vs {cur_acc:.3f}"

        cand_acc = candidate["by_kind"][kind]["accuracy"]

        if cand_acc < cur_acc - tolerance:
            return False, f"worse on {kind}: {cand_acc:.3f} vs {cur_acc:.3f}"

    # 4. All checks passed
    return (
        True,
        f"accuracy {candidate['accuracy']:.3f} vs {current['accuracy']:.3f}",
    )
