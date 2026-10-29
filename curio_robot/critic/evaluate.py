"""Scoring a critic on episodes: when it raises the alarm, how often it is
right, how far ahead of the spoken "no", and whether it names the block the
person wants. The phase 1 gate lives here. Task T4-06."""

from __future__ import annotations

import numpy as np

from curio_robot.critic.episodes import Episode

Scored = tuple[np.ndarray, np.ndarray]  # (p_wrong (F,), intended (F, N)) for one episode

GATE = {"precision": 0.90, "recall": 0.70, "lead_s": 0.4, "target_acc": 0.85}
THRESHOLDS = np.round(np.linspace(0.05, 0.99, 95), 2)


def alarm_frame(p_wrong: np.ndarray, threshold: float) -> int | None:
    """Index of the first frame with p_wrong >= threshold, else None."""
    raise NotImplementedError("T4-06")


def evaluate(episodes: list[Episode], scored: list[Scored], threshold: float) -> dict:
    """Episode-level metrics at one threshold. An episode is flagged if
    alarm_frame is not None.

    Returns {"n", "n_wrong", "tp", "fp", "fn", "precision", "recall",
    "false_alarm_rate", "lead_s", "target_acc", "threshold"}:
    - tp: wrong and flagged; fp: right and flagged; fn: wrong, not flagged.
    - precision = tp / (tp + fp), 0.0 if nothing flagged; recall = tp / n_wrong,
      0.0 if n_wrong is 0; false_alarm_rate = fp / (n - n_wrong), 0.0 if 0.
    - lead_s: median of (ep.t_no - ep.t[alarm]) over tp episodes whose t_no
      is not None; None if there are none.
    - target_acc: over tp episodes, the fraction where argmax(intended[alarm])
      == ep.intended; None if tp is 0.
    ValueError if the two lists differ in length."""
    raise NotImplementedError("T4-06")


def threshold_for_precision(
    episodes: list[Episode], scored: list[Scored], precision: float = GATE["precision"]
) -> float | None:
    """The smallest t in THRESHOLDS whose evaluate(...) has tp + fp > 0 and
    precision >= the target; None if none does."""
    raise NotImplementedError("T4-06")


def auroc(episodes: list[Episode], scored: list[Scored]) -> float:
    """Area under the ROC curve with score = max(p_wrong) per episode and
    wrong episodes as positives: the fraction of (wrong, right) pairs where
    the wrong one scores higher, ties counting 0.5. ValueError if either
    class is empty."""
    raise NotImplementedError("T4-06")


def gate(metrics: dict) -> dict[str, bool]:
    """For each key in GATE: metrics[key] >= GATE[key] (None -> False)."""
    raise NotImplementedError("T4-06")
