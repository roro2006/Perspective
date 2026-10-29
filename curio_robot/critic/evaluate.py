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
    idx = np.flatnonzero(p_wrong >= threshold)
    if len(idx) == 0:
        return None
    return int(idx[0])


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
    if len(episodes) != len(scored):
        raise ValueError("episodes and scored must have the same length")

    n = len(episodes)
    tp = 0
    fp = 0
    fn = 0
    n_wrong = 0
    lead_times = []
    target_correct = 0

    for i, ep in enumerate(episodes):
        p_wrong, intended = scored[i]
        alarm = alarm_frame(p_wrong, threshold)
        flagged = alarm is not None

        if ep.wrong:
            n_wrong += 1
            if flagged:
                tp += 1
                if ep.t_no is not None:
                    lead_times.append(ep.t_no - float(ep.t[alarm]))
                if int(np.argmax(intended[alarm])) == ep.intended:
                    target_correct += 1
            else:
                fn += 1
        elif flagged:
            fp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / n_wrong if n_wrong > 0 else 0.0
    false_alarm_rate = fp / (n - n_wrong) if (n - n_wrong) > 0 else 0.0

    lead_s = float(np.median(lead_times)) if lead_times else None
    target_acc = target_correct / tp if tp > 0 else None

    return {
        "n": n,
        "n_wrong": n_wrong,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "false_alarm_rate": float(false_alarm_rate),
        "lead_s": lead_s,
        "target_acc": target_acc,
        "threshold": float(threshold),
    }


def threshold_for_precision(
    episodes: list[Episode], scored: list[Scored], precision: float = GATE["precision"]
) -> float | None:
    """The smallest t in THRESHOLDS whose evaluate(...) has tp + fp > 0 and
    precision >= the target; None if none does."""
    for t in THRESHOLDS:
        m = evaluate(episodes, scored, float(t))
        if m["tp"] + m["fp"] > 0 and m["precision"] >= precision:
            return float(t)
    return None


def auroc(episodes: list[Episode], scored: list[Scored]) -> float:
    """Area under the ROC curve with score = max(p_wrong) per episode and
    wrong episodes as positives: the fraction of (wrong, right) pairs where
    the wrong one scores higher, ties counting 0.5. ValueError if either
    class is empty."""
    pos_scores = []
    neg_scores = []

    for i, ep in enumerate(episodes):
        p_wrong, _ = scored[i]
        score = float(np.max(p_wrong))
        if ep.wrong:
            pos_scores.append(score)
        else:
            neg_scores.append(score)

    if not pos_scores or not neg_scores:
        raise ValueError("Both positive and negative classes must be non-empty")

    total = 0.0
    for p in pos_scores:
        for neg in neg_scores:
            if p > neg:
                total += 1.0
            elif p == neg:
                total += 0.5

    return float(total / (len(pos_scores) * len(neg_scores)))


def gate(metrics: dict) -> dict[str, bool]:
    """For each key in GATE: metrics[key] >= GATE[key] (None -> False)."""
    return {key: bool(metrics[key] is not None and metrics[key] >= GATE[key]) for key in GATE}
