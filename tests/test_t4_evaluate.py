"""T4-06: scoring a critic and the phase 1 gate."""

from __future__ import annotations

import numpy as np
import pytest
from critic_util import make_ep

from curio_robot.critic.evaluate import (
    GATE,
    alarm_frame,
    auroc,
    evaluate,
    gate,
    threshold_for_precision,
)


def _scored(peak: float, at: int, top: int) -> tuple[np.ndarray, np.ndarray]:
    p = np.zeros(31)
    p[at:] = peak
    intended = np.zeros((31, 3))
    intended[:, top] = 1.0
    return p, intended


@pytest.fixture
def case():
    eps = [
        make_ep(intended=1, arm_target=0, t_no=1.5),  # wrong, flagged at 1.0 s, right block
        make_ep(intended=1, arm_target=0, t_no=None),  # wrong, silent, flagged, wrong block
        make_ep(intended=1, arm_target=0, t_no=1.2),  # wrong, never flagged above 0.3
        make_ep(intended=0, arm_target=0),  # right, false alarm at 0.6
    ]
    scored = [_scored(0.9, 10, 1), _scored(0.8, 12, 0), _scored(0.3, 5, 1), _scored(0.6, 5, 0)]
    return eps, scored


def test_alarm_frame():
    p = np.array([0.1, 0.4, 0.7, 0.2])
    assert alarm_frame(p, 0.5) == 2 and alarm_frame(p, 0.4) == 1 and alarm_frame(p, 0.8) is None


def test_evaluate_at_a_low_threshold(case):
    m = evaluate(*case, threshold=0.5)
    assert (m["n"], m["n_wrong"], m["tp"], m["fp"], m["fn"]) == (4, 3, 2, 1, 1)
    assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(2 / 3)
    assert m["false_alarm_rate"] == 1.0
    assert m["lead_s"] == pytest.approx(0.5) and m["target_acc"] == pytest.approx(0.5)
    assert m["threshold"] == 0.5


def test_evaluate_at_a_high_threshold(case):
    m = evaluate(*case, threshold=0.7)
    assert m["precision"] == 1.0 and m["fp"] == 0


def test_nothing_flagged_and_length_mismatch(case):
    m = evaluate(*case, threshold=0.95)
    assert m["precision"] == 0.0 and m["lead_s"] is None and m["target_acc"] is None
    with pytest.raises(ValueError):
        evaluate(case[0], case[1][:2], threshold=0.5)


def test_threshold_for_precision(case):
    assert threshold_for_precision(*case, precision=0.9) == pytest.approx(0.61)
    eps, scored = case
    assert threshold_for_precision([eps[3]], [scored[3]], precision=0.9) is None


def test_auroc(case):
    assert auroc(*case) == pytest.approx(2 / 3)
    ties = [make_ep(intended=1, arm_target=0), make_ep(intended=0, arm_target=0)]
    assert auroc(ties, [_scored(0.5, 0, 0), _scored(0.5, 0, 0)]) == 0.5
    with pytest.raises(ValueError):
        auroc(ties[:1], [_scored(0.5, 0, 0)])


def test_gate():
    good = {"precision": 0.95, "recall": 0.8, "lead_s": 0.5, "target_acc": 0.9}
    assert gate(good) == {k: True for k in GATE}
    assert gate({**good, "lead_s": None})["lead_s"] is False
    assert gate({**good, "recall": 0.5})["recall"] is False
