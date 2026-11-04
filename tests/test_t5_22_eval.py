"""T5-22: the ask-back replay, split, tuning rule and statistics (CPU only, no grounder)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ask_back_eval as E

from curio_robot.types import CameraPose, ObjectMap, WorldObject

# person at x=0.7 facing -x (their left is -y), robot at x=-0.2 facing +x, as tests/scene.py
HUMAN = CameraPose(np.array([0.7, 0.0, 0.4]),
                   np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
ROBOT = CameraPose(np.array([-0.2, 0.0, 0.4]),
                   np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))


def item(i: int, kind: str, target: str) -> SimpleNamespace:
    objs = [WorldObject("red_1", "red", np.array([0.3, 0.08, 0.0125])),
            WorldObject("red_2", "red", np.array([0.3, -0.08, 0.0125]))]
    return SimpleNamespace(id=f"{kind}-{i:04d}", kind=kind, target_id=target,
                           object_map=ObjectMap(objs, ROBOT, HUMAN),
                           robot_visible=("red_1", "red_2"))


def test_split_is_even_odd_within_each_kind():
    items = [item(i, k, "red_1") for k in ("a", "b") for i in range(4)]
    dev, test = E.split(items)
    assert [x.id for x in dev] == ["a-0000", "a-0002", "b-0000", "b-0002"]
    assert [x.id for x in test] == ["a-0001", "a-0003", "b-0001", "b-0003"]


def test_candidates_are_in_the_speaker_frame():
    c = {x.object_id: x for x in E.candidates_for(item(0, "habit", "red_1"))}
    assert c["red_2"].lateral < 0 < c["red_1"].lateral  # red_2 is on the person's left


def test_replay_asks_only_when_unsure_and_truthful_answers_fix_it():
    items = [item(0, "habit", "red_2"), item(1, "habit", "red_1")]
    scores = {"habit-0000": {"probs": {"red_1": 0.7, "red_2": 0.3}},
              "habit-0001": {"probs": {"red_1": 0.99, "red_2": 0.01}}}
    runs = E.replay(items, scores, 0.95)
    assert runs[0].n_q >= 1 and runs[0].ok and not runs[0].base_ok
    assert runs[0].asked[0][0] == "The one on your left?"
    assert runs[1].n_q == 0 and runs[1].ok
    s = E.summary(runs)
    assert s["acc"] == 1.0 and s["base_acc"] == 0.5 and s["ask_rate"] == 0.5


def test_tune_picks_the_smallest_best_tau_within_budget():
    items = [item(i, "habit", "red_2") for i in range(8)]
    scores = {x.id: {"probs": {"red_1": 0.6, "red_2": 0.4}} for x in items[:1]}
    scores |= {x.id: {"probs": {"red_1": 0.01, "red_2": 0.99}} for x in items[1:]}
    tau, table = E.tune(items, scores)
    assert len(table) == len(E.TAUS)
    assert tau == 0.65  # first tau above 0.6: asks on 1/8 = 0.125 <= 0.25 and is right


def test_wilson_and_bootstrap():
    lo, hi = E.wilson(171, 175)
    assert lo < 171 / 175 < hi <= 1.0
    assert E.wilson(0, 10)[0] == 0.0
    lo, hi = E.boot_ci(np.array([0.0, 1.0] * 50))
    assert lo < 0.5 < hi
    assert hi - lo == pytest.approx(0.2, abs=0.08)
