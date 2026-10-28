"""Acceptance tests for T1-07 (recorder.py)."""

import json

import numpy as np

from curio_robot.conventions.decide import Decision
from curio_robot.conventions.readings import Reading
from curio_robot.recorder import VERSIONS, DecisionRecord, DecisionRecorder, load_records


def _rec(utterance: int, stage="final") -> DecisionRecord:
    return DecisionRecord(
        utterance=utterance,
        t=1.5,
        text="the red one on the left",
        stage=stage,
        readings=[Reading("red_2", "speaker", "left", False), Reading("red_1", "robot", "left", False)],
        posterior={"red_2": 0.8, "red_1": 0.2},
        decision=Decision("act", "red_2", np.array([0.25, -0.0800004, 0.0125]), 0.8),
        belief={"own_frame_a": 1.6, "own_frame_b": 0.4},
        person="ana",
    )


def test_lines_are_plain_json(tmp_path):
    path = tmp_path / "decisions.jsonl"
    r = DecisionRecorder(path)
    r.record(_rec(1))
    r.outcome(1, "corrected", "red_1", 2.0)
    r.close()
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert [line["kind"] for line in lines] == ["decision", "outcome"]
    d = lines[0]
    assert d["readings"][0] == {
        "object_id": "red_2", "frame": "speaker", "relation": "left", "explicit": False
    }
    assert d["decision"]["kind"] == "act"
    assert d["decision"]["position"] == [0.25, -0.08, 0.0125]
    assert d["versions"] == VERSIONS and d["person"] == "ana"


def test_load_joins_outcomes(tmp_path):
    path = tmp_path / "decisions.jsonl"
    r = DecisionRecorder(path)
    r.record(_rec(1, "partial"))
    r.record(_rec(1, "final"))
    r.record(_rec(2))
    r.outcome(1, "accepted", None, 2.0)
    r.outcome(1, "corrected", "red_1", 3.0)  # the last outcome wins
    r.close()
    recs = load_records(path)
    assert [(x["utterance"], x["stage"]) for x in recs] == [(1, "partial"), (1, "final"), (2, "final")]
    assert [x["outcome"] for x in recs] == ["corrected", "corrected", None]
    assert recs[0]["right_id"] == "red_1" and recs[2]["right_id"] is None


def test_appends_across_sessions(tmp_path):
    path = tmp_path / "sub" / "decisions.jsonl"
    for u in (1, 2):
        r = DecisionRecorder(path)
        r.record(_rec(u))
        r.close()
    assert [x["utterance"] for x in load_records(path)] == [1, 2]
