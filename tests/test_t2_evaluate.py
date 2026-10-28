"""Acceptance tests for T2-08 (grounding/evaluate.py)."""

import pytest

from curio_robot.grounding.evaluate import Example, Prediction, evaluate, passes_gate

EXAMPLES = [
    Example("e1", "relation", "A"),
    Example("e2", "relation", "B"),
    Example("e3", "deictic", "A"),
    Example("e4", "color", "C"),
]


def _predictor(answers: dict[str, str | None], ms: dict[str, float]):
    return lambda ex: Prediction(answers[ex.id], ms[ex.id])


def test_evaluate():
    predict = _predictor({"e1": "A", "e2": "A", "e3": None, "e4": "C"},
                         {"e1": 100.0, "e2": 200.0, "e3": 300.0, "e4": 400.0})
    r = evaluate(predict, EXAMPLES)
    assert r["n"] == 4 and r["accuracy"] == pytest.approx(0.5)
    assert r["by_kind"] == {
        "relation": {"n": 2, "accuracy": pytest.approx(0.5)},
        "deictic": {"n": 1, "accuracy": pytest.approx(0.0)},
        "color": {"n": 1, "accuracy": pytest.approx(1.0)},
    }
    assert r["p50_ms"] == pytest.approx(250.0) and r["p95_ms"] == pytest.approx(385.0)
    assert r["failures"] == ["e2", "e3"]


def test_evaluate_empty():
    r = evaluate(lambda ex: Prediction(None, 0.0), [])
    assert r == {"n": 0, "accuracy": 0.0, "by_kind": {}, "p50_ms": 0.0, "p95_ms": 0.0,
                 "failures": []}


def _result(acc, kinds, p95):
    return {"accuracy": acc, "p95_ms": p95,
            "by_kind": {k: {"n": 10, "accuracy": a} for k, a in kinds.items()}}


CURRENT = _result(0.8, {"relation": 0.9, "deictic": 0.7}, 600.0)


def test_gate_latency_first():
    ok, why = passes_gate(_result(0.99, {"relation": 1.0, "deictic": 1.0}, 900.0), CURRENT,
                          max_p95_ms=800)
    assert not ok and why == "p95 900 ms over budget 800 ms"


def test_gate_without_a_current_model():
    assert passes_gate(_result(0.1, {}, 100.0), None, max_p95_ms=800) == (True, "no current model")


def test_gate_rejects_a_worse_kind():
    ok, why = passes_gate(_result(0.85, {"relation": 0.95, "deictic": 0.65}, 500.0), CURRENT,
                          max_p95_ms=800)
    assert not ok and why == "worse on deictic: 0.650 vs 0.700"
    ok, why = passes_gate(_result(0.85, {"relation": 0.95}, 500.0), CURRENT, max_p95_ms=800)
    assert not ok and why.startswith("worse on deictic")


def test_gate_tolerance_and_pass():
    cand = _result(0.81, {"relation": 0.89, "deictic": 0.75}, 500.0)
    assert not passes_gate(cand, CURRENT, max_p95_ms=800)[0]
    assert passes_gate(cand, CURRENT, max_p95_ms=800, tolerance=0.02) == (
        True, "accuracy 0.810 vs 0.800")
