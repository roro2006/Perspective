"""Acceptance tests for T2-06 (grounding/fuse.py)."""

import pytest

from curio_robot.grounding.fuse import fuse


def test_product_of_experts():
    out = fuse({"a": 0.5, "b": 0.5}, {"a": 0.9, "b": 0.1}, weight=1.0, floor=0.0)
    assert out == pytest.approx({"a": 0.9, "b": 0.1})


def test_floor_keeps_a_vlm_zero_from_vetoing():
    out = fuse({"a": 0.8, "b": 0.2}, {"b": 1.0}, weight=1.0, floor=0.05)
    # a: 0.8 * 0.05 = 0.04, b: 0.2 * 1.05 = 0.21
    assert out == pytest.approx({"a": 0.04 / 0.25, "b": 0.21 / 0.25})


def test_explicit_frame_cannot_move():
    assert fuse({"a": 1.0}, {"b": 0.99, "a": 0.01}) == pytest.approx({"a": 1.0})


def test_vlm_cannot_add_candidates():
    out = fuse({"a": 0.5, "b": 0.5}, {"c": 1.0}, floor=0.05)
    assert set(out) == {"a", "b"} and out == pytest.approx({"a": 0.5, "b": 0.5})


def test_no_readings_means_the_vlm_decides():
    assert fuse({}, {"a": 3.0, "b": 1.0}) == pytest.approx({"a": 0.75, "b": 0.25})
    assert fuse({}, {}) == {}


def test_unavailable_vlm_keeps_the_belief():
    assert fuse({"a": 0.7, "b": 0.3}, {}) == {"a": 0.7, "b": 0.3}


def test_weight_zero_is_the_belief():
    assert fuse({"a": 0.7, "b": 0.3}, {"b": 1.0}, weight=0.0) == pytest.approx({"a": 0.7, "b": 0.3})


def test_weight_sharpens():
    soft = fuse({"a": 0.5, "b": 0.5}, {"a": 0.6, "b": 0.4}, weight=1.0, floor=0.0)
    sharp = fuse({"a": 0.5, "b": 0.5}, {"a": 0.6, "b": 0.4}, weight=3.0, floor=0.0)
    assert sharp["a"] > soft["a"] > 0.5


def test_all_zero_keeps_the_belief():
    assert fuse({"a": 0.0, "b": 0.0}, {"a": 1.0}, floor=0.0) == {"a": 0.0, "b": 0.0}
