"""Acceptance tests for T1-04 (conventions/decide.py)."""

import numpy as np
import pytest

from curio_robot.conventions.decide import Thresholds, decide


def test_confident_acts(pair_map):
    d = decide({"red_2": 0.8, "red_1": 0.2}, pair_map, "the red one on the left")
    assert (d.kind, d.object_id) == ("act", "red_2")
    assert d.confidence == pytest.approx(0.8)
    assert np.allclose(d.position, [0.25, -0.08, 0.0125])
    assert d.question is None


def test_unsure_hedges_toward_the_weighted_centre(pair_map):
    d = decide({"red_2": 0.6, "red_1": 0.4}, pair_map, "the red one on the left")
    assert (d.kind, d.object_id) == ("hedge", "red_2")
    assert np.allclose(d.position, [0.25, 0.6 * -0.08 + 0.4 * 0.08, 0.0125])


def test_tie_breaks_on_the_smallest_id(pair_map):
    d = decide({"red_2": 0.5, "red_1": 0.5}, pair_map, "the red one on the left")
    assert (d.kind, d.object_id) == ("hedge", "red_1")


def test_lost_asks_whose_side(pair_map):
    post = {"red_1": 0.4, "red_2": 0.3, "blue_1": 0.3}
    d = decide(post, pair_map, "the one on the left")
    assert (d.kind, d.object_id, d.position) == ("ask", None, None)
    assert d.confidence == pytest.approx(0.4)
    assert d.question == "Your left or mine?"
    d = decide(post, pair_map, "that one")
    assert d.kind == "ask" and d.question == "Which one do you mean?"


def test_empty_is_none(pair_map):
    d = decide({}, pair_map, "the red one")
    assert (d.kind, d.object_id, d.position, d.confidence) == ("none", None, None, 0.0)


def test_ids_not_in_the_map_are_dropped(pair_map):
    d = decide({"ghost_1": 0.9, "red_1": 0.1}, pair_map, "the red one")
    assert (d.kind, d.object_id) == ("act", "red_1")
    assert d.confidence == pytest.approx(1.0)
    assert decide({"ghost_1": 1.0}, pair_map, "x").kind == "none"


def test_custom_thresholds(pair_map):
    d = decide({"red_2": 0.8, "red_1": 0.2}, pair_map, "left", Thresholds(act=0.9, hedge=0.85))
    assert d.kind == "ask"
