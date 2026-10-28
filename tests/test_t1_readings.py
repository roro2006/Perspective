"""Acceptance tests for T1-01 (conventions/readings.py)."""

import pytest
from scene import obj

from curio_robot.conventions.readings import Reading, enumerate_readings, frame_marker
from curio_robot.types import ObjectMap


@pytest.mark.parametrize(
    ("text", "frame"),
    [
        ("Hand me the red one on my left", "speaker"),
        ("the red one on your left", "robot"),
        ("Hand me the red one on the left", None),
        ("can you hand me the red one on the left", None),  # bare "you"/"me" aren't markers
        ("the one closest to me", "speaker"),
        ("the one closest to you", "robot"),
        ("not my left, your left", "robot"),  # the last marker wins
        ("no, YOUR left... my left!", "speaker"),
        ("hand me that", None),
    ],
)
def test_frame_marker(text, frame):
    assert frame_marker(text) == frame


def test_bare_left_gives_one_reading_per_frame(pair_map):
    assert enumerate_readings("Hand me the red one on the left", pair_map) == [
        Reading("red_2", "speaker", "left", False),
        Reading("red_1", "robot", "left", False),
    ]


def test_my_left_is_explicit(pair_map):
    assert enumerate_readings("the red one on my left", pair_map) == [
        Reading("red_2", "speaker", "left", True)
    ]


def test_your_right_is_explicit(pair_map):
    assert enumerate_readings("the red one on your right", pair_map) == [
        Reading("red_2", "robot", "right", True)
    ]


def test_no_relation_lists_every_candidate(pair_map):
    assert enumerate_readings("the red one", pair_map) == [
        Reading("red_1", None, None, False),
        Reading("red_2", None, None, False),
    ]


def test_unknown_colour_gives_nothing(pair_map):
    assert enumerate_readings("the green one on the left", pair_map) == []


def test_no_colour_uses_every_object(pair_map):
    readings = enumerate_readings("the one on the left", pair_map)
    assert [r.frame for r in readings] == ["speaker", "robot"]
    assert readings[0].object_id == "red_2"  # leftmost of all three, speaker's view
    assert readings[1].object_id == "red_1"


def test_single_candidate_keeps_both_frames(pair_map):
    assert enumerate_readings("the blue one on the left", pair_map) == [
        Reading("blue_1", "speaker", "left", False),
        Reading("blue_1", "robot", "left", False),
    ]


def test_without_the_human_view_only_the_robot_frame(robot_only_map):
    assert enumerate_readings("the red one on the left", robot_only_map) == [
        Reading("red_1", "robot", "left", False)
    ]


def test_my_left_without_the_human_view_falls_back_to_robot(robot_only_map):
    assert enumerate_readings("the red one on my left", robot_only_map) == [
        Reading("red_1", "robot", "left", False)
    ]


def test_nearest_to_me(pair_map):
    near_human = obj("red_3", "red", (0.45, 0.0))
    m = ObjectMap([*pair_map.objects, near_human], pair_map.robot_pose, pair_map.human_pose)
    assert enumerate_readings("the red one closest to me", m) == [
        Reading("red_3", "speaker", "nearest", True)
    ]
