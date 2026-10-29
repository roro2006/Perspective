"""T3-05: what the person's head points at, and when a look means "not that".
Angles in pair_map from a head at (0.7, 0, 0.4): looking at blue_1, both reds
are 8.37 deg off; looking at red_2, blue_1 is 8.37 and red_1 15.34 deg off."""

from __future__ import annotations

from scene import HUMAN, ROBOT, make_pose, obj

from curio_robot.collab.gaze import DWELL_S, GazeWatch, gaze_target
from curio_robot.types import ObjectMap

EYE = [0.7, 0.0, 0.4]


def test_looks_at_the_nearest_to_the_line_of_sight(pair_map):
    assert gaze_target(make_pose(EYE, [0.3, 0.0, 0.0125]), pair_map) == "blue_1"
    assert gaze_target(make_pose(EYE, [0.25, -0.08, 0.0125]), pair_map) == "red_2"


def test_nothing_within_the_cone(pair_map):
    away = make_pose(EYE, [0.25, 0.4, 0.0])  # nearest object is 25.9 deg off
    assert gaze_target(away, pair_map) is None
    assert gaze_target(away, pair_map, max_deg=30.0) == "red_1"


def test_cone_is_inclusive_and_configurable(pair_map):
    at_blue = make_pose(EYE, [0.3, 0.0, 0.0125])
    reds = ObjectMap([o for o in pair_map.objects if o.color == "red"], ROBOT, HUMAN)
    assert gaze_target(at_blue, reds, max_deg=5.0) is None
    assert gaze_target(at_blue, reds, max_deg=8.4) == "red_1"  # a tie: smaller id


def test_tie_goes_to_the_smaller_id():
    m = ObjectMap([obj("b_2", "red", (0.25, 0.05)), obj("b_1", "red", (0.25, -0.05))],
                  ROBOT, HUMAN)
    assert gaze_target(make_pose([0.0, 0.0, 0.4], [0.25, 0.0, 0.0125]), m, max_deg=30) == "b_1"


def test_empty_map():
    assert gaze_target(make_pose(EYE, [0.3, 0, 0]), ObjectMap([], ROBOT, HUMAN)) is None


def test_fires_once_after_the_dwell():
    w = GazeWatch(0.4)
    assert w.update(0.0, "a", "b") is None
    assert w.update(0.3, "a", "b") is None
    assert w.update(0.4, "a", "b") == "a"
    assert w.update(0.5, "a", "b") is None
    assert w.update(2.0, "a", "b") is None


def test_a_new_look_restarts_the_clock():
    w = GazeWatch(0.4)
    w.update(0.0, "a", "b")
    assert w.update(0.6, "c", "b") is None
    assert w.update(0.9, "c", "b") is None
    assert w.update(1.0, "c", "b") == "c"


def test_looking_at_the_goal_or_nothing_forgets():
    w = GazeWatch(0.4)
    w.update(0.0, "a", "b")
    assert w.update(0.2, "b", "b") is None  # looking where the arm goes
    assert w.update(0.5, "a", "b") is None  # timing restarts here
    assert w.update(0.8, "a", "b") is None
    assert w.update(0.9, "a", "b") == "a"
    assert w.update(1.0, None, "b") is None
    assert w.update(1.1, "a", "b") is None
    assert w.update(1.2, "a", None) is None  # arm idle: nothing to object to
    assert w.update(1.3, "a", "b") is None
    assert w.update(1.7, "a", "b") == "a"


def test_reset_and_default_dwell():
    w = GazeWatch()
    assert w.dwell_s == DWELL_S
    w.update(0.0, "a", "b")
    w.reset()
    assert w.update(DWELL_S, "a", "b") is None
    assert w.update(2 * DWELL_S, "a", "b") == "a"
