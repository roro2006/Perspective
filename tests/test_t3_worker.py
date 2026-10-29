"""T3-03: how the simulated person names each goal. In pair_map the robot
faces +x (its left is red_1) and the person faces -x (their left is red_2)."""

from __future__ import annotations

import numpy as np
import pytest
from scene import HUMAN, ROBOT, obj

from curio_robot.collab.worker import Goal, describe, plan_task
from curio_robot.sim.users import SimUser
from curio_robot.types import ObjectMap

OWN = SimUser("own", p_own_frame=1.0)
ROBOTS = SimUser("robot", p_own_frame=0.0)


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def test_unique_colour_needs_no_side_and_no_draws(pair_map):
    r = rng(5)
    assert describe("blue_1", pair_map, OWN, r) == Goal("blue_1", "Hand me the blue one", None,
                                                       False)
    assert r.random() == rng(5).random()


def test_deictic_needs_no_draws(pair_map):
    r = rng(5)
    assert describe("red_1", pair_map, OWN, r, deictic=True) == Goal(
        "red_1", "Hand me that one", None, True)
    assert r.random() == rng(5).random()


def test_own_frame_user_names_their_side(pair_map):
    assert describe("red_2", pair_map, OWN, rng()) == Goal(
        "red_2", "Hand me the red one on the left", "speaker", False)
    assert describe("red_1", pair_map, OWN, rng()).text == "Hand me the red one on the right"


def test_robot_frame_user_names_the_robots_side(pair_map):
    assert describe("red_1", pair_map, ROBOTS, rng()) == Goal(
        "red_1", "Hand me the red one on the left", "robot", False)
    assert describe("red_2", pair_map, ROBOTS, rng()).text == "Hand me the red one on the right"


def test_markers(pair_map):
    mine = SimUser("m", p_own_frame=1.0, p_marker=1.0)
    yours = SimUser("y", p_own_frame=0.0, p_marker=1.0)
    assert describe("red_2", pair_map, mine, rng()).text == "Hand me the red one on my left"
    assert describe("red_1", pair_map, yours, rng()).text == "Hand me the red one on your left"


@pytest.mark.parametrize("seed", range(6))
def test_frame_then_marker_draw_order(pair_map, seed):
    user = SimUser("half", p_own_frame=0.5, p_marker=0.5)
    ref = rng(seed)
    frame = "speaker" if ref.random() < 0.5 else "robot"
    marked = ref.random() < 0.5
    side = "left" if frame == "speaker" else "right"  # red_2 is left for the person only
    m = "my" if frame == "speaker" else "your"
    text = f"Hand me the red one on {m} {side}" if marked else f"Hand me the red one on the {side}"
    r = rng(seed)
    assert describe("red_2", pair_map, user, r) == Goal("red_2", text, frame, False)
    assert r.random() == ref.random()  # exactly two draws


def test_errors(pair_map, robot_only_map):
    with pytest.raises(ValueError):
        describe("blue_1", robot_only_map, OWN, rng())
    with pytest.raises(KeyError):
        describe("green_9", pair_map, OWN, rng())
    three = ObjectMap([obj("red_1", "red", (0.25, 0.08)), obj("red_2", "red", (0.25, 0.0)),
                       obj("red_3", "red", (0.25, -0.08))], ROBOT, HUMAN)
    with pytest.raises(ValueError):
        describe("red_2", three, OWN, rng())


def test_plan_order_and_reproducible(pair_map):
    goals = plan_task(pair_map, OWN, 3, rng(4))
    ids = sorted(o.object_id for o in pair_map.objects)
    order = rng(4).permutation(3)
    assert [g.object_id for g in goals] == [ids[i] for i in order]
    assert plan_task(pair_map, OWN, 3, rng(4)) == goals
    assert all(not g.deictic for g in goals)


def test_plan_partial_and_deictic(pair_map):
    goals = plan_task(pair_map, OWN, 2, rng(1), p_deictic=1.0)
    assert len(goals) == 2 and len({g.object_id for g in goals}) == 2
    assert all(g.deictic and g.text == "Hand me that one" for g in goals)


def test_plan_bad_counts(pair_map):
    with pytest.raises(ValueError):
        plan_task(pair_map, OWN, 4, rng())
    with pytest.raises(ValueError):
        plan_task(pair_map, OWN, 0, rng())
