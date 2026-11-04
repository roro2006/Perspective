"""T5-23: teach by doing through the head camera. Pure checks of naming, relations, retargeting
and the goal check, then rendered perception, the new place(), and one full episode."""

from __future__ import annotations

import numpy as np
import pytest

from curio_robot.ego_plan import (
    BLOCK_HALF,
    ROBOT_FRAME,
    PersonFrame,
    Step,
    classify,
    detect,
    goal_for,
    name_objects,
)
from curio_robot.sim.ego_tasks import FAMILIES, Goal, make_episode, satisfied

# The person across the table (facing -x): their left is -y, the robot's left is +y.
ACROSS = PersonFrame.from_left(np.array([0.0, -1.0, 0.0]))


def p(x: float, y: float, z: float = BLOCK_HALF) -> np.ndarray:
    return np.array([x, y, z])


def test_person_frame_axes():
    assert np.allclose(ACROSS.forward, [-1, 0, 0])  # away from the person, towards the robot
    assert np.allclose(ROBOT_FRAME.forward, [1, 0, 0])
    s, t = ACROSS.coords(p(0.24, -0.1))
    assert s == pytest.approx(0.1) and t == pytest.approx(0.0)


def test_names_count_from_the_persons_left():
    blobs = [("red", p(0.2, 0.1)), ("red", p(0.2, -0.1)), ("blue", p(0.3, 0.0))]
    assert name_objects(blobs, ACROSS) == ["red_2", "red_1", "blue_1"]
    assert name_objects(blobs, ROBOT_FRAME) == ["red_1", "red_2", "blue_1"]


def test_classify_on_beside_zone():
    others = {"blue_1": p(0.24, 0.0)}
    assert classify(p(0.24, 0.0, 3 * BLOCK_HALF), others, ACROSS) == Step("", "on", "blue_1")
    assert classify(p(0.24, -0.06), others, ACROSS) == Step("", "left_of", "blue_1")
    assert classify(p(0.24, 0.06), others, ACROSS) == Step("", "right_of", "blue_1")
    assert classify(p(0.24, 0.06), others, ROBOT_FRAME) == Step("", "left_of", "blue_1")
    far = {"blue_1": p(0.32, 0.12)}
    assert classify(p(0.24, -0.1), far, ACROSS) == Step("", "zone", "left")
    assert classify(p(0.24, 0.1), far, ACROSS) == Step("", "zone", "right")
    assert classify(p(0.24, 0.0), far, ACROSS).relation == "at"


def test_goal_for_reads_left_in_the_given_frame():
    pos = {"red_1": p(0.2, 0.1), "blue_1": p(0.24, 0.0)}
    step = Step("red_1", "left_of", "blue_1")
    curio = goal_for(step, pos, ACROSS)
    naive = goal_for(step, pos, ROBOT_FRAME)
    assert curio is not None and naive is not None
    assert curio[1] < -0.04 and naive[1] > 0.04  # opposite sides of blue_1
    on = goal_for(Step("red_1", "on", "blue_1"), pos, ACROSS)
    assert np.allclose(on, p(0.24, 0.0, 3 * BLOCK_HALF))
    zone = goal_for(Step("red_1", "zone", "left"), pos, ACROSS)
    assert ACROSS.coords(zone)[0] == pytest.approx(0.10)
    assert goal_for(Step("red_1", "on", "green_1"), pos, ACROSS) is None


def test_goal_check_takes_any_block_of_the_colour():
    colors = ("red", "red", "blue")
    goals = (Goal(0, "left_of", 2),)
    start = [p(0.2, 0.1), p(0.3, 0.1), p(0.24, 0.0)]
    moved_other = [start[0], p(0.24, -0.06), start[2]]  # the other red did it
    assert satisfied(goals, colors, start, moved_other, ACROSS)
    wrong_side = [start[0], p(0.24, 0.06), start[2]]
    assert not satisfied(goals, colors, start, wrong_side, ACROSS)
    fell = [p(0.24, -0.06), start[1], p(0.24, 0.0, -0.7)]
    assert not satisfied(goals, colors, start, fell, ACROSS)


@pytest.mark.parametrize("family", FAMILIES)
def test_episodes_are_deterministic_and_unambiguous(family):
    a, b = make_episode(family, 1000), make_episode(family, 1000)
    assert a.plan == b.plan and a.new_xy == b.new_xy and a.plan
    assert len(a.moves) == len(a.plan) == len(a.goals)
    assert a.demo_xy != a.new_xy


def test_detect_finds_rendered_blocks_from_the_head_camera():
    from curio_robot.sim.ego_world import EgoWorld

    ep = make_episode("c_beside", 1001)
    with EgoWorld() as w:
        w.reset(ep.scene(new=False))
        blobs, _ = detect(w.ego_frame())
        truth = [w.block_position(i) for i in range(len(ep.colors))]
    on_table = [b for b in blobs if b.position[2] < 0.03]
    assert len(on_table) == len(truth)
    for c, q in zip(ep.colors, truth):
        err = min(np.linalg.norm((b.position - q)[:2]) for b in on_table if b.color == c)
        assert err < 0.01


def test_place_puts_a_held_block_beside_and_on_another():
    from curio_robot.collab.tasks import TaskScene
    from curio_robot.sim.expert import pick
    from curio_robot.sim.physics import PhysWorld
    from curio_robot.sim.place import place, resting_z
    from curio_robot.sim.world import BlockSpec

    scene = TaskScene("t", (BlockSpec("red", (0.2, 0.05)), BlockSpec("blue", (0.28, -0.08)),
                            BlockSpec("green", (0.3, 0.1))),
                      np.array([0.7, 0, 0.38]), np.array([0.24, 0, 0]), None, 0.0)
    with PhysWorld() as w:
        w.reset(scene)
        assert pick(w, 0).success
        goal = w.block_position(1) + [0, 0.06, 0]
        goal[2] = resting_z()
        assert place(w, 0, goal).success
        assert pick(w, 2).success
        assert place(w, 2, w.block_position(1) + [0, 0, 2 * BLOCK_HALF]).success
        top, base = w.block_position(2), w.block_position(1)
    assert np.linalg.norm((top - base)[:2]) < 0.01 and top[2] - base[2] > 0.02


@pytest.mark.slow
def test_one_episode_end_to_end():
    from curio_robot.sim.ego_exec import episode
    from curio_robot.sim.ego_world import EgoWorld

    ep = make_episode("c_beside", 1002)
    with EgoWorld() as w:
        out = episode(w, ep)
    assert out["exact"], out
    assert out["curio"]["success"], out
    assert not out["baseline"]["success"], out
