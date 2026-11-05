"""T5-27: Curio as a plug-in layer. CPU only: a stub grounder stands in for the server."""

from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from scene import HUMAN, ROBOT, make_pose, obj

from curio_robot import belief as belief_mod
from curio_robot.geometry import intrinsics_from_fovy, project
from curio_robot.grounding.client import GroundingAnswer
from curio_robot.layer import (
    Base,
    Goal,
    GroundingUnavailable,
    base_from_camera,
    check,
    pose,
    resolve,
    ring,
    robot_words,
)
from curio_robot.layer.adapters import ring_pixel
from curio_robot.layer.resolve import geometry_request, production_adapter
from curio_robot.layer.robot_words import instruction, relation
from curio_robot.sim.demos import draw_marker
from curio_robot.types import HUMAN_CAM, ROBOT_CAM, CameraFrame, ObjectMap


class Stub:
    def __init__(self, probs: dict[str, float]) -> None:
        self.probs = probs
        self.requests = []

    def score(self, request) -> GroundingAnswer:
        self.requests.append(request)
        return GroundingAnswer(dict(self.probs), None, 1.0, "stub")


def frame(pose, camera: str, t: float = 0.0) -> CameraFrame:
    k = intrinsics_from_fovy(60.0, 320, 240)
    rgb = np.full((240, 320, 3), 90, np.uint8)
    return CameraFrame(camera, rgb, k, pose, t)


@pytest.fixture
def frames():
    return frame(HUMAN, HUMAN_CAM), {ROBOT_CAM: frame(ROBOT, ROBOT_CAM)}


def test_resolve_asks_the_live_pipeline_request_and_fills_the_goal(pair_map, frames):
    head, robots = frames
    g = Stub({"red_1": 0.97, "red_2": 0.02, "blue_1": 0.01})
    goal = resolve("Hand me the red one on my left", head, robots, pair_map, None, grounder=g)
    assert g.requests[0].messages == geometry_request(
        "Hand me the red one on my left", pair_map, head).messages
    assert goal.target_id == "red_1"
    assert goal.frame == "speaker"
    assert goal.confidence == pytest.approx(0.97)
    assert goal.alternatives == (("red_2", 0.02), ("blue_1", 0.01))
    assert goal.ask is None and goal.note is None
    assert goal.adapter == production_adapter() == "r16-v4"
    # robot below its camera at (-0.2, 0), facing +x: red_1 is 0.45 ahead and 0.08 to its left
    assert goal.target_xyz_robot == pytest.approx((0.45, 0.08, 0.0125))


def test_habit_line_is_passed_through(pair_map, frames):
    head, robots = frames
    g = Stub({"red_1": 1.0})
    resolve("the red one on the left", head, robots, pair_map, None, grounder=g, p_own_frame=0.85)
    assert g.requests[0].messages == geometry_request(
        "the red one on the left", pair_map, head, 0.85).messages


def test_without_a_head_frame_it_grounds_on_the_robot_view(pair_map, frames):
    _, robots = frames
    g = Stub({"red_2": 1.0})
    goal = resolve("the red one on your right", None, robots, pair_map, None, grounder=g)
    text = g.requests[0].messages[0]["content"][1]["text"]
    assert "speaker's" not in text and "robot's" in text
    assert goal.frame == "robot" and goal.target_id == "red_2"


def test_unclear_answer_attaches_the_ask_back_question(pair_map, frames):
    head, robots = frames
    goal = resolve("the red one", head, robots, pair_map, None,
                   grounder=Stub({"red_1": 0.55, "red_2": 0.45}))
    assert goal.should_ask
    assert goal.ask in ("The one on your left?", "The one on your right?",
                        "The one nearer you?", "The one farther from you?")
    sure = resolve("the red one", head, robots, pair_map, None,
                   grounder=Stub({"red_1": 0.99, "red_2": 0.01}))
    assert sure.ask is None


def test_no_answer_raises(pair_map, frames):
    head, robots = frames
    with pytest.raises(GroundingUnavailable):
        resolve("the red one", head, robots, pair_map, None, grounder=Stub({}))


def test_belief_route_gives_the_moved_it_note(pair_map, frames):
    head, robots = frames
    person = belief_mod.Belief("p").observe(head, pair_map, ["red_1", "red_2", "blue_1"])
    moved = ObjectMap([obj("red_1", "red", (0.25, -0.2)), *pair_map.objects[1:]],
                      ROBOT, HUMAN)
    text = "Get the red one I left on my right"
    g = Stub({"red_1": 0.98, "red_2": 0.02})
    goal = resolve(text, head, robots, moved, person, grounder=g, movers={"red_1": "robot"})
    expected = belief_mod.resolve(text, Stub(g.probs), head, moved, person, {"red_1": "robot"})
    assert goal.target_id == expected.object_id == "red_1"
    assert goal.note == expected.note and goal.note.startswith("I moved it")
    assert goal.target_xyz_robot == pytest.approx((0.45, -0.2, 0.0125))


def test_goal_is_frozen_plain_data(pair_map, frames):
    head, robots = frames
    goal = resolve("the blue one", head, robots, pair_map, None, grounder=Stub({"blue_1": 1.0}))
    with pytest.raises(FrozenInstanceError):
        goal.target_id = "red_1"
    assert json.loads(json.dumps(goal.to_dict()))["target_id"] == "blue_1"


# ---- adapters ------------------------------------------------------------------------------


def _goal(xyz, target="red_1") -> Goal:
    return Goal(target, tuple(xyz), "speaker", 1.0, (), None, None, "r16-v4")


def test_base_round_trip_and_camera_base():
    base = Base((0.3, -1.0, 0.2), 2.1)
    p = np.array([1.0, 2.0, 0.5])
    assert np.allclose(base.to_world(base.to_base(p)), p, atol=1e-12)
    cam = base_from_camera(make_pose([1.0, 1.0, 0.5], [1.0, 2.0, 0.0]))  # looks along +y
    assert cam.yaw == pytest.approx(math.pi / 2)
    assert cam.position == (1.0, 1.0, 0.0)
    assert np.allclose(cam.to_base([1.0, 1.5, 0.0]), [0.5, 0.0, 0.0])


def test_pose_adapter_gives_robot_frame_or_world():
    base = base_from_camera(ROBOT)
    g = _goal(base.to_base([0.25, 0.08, 0.0125]))
    assert np.allclose(pose(g), [0.45, 0.08, 0.0125])
    assert np.allclose(pose(g, base), [0.25, 0.08, 0.0125], atol=1e-12)


def test_ring_matches_the_smolvla_marker():
    robot = frame(ROBOT, ROBOT_CAM)
    world = np.array([0.25, 0.08, 0.0125])
    g = _goal(base_from_camera(ROBOT).to_base(world))
    u, v = project(world, robot.intrinsics, robot.pose)
    expected = draw_marker(robot.rgb, (round(float(u), 2), round(float(v), 2)))
    out = ring(g, robot)
    assert np.array_equal(out, expected)
    assert (out != robot.rgb).any() and (robot.rgb == 90).all()


def test_ring_off_image_is_unmarked():
    robot = frame(ROBOT, ROBOT_CAM)
    g = _goal(base_from_camera(ROBOT).to_base([-1.0, 0.0, 0.0]))  # behind the camera
    assert ring_pixel(g, robot) is None
    assert np.array_equal(ring(g, robot), robot.rgb)


# ---- robot words ---------------------------------------------------------------------------


def test_words_use_the_robot_frame(pair_map):
    assert instruction("red_1", pair_map) == "pick up the red block on your left"
    assert instruction("red_2", pair_map) == "pick up the red block on your right"
    assert instruction("blue_1", pair_map) == "pick up the blue block"
    assert robot_words(_goal((0, 0, 0)), pair_map) == "pick up the red block on your left"


def test_distance_when_left_right_does_not_separate():
    om = ObjectMap([obj("red_1", "red", (0.10, 0.0)), obj("red_2", "red", (0.40, 0.005))],
                   ROBOT, HUMAN)
    assert instruction("red_1", om) == "pick up the red block closest to you"
    assert instruction("red_2", om) == "pick up the red block furthest from you"
    assert check(instruction("red_2", om), om) == "red_2"


def test_check_rejects_words_that_do_not_single_out_one():
    om = ObjectMap([obj("red_1", "red", (0.25, 0.0)), obj("red_2", "red", (0.25, 0.005))],
                   ROBOT, HUMAN)
    assert check(instruction("red_1", om), om) is None
    pair = ObjectMap([obj("red_1", "red", (0.25, 0.08)), obj("red_2", "red", (0.25, -0.08))],
                     ROBOT, HUMAN)
    assert check("pick up the red block", pair) is None
    assert check("pick up the green block", pair) is None


def test_words_never_mention_the_speaker_and_round_trip():
    rng = np.random.default_rng(7)
    colours = ("red", "blue", "green", "yellow")
    for _ in range(200):
        objs = []
        for i, c in enumerate(rng.choice(colours, 5)):
            objs.append(obj(f"{c}_{i}", str(c), rng.uniform([0.05, -0.2], [0.45, 0.2])))
        om = ObjectMap(objs, ROBOT, HUMAN)
        for o in objs:
            words = instruction(o.object_id, om)
            assert words == instruction(o.object_id, om)  # deterministic
            assert not set(words.split()) & {"my", "me", "i", "person", "speaker", "their"}
            if relation(o.object_id, om) is not None:  # a separating relation exists
                assert check(words, om) == o.object_id
            else:  # the fallback words never claim to single out the target
                assert check(words, om) != o.object_id
