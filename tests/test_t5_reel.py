"""T5-19: the versatility reel's scene models and the geometry its scripted beats rely on."""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "poc"))

from kit import side_of
from make_reel import all_scenes
from reel_kit import REACH, arm_angles, in_reach, person_point, pointed_door
from reel_scenes_a import WALL_Y, Pointing, Reach
from reel_scenes_b import Placement


def _sim(scene):
    model = mujoco.MjModel.from_xml_string(scene.xml())
    data = mujoco.MjData(model)
    scene.setup(model, data)
    mujoco.mj_forward(model, data)
    return model, data


@pytest.mark.parametrize("scene", all_scenes(), ids=lambda s: s.name)
def test_every_scene_compiles_and_animates(scene):
    model, data = _sim(scene)
    assert model.camera("cine") is not None
    assert model.camera(scene.inset_cam()) is not None
    for t in (0.0, scene.duration / 2, scene.duration - 1e-3):
        scene.animate(model, data, t, {})
        mujoco.mj_forward(model, data)
    assert np.all(np.isfinite(data.qpos))
    assert scene.beats({}), scene.name


def test_nine_scenes_with_the_payoff_words_in_order():
    words = [s.word for s in all_scenes()]
    assert words == ["see", "reach", "look", "point", "remember", "place", "face", "move",
                     "learn"]


def test_only_visibility_and_your_frame_ask_the_grounder():
    live = {s.name for s in all_scenes() if s.queries}
    assert live == {"visibility", "yourframe"}


def test_reach_far_jar_is_outside_the_reach_radius():
    scene = Reach()
    shoulders = scene.shoulders()
    near, far = (np.asarray(scene.cand(k).pos) for k in ("jar_near", "jar_far"))
    assert any(in_reach(s, near) for s in shoulders)
    assert all(np.linalg.norm(far - s) > REACH for s in shoulders)
    assert scene.target() == "jar_far"


def test_reach_shoulders_match_the_posed_person():
    scene = Reach()
    model, data = _sim(scene)
    for side, expect in (("l", scene.shoulders()[0]), ("r", scene.shoulders()[1])):
        got = data.xpos[model.body(f"p_{side}_upper").id]
        np.testing.assert_allclose(got, expect, atol=0.03)


def test_person_point_is_in_the_persons_frame():
    # facing +y: forward is +y and their right is +x
    p = person_point((1.0, 2.0, 0.0), np.pi / 2, (0.5, -0.2, 0.0))
    np.testing.assert_allclose(p, [1.2, 2.5, 0.0], atol=1e-9)


def test_placement_right_hand_target_is_on_the_persons_right():
    scene = Placement()
    model, data = _sim(scene)
    head = scene.head_pose(model, data)
    target = scene.right_hand_target()
    expect = person_point(scene.PERSON_POS, scene.PERSON_YAW, (0.45, -0.24, 0.0))
    np.testing.assert_allclose(target[:2], expect[:2], atol=1e-9)
    assert side_of(target, head) == "right"
    # the robot-only reading (its own right) lands on the person's left, and on the table
    assert side_of(scene.robot_right_target(), head) == "left"
    assert np.all(np.abs(target[:2]) < 0.6)


def test_placement_targets_are_within_the_arms_reach():
    scene = Placement()
    for target in (scene.right_hand_target(), scene.robot_right_target()):
        a = arm_angles(scene.BASE, target)
        assert np.all(np.isfinite(list(a.values())))


def test_pointing_eye_to_fingertip_ray_picks_the_intended_door():
    scene = Pointing()
    _sim(scene)
    r = scene.rays
    assert r["pick"] == "door_near"
    assert pointed_door(r["eye"], r["tip"], scene.DOORS, WALL_Y) == "door_near"
    # the arm's own line (shoulder to fingertip) lands on the other door: the naive beat
    assert r["naive"] == "door_far"
    assert abs(r["eye_hit"][0] - scene.DOORS["door_near"]) < 0.45  # within the door's width


def test_pointed_door_ignores_rays_away_from_the_wall():
    assert pointed_door((0, 0, 1.6), (1.0, -0.5, 1.6), {"a": 1.0}, 1.3) is None
