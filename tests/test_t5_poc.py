"""T5-18: the PoC video's head-frame maths and scene models."""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "poc"))

from kit import euler_quat, head_coords, side_of
from scenes_b import FaceToFace, all_scenes

from curio_robot.geometry import look_at
from curio_robot.perspective import lateral
from curio_robot.types import CameraPose, WorldObject


def _pose(rot: np.ndarray, pos=(0.0, 0.0, 0.0)) -> CameraPose:
    return CameraPose(np.asarray(pos, float), rot)


def _quat_mat(q: np.ndarray) -> np.ndarray:
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, q)
    return m.reshape(3, 3)


# A head looking along +x, upright: camera x = -y_world (right), y = up, -z = forward.
UPRIGHT = look_at(np.zeros(3), np.array([1.0, 0, 0]))


def test_upright_head_left_is_plus_y():
    pose = _pose(UPRIGHT)
    assert side_of((2.0, 1.0, 0.0), pose) == "left"
    assert side_of((2.0, -1.0, 0.0), pose) == "right"
    np.testing.assert_allclose(head_coords((2.0, 1.0, 0.5), pose), [-1.0, 0.5, 2.0], atol=1e-9)


def test_turned_person_left_follows_the_head_not_the_world():
    # facing -x (turned 180 degrees): their left is -y, the opposite of the robot's
    rot = _quat_mat(euler_quat(yaw=np.pi)) @ UPRIGHT
    assert side_of((-2.0, -1.0, 0.0), _pose(rot)) == "left"
    assert side_of((-2.0, 1.0, 0.0), _pose(rot)) == "right"


@pytest.mark.parametrize("roll_deg", [40, -40, 120, 180])
def test_rolled_head_left_is_the_heads_left(roll_deg):
    """Roll the head about its forward axis: a point on the head's own left (its -x camera
    axis) stays 'left' however the world sees it, and agrees with perspective.lateral."""
    rot = _quat_mat(euler_quat(roll=np.radians(roll_deg))) @ UPRIGHT
    pose = _pose(rot, (0.5, 0.2, 1.0))
    own_left = pose.position + 2.0 * pose.forward - 0.7 * pose.right
    assert side_of(own_left, pose) == "left"
    assert lateral(WorldObject("x", "x", own_left), pose) < 0
    if roll_deg == 180:  # upside down: the world's +y is now on the head's right
        assert side_of(pose.position + [2.0, 1.0, 0.0], pose) == "right"


def test_behind_is_negative_forward():
    pose = _pose(UPRIGHT)
    assert head_coords((-3.0, 0.0, 0.0), pose)[2] < 0


SCENES = all_scenes() + [FaceToFace()]


@pytest.mark.parametrize("scene", SCENES, ids=[s.name for s in SCENES])
def test_scene_compiles_and_poses(scene):
    model = mujoco.MjModel.from_xml_string(scene.xml())
    data = mujoco.MjData(model)
    scene.setup(model, data)
    mujoco.mj_forward(model, data)
    assert model.camera("cine").id >= 0


@pytest.mark.parametrize("scene", all_scenes(), ids=[s.name for s in all_scenes()])
def test_scene_answer_key_follows_the_speakers_frame(scene):
    """Each scene's correct object is the one the speaker's head frame names, and for every
    left/right query the robot's own frame names the other one."""
    model = mujoco.MjModel.from_xml_string(scene.xml())
    data = mujoco.MjData(model)
    scene.setup(model, data)
    mujoco.mj_forward(model, data)
    head, robot = scene.head_pose(model, data), scene.robot_pose(model, data)
    for q in scene.queries:
        words = q.text.lower()
        if "left" not in words and "right" not in words:
            continue
        side = "left" if "left" in words else "right"
        label = scene.cand(q.correct).label
        same = [c for c in scene.candidates if c.oid in q.ids and c.label == label]
        speaker_pick = [c.oid for c in same if side_of(c.pos, head) == side]
        robot_pick = [c.oid for c in same if side_of(c.pos, robot) == side]
        assert speaker_pick == [q.correct]
        assert q.correct not in robot_pick
