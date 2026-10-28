import numpy as np

from curio_robot.fusion import build_object_map
from curio_robot.geometry import intrinsics_from_fovy, lift_to_plane, look_at, project
from curio_robot.monitor import check_speech
from curio_robot.resolver import parse_request, parse_target_tag, resolve
from curio_robot.types import (
    HUMAN_CAM,
    ROBOT_CAM,
    CameraFrame,
    CameraPose,
    Detection,
    ObjectMap,
    WorldObject,
)

K = intrinsics_from_fovy(60, 320, 240)


def _pose(eye, look):
    eye = np.asarray(eye, float)
    return CameraPose(eye, look_at(eye, np.asarray(look, float)))


ROBOT = _pose([-0.2, 0, 0.5], [0.25, 0, 0])  # robot faces +x: its left is +y
HUMAN = _pose([0.7, 0, 0.4], [0.25, 0, 0])  # human faces the robot: their left is -y


def _obj(oid, color, xy, seen=(ROBOT_CAM,)):
    return WorldObject(oid, color, np.array([*xy, 0.0125]), set(seen))


def test_project_and_lift_round_trip():
    p = np.array([0.3, 0.05, 0.0125])
    uv = project(p, K, HUMAN)
    assert np.allclose(lift_to_plane(uv, K, HUMAN), p, atol=1e-6)


def test_my_left_uses_the_speakers_view():
    reds = [_obj("red_1", "red", (0.25, 0.08)), _obj("red_2", "red", (0.25, -0.08))]
    with_human = resolve("the red one on my left", ObjectMap(reds, ROBOT, HUMAN))
    robot_only = resolve("the red one on my left", ObjectMap(reds, ROBOT, None))
    assert with_human.object_id == "red_2"  # -y is the human's left
    assert robot_only.object_id == "red_1"  # +y is the robot's left


def test_your_left_uses_the_robots_view():
    reds = [_obj("red_1", "red", (0.25, 0.08)), _obj("red_2", "red", (0.25, -0.08))]
    assert resolve("the red one on your left", ObjectMap(reds, ROBOT, HUMAN)).object_id == "red_1"


def test_missing_colour_is_not_found():
    target = resolve("the blue one", ObjectMap([_obj("red_1", "red", (0.2, 0))], ROBOT, HUMAN))
    assert target.object_id is None


def test_parse_request():
    r = parse_request("Hand me the green one closest to me")
    assert (r.color, r.relation, r.speaker_frame) == ("green", "nearest", True)


def test_target_tag():
    assert parse_target_tag("Got it, the red one. [TARGET:red_2]") == ("Got it, the red one.", "red_2")
    assert parse_target_tag("I can't see it. [TARGET:none]") == ("I can't see it.", None)
    assert parse_target_tag("no tag") == ("no tag", None)


def test_fusion_merges_and_respects_switch():
    p = np.array([0.25, 0.02, 0.0125])
    frames = {
        cam: CameraFrame(cam, np.zeros((240, 320, 3), np.uint8), K, pose, 0.0)
        for cam, pose in ((ROBOT_CAM, ROBOT), (HUMAN_CAM, HUMAN))
    }
    hidden = np.array([0.3, -0.05, 0.0125])
    dets = {
        ROBOT_CAM: [Detection(ROBOT_CAM, "red", project(p, K, ROBOT), 100)],
        HUMAN_CAM: [
            Detection(HUMAN_CAM, "red", project(p, K, HUMAN), 100),
            Detection(HUMAN_CAM, "blue", project(hidden, K, HUMAN), 100),
        ],
    }
    on = build_object_map(frames, dets, use_human_view=True)
    off = build_object_map(frames, dets, use_human_view=False)
    assert sorted(o.object_id for o in on.objects) == ["blue_1", "red_1"]
    assert on.objects[0].seen_by == {ROBOT_CAM, HUMAN_CAM}
    assert [o.object_id for o in off.objects] == ["red_1"] and off.human_pose is None


def test_speech_interrupt():
    assert check_speech("no, the other one").source == "speech"
    assert check_speech("great, thanks") is None
