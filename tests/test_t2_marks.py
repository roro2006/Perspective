"""Acceptance tests for T2-03 (grounding/marks.py)."""

import math

import numpy as np
import pytest
from scene import HUMAN, obj

from curio_robot.geometry import intrinsics_from_fovy, project
from curio_robot.grounding.marks import (
    CROP_MARGIN_PX,
    MAX_SIDE_PX,
    Crop,
    Mark,
    place_marks,
    render_marks,
)
from curio_robot.types import HUMAN_CAM, CameraFrame, ObjectMap


def _frame(w=320, h=240) -> CameraFrame:
    k = intrinsics_from_fovy(70, w, h)
    return CameraFrame(HUMAN_CAM, np.zeros((h, w, 3), np.uint8), k, HUMAN, 0.0)


def test_marks_follow_projection_and_id_order(pair_map):
    frame = _frame()
    marks = place_marks(pair_map, frame)
    assert [(m.letter, m.object_id) for m in marks] == [("A", "blue_1"), ("B", "red_1"), ("C", "red_2")]
    for m in marks:
        o = next(o for o in pair_map.objects if o.object_id == m.object_id)
        assert np.allclose(m.pixel, project(o.position, frame.intrinsics, frame.pose))


def test_objects_out_of_view_get_no_mark(pair_map):
    behind = obj("green_1", "green", (1.5, 0.0))  # behind the person
    far_side = obj("yellow_1", "yellow", (0.3, 3.0))  # way off to the side
    m = ObjectMap([*pair_map.objects, behind, far_side], pair_map.robot_pose, pair_map.human_pose)
    assert {mk.object_id for mk in place_marks(m, _frame())} == {"blue_1", "red_1", "red_2"}


def test_at_most_26_marks(pair_map):
    many = [obj(f"red_{i:02d}", "red", (0.2 + 0.004 * i, 0.0)) for i in range(30)]
    m = ObjectMap(many, pair_map.robot_pose, pair_map.human_pose)
    marks = place_marks(m, _frame())
    assert len(marks) == 26 and marks[-1].letter == "Z" and marks[-1].object_id == "red_25"


def test_render_crops_around_the_marks(pair_map):
    frame = _frame()
    marks = place_marks(pair_map, frame)
    image, crop = render_marks(frame, marks)
    us = [m.pixel[0] for m in marks]
    vs = [m.pixel[1] for m in marks]
    x0 = max(0, math.floor(min(us) - CROP_MARGIN_PX))
    y0 = max(0, math.floor(min(vs) - CROP_MARGIN_PX))
    x1 = min(320, math.ceil(max(us) + CROP_MARGIN_PX))
    y1 = min(240, math.ceil(max(vs) + CROP_MARGIN_PX))
    assert crop == Crop(x0, y0, 1.0)
    assert image.shape == (y1 - y0, x1 - x0, 3) and image.dtype == np.uint8
    for m in marks:  # the disc is drawn where the mark is
        u, v = round(m.pixel[0] - x0), round(m.pixel[1] - y0)
        assert image[v - 6 : v + 7, u - 6 : u + 7].max() > 200
    assert frame.rgb.max() == 0  # the input frame is untouched


def test_render_downscales_big_crops():
    frame = _frame(1280, 960)
    marks = [Mark("A", "red_1", (100.0, 100.0)), Mark("B", "red_2", (1100.0, 700.0))]
    image, crop = render_marks(frame, marks)
    w, h = 1140 - 60, 740 - 60  # marks +- 40 px
    assert (crop.x0, crop.y0) == (60, 60)
    assert crop.scale == pytest.approx(MAX_SIDE_PX / w)
    assert image.shape == (round(h * crop.scale), MAX_SIDE_PX, 3)


def test_no_marks_is_the_whole_frame():
    frame = _frame(640, 480)
    image, crop = render_marks(frame, [])
    assert (crop.x0, crop.y0) == (0, 0)
    assert crop.scale == MAX_SIDE_PX / 640
    assert image.shape == (336, 448, 3)
