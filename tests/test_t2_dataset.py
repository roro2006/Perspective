"""Acceptance tests for T2-07 (sim/grounding_data.py). Renders the sim, so a
few seconds."""

import json

import numpy as np
import pytest
from PIL import Image

from curio_robot.grounding.marks import place_marks
from curio_robot.perspective import pick
from curio_robot.sim.grounding_data import load_dataset, make_example, write_dataset
from curio_robot.sim.world import SimWorld
from curio_robot.types import HUMAN_CAM, ObjectMap, WorldObject


@pytest.fixture(scope="module")
def world():
    return SimWorld(320, 240)


def _live_map(world) -> tuple[ObjectMap, object]:
    frame = world.render_camera(HUMAN_CAM)
    counts: dict[str, int] = {}
    objs = []
    for color, pos in world.block_positions():
        counts[color] = counts.get(color, 0) + 1
        objs.append(WorldObject(f"{color}_{counts[color]}", color, pos, set()))
    robot = world.render_camera("robot").pose
    return ObjectMap(objs, robot, frame.pose), frame


def test_unknown_kind(world):
    with pytest.raises(ValueError):
        make_example(world, np.random.default_rng(0), "juggling", "x")


def test_color_examples_name_a_unique_colour(world):
    rng = np.random.default_rng(1)
    for i in range(4):
        ex = make_example(world, rng, "color", f"c{i}")
        if ex is None:
            continue
        color = ex.target_id.split("_")[0]
        assert ex.text == f"Hand me the {color} one"
        assert [c for c, _ in world.block_positions()].count(color) == 1
        assert ex.request.letters[ex.answer] == ex.target_id


def test_deictic_target_is_where_the_person_looks(world):
    rng = np.random.default_rng(2)
    ex = next(e for e in (make_example(world, rng, "deictic", f"d{i}") for i in range(6)) if e)
    assert ex.text == "Hand me that one"
    object_map, frame = _live_map(world)
    marks = {m.object_id: m for m in place_marks(object_map, frame)}
    u, v = marks[ex.target_id].pixel
    assert abs(u - 160) < 3 and abs(v - 120) < 3  # dead centre of the person's view


def test_relation_targets_follow_the_frame(world):
    rng = np.random.default_rng(3)
    for i in range(6):
        ex = make_example(world, rng, "relation", f"r{i}")
        if ex is None:
            continue
        object_map, _ = _live_map(world)
        pair = object_map.objects[:2]
        side = ex.text.split()[-1]
        assert ex.target_id in {o.object_id for o in pair}
        if " my " in ex.text:
            assert ex.target_id == pick(pair, side, object_map.human_pose).object_id
        if " your " in ex.text:
            assert ex.target_id == pick(pair, side, object_map.robot_pose).object_id
        assert "usually mean" in ex.request.messages[0]["content"][1]["text"]


def test_write_and_load(tmp_path):
    path = write_dataset(tmp_path, n=6, seed=0)
    rows = load_dataset(path)
    assert [r["kind"] for r in rows] == ["relation", "deictic", "color"] * 2
    assert [r["id"] for r in rows] == [f"0-{i:05d}" for i in range(6)]
    for r in rows:
        assert r["letters"][r["answer"]] == r["target_id"]
        img = Image.open(tmp_path / r["image"])
        assert img.format == "PNG"
        user, assistant = r["messages"]
        assert user["content"][0]["image_url"]["url"] == r["image"]
        assert assistant == {"role": "assistant", "content": r["answer"]}
    raw = path.read_text().splitlines()
    assert len(raw) == 6 and "base64" not in raw[0]
    assert json.loads(raw[0])["prompt_version"]
