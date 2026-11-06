"""T5-35: seeded building layouts for the dog demo. Layout generation is plain numpy and runs
anywhere; building, deciding and walking need the Go2 assets and skip without them."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from curio_robot.demo import dog, dog_scenes, voice
from curio_robot.demo.dog_scenes import HALF_W, NEW_KINDS, SLOT
from curio_robot.resolver import parse_request

ROOT = Path(__file__).resolve().parents[1]
needs_go2 = pytest.mark.skipif(not dog.have_assets(), reason="Go2 assets not downloaded")


@pytest.mark.parametrize("kind", NEW_KINDS)
def test_layouts_are_seeded_and_sane(kind):
    for seed in range(30):
        a, b = dog_scenes.make(kind, seed), dog_scenes.make(kind, seed)
        assert a.request == b.request and a.rooms == b.rooms  # same seed, same building
        req = parse_request(a.request)
        same = [r for r in a.rooms.values() if r.color == req.color]
        assert len(same) >= 2, "the words must need a relation to pick a room"
        slots = [(r.slot, r.side) for r in a.rooms.values()]
        assert len(slots) == len(set(slots)), "one room per slot and side"
        for r in a.rooms.values():  # every door sits inside its slot, west of the dog
            lo = dog_scenes.FIRST + SLOT * r.slot
            assert lo + 0.5 < r.door_x < lo + SLOT - 0.5
            assert r.door_x + 1.0 < a.start[0]
        assert abs(a.ff[1]) < HALF_W
    assert len({dog_scenes.make(kind, s).request for s in range(30)}) > 3  # they do vary


def test_kinds_set_up_their_disagreement():
    c = dog_scenes.make("corridor", 0)
    assert c.ff[2] == 0.0 and c.start[2] == np.pi  # face to face
    d = dog_scenes.make("doorway", 0)
    assert abs(abs(d.ff[2]) - np.pi / 2) < 1e-9  # facing across the corridor
    same = [r for r in d.rooms.values() if r.color == parse_request(d.request).color]
    assert len({r.side for r in same}) == 1 and np.sign(d.ff[1]) == -same[0].side
    assert d.lane == pytest.approx(-dog_scenes.LANE * np.sign(d.ff[1]))
    k = dog_scenes.make("closest", 0)
    assert parse_request(k.request).relation in ("nearest", "farthest")


def test_voice_knows_the_colour_rooms():
    reqs = voice.requests("dog")
    for kind in NEW_KINDS:
        for seed in range(10):
            words = dog_scenes.make(kind, seed).request.lower().rstrip(".")
            assert words in reqs, words
    assert voice.to_command("curio next scene", "dog") == "n"


@needs_go2
@pytest.mark.parametrize("kind", [*NEW_KINDS, "rescue"])
def test_curio_on_picks_the_meant_room(kind):
    for seed in range(4):
        w = dog.DogWorld(kind, seed)
        on = dog.decide(w, w.request, True)
        assert on.verdict == "correct", (kind, seed, on.target, on.oracle_why)
        w.close()


@needs_go2
def test_survivor_is_in_the_meant_room():
    w = dog.DogWorld("corridor", 0)
    rid, _ = dog.oracle_room(w.request, w)
    xy = w.data.joint("s_free").qpos[:2]
    r = w.layout.rooms[rid]
    assert np.sign(xy[1]) == r.side and abs(xy[0] - r.door_x) < SLOT / 2
    w.close()


@needs_go2
def test_walk_and_switch_scene_in_the_demo():
    sys.path.insert(0, str(ROOT / "scripts"))
    from dog_demo_gate import HeadlessViewer

    w = dog.DogWorld("doorway", 1)
    demo = dog.DogDemo(HeadlessViewer(), w)
    demo.paced = False
    demo.load()
    demo.command("")
    for _ in range(600):
        demo.tick(0.0)
        demo.draw(0.0)
        if not w.walking:
            break
    assert not w.fell and demo.result_line == "in the room the firefighter meant"
    demo.command("scene closest 3")
    assert demo.switch_to == ("closest", 3)
    demo.switch_to = None
    demo.command("scene nowhere 1")
    assert demo.switch_to is None
    w.close()
