"""T5-34: the live rescue-dog demo. The voice grammar and the geometry run anywhere; the scene
tests need the Go2 model and walking policy (third_party/, see its README) and skip without."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from curio_robot.demo import dog, voice
from curio_robot.resolver import parse_request

ROOT = Path(__file__).resolve().parents[1]
needs_go2 = pytest.mark.skipif(not dog.have_assets(), reason="Go2 assets not downloaded")


def test_dog_phrases_are_what_the_resolver_reads():
    reqs = voice.requests("dog")
    # T5-35 added colour rooms with closest / furthest; every phrase is still one the rule
    # resolver reads (a relation, and a colour when it names one)
    assert reqs and all(parse_request(r).relation in ("left", "right", "nearest", "farthest")
                        for r in reqs)
    assert all(p.startswith("curio ") for p in voice.phrases("dog"))
    assert voice.to_command("curio turn around", "dog") == "turn"
    assert voice.to_command("curio compare", "dog") == "c"
    assert voice.to_command("curio search the room on my left", "dog") == \
        "search the room on my left"
    # the arm's grammar is unchanged and doesn't hear the dog's words
    assert voice.phrases() == voice.phrases("arm")
    assert not any("room" in p for p in voice.phrases())
    assert voice.to_command("curio turn around") is not None  # still a plain request there


def test_left_of_yaw():
    assert np.allclose(dog.left_of(0.0), (0, 1))
    assert np.allclose(dog.left_of(np.pi), (0, -1), atol=1e-9)


@pytest.fixture(scope="module")
def world():
    w = dog.DogWorld()
    yield w
    w.close()


@needs_go2
def test_poses_face_each_other(world):
    world.facing = 0.0
    world.home()
    om = dog.room_map(world, True)
    assert np.allclose(om.human_pose.right, (0, -1, 0), atol=1e-3)  # firefighter's right is -y
    assert np.allclose(om.robot_pose.right, (0, 1, 0), atol=1e-3)  # the dog faces them
    assert {o.object_id for o in om.objects} == {"room_n", "room_s"}
    assert all(o.seen_by == {"map"} for o in om.objects)
    assert dog.room_map(world, False).human_pose is None


@needs_go2
@pytest.mark.parametrize("turned", [False, True])
def test_decisions_follow_the_firefighter(world, turned):
    world.facing = np.pi if turned else 0.0
    world.home()
    left = dog.decide(world, "Survivor's in the room on my left.", True)
    right = dog.decide(world, "Survivor's in the room on my right.", True)
    assert left.verdict == right.verdict == "correct"
    assert left.target.object_id == ("room_s" if turned else "room_n")
    assert right.target.object_id != left.target.object_id
    off = dog.decide(world, "Survivor's in the room on my left.", False)
    assert off.target.object_id == "room_s"  # the dog's own left, whichever way they face
    assert off.verdict == ("correct" if turned else "wrong")
    assert dog.decide(world, "Survivor's in the room.", True).verdict == "unknown"
    world.facing = 0.0
    world.home()


@needs_go2
def test_turn_moves_the_survivor_and_flips_the_truth(world):
    world.facing = 0.0
    world.home()
    a = dog.oracle_room("the room on my left", world)[0]
    y0 = float(world.data.joint("s_free").qpos[1])
    world.turn_firefighter()
    b = dog.oracle_room("the room on my left", world)[0]
    y1 = float(world.data.joint("s_free").qpos[1])
    assert (a, b) == ("room_n", "room_s")
    assert y0 > 0 > y1  # the survivor is in the room the firefighter's left points at
    world.turn_firefighter()
    assert world.facing == 0.0


@needs_go2
def test_demo_runs_a_request_end_to_end(world):
    sys.path.insert(0, str(ROOT / "scripts"))
    from dog_demo_gate import HeadlessViewer

    world.facing = 0.0
    v = HeadlessViewer()
    demo = dog.DogDemo(v, world)
    demo.paced = False
    demo.load()
    assert demo.command("")  # the scene's own request
    assert world.walking and demo.decision.target.object_id == "room_n"
    demo.command("turn")  # refused while the dog walks
    assert world.facing == 0.0
    for _ in range(400):
        demo.tick(0.0)
        demo.draw(0.0)
        if not world.walking:
            break
    assert not world.walking and not world.fell
    assert dog.arrived_at(world, "room_n")
    assert demo.result_line == "in the room the firefighter meant"
    assert v.images and v.texts
    assert "no VLM" in v.texts[-1][3] and "cagataydev" in v.texts[-1][3]
    assert not demo.command("q")
