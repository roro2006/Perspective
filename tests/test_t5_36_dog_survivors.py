"""T5-36: survivor variety. Drawing survivors is plain numpy; placing and walking need the
Go2 assets and skip without them."""

from __future__ import annotations

import numpy as np
import pytest

from curio_robot.demo import dog, dog_scenes
from curio_robot.demo.dog_scenes import DOG_HALF, NEW_KINDS, POSES, SITTER_HALF, SLOT

needs_go2 = pytest.mark.skipif(not dog.have_assets(), reason="Go2 assets not downloaded")


@pytest.mark.parametrize("kind", NEW_KINDS)
def test_survivors_vary_and_are_seeded(kind):
    counts, outfits, poses = set(), set(), set()
    for seed in range(50):
        a, b = dog_scenes.make(kind, seed), dog_scenes.make(kind, seed)
        assert a.survivors == b.survivors
        meant = [sv for sv in a.survivors if sv.meant]
        assert 1 <= len(meant) <= 3 and len(a.survivors) - len(meant) <= 1
        assert len({sv.prefix for sv in a.survivors}) == len(a.survivors)
        assert meant[0].prefix == "s_"
        counts.add(len(meant))
        outfits |= {sv.outfit.name for sv in a.survivors}
        poses |= {sv.pose for sv in a.survivors}
    assert counts == {1, 2, 3} and len(outfits) >= 5 and poses == set(POSES)


def test_survivors_draw_from_their_own_stream():
    """Adding survivors must not move rooms or change words (T5-35's results stand)."""
    for kind in NEW_KINDS:
        gen = getattr(dog_scenes, kind)
        for seed in range(20):
            plain, full = gen(seed), dog_scenes.make(kind, seed)
            assert plain.rooms == full.rooms and plain.request == full.request


def test_spots_keep_off_the_route():
    for kind in NEW_KINDS:
        for seed in range(30):
            lay = dog_scenes.make(kind, seed)
            for rid in lay.rooms:
                pts, goal = lay.route(rid)
                for _, room, pos, _, _ in lay.place_survivors(rid):
                    r = lay.rooms[room]
                    a = dog_scenes.FIRST + SLOT * r.slot
                    assert a < pos[0] < a + SLOT and np.sign(pos[1]) == r.side
                    if room == rid and pos[2] < 0.14:  # a sitter: clear of the dog's path
                        for p in (pts[-2], goal):  # through the door and at its goal
                            gap = abs(pos[0] - p[0]) - SITTER_HALF - DOG_HALF
                            assert gap > 0.1, (kind, seed, rid, gap)
                    elif room == rid:  # lying at the back: well behind the goal
                        assert abs(pos[1]) - abs(goal[1]) > 0.8


@needs_go2
def test_survivors_follow_the_meaning_and_are_counted():
    w = dog.DogWorld("closest", 0)
    meant, _ = dog.oracle_room(w.request, w)
    n = sum(sv.meant for sv in w.layout.survivors)
    assert w.survivors_in(meant) == n
    assert "in " + meant in w.survivor_summary()
    w.close()
    w = dog.DogWorld("corridor", 2)
    before, _ = dog.oracle_room(w.request, w)
    w.turn_firefighter()
    after, _ = dog.oracle_room(w.request, w)
    assert before != after and w.survivors_in(after) == sum(sv.meant for sv in w.layout.survivors)
    w.close()


@needs_go2
def test_walk_into_a_crowded_room_without_touching_anyone():
    w = dog.DogWorld("corridor", 0)  # three survivors in the meant room
    assert sum(sv.meant for sv in w.layout.survivors) == 3
    d = dog.decide(w, w.request, True)
    w.go(d.target.object_id)
    while w.walking and w.sim_t < 40:
        w.step(0.1)
    assert dog.arrived_at(w, d.oracle_id) and not w.fell and w.survivor_contacts == 0
    w.close()
